using SolarPipe.Core.Domain;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Training.Physics;

// Drag-Based Model (DBM) for CME propagation.
// ODE: dv/dt = -γ(v - w)|v - w|  (Vrsnak et al. 2013, A&A 512)
// RULE-030: Uses Dormand-Prince RK4(5) adaptive solver.
// RULE-031: Bz inputs must be GSM-frame (enforced via column name requirement).
// RULE-032: Parameter ranges validated at entry.
public sealed class DragBasedModel : ITrainedModel
{
    private readonly StageConfig _config;

    // Extracted hyperparameters
    private double _gammaKmInv;
    private double _ambientWindKmPerSec;
    private double _startDistanceSolarRadii;
    private double _targetDistanceSolarRadii;

    public string ModelId { get; }
    public string StageName { get; }
    public ModelMetrics Metrics { get; private set; }

    public DragBasedModel(StageConfig config)
    {
        _config = config;
        ModelId = $"drag_based_{config.Name}_{DateTime.UtcNow:yyyyMMddHHmmss}";
        StageName = config.Name;
        Metrics = new ModelMetrics(0.0, 0.0, 0.0);

        ExtractHyperparameters(config.Hyperparameters);
    }

    // Internal constructor for LoadAsync
    private DragBasedModel(string modelId, string stageName, ModelMetrics metrics,
        double gamma, double ambientWind, double startDist, double targetDist)
    {
        _config = new StageConfig(stageName, "Physics", "DragBased", "", [], "");
        ModelId = modelId;
        StageName = stageName;
        Metrics = metrics;
        _gammaKmInv = gamma;
        _ambientWindKmPerSec = ambientWind;
        _startDistanceSolarRadii = startDist;
        _targetDistanceSolarRadii = targetDist;
    }

    // Runs ODE per row. Reads radial_speed_km_s column; outputs arrival_time_hours.
    public Task<PredictionResult> PredictAsync(IDataFrame input, CancellationToken ct)
    {
        var speedCol = FindColumn(input, "radial_speed_km_s", "v0_km_s", "speed_km_s", "cme_speed_kms");
        float[] speeds = input.GetColumn(speedCol);

        var results = new float[speeds.Length];
        for (int i = 0; i < speeds.Length; i++)
        {
            ct.ThrowIfCancellationRequested();

            float v0 = speeds[i];
            if (float.IsNaN(v0))
            {
                results[i] = float.NaN;
                continue;
            }

            // RULE-032: speeds outside physical range produce NaN (not throw) in batch mode.
            // Slow CMEs (<200 km/s) and extremely fast events (>3500 km/s) cannot be integrated
            // by this ODE — propagate as missing rather than aborting the whole batch.
            if (v0 < 200f || v0 > 3500f)
            {
                results[i] = float.NaN;
                continue;
            }

            var (_, arrivalHours) = RunOde(v0, _ambientWindKmPerSec, _gammaKmInv,
                _startDistanceSolarRadii, _targetDistanceSolarRadii);

            results[i] = (float)arrivalHours;
        }

        return Task.FromResult(new PredictionResult(
            results, null, null, ModelId, DateTime.UtcNow,
            new Dictionary<string, object>
            {
                ["gamma_km_inv"] = _gammaKmInv,
                ["ambient_wind_km_s"] = _ambientWindKmPerSec,
                ["start_dist_solar_radii"] = _startDistanceSolarRadii
            }));
    }

    public Task SaveAsync(string path, CancellationToken ct)
    {
        var state = new DragModelState(
            ModelId, StageName, _gammaKmInv, _ambientWindKmPerSec,
            _startDistanceSolarRadii, _targetDistanceSolarRadii,
            Metrics.Rmse, Metrics.Mae, Metrics.R2);

        var json = System.Text.Json.JsonSerializer.Serialize(state,
            new System.Text.Json.JsonSerializerOptions { WriteIndented = true });

        string tempPath = Path.Combine(Path.GetDirectoryName(path)!, $".tmp_{Guid.NewGuid():N}");
        File.WriteAllText(tempPath, json);
        File.Move(tempPath, path, overwrite: true); // RULE-040: atomic write
        return Task.CompletedTask;
    }

    public Task LoadAsync(string path, CancellationToken ct)
    {
        throw new NotSupportedException(
            "Use DragBasedModel.FromFile() to load — ITrainedModel.LoadAsync cannot mutate a sealed instance (stage=" + StageName + ").");
    }

    public static DragBasedModel FromFile(string path)
    {
        var json = File.ReadAllText(path);
        var state = System.Text.Json.JsonSerializer.Deserialize<DragModelState>(json)
            ?? throw new InvalidOperationException($"Failed to deserialize DragModelState from {path}.");

        return new DragBasedModel(state.ModelId, state.StageName,
            new ModelMetrics(state.Rmse, state.Mae, state.R2),
            state.GammaKmInv, state.AmbientWindKmPerSec,
            state.StartDistanceSolarRadii, state.TargetDistanceSolarRadii);
    }

    // Compute ODE result for a single CME.
    // Returns (tFinal, arrivalTimeHours).
    public static (double TFinal, double ArrivalTimeHours) RunOde(
        double v0KmPerSec,
        double wKmPerSec,
        double gammaKmInv,
        double startSolarRadii,
        double targetSolarRadii)
    {
        // RULE-032: validate parameter ranges
        if (v0KmPerSec < 200.0 || v0KmPerSec > 3500.0)
            throw new ArgumentOutOfRangeException(nameof(v0KmPerSec),
                $"Initial speed {v0KmPerSec} km/s outside [200, 3500] (stage=DragBasedModel).");
        if (gammaKmInv < 0.19e-7 || gammaKmInv > 2.1e-7)
            throw new ArgumentOutOfRangeException(nameof(gammaKmInv),
                $"Drag coefficient {gammaKmInv:E3} km⁻¹ outside [0.2e-7, 2e-7] (stage=DragBasedModel).");
        if (startSolarRadii < 20.0)
            throw new ArgumentOutOfRangeException(nameof(startSolarRadii),
                $"Start distance {startSolarRadii} R☉ must be ≥ 20 (stage=DragBasedModel).");

        // ODE state: y[0] = distance (solar radii), y[1] = speed (km/s)
        // dy[0]/dt = y[1]  (in units of solar_radii/hour)
        // dy[1]/dt = -γ * (y[1] - w) * |y[1] - w|  (km/s per hour)
        //
        // Unit conversion: 1 solar radius = 695700 km, time in hours
        // dy[0]/dt [R☉/h] = y[1] [km/s] * 3600 [s/h] / 695700 [km/R☉]
        const double kmPerHourPerSolarRadius = 3600.0 / PhysicalConstants.SolarRadiusKm;

        double[] y0 = [startSolarRadii, v0KmPerSec];
        double tEnd = 200.0; // max 200 hours (~8 days), enough for any CME

        double[] yFinal = [];

        // Terminate early when CME reaches target distance
        // We integrate to tEnd but check for early termination via event detection
        // by using short segments and checking after each segment.
        double t = 0.0;
        double[] yCurr = y0;
        double h0 = 0.1; // initial step 0.1 hours

        // Segment integration with target-crossing detection
        double tArrival = double.NaN;
        double vArrival = double.NaN;

        while (t < tEnd)
        {
            double tSegEnd = Math.Min(t + 1.0, tEnd); // 1-hour segments for crossing detection

            var (tNew, yNew) = DormandPrinceSolver.IntegrateVector(
                (tau, yv) =>
                {
                    double v = yv[1];
                    double dDist = v * kmPerHourPerSolarRadius;
                    double dSpeed = -gammaKmInv * (v - wKmPerSec) * Math.Abs(v - wKmPerSec) * 3600.0;
                    // RULE-121: NaN propagation guard
                    if (double.IsNaN(dDist) || double.IsNaN(dSpeed))
                        throw new InvalidOperationException(
                            $"NaN in drag ODE at t={tau:F3}h, v={v:F1} km/s (stage=DragBasedModel).");
                    return [dDist, dSpeed];
                },
                t, tSegEnd, yCurr, h0);

            // Check for target crossing between yCurr and yNew
            if (yCurr[0] < targetSolarRadii && yNew[0] >= targetSolarRadii)
            {
                // Linear interpolation for crossing time
                double frac = (targetSolarRadii - yCurr[0]) / (yNew[0] - yCurr[0]);
                tArrival = t + frac * (tNew - t);
                vArrival = yCurr[1] + frac * (yNew[1] - yCurr[1]);
                break;
            }

            t = tNew;
            yCurr = yNew;
            h0 = 0.1; // reset hint for next segment
        }

        if (double.IsNaN(tArrival))
        {
            // Never reached target — return final state time as upper bound
            tArrival = t;
            vArrival = yCurr.Length > 1 ? yCurr[1] : double.NaN;
        }

        return (t, tArrival);
    }

    // Like RunOde but returns (ArrivalTimeHours, ArrivalSpeedKms) instead of (tFinal, tArrival).
    // Used by DragBasedModelWithArrivalSpeed to generate Domain 1 pseudo-labels.
    internal static (double ArrivalHours, double ArrivalSpeedKms) RunOdeWithArrivalSpeed(
        double v0KmPerSec,
        double wKmPerSec,
        double gammaKmInv,
        double startSolarRadii,
        double targetSolarRadii)
    {
        var (_, arrivalHours) = RunOde(v0KmPerSec, wKmPerSec, gammaKmInv, startSolarRadii, targetSolarRadii);

        // Re-run to capture vArrival — same ODE, same crossing logic.
        const double kmPerHourPerSolarRadius = 3600.0 / PhysicalConstants.SolarRadiusKm;
        double[] yCurr = [startSolarRadii, v0KmPerSec];
        double t = 0.0;
        double h0 = 0.1;
        double vArrival = v0KmPerSec;

        while (t < 200.0)
        {
            double tSegEnd = Math.Min(t + 1.0, 200.0);
            var (tNew, yNew) = DormandPrinceSolver.IntegrateVector(
                (tau, yv) =>
                {
                    double v = yv[1];
                    double dDist  = v * kmPerHourPerSolarRadius;
                    double dSpeed = -gammaKmInv * (v - wKmPerSec) * Math.Abs(v - wKmPerSec) * 3600.0;
                    if (double.IsNaN(dDist) || double.IsNaN(dSpeed))
                        throw new InvalidOperationException(
                            $"NaN in drag ODE (arrival speed) at t={tau:F3}h (stage=DragBasedModel).");
                    return [dDist, dSpeed];
                },
                t, tSegEnd, yCurr, h0);

            if (yCurr[0] < targetSolarRadii && yNew[0] >= targetSolarRadii)
            {
                double frac = (targetSolarRadii - yCurr[0]) / (yNew[0] - yCurr[0]);
                vArrival = yCurr[1] + frac * (yNew[1] - yCurr[1]);
                break;
            }

            t = tNew;
            yCurr = yNew;
            h0 = 0.1;

            if (t >= 200.0)
            {
                vArrival = yCurr.Length > 1 ? yCurr[1] : double.NaN;
                break;
            }
        }

        return (arrivalHours, vArrival);
    }

    private void ExtractHyperparameters(IReadOnlyDictionary<string, object>? hp)
    {
        _gammaKmInv = FindHyperValue(hp, 0.5e-7, "gamma_km_inv", "drag_parameter");
        _ambientWindKmPerSec = FindHyperValue(hp, 400.0, "ambient_wind_km_s", "background_speed_km_s");
        _startDistanceSolarRadii = FindHyperValue(hp, 21.5, "start_distance_solar_radii", "r_start_rs");
        _targetDistanceSolarRadii = FindHyperValue(hp, 215.0, "target_distance_solar_radii", "r_stop_rs");
    }

    // RULE-012: OrdinalIgnoreCase lookup for hyperparameter maps.
    // Accepts multiple candidate keys to support both code-style and YAML-style naming.
    private static double FindHyperValue(
        IReadOnlyDictionary<string, object>? hp, double defaultValue, params string[] keys)
    {
        if (hp == null) return defaultValue;
        foreach (var key in keys)
            foreach (var kvp in hp)
                if (string.Equals(kvp.Key, key, StringComparison.OrdinalIgnoreCase))
                    return Convert.ToDouble(kvp.Value);
        return defaultValue;
    }

    private static string FindColumn(IDataFrame frame, params string[] candidates)
    {
        foreach (var name in candidates)
            if (frame.Schema.HasColumn(name))
                return name;
        throw new InvalidOperationException(
            $"DragBasedModel requires one of [{string.Join(", ", candidates)}] in the input DataFrame. " +
            $"Available: [{string.Join(", ", frame.Schema.Columns.Select(c => c.Name))}].");
    }

    private record DragModelState(
        string ModelId,
        string StageName,
        double GammaKmInv,
        double AmbientWindKmPerSec,
        double StartDistanceSolarRadii,
        double TargetDistanceSolarRadii,
        double Rmse,
        double Mae,
        double R2);
}
