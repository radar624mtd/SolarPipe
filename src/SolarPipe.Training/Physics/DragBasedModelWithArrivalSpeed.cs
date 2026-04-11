using SolarPipe.Core.Domain;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Training.Physics;

// Variant of DragBasedModel that returns CME arrival speed at L1 (km/s) instead of transit time.
//
// Used by DomainFoldExecutor to generate physics-derived pseudo-labels for the Origination
// domain model. The Origination RF is trained to predict these pseudo-labels using source
// properties (SHARP, flare class, CME geometry) as features.
//
// The arrival speed is computed from the same Dormand-Prince ODE as DragBasedModel, via
// DragBasedModel.RunOdeWithArrivalSpeed. At the 215 R☉ crossing, velocity is linearly
// interpolated to give the L1 arrival speed in km/s.
//
// RULE-030: Uses Dormand-Prince RK4(5) via DragBasedModel.RunOdeWithArrivalSpeed.
// RULE-032: Speed range validated ([200, 3500] km/s); NaN returned on out-of-range input.
public sealed class DragBasedModelWithArrivalSpeed : ITrainedModel
{
    private readonly double _gammaKmInv;
    private readonly double _ambientWindKmPerSec;
    private readonly double _startDistanceSolarRadii;
    private readonly double _targetDistanceSolarRadii;

    public string ModelId { get; }
    public string StageName { get; }
    public ModelMetrics Metrics { get; }

    public DragBasedModelWithArrivalSpeed(StageConfig config)
    {
        StageName = config.Name;
        ModelId   = $"drag_v2_{config.Name}_{DateTime.UtcNow:yyyyMMddHHmmss}";
        Metrics   = new ModelMetrics(0.0, 0.0, 0.0);

        // Same hyperparameter keys as DragBasedModel for config compatibility.
        _gammaKmInv              = FindHyper(config.Hyperparameters, 0.2e-7, "gamma_km_inv", "drag_parameter");
        _ambientWindKmPerSec     = FindHyper(config.Hyperparameters, 400.0,  "ambient_wind_km_s", "background_speed_km_s");
        _startDistanceSolarRadii = FindHyper(config.Hyperparameters, 21.5,   "start_distance_solar_radii", "r_start_rs");
        _targetDistanceSolarRadii = FindHyper(config.Hyperparameters, 215.0, "target_distance_solar_radii", "r_stop_rs");
    }

    // PredictAsync: returns arrival speed at L1 (km/s) per row.
    // Input column: cme_speed_kms (or any DragBasedModel-compatible speed column name).
    // Output: float[] of arrival speeds; NaN for invalid inputs.
    public Task<PredictionResult> PredictAsync(IDataFrame input, CancellationToken ct)
    {
        var speedCol = FindColumn(input, "cme_speed_kms", "radial_speed_km_s", "v0_km_s", "speed_km_s");
        float[] speeds = input.GetColumn(speedCol);
        var results = new float[speeds.Length];

        for (int i = 0; i < speeds.Length; i++)
        {
            ct.ThrowIfCancellationRequested();

            float v0 = speeds[i];
            if (float.IsNaN(v0) || v0 < 200f || v0 > 3500f)
            {
                results[i] = float.NaN;
                continue;
            }

            try
            {
                var (_, arrivalSpeedKms) = DragBasedModel.RunOdeWithArrivalSpeed(
                    v0, _ambientWindKmPerSec, _gammaKmInv,
                    _startDistanceSolarRadii, _targetDistanceSolarRadii);

                results[i] = double.IsNaN(arrivalSpeedKms) ? float.NaN : (float)arrivalSpeedKms;
            }
            catch
            {
                results[i] = float.NaN;
            }
        }

        return Task.FromResult(new PredictionResult(results, null, null, ModelId, DateTime.UtcNow));
    }

    public Task SaveAsync(string path, CancellationToken ct) => Task.CompletedTask;
    public Task LoadAsync(string path, CancellationToken ct) => Task.CompletedTask;

    // ── Helpers ──────────────────────────────────────────────────────────────

    private static string FindColumn(IDataFrame frame, params string[] candidates)
    {
        foreach (var c in candidates)
            if (frame.Schema.HasColumn(c)) return c;
        throw new InvalidOperationException(
            $"DragBasedModelWithArrivalSpeed: none of [{string.Join(", ", candidates)}] found in frame " +
            $"(columns: [{string.Join(", ", frame.Schema.Columns.Select(c => c.Name))}]).");
    }

    private static double FindHyper(IReadOnlyDictionary<string, object>? hp, double def, params string[] keys)
    {
        if (hp is null) return def;
        foreach (var k in keys)
            if (hp.TryGetValue(k, out var v))
                return Convert.ToDouble(v);
        return def;
    }
}
