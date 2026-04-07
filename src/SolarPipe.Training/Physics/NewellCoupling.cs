using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Training.Physics;

// Newell coupling function: dΦMP/dt = v^(4/3) × B_T^(2/3) × sin^(8/3)(θ_c/2)
// Reference: Newell et al. (2007), JGR 112, A01206.
// RULE-031: Bz inputs must be GSM-frame — enforced by column name assertion.
// RULE-032: Input ranges validated at entry.
//
// Physical interpretation:
//   - Φ: dayside reconnection rate (Wb/s ≈ V)
//   - v: solar wind speed (km/s)
//   - B_T = sqrt(By^2 + Bz^2): IMF transverse magnitude (nT)
//   - θ_c = clock angle = atan2(By_gsm, Bz_gsm): IMF orientation
//   - sin^(8/3)(θ_c/2) → 0 northward (θ=0°), max southward (θ=180°)
//
// Normalization constant k = 2.11 (Newell et al. 2007 eq. 4)
// When k included: coupling in mWb/s. Without k: raw dimensionless coupling.
public sealed class NewellCoupling : ITrainedModel
{
    // Newell et al. 2007 normalization constant (eq. 4)
    // Units: Wb/s per (km/s)^(4/3) per nT^(2/3) — omitted for dimensionless form
    private const double NormalizationK = 2.11;

    // Exponents from Newell et al. (2007)
    private const double ExpV = 4.0 / 3.0;
    private const double ExpBt = 2.0 / 3.0;
    private const double ExpSin = 8.0 / 3.0;

    private bool _applyNormalization;

    public string ModelId { get; }
    public string StageName { get; }
    public ModelMetrics Metrics => new(0.0, 0.0, 0.0);

    public NewellCoupling(StageConfig config)
    {
        ModelId = $"newell_coupling_{config.Name}_{DateTime.UtcNow:yyyyMMddHHmmss}";
        StageName = config.Name;
        ExtractHyperparameters(config.Hyperparameters);
    }

    private NewellCoupling(string modelId, string stageName, bool applyNorm)
    {
        ModelId = modelId;
        StageName = stageName;
        _applyNormalization = applyNorm;
    }

    // PredictAsync: computes Newell coupling for each row.
    // Required columns: by_gsm (nT), bz_gsm (nT), v_km_s (km/s)
    // Output: coupling rate (Wb/s or dimensionless)
    public Task<PredictionResult> PredictAsync(IDataFrame input, CancellationToken ct)
    {
        var byCol = FindColumn(input, "by_gsm", "by_gsm_nt", "By_gsm");
        var bzCol = FindColumn(input, "bz_gsm", "bz_gsm_nt", "Bz_gsm");
        var vCol = FindColumn(input, "v_km_s", "speed_km_s", "v_sw_km_s");

        float[] byArr = input.GetColumn(byCol);
        float[] bzArr = input.GetColumn(bzCol);
        float[] vArr = input.GetColumn(vCol);

        if (byArr.Length != bzArr.Length || byArr.Length != vArr.Length)
            throw new ArgumentException(
                $"NewellCoupling: column length mismatch " +
                $"(by={byArr.Length}, bz={bzArr.Length}, v={vArr.Length}) " +
                $"(stage={StageName}).");

        int n = byArr.Length;
        var results = new float[n];

        for (int i = 0; i < n; i++)
        {
            ct.ThrowIfCancellationRequested();

            float by = byArr[i];
            float bz = bzArr[i];
            float v = vArr[i];

            if (float.IsNaN(by) || float.IsNaN(bz) || float.IsNaN(v))
            {
                results[i] = float.NaN;
                continue;
            }

            // RULE-032: validate solar wind speed
            if (v < 200f || v > 3000f)
                throw new ArgumentOutOfRangeException(vCol,
                    $"Solar wind speed {v} km/s outside [200, 3000] at row {i} (stage={StageName}).");

            results[i] = (float)Compute(v, by, bz, _applyNormalization);
        }

        return Task.FromResult(new PredictionResult(
            results, null, null, ModelId, DateTime.UtcNow,
            new Dictionary<string, object> { ["apply_normalization"] = _applyNormalization }));
    }

    // Compute coupling for a single observation.
    // v: solar wind speed (km/s), by: IMF By GSM (nT), bz: IMF Bz GSM (nT)
    // Returns coupling rate (Wb/s when normalized, dimensionless otherwise).
    public static double Compute(double vKmPerSec, double byGsmNt, double bzGsmNt, bool normalize = true)
    {
        // B_T: transverse IMF magnitude
        double bT = Math.Sqrt(byGsmNt * byGsmNt + bzGsmNt * bzGsmNt);

        if (bT < 1e-9)
            return 0.0; // negligible IMF → no reconnection

        // Clock angle θ_c = atan2(By, Bz) in GSM
        double thetaC = Math.Atan2(byGsmNt, bzGsmNt);

        // sin(θ_c / 2)
        double sinHalfTheta = Math.Sin(thetaC / 2.0);

        // Coupling: v^(4/3) × B_T^(2/3) × |sin(θ_c/2)|^(8/3)
        // Use absolute value to ensure real-valued result
        double coupling = Math.Pow(vKmPerSec, ExpV)
                        * Math.Pow(bT, ExpBt)
                        * Math.Pow(Math.Abs(sinHalfTheta), ExpSin);

        return normalize ? NormalizationK * coupling : coupling;
    }

    public Task SaveAsync(string path, CancellationToken ct)
    {
        var state = new NewellState(ModelId, StageName, _applyNormalization);
        var json = System.Text.Json.JsonSerializer.Serialize(state,
            new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
        string tempPath = Path.Combine(Path.GetDirectoryName(path)!, $".tmp_{Guid.NewGuid():N}");
        File.WriteAllText(tempPath, json);
        File.Move(tempPath, path, overwrite: true); // RULE-040: atomic write
        return Task.CompletedTask;
    }

    public Task LoadAsync(string path, CancellationToken ct) =>
        throw new NotSupportedException(
            $"Use NewellCoupling.FromFile() to load (stage={StageName}).");

    public static NewellCoupling FromFile(string path)
    {
        var json = File.ReadAllText(path);
        var state = System.Text.Json.JsonSerializer.Deserialize<NewellState>(json)
            ?? throw new InvalidOperationException($"Failed to deserialize NewellState from {path}.");
        return new NewellCoupling(state.ModelId, state.StageName, state.ApplyNormalization);
    }

    private void ExtractHyperparameters(IReadOnlyDictionary<string, object>? hp)
    {
        _applyNormalization = FindHyperBool(hp, "apply_normalization", defaultValue: true);
    }

    private static bool FindHyperBool(IReadOnlyDictionary<string, object>? hp, string key, bool defaultValue)
    {
        if (hp == null) return defaultValue;
        foreach (var kvp in hp)
            if (string.Equals(kvp.Key, key, StringComparison.OrdinalIgnoreCase))
                return Convert.ToBoolean(kvp.Value);
        return defaultValue;
    }

    private static string FindColumn(IDataFrame frame, params string[] candidates)
    {
        foreach (var name in candidates)
            if (frame.Schema.HasColumn(name))
                return name;
        throw new InvalidOperationException(
            $"NewellCoupling requires one of [{string.Join(", ", candidates)}] in the input DataFrame. " +
            $"Available: [{string.Join(", ", frame.Schema.Columns.Select(c => c.Name))}]. " +
            "By/Bz must be GSM-frame (RULE-031).");
    }

    private record NewellState(string ModelId, string StageName, bool ApplyNormalization);
}
