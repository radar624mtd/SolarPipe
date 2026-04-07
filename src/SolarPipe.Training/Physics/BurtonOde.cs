using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Training.Physics;

// Burton ODE: dDst*/dt = Q(t) - Dst*/τ(VBs)
// Reference: O'Brien & McPherron (2000), JGR 105, 7707–7719.
// RULE-030: Uses Dormand-Prince RK4(5) adaptive solver.
// RULE-031: Bz values must be GSM-frame — enforced by column name assertion.
// RULE-032: Input ranges validated at entry.
//
// O'Brien & McPherron (2000) Table 1 coefficients:
//   Q activates when VBs > VBs_threshold (0.5 mV/m)
//   Q(VBs) = α × (VBs − VBs_threshold)    α ≈ −4.4 nT/hr per mV/m
//   τ = a × Dst* + b    a = 0.060, b = 16.65 hr   (floor: τ_min = 6.15 hr)
//   Pressure correction: Dst* = Dst − p_b × √Pdyn + p_c
//     p_b = 7.26 nT/√nPa, p_c = 11.0 nT
public sealed class BurtonOde : ITrainedModel
{
    // O'Brien & McPherron (2000) Table 1
    private const double Alpha = -4.4;              // nT/hr per mV/m
    private const double VBsThresholdMvPerM = 0.5; // mV/m
    private const double TauA = 0.060;              // hr/nT
    private const double TauB = 16.65;              // hr
    private const double TauMinHours = 6.15;        // hr (floor)
    private const double PressureB = 7.26;          // nT/√nPa
    private const double PressureC = 11.0;          // nT

    // Hyperparameter defaults
    private const double DefaultDst0Nt = 0.0;
    private const double DefaultDtHours = 1.0;
    private const double DefaultPdynNpa = 2.0;

    private double _dst0Nt;
    private double _dtHours;
    private double _pdynNpa;

    public string ModelId { get; }
    public string StageName { get; }
    public ModelMetrics Metrics => new(0.0, 0.0, 0.0);

    public BurtonOde(StageConfig config)
    {
        ModelId = $"burton_ode_{config.Name}_{DateTime.UtcNow:yyyyMMddHHmmss}";
        StageName = config.Name;
        ExtractHyperparameters(config.Hyperparameters);
    }

    // Private constructor for deserialization
    private BurtonOde(string modelId, string stageName, double dst0, double dtHours, double pdyn)
    {
        ModelId = modelId;
        StageName = stageName;
        _dst0Nt = dst0;
        _dtHours = dtHours;
        _pdynNpa = pdyn;
    }

    // PredictAsync: integrates Burton ODE for each VBs time series in input.
    // Required columns: bz_gsm (nT), v_km_s (km/s) — computes VBs = v * |Bz_gsm| * 1e-3 (mV/m)
    // Output: Dst_min (nT) per row (minimum Dst* reached during integration).
    public Task<PredictionResult> PredictAsync(IDataFrame input, CancellationToken ct)
    {
        var bzCol = FindColumn(input, "bz_gsm", "bz_gsm_nt", "Bz_gsm");
        var vCol = FindColumn(input, "v_km_s", "speed_km_s", "v_sw_km_s");

        float[] bzArr = input.GetColumn(bzCol);
        float[] vArr = input.GetColumn(vCol);

        if (bzArr.Length != vArr.Length)
            throw new ArgumentException(
                $"BurtonOde: column length mismatch: {bzCol}={bzArr.Length} vs {vCol}={vArr.Length} " +
                $"(stage={StageName}).");

        int n = bzArr.Length;
        var results = new float[n];

        // Build piecewise constant VBs time series from the data columns
        for (int i = 0; i < n; i++)
        {
            ct.ThrowIfCancellationRequested();

            float bz = bzArr[i];
            float v = vArr[i];

            if (float.IsNaN(bz) || float.IsNaN(v))
            {
                results[i] = float.NaN;
                continue;
            }

            // RULE-032: validate solar wind speed
            if (v < 200f || v > 3000f)
                throw new ArgumentOutOfRangeException(vCol,
                    $"Solar wind speed {v} km/s outside [200, 3000] at row {i} (stage={StageName}).");

            // VBs [mV/m] = v [km/s] × |Bz_gsm| [nT] × 1e-3  (only when Bz southward)
            double vBsMvPerM = bz < 0.0f ? (double)v * Math.Abs(bz) * 1e-3 : 0.0;

            double dstMin = IntegrateOde(vBsMvPerM, _dst0Nt, _dtHours, _pdynNpa);
            results[i] = (float)dstMin;
        }

        return Task.FromResult(new PredictionResult(
            results, null, null, ModelId, DateTime.UtcNow,
            new Dictionary<string, object>
            {
                ["dst0_nt"] = _dst0Nt,
                ["dt_hours"] = _dtHours,
                ["pdyn_npa"] = _pdynNpa
            }));
    }

    // IntegrateOde: integrate Burton ODE for a single constant VBs value over _dtHours.
    // Returns the minimum Dst* reached (most negative).
    internal double IntegrateOde(double vBsMvPerM, double dst0Nt, double durationHours, double pdynNpa)
    {
        // Pressure correction: Dst* = Dst - p_b*sqrt(Pdyn) + p_c
        // We track Dst* directly; initial Dst* from Dst0 via inverse:
        // Dst*_0 = Dst0 - p_b*sqrt(Pdyn) + p_c
        double dstStar0 = dst0Nt - PressureB * Math.Sqrt(pdynNpa) + PressureC;

        double dstMinStar = dstStar0;

        double dDstStarDt(double t, double dstStar)
        {
            // Injection function Q (activated when VBs > threshold)
            double q = vBsMvPerM > VBsThresholdMvPerM
                ? Alpha * (vBsMvPerM - VBsThresholdMvPerM)
                : 0.0;

            // Recovery time constant τ (floored at τ_min)
            double tau = Math.Max(TauMinHours, TauA * dstStar + TauB);
            if (tau <= 0.0) tau = TauMinHours;

            return q - dstStar / tau;
        }

        var (_, dstStarFinal) = DormandPrinceSolver.Integrate(
            dDstStarDt,
            t0: 0.0,
            tEnd: durationHours,
            y0: dstStar0,
            h0: 0.1);

        // Track minimum (step tracking not needed — we return endpoint for single-row prediction)
        if (dstStarFinal < dstMinStar)
            dstMinStar = dstStarFinal;

        // Convert Dst* back to Dst (reverse pressure correction)
        double dstFinal = dstMinStar + PressureB * Math.Sqrt(pdynNpa) - PressureC;
        return dstFinal;
    }

    // RunOde: static helper for direct ODE integration over a time series of VBs.
    // vBsTimeSeries: array of VBs values (mV/m) at each _dtHours interval.
    // Returns (Dst_min, Dst_final) both in nT.
    public static (double DstMinNt, double DstFinalNt) RunOdeTimeSeries(
        double[] vBsTimeSeries,
        double dst0Nt,
        double dtHours,
        double pdynNpa)
    {
        if (vBsTimeSeries.Length == 0)
            throw new ArgumentException("VBs time series must not be empty (stage=BurtonOde).");

        double dstStar = dst0Nt - PressureB * Math.Sqrt(pdynNpa) + PressureC;
        double dstMinStar = dstStar;

        for (int i = 0; i < vBsTimeSeries.Length; i++)
        {
            if (double.IsNaN(vBsTimeSeries[i]))
                continue;

            double vBs = vBsTimeSeries[i];

            double dDstStarDt(double t, double d)
            {
                double q = vBs > VBsThresholdMvPerM
                    ? Alpha * (vBs - VBsThresholdMvPerM)
                    : 0.0;
                double tau = Math.Max(TauMinHours, TauA * d + TauB);
                if (tau <= 0.0) tau = TauMinHours;
                return q - d / tau;
            }

            var (_, dstStarNext) = DormandPrinceSolver.Integrate(
                dDstStarDt, 0.0, dtHours, dstStar, h0: 0.1);

            dstStar = dstStarNext;
            if (dstStar < dstMinStar) dstMinStar = dstStar;
        }

        double pCorr = PressureB * Math.Sqrt(pdynNpa) - PressureC;
        return (dstMinStar + pCorr, dstStar + pCorr);
    }

    public Task SaveAsync(string path, CancellationToken ct)
    {
        var state = new BurtonState(ModelId, StageName, _dst0Nt, _dtHours, _pdynNpa);
        var json = System.Text.Json.JsonSerializer.Serialize(state,
            new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
        string tempPath = Path.Combine(Path.GetDirectoryName(path)!, $".tmp_{Guid.NewGuid():N}");
        File.WriteAllText(tempPath, json);
        File.Move(tempPath, path, overwrite: true); // RULE-040: atomic write
        return Task.CompletedTask;
    }

    public Task LoadAsync(string path, CancellationToken ct) =>
        throw new NotSupportedException(
            $"Use BurtonOde.FromFile() to load (stage={StageName}).");

    public static BurtonOde FromFile(string path)
    {
        var json = File.ReadAllText(path);
        var state = System.Text.Json.JsonSerializer.Deserialize<BurtonState>(json)
            ?? throw new InvalidOperationException($"Failed to deserialize BurtonState from {path}.");
        return new BurtonOde(state.ModelId, state.StageName, state.Dst0Nt, state.DtHours, state.PdynNpa);
    }

    private void ExtractHyperparameters(IReadOnlyDictionary<string, object>? hp)
    {
        _dst0Nt = FindHyperValue(hp, "dst0_nt", DefaultDst0Nt);
        _dtHours = FindHyperValue(hp, "dt_hours", DefaultDtHours);
        _pdynNpa = FindHyperValue(hp, "pdyn_npa", DefaultPdynNpa);
    }

    private static double FindHyperValue(IReadOnlyDictionary<string, object>? hp, string key, double def)
    {
        if (hp == null) return def;
        foreach (var kvp in hp)
            if (string.Equals(kvp.Key, key, StringComparison.OrdinalIgnoreCase))
                return Convert.ToDouble(kvp.Value);
        return def;
    }

    private static string FindColumn(IDataFrame frame, params string[] candidates)
    {
        foreach (var name in candidates)
            if (frame.Schema.HasColumn(name))
                return name;
        throw new InvalidOperationException(
            $"BurtonOde requires one of [{string.Join(", ", candidates)}] in the input DataFrame. " +
            $"Available: [{string.Join(", ", frame.Schema.Columns.Select(c => c.Name))}]. " +
            "Bz must be GSM-frame (RULE-031).");
    }

    private record BurtonState(
        string ModelId,
        string StageName,
        double Dst0Nt,
        double DtHours,
        double PdynNpa);
}
