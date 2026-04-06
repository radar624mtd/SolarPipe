using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Data.Transforms;

// Solar wind–magnetosphere coupling functions as ITransform implementations.
// All operate on GSM-frame Bz (RULE-031).
//
// References:
//   Newell:   Newell et al. (2007), JGR 112, A01206
//   VBs:      Fairfield & Cahill (1966) / Burton et al. (1975)
//   Borovsky: Borovsky (2008), JGR 113, A08228

// ─── NewellCoupling ─────────────────────────────────────────────────────────────
// Φ = v^(4/3) × B_T^(2/3) × sin^(8/3)(θ/2)
// where θ = clock angle of IMF (atan2(By, Bz) in GSM), B_T = sqrt(By² + Bz²)
// Inputs: columns for solar wind speed (km/s), GSM By (nT), GSM Bz (nT).
// Output: coupling rate in Wb/s (Weber per second).
public sealed class NewellCouplingTransform : ITransform
{
    private readonly string _speedCol;
    private readonly string _byCol;
    private readonly string _bzCol;

    public string OutputColumnName { get; }

    public NewellCouplingTransform(string speedColumn, string byColumn, string bzColumn,
        string? outputColumn = null)
    {
        _speedCol = speedColumn ?? throw new ArgumentNullException(nameof(speedColumn));
        _byCol = byColumn ?? throw new ArgumentNullException(nameof(byColumn));
        _bzCol = bzColumn ?? throw new ArgumentNullException(nameof(bzColumn));
        OutputColumnName = outputColumn ?? "newell_coupling";
    }

    public IDataFrame Apply(IDataFrame input)
    {
        var v = input.GetColumn(_speedCol);
        var by = input.GetColumn(_byCol);
        var bz = input.GetColumn(_bzCol);

        ValidateLengths(v, by, bz, input.RowCount);

        var result = new float[v.Length];
        for (int i = 0; i < v.Length; i++)
        {
            if (float.IsNaN(v[i]) || float.IsNaN(by[i]) || float.IsNaN(bz[i]))
            {
                result[i] = float.NaN;
                continue;
            }

            double bt = Math.Sqrt((double)by[i] * by[i] + (double)bz[i] * bz[i]);
            if (bt < 1e-10) { result[i] = 0f; continue; }

            // Clock angle θ: atan2(By, Bz) — southward Bz → θ ~ π, maximises sin^8/3
            double theta = Math.Atan2(by[i], bz[i]);
            double sinHalf = Math.Abs(Math.Sin(theta / 2.0));

            // Φ = v^(4/3) * Bt^(2/3) * sin^(8/3)(θ/2)
            double phi = Math.Pow(v[i], 4.0 / 3.0)
                       * Math.Pow(bt, 2.0 / 3.0)
                       * Math.Pow(sinHalf, 8.0 / 3.0);

            result[i] = (float)phi;
        }
        return input.AddColumn(OutputColumnName, result);
    }

    private static void ValidateLengths(float[] v, float[] by, float[] bz, int expectedRows)
    {
        if (v.Length != expectedRows || by.Length != expectedRows || bz.Length != expectedRows)
            throw new InvalidOperationException(
                $"NewellCouplingTransform: column length mismatch (v={v.Length}, by={by.Length}, bz={bz.Length}, rows={expectedRows}).");
    }
}

// ─── VBsCoupling ────────────────────────────────────────────────────────────────
// E_KL = v × Bs  where Bs = max(0, -Bz_GSM) (southward component only, mV/m)
// Inputs: solar wind speed (km/s), GSM Bz (nT).
// Output: rectified dawn-to-dusk electric field in mV/m.
// Reference: Burton et al. (1975), multiplied by 1e-3 to get V/m if needed.
public sealed class VBsCouplingTransform : ITransform
{
    private readonly string _speedCol;
    private readonly string _bzCol;

    public string OutputColumnName { get; }

    public VBsCouplingTransform(string speedColumn, string bzColumn,
        string? outputColumn = null)
    {
        _speedCol = speedColumn ?? throw new ArgumentNullException(nameof(speedColumn));
        _bzCol = bzColumn ?? throw new ArgumentNullException(nameof(bzColumn));
        OutputColumnName = outputColumn ?? "vbs_coupling";
    }

    public IDataFrame Apply(IDataFrame input)
    {
        var v = input.GetColumn(_speedCol);
        var bz = input.GetColumn(_bzCol);

        if (v.Length != bz.Length)
            throw new InvalidOperationException(
                $"VBsCouplingTransform: speed column length {v.Length} != bz column length {bz.Length}.");

        var result = new float[v.Length];
        for (int i = 0; i < v.Length; i++)
        {
            if (float.IsNaN(v[i]) || float.IsNaN(bz[i])) { result[i] = float.NaN; continue; }
            // Bs = rectified southward component (positive means southward)
            float bs = Math.Max(0f, -bz[i]);
            // v is in km/s, bz in nT → v*Bs in km/s · nT = 1e3 m/s · 1e-9 T = 1e-6 V/m = 1 µV/m
            // Conventional VBs in mV/m = v[km/s] × Bs[nT] × 1e-3
            result[i] = v[i] * bs * 1e-3f;
        }
        return input.AddColumn(OutputColumnName, result);
    }
}

// ─── BorovskyCoupling ────────────────────────────────────────────────────────────
// Borovsky (2008): Φ_B = v × B_T × sin^4(θ/2) × f(ρ_sw)
// where f(ρ) = sqrt(ρ/ρ_ref), ρ_ref = 7 cm⁻³ (nominal solar wind density)
// Inputs: speed (km/s), By (nT), Bz (nT), proton density (cm⁻³).
// Output: coupling parameter (dimensionless relative to reference).
public sealed class BorovskyCouplingTransform : ITransform
{
    private const float ReferenceDensityCm3 = 7.0f; // O'Brien & McPherron 2000 reference
    private readonly string _speedCol;
    private readonly string _byCol;
    private readonly string _bzCol;
    private readonly string _densityCol;

    public string OutputColumnName { get; }

    public BorovskyCouplingTransform(string speedColumn, string byColumn, string bzColumn,
        string densityColumn, string? outputColumn = null)
    {
        _speedCol = speedColumn ?? throw new ArgumentNullException(nameof(speedColumn));
        _byCol = byColumn ?? throw new ArgumentNullException(nameof(byColumn));
        _bzCol = bzColumn ?? throw new ArgumentNullException(nameof(bzColumn));
        _densityCol = densityColumn ?? throw new ArgumentNullException(nameof(densityColumn));
        OutputColumnName = outputColumn ?? "borovsky_coupling";
    }

    public IDataFrame Apply(IDataFrame input)
    {
        var v = input.GetColumn(_speedCol);
        var by = input.GetColumn(_byCol);
        var bz = input.GetColumn(_bzCol);
        var rho = input.GetColumn(_densityCol);

        if (v.Length != by.Length || v.Length != bz.Length || v.Length != rho.Length)
            throw new InvalidOperationException(
                $"BorovskyCouplingTransform: column length mismatch among {_speedCol}, {_byCol}, {_bzCol}, {_densityCol}.");

        var result = new float[v.Length];
        for (int i = 0; i < v.Length; i++)
        {
            if (float.IsNaN(v[i]) || float.IsNaN(by[i]) || float.IsNaN(bz[i]) || float.IsNaN(rho[i]))
            {
                result[i] = float.NaN;
                continue;
            }

            double bt = Math.Sqrt((double)by[i] * by[i] + (double)bz[i] * bz[i]);
            if (bt < 1e-10) { result[i] = 0f; continue; }

            double theta = Math.Atan2(by[i], bz[i]);
            double sinHalf = Math.Abs(Math.Sin(theta / 2.0));
            double densityFactor = Math.Sqrt(Math.Max(0.0, rho[i]) / ReferenceDensityCm3);

            double phi = v[i] * bt * Math.Pow(sinHalf, 4.0) * densityFactor;
            result[i] = (float)phi;
        }
        return input.AddColumn(OutputColumnName, result);
    }
}
