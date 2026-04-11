using SolarPipe.Core.Interfaces;

namespace SolarPipe.Training.Features;

// Higher-order and derivative feature computation for all three CME prediction domains.
//
// All computations are NaN-safe: if any operand is NaN, the output is NaN.
// All methods return a new IDataFrame (AddColumn is immutable).
//
// Origination domain: CME source properties → higher-order magnetic and kinematic features.
// Transit domain:     Solar wind coupling → pressure and momentum-transfer proxies.
// Impact domain:      Magnetospheric coupling → ring-current and storm-driver proxies.
// Synthetic targets:  Storm duration from Burton ODE recovery formula.
public static class DomainFeatureTransforms
{
    // ── Origination features ──────────────────────────────────────────────────

    // Adds: magnetic_complexity_ratio, helicity_proxy, flare_speed_coupling, source_geometry_factor.
    // Reads: totpot, usflux, meanshr, totusjz, flare_class_numeric, cme_speed_kms,
    //        cme_latitude (degrees), cme_longitude (degrees).
    public static IDataFrame AddOriginationFeatures(IDataFrame frame)
    {
        int n = frame.RowCount;

        // magnetic_complexity_ratio = totpot / (usflux + ε)
        // Measures free energy density relative to total unsigned flux.
        float[] totpot = GetOrNaN(frame, "totpot", n);
        float[] usflux = GetOrNaN(frame, "usflux", n);
        var complexityRatio = new float[n];
        for (int i = 0; i < n; i++)
            complexityRatio[i] = (float.IsNaN(totpot[i]) || float.IsNaN(usflux[i]))
                ? float.NaN
                : totpot[i] / (usflux[i] + 1e-10f);

        // helicity_proxy = meanshr * totusjz
        // Product of mean shear angle and total unsigned current helicity.
        float[] meanshr = GetOrNaN(frame, "meanshr", n);
        float[] totusjz = GetOrNaN(frame, "totusjz", n);
        var helicityProxy = new float[n];
        for (int i = 0; i < n; i++)
            helicityProxy[i] = (float.IsNaN(meanshr[i]) || float.IsNaN(totusjz[i]))
                ? float.NaN
                : meanshr[i] * totusjz[i];

        // flare_speed_coupling = flare_class_numeric * cme_speed_kms
        // Scaled energy of CME launch event.
        float[] flareClass = GetOrNaN(frame, "flare_class_numeric", n);
        float[] cmeSpeed   = GetOrNaN(frame, "cme_speed_kms", n);
        var flareSpeedCoupling = new float[n];
        for (int i = 0; i < n; i++)
            flareSpeedCoupling[i] = (float.IsNaN(flareClass[i]) || float.IsNaN(cmeSpeed[i]))
                ? float.NaN
                : flareClass[i] * cmeSpeed[i];

        // source_geometry_factor = cos(lat * π/180) * clamp(1 - |lon| / 90, 0, 1)
        // Peaks for equatorial, central-meridian CMEs (most geoeffective geometry).
        float[] lat = GetOrNaN(frame, "cme_latitude", n);
        float[] lon = GetOrNaN(frame, "cme_longitude", n);
        var geometryFactor = new float[n];
        for (int i = 0; i < n; i++)
        {
            if (float.IsNaN(lat[i]) || float.IsNaN(lon[i]))
            {
                geometryFactor[i] = float.NaN;
                continue;
            }
            float cosLat  = (float)Math.Cos(lat[i] * Math.PI / 180.0);
            float lonFrac  = Math.Clamp(1f - Math.Abs(lon[i]) / 90f, 0f, 1f);
            geometryFactor[i] = cosLat * lonFrac;
        }

        return frame
            .AddColumn("magnetic_complexity_ratio", complexityRatio)
            .AddColumn("helicity_proxy",             helicityProxy)
            .AddColumn("flare_speed_coupling",       flareSpeedCoupling)
            .AddColumn("source_geometry_factor",     geometryFactor);
    }

    // ── Transit features ──────────────────────────────────────────────────────

    // Adds: dynamic_pressure_sw, newell_coupling_approx.
    // delta_v_kms is NOT added here; it requires pred_arrival_speed_kms from Domain 1 output
    // and is added by DomainFoldExecutor after D1 predictions are available.
    // Reads: sw_density_ambient, sw_speed_ambient, sw_bz_ambient.
    public static IDataFrame AddTransitFeatures(IDataFrame frame)
    {
        int n = frame.RowCount;

        // dynamic_pressure_sw = 1.67e-6 * density * speed²  (nPa)
        // Solar wind ram pressure; primary transit driver alongside CME speed.
        float[] density = GetOrNaN(frame, "sw_density_ambient", n);
        float[] speed   = GetOrNaN(frame, "sw_speed_ambient", n);
        var dynPressure = new float[n];
        for (int i = 0; i < n; i++)
            dynPressure[i] = (float.IsNaN(density[i]) || float.IsNaN(speed[i]))
                ? float.NaN
                : 1.67e-6f * density[i] * speed[i] * speed[i];

        // newell_coupling_approx = v^(4/3) * |Bz|^(2/3)
        // Approximation of Newell et al. (2007) coupling with By=0 (only Bz available).
        // Full formula: v^(4/3) * BT^(2/3) * |sin(θ/2)|^(8/3). When By=0, BT=|Bz|, sin(θ/2)=1 for Bz<0.
        float[] bz = GetOrNaN(frame, "sw_bz_ambient", n);
        var newellApprox = new float[n];
        for (int i = 0; i < n; i++)
        {
            if (float.IsNaN(speed[i]) || float.IsNaN(bz[i]))
            {
                newellApprox[i] = float.NaN;
                continue;
            }
            float absBz = Math.Abs(bz[i]);
            if (absBz < 1e-6f) { newellApprox[i] = 0f; continue; }
            newellApprox[i] = (float)(Math.Pow(speed[i], 4.0 / 3.0) * Math.Pow(absBz, 2.0 / 3.0));
        }

        return frame
            .AddColumn("dynamic_pressure_sw",     dynPressure)
            .AddColumn("newell_coupling_approx",   newellApprox);
    }

    // Adds: delta_v_kms — difference between predicted arrival speed and ambient wind.
    // Called after DomainFoldExecutor has appended pred_arrival_speed_kms to the frame.
    public static IDataFrame AddDeltaV(IDataFrame frame)
    {
        if (!frame.Schema.HasColumn("pred_arrival_speed_kms") ||
            !frame.Schema.HasColumn("sw_speed_ambient"))
            return frame;

        int n = frame.RowCount;
        float[] predSpeed = frame.GetColumn("pred_arrival_speed_kms");
        float[] swSpeed   = frame.GetColumn("sw_speed_ambient");
        var deltaV = new float[n];
        for (int i = 0; i < n; i++)
            deltaV[i] = (float.IsNaN(predSpeed[i]) || float.IsNaN(swSpeed[i]))
                ? float.NaN
                : predSpeed[i] - swSpeed[i];

        return frame.AddColumn("delta_v_kms", deltaV);
    }

    // ── Impact features ───────────────────────────────────────────────────────

    // Adds: vbs_coupling, ring_current_proxy.
    // Reads: sw_speed_ambient, sw_bz_ambient, sw_density_ambient.
    public static IDataFrame AddImpactFeatures(IDataFrame frame)
    {
        int n = frame.RowCount;

        // vbs_coupling = v * max(0, -Bz) * 1e-3  (mV/m)
        // Burton ODE driver: only activated by southward Bz. Zero for northward field.
        float[] speed = GetOrNaN(frame, "sw_speed_ambient", n);
        float[] bz    = GetOrNaN(frame, "sw_bz_ambient", n);
        var vbsCoupling = new float[n];
        for (int i = 0; i < n; i++)
            vbsCoupling[i] = (float.IsNaN(speed[i]) || float.IsNaN(bz[i]))
                ? float.NaN
                : speed[i] * Math.Max(0f, -bz[i]) * 1e-3f;

        // ring_current_proxy = Bz * density
        // Negative when southward + dense plasma: stronger ring-current injection.
        float[] density = GetOrNaN(frame, "sw_density_ambient", n);
        var ringCurrentProxy = new float[n];
        for (int i = 0; i < n; i++)
            ringCurrentProxy[i] = (float.IsNaN(bz[i]) || float.IsNaN(density[i]))
                ? float.NaN
                : bz[i] * density[i];

        return frame
            .AddColumn("vbs_coupling",       vbsCoupling)
            .AddColumn("ring_current_proxy", ringCurrentProxy);
    }

    // ── Synthetic target ──────────────────────────────────────────────────────

    // Adds: storm_duration_hours — derived from Burton ODE recovery formula.
    // τ = max(6.15, 0.060 * |Dst_min| + 16.65)  (O'Brien & McPherron 2000, eq. 9)
    // NaN when dst_min_nt is NaN. icme_end_time is entirely NULL in the DB.
    public static IDataFrame AddStormDuration(IDataFrame frame)
    {
        if (frame.Schema.HasColumn("storm_duration_hours"))
            return frame;

        if (!frame.Schema.HasColumn("dst_min_nt"))
            return frame.AddColumn("storm_duration_hours", new float[frame.RowCount].Select(_ => float.NaN).ToArray());

        int n = frame.RowCount;
        float[] dst = frame.GetColumn("dst_min_nt");
        var duration = new float[n];
        for (int i = 0; i < n; i++)
        {
            if (float.IsNaN(dst[i]))
            {
                duration[i] = float.NaN;
                continue;
            }
            float dstAbs = Math.Abs(dst[i]);
            duration[i] = Math.Max(6.15f, 0.060f * dstAbs + 16.65f);
        }

        return frame.AddColumn("storm_duration_hours", duration);
    }

    // ── Helper ────────────────────────────────────────────────────────────────

    private static float[] GetOrNaN(IDataFrame frame, string col, int n)
    {
        if (!frame.Schema.HasColumn(col))
        {
            var arr = new float[n];
            for (int i = 0; i < n; i++) arr[i] = float.NaN;
            return arr;
        }
        return frame.GetColumn(col);
    }
}
