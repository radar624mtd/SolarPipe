using SolarPipe.Core.Interfaces;

namespace SolarPipe.Training.Features;

// Adds binary missingness indicator columns (0.0 = absent, 1.0 = present) alongside NaN values.
//
// These indicators allow ML models to distinguish "unknown" from "zero" for high-missingness
// feature groups. The indicators are ordinary float columns so no special adapter support is needed.
//
// Missingness groups:
//   has_sharp_obs  — 1 if usflux is non-NaN (SHARP keywords present, ~73% in staging)
//   has_flare_obs  — 1 if flare_class_numeric is non-NaN (~32% in staging)
//   has_bz_obs     — 1 if sw_bz_ambient is non-NaN (~87% in staging)
//   has_mass_obs   — 1 if cme_mass_grams is non-NaN (always 0 in staging; reserved for future)
public sealed class MissingnessFeatureEnricher
{
    // Sentinel column for SHARP group — if this is present, all SHARP keywords were retrieved.
    private const string SharpSentinel = "usflux";

    // Returns a new IDataFrame with four boolean float columns appended.
    // Idempotent: if a column already exists, it is NOT re-added (returns input unchanged for that column).
    public IDataFrame Enrich(IDataFrame input)
    {
        var frame = input;
        frame = AddPresenceIfMissing(frame, "has_sharp_obs", SharpSentinel);
        frame = AddPresenceIfMissing(frame, "has_flare_obs", "flare_class_numeric");
        frame = AddPresenceIfMissing(frame, "has_bz_obs",    "sw_bz_ambient");
        frame = AddPresenceIfMissing(frame, "has_mass_obs",  "cme_mass_grams");
        return frame;
    }

    private static IDataFrame AddPresenceIfMissing(IDataFrame frame, string indicatorName, string sourceColumn)
    {
        if (frame.Schema.HasColumn(indicatorName))
            return frame;

        var indicator = ComputePresence(frame, sourceColumn);
        return frame.AddColumn(indicatorName, indicator);
    }

    private static float[] ComputePresence(IDataFrame frame, string col)
    {
        if (!frame.Schema.HasColumn(col))
            return new float[frame.RowCount]; // all zeros — column not present at all

        var data = frame.GetColumn(col);
        var result = new float[data.Length];
        for (int i = 0; i < data.Length; i++)
            result[i] = float.IsNaN(data[i]) ? 0f : 1f;
        return result;
    }
}
