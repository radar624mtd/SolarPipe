using Microsoft.Data.Sqlite;

namespace SolarPipe.Data;

// Phase 9: hourly L1 observation stream from solar_data.db:omni_hourly.
//
// Responsibilities:
//   - Enforce no-leakage time window: observations can only be read in
//     [launch_time, launch_time + min(72h, max_lookahead)]. Queries outside
//     that window throw.
//   - Map hour index (0 = launch_time hour) to (n_obs, v_obs, bz_gsm) tuples.
//   - Report coverage stats for the 20%-null fallback rule (§3.5).
//   - Drive the Phase 9 shock detector (§3.3 step 4): Δv ≥ 200 km/s AND
//     density ratio ≥ 3× hour-over-hour.
//
// Not an IDataFrame — this is a narrow, pre-loaded per-event observation window
// optimised for step-by-step propagation.
public sealed class L1ObservationStream
{
    public const double MaxWindowHours = 72.0;
    public const double ShockDeltaVKmPerSec = 200.0;
    public const double ShockDensityRatio = 3.0;

    private readonly DateTime _launchTime;
    private readonly DateTime _windowEnd;
    private readonly IReadOnlyList<L1Observation> _hourly;
    private readonly int _hoursCovered;
    private readonly int _hoursWithDensity;

    public DateTime LaunchTime => _launchTime;
    public DateTime WindowEnd => _windowEnd;
    public int HourCount => _hourly.Count;
    public int HoursWithDensity => _hoursWithDensity;
    public double DensityCoverage => _hourly.Count == 0 ? 0.0 : (double)_hoursWithDensity / _hourly.Count;

    private L1ObservationStream(DateTime launchTime, DateTime windowEnd, IReadOnlyList<L1Observation> hourly)
    {
        _launchTime = launchTime;
        _windowEnd = windowEnd;
        _hourly = hourly;
        _hoursCovered = hourly.Count;
        _hoursWithDensity = hourly.Count(o => o.ProtonDensity.HasValue);
    }

    // Load the stream for a single event. max_lookahead_hours is usually
    // min(72, labeled_transit * 1.1) during backtest (no leakage), or 72 during
    // live prediction (no label).
    public static L1ObservationStream LoadFromSqlite(
        string connectionString,
        DateTime launchTime,
        double maxLookaheadHours)
    {
        if (maxLookaheadHours <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(maxLookaheadHours),
                $"maxLookaheadHours must be > 0 (stage=L1ObservationStream).");

        double capped = Math.Min(maxLookaheadHours, MaxWindowHours);
        var windowStartHour = FloorToHour(launchTime);
        var windowEnd = windowStartHour.AddHours(capped);

        var rows = new List<L1Observation>();
        using var conn = new SqliteConnection(connectionString);
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            @"SELECT datetime, flow_speed, proton_density, Bz_GSM
              FROM omni_hourly
              WHERE datetime >= @start AND datetime < @end
              ORDER BY datetime ASC";
        cmd.Parameters.AddWithValue("@start", windowStartHour.ToString("yyyy-MM-dd HH:mm"));
        cmd.Parameters.AddWithValue("@end", windowEnd.ToString("yyyy-MM-dd HH:mm"));

        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            var dtStr = reader.GetString(0);
            var dt = DateTime.ParseExact(dtStr, "yyyy-MM-dd HH:mm", null);
            double? v   = reader.IsDBNull(1) ? null : reader.GetDouble(1);
            double? n   = reader.IsDBNull(2) ? null : reader.GetDouble(2);
            double? bz  = reader.IsDBNull(3) ? null : reader.GetDouble(3);
            rows.Add(new L1Observation(dt, v, n, bz));
        }

        // Build an index keyed by hour offset from launch hour. Missing hours are
        // represented by null entries so downstream code can count them as "fallback".
        int nHours = (int)Math.Round((windowEnd - windowStartHour).TotalHours);
        var byHour = new L1Observation?[nHours];
        foreach (var r in rows)
        {
            int idx = (int)Math.Round((r.Timestamp - windowStartHour).TotalHours);
            if (idx >= 0 && idx < nHours) byHour[idx] = r;
        }
        var dense = new L1Observation[nHours];
        for (int i = 0; i < nHours; i++)
        {
            dense[i] = byHour[i] ?? new L1Observation(windowStartHour.AddHours(i), null, null, null);
        }
        return new L1ObservationStream(launchTime, windowEnd, dense);
    }

    // Get observation at `hourIndex` (0 = launch hour). Throws if outside the allowed window.
    public L1Observation GetAtHour(int hourIndex)
    {
        if (hourIndex < 0)
            throw new InvalidOperationException(
                $"L1ObservationStream: hour index {hourIndex} is before launch (no-leakage guard; stage=L1ObservationStream).");
        if (hourIndex >= _hourly.Count)
            throw new InvalidOperationException(
                $"L1ObservationStream: hour index {hourIndex} exceeds window {_hourly.Count}h (no-leakage guard; stage=L1ObservationStream).");
        return _hourly[hourIndex];
    }

    public bool TryGetAtHour(int hourIndex, out L1Observation obs)
    {
        if (hourIndex < 0 || hourIndex >= _hourly.Count)
        {
            obs = default!;
            return false;
        }
        obs = _hourly[hourIndex];
        return true;
    }

    // Spec §3.3 step 4 — hour-over-hour shock detection.
    // Returns true iff both Δv ≥ 200 km/s and density ratio ≥ 3× between
    // hourIndex-1 and hourIndex. Missing values return false.
    public bool ShockDetectedAtHour(int hourIndex)
    {
        if (hourIndex <= 0 || hourIndex >= _hourly.Count) return false;
        var prev = _hourly[hourIndex - 1];
        var cur  = _hourly[hourIndex];
        if (!prev.FlowSpeed.HasValue || !cur.FlowSpeed.HasValue) return false;
        if (!prev.ProtonDensity.HasValue || !cur.ProtonDensity.HasValue) return false;
        if (prev.ProtonDensity.Value <= 0.0) return false;

        double dv = cur.FlowSpeed.Value - prev.FlowSpeed.Value;
        double ratio = cur.ProtonDensity.Value / prev.ProtonDensity.Value;
        return dv >= ShockDeltaVKmPerSec && ratio >= ShockDensityRatio;
    }

    // Maximum CME transit time in dataset (~6.3 days). Used as the pre-launch lookback
    // window so the ambient density profile captures all prior events that could have
    // modified the solar wind before the new CME launched.
    public const double PreLaunchLookbackHours = 150.0;

    // Compute the median proton density in [launchTime - lookbackHours, launchTime).
    // Missing hourly slots are gap-filled via linear interpolation between the nearest
    // valid readings on either side; edge gaps use flat extrapolation.
    // Returns NaN only when the entire window has zero valid readings (data ingest flag).
    // Sentinels (≤ 0 or ≥ 9999) excluded per RULE-120.
    public static double ComputePreLaunchDensityMedian(
        string connectionString,
        DateTime launchTime,
        double lookbackHours = PreLaunchLookbackHours)
    {
        var windowEnd   = FloorToHour(launchTime);
        var windowStart = windowEnd.AddHours(-lookbackHours);
        int nSlots      = (int)Math.Round(lookbackHours);

        var readings = new SortedDictionary<int, double>(); // slot → density
        using (var conn = new SqliteConnection(connectionString))
        {
            conn.Open();
            using var cmd = conn.CreateCommand();
            cmd.CommandText =
                @"SELECT datetime, proton_density FROM omni_hourly
                  WHERE datetime >= @start AND datetime < @end
                  ORDER BY datetime ASC";
            cmd.Parameters.AddWithValue("@start", windowStart.ToString("yyyy-MM-dd HH:mm"));
            cmd.Parameters.AddWithValue("@end",   windowEnd.ToString("yyyy-MM-dd HH:mm"));
            using var r = cmd.ExecuteReader();
            while (r.Read())
            {
                var dt   = DateTime.ParseExact(r.GetString(0), "yyyy-MM-dd HH:mm", null);
                int slot = (int)Math.Round((dt - windowStart).TotalHours);
                if (slot < 0 || slot >= nSlots) continue;
                double v = r.IsDBNull(1) ? -1.0 : r.GetDouble(1);
                if (v > 0.0 && v < 9999.0) readings[slot] = v;
            }
        }

        if (readings.Count == 0)
        {
            Console.Error.WriteLine(
                $"PREDICT_PROGRESSIVE_WARN type=EmptyPreLaunchWindow " +
                $"launch={launchTime:O} " +
                $"message=\"n_ref_derived=NaN; falling back to DefaultNReference\"");
            return double.NaN;
        }

        // Fill all slots: interpolate gaps, flat-extrapolate edge gaps.
        var keys   = readings.Keys.ToList(); // sorted ascending (SortedDictionary)
        var filled = new double[nSlots];
        for (int i = 0; i < nSlots; i++)
        {
            if (readings.TryGetValue(i, out double v)) { filled[i] = v; continue; }

            int leftSlot  = -1;
            int rightSlot = -1;
            for (int k = keys.Count - 1; k >= 0; k--)
                if (keys[k] < i) { leftSlot  = keys[k]; break; }
            for (int k = 0; k < keys.Count; k++)
                if (keys[k] > i) { rightSlot = keys[k]; break; }

            if (leftSlot >= 0 && rightSlot >= 0)
            {
                double t = (double)(i - leftSlot) / (rightSlot - leftSlot);
                filled[i] = readings[leftSlot] + t * (readings[rightSlot] - readings[leftSlot]);
            }
            else if (leftSlot >= 0)  filled[i] = readings[leftSlot];
            else                     filled[i] = readings[rightSlot];
        }

        var sorted = filled.OrderBy(x => x).ToArray();
        return sorted.Length % 2 == 1
            ? sorted[sorted.Length / 2]
            : (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2.0;
    }

    private static DateTime FloorToHour(DateTime dt) =>
        new(dt.Year, dt.Month, dt.Day, dt.Hour, 0, 0, dt.Kind);
}

public sealed record L1Observation(
    DateTime Timestamp,
    double? FlowSpeed,
    double? ProtonDensity,
    double? BzGsm);
