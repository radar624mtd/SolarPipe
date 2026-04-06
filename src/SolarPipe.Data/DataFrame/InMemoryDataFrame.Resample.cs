using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Data.DataFrame;

// ResampleAndAlign: temporal alignment primitive (RULE-122, ADR-016).
//
// Resamples an IDataFrame to a uniform cadence. Requires a timestamp column
// (ColumnType.DateTime or case-insensitive name "timestamp") whose values
// are stored as float = Unix epoch seconds.
//
// Interpolation strategy:
//   Float columns  → linear interpolation (best for continuous solar wind data)
//   Other columns  → forward-fill (for categorical/integer data)
//   NaN values     → propagated as-is (RULE-121: NaN guard)
//
// The output frame has the same schema as the input, with the timestamp column
// resampled to a uniform grid [t_min, t_min + cadence, ..., t_max].
public sealed partial class InMemoryDataFrame
{
    public IDataFrame ResampleAndAlign(TimeSpan cadence)
    {
        if (cadence <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(cadence),
                $"ResampleAndAlign: cadence must be positive, got {cadence}. " +
                $"Schema: [{string.Join(", ", _schema.Columns.Select(c => c.Name))}].");

        int tsIdx = FindTimestampColumn();

        float[] srcTimes = _data[tsIdx];
        if (srcTimes.Length == 0)
            return new InMemoryDataFrame(_schema, _schema.Columns.Select(_ => Array.Empty<float>()).ToArray());

        // Sort input by timestamp (solar wind data may arrive slightly out of order).
        int[] order = BuildSortOrder(srcTimes);
        float[] sortedTimes = order.Select(i => srcTimes[i]).ToArray();
        float[][] sortedData = BuildSortedData(order);

        // Build uniform target grid.
        double tMin = sortedTimes[0];
        double tMax = sortedTimes[sortedTimes.Length - 1];
        double cadenceSec = cadence.TotalSeconds;

        int gridSize = (int)Math.Floor((tMax - tMin) / cadenceSec) + 1;
        float[] gridTimes = new float[gridSize];
        for (int i = 0; i < gridSize; i++)
            gridTimes[i] = (float)(tMin + i * cadenceSec);

        // Resample each column.
        float[][] outData = new float[_data.Length][];
        for (int c = 0; c < _data.Length; c++)
        {
            if (c == tsIdx)
            {
                outData[c] = gridTimes;
            }
            else
            {
                outData[c] = _schema.Columns[c].Type == ColumnType.Float
                    ? LinearInterpolate(sortedTimes, sortedData[c], gridTimes)
                    : ForwardFill(sortedTimes, sortedData[c], gridTimes);
            }
        }

        return new InMemoryDataFrame(_schema, outData);
    }

    // ─── private helpers ──────────────────────────────────────────────────────────

    private int FindTimestampColumn()
    {
        // Prefer ColumnType.DateTime, then fall back to name "timestamp".
        for (int i = 0; i < _schema.Columns.Count; i++)
            if (_schema.Columns[i].Type == ColumnType.DateTime)
                return i;

        int byName = _schema.IndexOf("timestamp");
        if (byName >= 0) return byName;

        throw new InvalidOperationException(
            $"ResampleAndAlign requires a timestamp column (ColumnType.DateTime or name 'timestamp'). " +
            $"Schema columns: [{string.Join(", ", _schema.Columns.Select(c => $"{c.Name}:{c.Type}"))}].");
    }

    private static int[] BuildSortOrder(float[] times)
    {
        int n = times.Length;
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, (a, b) => times[a].CompareTo(times[b]));
        return order;
    }

    private float[][] BuildSortedData(int[] order)
    {
        int n = order.Length;
        var result = new float[_data.Length][];
        for (int c = 0; c < _data.Length; c++)
        {
            result[c] = new float[n];
            for (int r = 0; r < n; r++)
                result[c][r] = _data[c][order[r]];
        }
        return result;
    }

    // LinearInterpolate: produces value at each grid time by linear interpolation
    // between the two nearest source points. NaN propagates when either neighbor is NaN.
    private static float[] LinearInterpolate(float[] srcTimes, float[] srcValues, float[] gridTimes)
    {
        int n = gridTimes.Length;
        float[] result = new float[n];

        for (int g = 0; g < n; g++)
        {
            float t = gridTimes[g];

            // Binary search for the insertion point.
            int lo = 0, hi = srcTimes.Length - 1;
            while (lo < hi)
            {
                int mid = (lo + hi) / 2;
                if (srcTimes[mid] < t) lo = mid + 1;
                else hi = mid;
            }

            if (srcTimes[lo] == t)
            {
                result[g] = srcValues[lo];
            }
            else if (lo == 0)
            {
                // Before first point: nearest-neighbor (no extrapolation).
                result[g] = srcValues[0];
            }
            else
            {
                int i1 = lo - 1, i2 = lo;
                float t1 = srcTimes[i1], t2 = srcTimes[i2];
                float v1 = srcValues[i1], v2 = srcValues[i2];

                // RULE-121: NaN guard — propagate NaN if either neighbor is NaN.
                if (float.IsNaN(v1) || float.IsNaN(v2))
                {
                    result[g] = float.NaN;
                }
                else
                {
                    float frac = (t - t1) / (t2 - t1);
                    result[g] = v1 + frac * (v2 - v1);
                }
            }
        }

        return result;
    }

    // ForwardFill: carries the last observed non-NaN value forward.
    // Used for categorical / integer columns where linear interpolation is meaningless.
    private static float[] ForwardFill(float[] srcTimes, float[] srcValues, float[] gridTimes)
    {
        int n = gridTimes.Length;
        float[] result = new float[n];
        int srcIdx = 0;
        float lastSeen = float.NaN;

        for (int g = 0; g < n; g++)
        {
            float t = gridTimes[g];

            // Advance source pointer to the last point at or before t.
            while (srcIdx + 1 < srcTimes.Length && srcTimes[srcIdx + 1] <= t)
                srcIdx++;

            if (srcTimes[srcIdx] <= t && !float.IsNaN(srcValues[srcIdx]))
                lastSeen = srcValues[srcIdx];

            result[g] = lastSeen;
        }

        return result;
    }
}
