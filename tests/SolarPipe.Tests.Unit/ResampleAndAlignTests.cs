using FluentAssertions;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;

namespace SolarPipe.Tests.Unit;

// Tests for IDataFrame.ResampleAndAlign() — temporal alignment primitive (RULE-122, ADR-016).
//
// Timestamps are stored as float (Unix epoch seconds). For precision, tests use small
// relative values (0, 60, 120 … seconds from epoch) rather than large absolute Unix times,
// because float only has ~7 decimal digits of precision and 1_434_614_400f + 60f may
// lose the 60-second offset entirely.
[Trait("Category", "Unit")]
public sealed class ResampleAndAlignTests
{
    // Small base timestamp (seconds from epoch — avoids float precision loss).
    private const float T0 = 0f;
    private const float OneMinute = 60f;
    private const float OneHour = 3600f;

    // ─── Basic grid construction ──────────────────────────────────────────────────

    [Fact]
    public void Resample_UniformInput_ReturnsSamePointCount_WhenCadenceMatches()
    {
        // 5 points at 60s cadence → resample at 60s = same 5 points.
        float[] timestamps = [T0, T0 + OneMinute, T0 + 2 * OneMinute, T0 + 3 * OneMinute, T0 + 4 * OneMinute];
        float[] values = [10f, 20f, 30f, 40f, 50f];

        using var frame = MakeFrame(timestamps, values);
        using var result = frame.ResampleAndAlign(TimeSpan.FromSeconds(60));

        result.RowCount.Should().Be(5);
        result.GetColumn("value")[0].Should().BeApproximately(10f, 0.01f);
        result.GetColumn("value")[4].Should().BeApproximately(50f, 0.01f);
    }

    [Fact]
    public void Resample_UpsamplesTo30s_MidpointsAreLinearlyInterpolated()
    {
        // 3 points at 60s → resample at 30s = 5 points (0s, 30s, 60s, 90s, 120s).
        float[] timestamps = [T0, T0 + OneMinute, T0 + 2 * OneMinute];
        float[] values = [100f, 200f, 300f];

        using var frame = MakeFrame(timestamps, values);
        using var result = frame.ResampleAndAlign(TimeSpan.FromSeconds(30));

        result.RowCount.Should().Be(5, "0s, 30s, 60s, 90s, 120s = 5 grid points");

        float[] resampled = result.GetColumn("value");
        resampled[0].Should().BeApproximately(100f, 0.1f, "t=0s: exact source");
        resampled[1].Should().BeApproximately(150f, 1.0f, "t=30s: midpoint of 100 and 200");
        resampled[2].Should().BeApproximately(200f, 0.1f, "t=60s: exact source");
        resampled[3].Should().BeApproximately(250f, 1.0f, "t=90s: midpoint of 200 and 300");
        resampled[4].Should().BeApproximately(300f, 0.1f, "t=120s: exact source");
    }

    [Fact]
    public void Resample_DownsamplesTo2h_ProducesFewerPoints()
    {
        // 7 points at 1h → resample at 2h = 4 grid points (0h, 2h, 4h, 6h).
        float[] timestamps = Enumerable.Range(0, 7).Select(i => T0 + i * OneHour).ToArray();
        float[] values = [10f, 20f, 30f, 40f, 50f, 60f, 70f];

        using var frame = MakeFrame(timestamps, values);
        using var result = frame.ResampleAndAlign(TimeSpan.FromHours(2));

        result.RowCount.Should().Be(4, "0h, 2h, 4h, 6h = 4 grid points");
    }

    // ─── Linear interpolation ────────────────────────────────────────────────────

    [Fact]
    public void Resample_LinearInterpolation_MidpointIsExactForLinearData()
    {
        // v(t) = t / 3600 (linear). Midpoint at 0.5h must be 0.5.
        float[] timestamps = [T0, T0 + OneHour, T0 + 2 * OneHour];
        float[] values = [0f, 1f, 2f];

        using var frame = MakeFrame(timestamps, values);
        using var result = frame.ResampleAndAlign(TimeSpan.FromMinutes(30));

        float[] resampled = result.GetColumn("value");
        // Grid: 0h(0), 0.5h(1), 1h(2), 1.5h(3), 2h(4)
        resampled[1].Should().BeApproximately(0.5f, 0.01f, "t=0.5h: linear midpoint");
        resampled[3].Should().BeApproximately(1.5f, 0.01f, "t=1.5h: linear midpoint");
    }

    [Fact]
    public void Resample_NanInSource_PropagatesNanToInterpolatedPoint()
    {
        // Middle point is NaN — interpolation between [t=0, v=10] and [t=60, v=NaN] must yield NaN.
        // Grid at 30s falls between t=0 and t=60(NaN) → NaN (RULE-121).
        float[] timestamps = [T0, T0 + OneMinute, T0 + 2 * OneMinute, T0 + 3 * OneMinute];
        float[] values = [10f, float.NaN, 30f, 40f];

        using var frame = MakeFrame(timestamps, values);
        using var result = frame.ResampleAndAlign(TimeSpan.FromSeconds(30));

        float[] resampled = result.GetColumn("value");
        // Grid point at 30s is between t=0 (v=10) and t=60 (v=NaN) → NaN.
        float.IsNaN(resampled[1]).Should().BeTrue(
            "interpolation with NaN neighbor must produce NaN (RULE-121)");
    }

    // ─── Forward-fill ─────────────────────────────────────────────────────────────

    [Fact]
    public void Resample_IntegerColumn_UsesForwardFill()
    {
        // Int column: forward-fill, not linear interpolation.
        float[] timestamps = [T0, T0 + 2 * OneHour];
        float[] events = [1f, 3f];

        var schema = new DataSchema([
            new ColumnInfo("timestamp", ColumnType.DateTime, false),
            new ColumnInfo("event_count", ColumnType.Int, false),
        ]);
        using var frame = new InMemoryDataFrame(schema, [timestamps, events]);
        using var result = frame.ResampleAndAlign(TimeSpan.FromHours(1));

        // Grid: 0h, 1h, 2h.
        result.RowCount.Should().Be(3);
        float[] resampled = result.GetColumn("event_count");
        resampled[0].Should().BeApproximately(1f, 0.001f, "t=0h: source value");
        resampled[1].Should().BeApproximately(1f, 0.001f, "t=1h: forward-fill from t=0h");
        resampled[2].Should().BeApproximately(3f, 0.001f, "t=2h: source value");
    }

    // ─── Timestamp column detection ───────────────────────────────────────────────

    [Fact]
    public void Resample_TimestampColumnByDateTimeType_Detected()
    {
        // Column with ColumnType.DateTime detected by type (highest priority).
        float[] timestamps = [T0, T0 + OneHour, T0 + 2 * OneHour];
        float[] values = [5f, 15f, 25f];

        var schema = new DataSchema([
            new ColumnInfo("ts", ColumnType.DateTime, false),
            new ColumnInfo("Dst_nT", ColumnType.Float, false),
        ]);
        using var frame = new InMemoryDataFrame(schema, [timestamps, values]);
        using var result = frame.ResampleAndAlign(TimeSpan.FromMinutes(30));

        result.RowCount.Should().Be(5, "5 grid points at 30-min cadence over 2h span");
    }

    [Fact]
    public void Resample_TimestampColumnByName_DetectedCaseInsensitive()
    {
        // Column named "Timestamp" (not DateTime type) detected by name (case-insensitive).
        float[] timestamps = [T0, T0 + OneHour, T0 + 2 * OneHour];
        float[] values = [5f, 15f, 25f];

        var schema = new DataSchema([
            new ColumnInfo("Timestamp", ColumnType.Float, false),
            new ColumnInfo("Dst_nT", ColumnType.Float, false),
        ]);
        using var frame = new InMemoryDataFrame(schema, [timestamps, values]);
        using var result = frame.ResampleAndAlign(TimeSpan.FromMinutes(30));

        // 5 grid points: 0, 30, 60, 90, 120 min
        result.RowCount.Should().Be(5, "Timestamp column detected by name");
    }

    [Fact]
    public void Resample_NoTimestampColumn_ThrowsInvalidOperationException()
    {
        var schema = new DataSchema([
            new ColumnInfo("speed_km_s", ColumnType.Float, false),
            new ColumnInfo("density", ColumnType.Float, false),
        ]);
        using var frame = new InMemoryDataFrame(schema, [[1f, 2f], [3f, 4f]]);

        Action act = () => frame.ResampleAndAlign(TimeSpan.FromMinutes(1));
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*timestamp*");
    }

    [Fact]
    public void Resample_NegativeCadence_ThrowsArgumentOutOfRangeException()
    {
        float[] timestamps = [T0, T0 + OneHour];
        float[] values = [10f, 20f];
        using var frame = MakeFrame(timestamps, values);

        Action act = () => frame.ResampleAndAlign(TimeSpan.FromMinutes(-1));
        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithMessage("*cadence*");
    }

    // ─── Sort tolerance: out-of-order input ──────────────────────────────────────

    [Fact]
    public void Resample_OutOfOrderInput_SortsBeforeInterpolating()
    {
        // Timestamps arrive out of order — sort step must correct this.
        float[] timestamps = [T0 + 2 * OneHour, T0, T0 + OneHour];
        float[] values = [300f, 100f, 200f];

        using var frame = MakeFrame(timestamps, values);
        using var result = frame.ResampleAndAlign(TimeSpan.FromHours(1));

        float[] resampled = result.GetColumn("value");
        result.RowCount.Should().Be(3, "3 grid points after sort: 0h, 1h, 2h");
        resampled[0].Should().BeApproximately(100f, 0.01f, "t=0h after sort");
        resampled[1].Should().BeApproximately(200f, 0.01f, "t=1h after sort");
        resampled[2].Should().BeApproximately(300f, 0.01f, "t=2h after sort");
    }

    // ─── helpers ──────────────────────────────────────────────────────────────────

    private static InMemoryDataFrame MakeFrame(float[] timestamps, float[] values)
    {
        var schema = new DataSchema([
            new ColumnInfo("timestamp", ColumnType.DateTime, false),
            new ColumnInfo("value", ColumnType.Float, false),
        ]);
        return new InMemoryDataFrame(schema, [timestamps, values]);
    }
}
