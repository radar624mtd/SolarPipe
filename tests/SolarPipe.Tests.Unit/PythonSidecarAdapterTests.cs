using Apache.Arrow;
using Apache.Arrow.Ipc;
using FluentAssertions;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Training.Adapters;

namespace SolarPipe.Tests.Unit;

// Task 14.1 — PythonSidecarAdapter unit tests.
//
// Tests cover the two key additions in Task 14.1:
//   1. ArrowIpcHelper.ReadAsync: round-trip fidelity (IDataFrame → Arrow → IDataFrame)
//   2. ArrowIpcHelper.ReadAsync: null Arrow values → float.NaN (RULE-120)
//   3. ArrowIpcHelper.ReadAsync: non-float32 column rejection (RULE-063)
//   4. GrpcSidecarAdapter.CheckHealthAsync: unreachable server → false within timeout
[Trait("Category", "Unit")]
public sealed class PythonSidecarAdapterTests : IDisposable
{
    private readonly string _tempDir;

    public PythonSidecarAdapterTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"sidecar_adapter_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, recursive: true); } catch { /* best-effort */ }
    }

    // ─── 1. Arrow IPC round-trip: write IDataFrame → read back → same values ─────────

    [Fact]
    public async Task ArrowIpcHelper_RoundTrip_PreservesFloat32Values()
    {
        float[] speeds  = [400f, 450f, 500f, float.NaN];
        float[] density = [5.0f, 6.0f, 7.0f, float.NaN];
        var schema = new DataSchema([
            new ColumnInfo("speed",   ColumnType.Float, false),
            new ColumnInfo("density", ColumnType.Float, true),
        ]);
        using var original = new InMemoryDataFrame(schema, [speeds, density]);

        string arrowPath = Path.Combine(_tempDir, "roundtrip.arrow");
        await ArrowIpcHelper.WriteAsync(original, arrowPath, CancellationToken.None);

        var restored = await ArrowIpcHelper.ReadAsync(arrowPath, CancellationToken.None);
        using var frame = (InMemoryDataFrame)restored;

        frame.RowCount.Should().Be(4);
        frame.Schema.Columns.Should().HaveCount(2);

        float[] restoredSpeeds = frame.GetColumn("speed");
        restoredSpeeds[0].Should().BeApproximately(400f, 1e-4f);
        restoredSpeeds[1].Should().BeApproximately(450f, 1e-4f);

        // NaN is preserved through the round-trip (null Arrow → NaN on read — RULE-120)
        float.IsNaN(restoredSpeeds[3]).Should().BeTrue("NaN must round-trip through Arrow IPC as null");

        float[] restoredDensity = frame.GetColumn("density");
        float.IsNaN(restoredDensity[3]).Should().BeTrue();
    }

    // ─── 2. Arrow IPC read: null values → float.NaN (RULE-120) ──────────────────────

    [Fact]
    public async Task ArrowIpcHelper_ReadAsync_NullArrowValues_BecomeNaN()
    {
        // Write an Arrow file with explicit null values using Apache.Arrow directly
        string arrowPath = Path.Combine(_tempDir, "null_test.arrow");

        var schema = new Apache.Arrow.Schema.Builder()
            .Field(new Field("bz_gsm", Apache.Arrow.Types.FloatType.Default, nullable: true))
            .Build();

        var builder = new FloatArray.Builder();
        builder.Append(-5.0f);
        builder.AppendNull();    // DSCOVR safe-hold gap
        builder.Append(-8.3f);
        var array = builder.Build();

        var batch = new RecordBatch(schema, [array], 3);
        await using (var stream = File.Create(arrowPath))
        {
            using var writer = new ArrowFileWriter(stream, schema);
            await writer.WriteRecordBatchAsync(batch);
            await writer.WriteEndAsync();
        }

        var restored = await ArrowIpcHelper.ReadAsync(arrowPath, CancellationToken.None);
        using var df = (InMemoryDataFrame)restored;

        float[] col = df.GetColumn("bz_gsm");
        col[0].Should().BeApproximately(-5.0f, 1e-4f);
        float.IsNaN(col[1]).Should().BeTrue("null Arrow value must become NaN per RULE-120");
        col[2].Should().BeApproximately(-8.3f, 1e-4f);
    }

    // ─── 3. Arrow IPC read: double64 column → InvalidOperationException (RULE-063) ───

    [Fact]
    public async Task ArrowIpcHelper_ReadAsync_Float64Column_ThrowsInvalidOperation()
    {
        // Write an Arrow file with float64 (double) column — violates RULE-063
        string arrowPath = Path.Combine(_tempDir, "double_col.arrow");

        var schema = new Apache.Arrow.Schema.Builder()
            .Field(new Field("speed", Apache.Arrow.Types.DoubleType.Default, nullable: false))
            .Build();

        var builder = new DoubleArray.Builder();
        builder.Append(400.0);
        builder.Append(450.0);
        var array = builder.Build();

        var batch = new RecordBatch(schema, [array], 2);
        await using (var stream = File.Create(arrowPath))
        {
            using var writer = new ArrowFileWriter(stream, schema);
            await writer.WriteRecordBatchAsync(batch);
            await writer.WriteEndAsync();
        }

        Func<Task> act = () => ArrowIpcHelper.ReadAsync(arrowPath, CancellationToken.None);

        await act.Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("*float32*RULE-063*");
    }

    // ─── 4. CheckHealthAsync: unreachable server returns false within timeout ─────────

    [Fact]
    public async Task GrpcSidecarAdapter_CheckHealthAsync_UnreachableServer_ReturnsFalse()
    {
        // Port 19997 should not be running anything
        using var adapter = new GrpcSidecarAdapter("http://localhost:19997", _tempDir);

        var timeout = TimeSpan.FromMilliseconds(500);
        bool healthy = await adapter.CheckHealthAsync(timeout, CancellationToken.None);

        healthy.Should().BeFalse("no server on port 19997 means health check must return false");
    }
}
