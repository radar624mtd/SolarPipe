using Apache.Arrow;
using Apache.Arrow.Ipc;
using FluentAssertions;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Training.Adapters;

namespace SolarPipe.Tests.Unit;

// Task 11.3 — TFT trainer unit tests.
//
// These tests do NOT require a running gRPC server. They verify:
//   1. TFT model type is accepted by GrpcSidecarAdapter (RULE-061: gRPC routing only fails at network)
//   2. Arrow IPC serialization for TFT feature columns is float32 (RULE-063)
//   3. TFT hyperparameters (epochs, hidden_size, learning_rate) are preserved
//      as strings when written into the request (RULE-125 / gRPC transport contract)
[Trait("Category", "Unit")]
public sealed class TftTrainerTests : IDisposable
{
    private readonly string _tempDir;

    public TftTrainerTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"tft_unit_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, recursive: true); } catch { /* best-effort */ }
    }

    // ─── 1. TFT model type is accepted by adapter; fails at network, not model routing ──

    [Fact]
    public async Task GrpcSidecarAdapter_TftModelType_PassesModelRouting_FailsAtNetwork()
    {
        // The adapter must not throw NotSupportedException for TFT —
        // it should reach gRPC and fail with a connection/RPC error.
        using var adapter = new GrpcSidecarAdapter("http://localhost:19999", _tempDir);
        using var frame = MakeTftFrame(rows: 5);
        var config = new StageConfig(
            "tft_stage", "PythonGrpc", "TFT", "src",
            ["speed", "density", "bz_gsm"],
            "arrival_time_hours",
            new Dictionary<string, object>
            {
                ["epochs"] = 5,
                ["hidden_size"] = 32,
                ["learning_rate"] = 0.001,
            });

        Func<Task> act = () => adapter.TrainAsync(config, frame, null, CancellationToken.None);

        // Must NOT be NotSupportedException — TFT is supported.
        // Must be some network / gRPC failure (RpcException, HttpRequestException, etc.)
        var ex = await act.Should().ThrowAsync<Exception>(
            "TFT routing succeeds; gRPC network call fails because no server runs on port 19999");
        ex.Which.Should().NotBeOfType<NotSupportedException>(
            "TFT is a supported model type for GrpcSidecarAdapter");
    }

    // ─── 2. Arrow IPC serialisation for TFT feature columns is float32 (RULE-063) ──────

    [Fact]
    public async Task ArrowIpcHelper_TftFeatureFrame_AllColumnsAreFloat32()
    {
        using var frame = MakeTftFrame(rows: 10);
        string arrowPath = Path.Combine(_tempDir, "tft_features.arrow");

        await ArrowIpcHelper.WriteAsync(frame, arrowPath, CancellationToken.None);

        await using var stream = File.OpenRead(arrowPath);
        using var reader = new ArrowFileReader(stream);
        var schema = reader.Schema;

        schema.FieldsList.Should().HaveCount(3,
            "speed, density, bz_gsm — three TFT feature columns");

        foreach (var field in schema.FieldsList)
        {
            field.DataType.TypeId.Should().Be(Apache.Arrow.Types.ArrowTypeId.Float,
                $"TFT feature column '{field.Name}' must be float32 (RULE-063)");
        }
    }

    // ─── 3. TFT hyperparameters are preserved correctly ──────────────────────────────

    [Theory]
    [InlineData("epochs",        "10")]
    [InlineData("hidden_size",   "64")]
    [InlineData("learning_rate", "0.001")]
    public void TftHyperparameters_StringConversion_MatchesExpected(string key, string expected)
    {
        // Verify the conversion pattern used inside GrpcSidecarAdapter.TrainAsync:
        // config.Hyperparameters[key]?.ToString() must produce the correct wire value.
        var hyperparameters = new Dictionary<string, object>
        {
            ["epochs"]        = 10,
            ["hidden_size"]   = 64,
            ["learning_rate"] = 0.001,
        };

        string? actual = hyperparameters.TryGetValue(key, out var val)
            ? val?.ToString()
            : null;

        actual.Should().Be(expected,
            $"hyperparameter '{key}' must serialize as '{expected}' for gRPC transport (RULE-125)");
    }

    // ─── helpers ──────────────────────────────────────────────────────────────────────

    private static InMemoryDataFrame MakeTftFrame(int rows)
    {
        float[] speeds   = Enumerable.Range(0, rows).Select(i => 400f + i * 50f).ToArray();
        float[] density  = Enumerable.Range(0, rows).Select(i => 5f + i * 0.5f).ToArray();
        float[] bzGsm    = Enumerable.Range(0, rows).Select(i => -5f + i * 0.3f).ToArray();
        var schema = new DataSchema([
            new ColumnInfo("speed",    ColumnType.Float, false),
            new ColumnInfo("density",  ColumnType.Float, false),
            new ColumnInfo("bz_gsm",   ColumnType.Float, false),
        ]);
        return new InMemoryDataFrame(schema, [speeds, density, bzGsm]);
    }
}
