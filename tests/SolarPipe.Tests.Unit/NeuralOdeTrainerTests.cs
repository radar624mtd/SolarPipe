using Apache.Arrow;
using Apache.Arrow.Ipc;
using FluentAssertions;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Training.Adapters;

namespace SolarPipe.Tests.Unit;

// Task 13.2 — Neural ODE trainer unit tests.
//
// These tests do NOT require a running gRPC server or Python. They verify:
//   1. NeuralOde model type is accepted by GrpcSidecarAdapter
//      (RULE-061: routing only fails at network, not at model-type check)
//   2. Arrow IPC serialization for NeuralOde state columns (state, t_start, t_end)
//      is float32 (RULE-063) — these are the column names OnnxTrainedModel expects
//      for the dynamics-network-only ONNX inference pattern (RULE-070)
[Trait("Category", "Unit")]
public sealed class NeuralOdeTrainerTests : IDisposable
{
    private readonly string _tempDir;

    public NeuralOdeTrainerTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"neural_ode_unit_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, recursive: true); } catch { /* best-effort */ }
    }

    // ─── 1. NeuralOde model type is accepted by adapter; fails at network ─────────────

    [Fact]
    public async Task GrpcSidecarAdapter_NeuralOdeModelType_PassesModelRouting_FailsAtNetwork()
    {
        // The adapter must not throw NotSupportedException for NeuralOde —
        // it should reach gRPC and fail with a connection/RPC error.
        using var adapter = new GrpcSidecarAdapter("http://localhost:19998", _tempDir);
        using var frame = MakeNeuralOdeFrame(rows: 4);
        var config = new StageConfig(
            "neural_ode_stage", "PythonGrpc", "NeuralOde", "src",
            ["state", "t_start", "t_end"],
            "state_final",
            new Dictionary<string, object>
            {
                ["epochs"]      = 50,
                ["hidden_size"] = 64,
                ["export_onnx"] = true,
            });

        Func<Task> act = () => adapter.TrainAsync(config, frame, null, CancellationToken.None);

        // Must NOT be NotSupportedException — NeuralOde is supported.
        var ex = await act.Should().ThrowAsync<Exception>(
            "NeuralOde routing succeeds; gRPC network call fails because no server runs on port 19998");
        ex.Which.Should().NotBeOfType<NotSupportedException>(
            "NeuralOde is a supported model type for GrpcSidecarAdapter");
    }

    // ─── 2. Arrow IPC for NeuralOde state columns is float32 (RULE-063) ─────────────
    // RULE-070: dynamics network f(y,t,θ) — OnnxTrainedModel.PredictNeuralOde expects
    //   columns "state", "t_start", "t_end" to be float32 in the Arrow IPC payload.

    [Fact]
    public async Task ArrowIpcHelper_NeuralOdeStateFrame_AllColumnsAreFloat32()
    {
        using var frame = MakeNeuralOdeFrame(rows: 8);
        string arrowPath = Path.Combine(_tempDir, "neural_ode_state.arrow");

        await ArrowIpcHelper.WriteAsync(frame, arrowPath, CancellationToken.None);

        await using var stream = File.OpenRead(arrowPath);
        using var reader = new ArrowFileReader(stream);
        var schema = reader.Schema;

        schema.FieldsList.Should().HaveCount(3,
            "state, t_start, t_end — three NeuralOde state columns");

        foreach (var field in schema.FieldsList)
        {
            field.DataType.TypeId.Should().Be(Apache.Arrow.Types.ArrowTypeId.Float,
                $"NeuralOde column '{field.Name}' must be float32 for ORT call compatibility (RULE-063, RULE-070)");
        }
    }

    // ─── helpers ──────────────────────────────────────────────────────────────────────

    private static InMemoryDataFrame MakeNeuralOdeFrame(int rows)
    {
        // State: Dst index (nT), t_start: integration start (hours), t_end: integration end (hours)
        float[] state  = Enumerable.Range(0, rows).Select(i => -50f - i * 10f).ToArray();
        float[] tStart = Enumerable.Range(0, rows).Select(i => (float)i).ToArray();
        float[] tEnd   = Enumerable.Range(0, rows).Select(i => (float)i + 24f).ToArray();
        var schema = new DataSchema([
            new ColumnInfo("state",   ColumnType.Float, false),
            new ColumnInfo("t_start", ColumnType.Float, false),
            new ColumnInfo("t_end",   ColumnType.Float, false),
        ]);
        return new InMemoryDataFrame(schema, [state, tStart, tEnd]);
    }
}
