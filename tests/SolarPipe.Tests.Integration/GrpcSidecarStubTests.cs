using Apache.Arrow;
using Apache.Arrow.Ipc;
using FluentAssertions;
using NSubstitute;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Prediction;
using SolarPipe.Training.Adapters;
using SolarPipe.Training.Physics;

namespace SolarPipe.Tests.Integration;

// Phase 2 gRPC sidecar stub integration tests (ADR-011).
//
// These tests do NOT start a real gRPC server (Python may not be available).
// They validate:
//   1. GrpcSidecarAdapter contract — implements IFrameworkAdapter with correct FrameworkType/SupportedModels
//   2. Arrow IPC schema enforcement — ArrowIpcHelper.WriteAsync produces float32 columns (RULE-063)
//   3. Composition algebra with mock gRPC-backed model — physics_model ^ grpc_stub_model end-to-end
//   4. Arrow IPC round-trip — C# writes float32, reads float32 back correctly
//
// Phase 4 will add live server tests once the Python sidecar is operational.
[Trait("Category", "Integration")]
public sealed class GrpcSidecarStubTests : IDisposable
{
    private readonly string _tempDir;

    public GrpcSidecarStubTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"grpc_stub_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, recursive: true); } catch { /* best-effort */ }
    }

    // ─── 1. Adapter contract ──────────────────────────────────────────────────────

    [Fact]
    public void GrpcSidecarAdapter_FrameworkType_IsPythonGrpc()
    {
        // Construction only requires a valid URI — does not connect eagerly.
        using var adapter = new GrpcSidecarAdapter("http://localhost:50051", _tempDir);

        adapter.FrameworkType.Should().Be(FrameworkType.PythonGrpc);
    }

    [Fact]
    public void GrpcSidecarAdapter_SupportedModels_ContainsTftAndNeuralOde()
    {
        using var adapter = new GrpcSidecarAdapter("http://localhost:50051", _tempDir);

        adapter.SupportedModels.Should().Contain("TFT");
        adapter.SupportedModels.Should().Contain("NeuralOde");
    }

    [Fact]
    public async Task GrpcSidecarAdapter_TrainAsync_ThrowsForUnsupportedModel()
    {
        using var adapter = new GrpcSidecarAdapter("http://localhost:50051", _tempDir);
        using var frame = MakeSmallFrame();

        var config = new StageConfig("s1", "GrpcSidecar", "FastForest", "src",
            ["speed"], "arrival", null);

        // FastForest is not supported by GrpcSidecarAdapter — must throw before touching gRPC.
        Func<Task> act = () => adapter.TrainAsync(config, frame, null, CancellationToken.None);
        await act.Should().ThrowAsync<NotSupportedException>()
            .WithMessage("*FastForest*");
    }

    // ─── 2. Arrow IPC schema enforcement (RULE-063) ───────────────────────────────

    [Fact]
    public async Task ArrowIpcHelper_Write_ProducesFloat32Columns()
    {
        using var frame = MakeSmallFrame();
        string arrowPath = Path.Combine(_tempDir, "test_schema.arrow");

        await ArrowIpcHelper.WriteAsync(frame, arrowPath, CancellationToken.None);

        File.Exists(arrowPath).Should().BeTrue("Arrow IPC file must be created");

        await using var fileStream = File.OpenRead(arrowPath);
        using var reader = new ArrowFileReader(fileStream);
        var schema = reader.Schema;

        schema.FieldsList.Should().HaveCount(frame.Schema.Columns.Count,
            "Arrow schema column count must match IDataFrame schema");

        foreach (var field in schema.FieldsList)
        {
            field.DataType.TypeId.Should().Be(Apache.Arrow.Types.ArrowTypeId.Float,
                $"Column '{field.Name}' must be float32 — never float64 (RULE-063)");
        }
    }

    [Fact]
    public async Task ArrowIpcHelper_Write_PreservesValuesAndNanAsNull()
    {
        float[] speeds = [800f, 1200f, 1800f, float.NaN];
        var schema = new DataSchema([
            new ColumnInfo("radial_speed_km_s", ColumnType.Float, false),
        ]);
        using var frame = new InMemoryDataFrame(schema, [speeds]);

        string arrowPath = Path.Combine(_tempDir, "nan_test.arrow");
        await ArrowIpcHelper.WriteAsync(frame, arrowPath, CancellationToken.None);

        await using var fileStream = File.OpenRead(arrowPath);
        using var reader = new ArrowFileReader(fileStream);
        var batch = await reader.ReadNextRecordBatchAsync(CancellationToken.None);

        batch.Should().NotBeNull();
        batch!.Length.Should().Be(4, "row count must be preserved");

        var col = (FloatArray)batch.Column(0);
        col.IsNull(3).Should().BeTrue("NaN float must be written as Arrow null");
        col.GetValue(0).Should().BeApproximately(800f, 0.001f);
        col.GetValue(2).Should().BeApproximately(1800f, 0.001f);
    }

    [Fact]
    public async Task ArrowIpcHelper_Write_MultipleColumnsPreservedCorrectly()
    {
        float[] speeds = [500f, 1000f, 2000f];
        float[] arrivals = [72f, 48f, 28f];
        var schema = new DataSchema([
            new ColumnInfo("radial_speed_km_s", ColumnType.Float, false),
            new ColumnInfo("arrival_time_hours", ColumnType.Float, false),
        ]);
        using var frame = new InMemoryDataFrame(schema, [speeds, arrivals]);

        string arrowPath = Path.Combine(_tempDir, "multi_col.arrow");
        await ArrowIpcHelper.WriteAsync(frame, arrowPath, CancellationToken.None);

        await using var fileStream = File.OpenRead(arrowPath);
        using var reader = new ArrowFileReader(fileStream);
        var batch = await reader.ReadNextRecordBatchAsync(CancellationToken.None);

        batch!.ColumnCount.Should().Be(2);
        var speedCol = (FloatArray)batch.Column(0);
        var arrivalCol = (FloatArray)batch.Column(1);
        speedCol.GetValue(2).Should().BeApproximately(2000f, 0.001f);
        arrivalCol.GetValue(0).Should().BeApproximately(72f, 0.001f);
    }

    // ─── 3. Composition algebra with mock gRPC-backed model ──────────────────────
    // Simulates: physics_baseline ^ grpc_stub_model (ResidualModel).
    // ITrainedModel is NSubstitute mock returning deterministic predictions (48h stub value).

    [Fact]
    public async Task ResidualComposition_WithMockGrpcModel_ProducesPhysicsPlusMockResidual()
    {
        var physicsAdapter = new PhysicsAdapter();
        using var trainFrame = MakeTrainingFrame(rows: 40);
        var baselineModel = await physicsAdapter.TrainAsync(
            MakeDragConfig(), trainFrame, null, CancellationToken.None);

        // Mock gRPC correction model: always returns 2h residual (deterministic stub)
        var mockGrpcModel = Substitute.For<ITrainedModel>();
        mockGrpcModel.StageName.Returns("grpc_correction");
        mockGrpcModel.ModelId.Returns("stub_grpc_v1");
        mockGrpcModel.PredictAsync(Arg.Any<IDataFrame>(), Arg.Any<CancellationToken>())
            .Returns(call =>
            {
                var f = call.Arg<IDataFrame>();
                return Task.FromResult(new PredictionResult(
                    Enumerable.Repeat(2.0f, f.RowCount).ToArray(),
                    null, null, "stub_grpc_v1", DateTime.UtcNow));
            });

        var composedModel = new ResidualModel(baselineModel, mockGrpcModel, "physics^grpc");

        using var testFrame = MakeSpeedOnlyFrame([700f, 1100f, 1900f]);
        var result = await composedModel.PredictAsync(testFrame, CancellationToken.None);

        result.Values.Should().HaveCount(3);
        result.Values.Should().OnlyContain(v => !float.IsNaN(v) && float.IsFinite(v),
            "residual composition with mock gRPC model must produce finite values");

        // Each prediction = physics_prediction + 2h (the mock residual)
        var physicsResult = await baselineModel.PredictAsync(testFrame, CancellationToken.None);
        for (int i = 0; i < 3; i++)
            result.Values[i].Should().BeApproximately(physicsResult.Values[i] + 2.0f, 0.01f,
                $"result[{i}] should equal physics + 2h mock residual");
    }

    [Fact]
    public async Task EnsembleComposition_PhysicsAndMockGrpc_EqualWeightMeanIsCorrect()
    {
        var physicsAdapter = new PhysicsAdapter();
        using var trainFrame = MakeTrainingFrame(rows: 40);
        var physicsModel = await physicsAdapter.TrainAsync(
            MakeDragConfig(), trainFrame, null, CancellationToken.None);

        // Mock gRPC model: always predicts 50h (deterministic stub)
        var mockGrpcModel = Substitute.For<ITrainedModel>();
        mockGrpcModel.StageName.Returns("grpc_stub");
        mockGrpcModel.ModelId.Returns("stub_v1");
        mockGrpcModel.PredictAsync(Arg.Any<IDataFrame>(), Arg.Any<CancellationToken>())
            .Returns(call =>
            {
                var f = call.Arg<IDataFrame>();
                return Task.FromResult(new PredictionResult(
                    Enumerable.Repeat(50.0f, f.RowCount).ToArray(),
                    null, null, "stub_v1", DateTime.UtcNow));
            });

        var ensemble = new EnsembleModel([physicsModel, mockGrpcModel], name: "physics+grpc");

        using var testFrame = MakeSpeedOnlyFrame([1000f]);
        var physResult = await physicsModel.PredictAsync(testFrame, CancellationToken.None);
        var ensembleResult = await ensemble.PredictAsync(testFrame, CancellationToken.None);

        float expected = (physResult.Values[0] + 50.0f) / 2f;
        ensembleResult.Values[0].Should().BeApproximately(expected, 0.01f,
            "equal-weight ensemble: (physics_pred + 50h) / 2");
    }

    // ─── helpers ──────────────────────────────────────────────────────────────────

    private static InMemoryDataFrame MakeSmallFrame()
    {
        var schema = new DataSchema([
            new ColumnInfo("feature1", ColumnType.Float, false),
            new ColumnInfo("feature2", ColumnType.Float, false),
        ]);
        return new InMemoryDataFrame(schema, [[1f, 2f, 3f], [4f, 5f, 6f]]);
    }

    private static InMemoryDataFrame MakeTrainingFrame(int rows)
    {
        float[] speeds = Enumerable.Range(0, rows).Select(i => 400f + i * 30f).ToArray();
        float[] arrivals = speeds.Select(v =>
        {
            var (_, hours) = DragBasedModel.RunOde(v, 400.0, 0.5e-7, 21.5, 215.0);
            return (float)hours;
        }).ToArray();
        var schema = new DataSchema([
            new ColumnInfo("radial_speed_km_s", ColumnType.Float, false),
            new ColumnInfo("arrival_time_hours", ColumnType.Float, false),
        ]);
        return new InMemoryDataFrame(schema, [speeds, arrivals]);
    }

    private static InMemoryDataFrame MakeSpeedOnlyFrame(float[] speeds)
    {
        var schema = new DataSchema([
            new ColumnInfo("radial_speed_km_s", ColumnType.Float, false),
        ]);
        return new InMemoryDataFrame(schema, [speeds]);
    }

    private static StageConfig MakeDragConfig() =>
        new("drag_baseline", "Physics", "DragBased", "src",
            ["radial_speed_km_s"], "arrival_time_hours",
            new Dictionary<string, object>
            {
                ["gamma_km_inv"] = 0.5e-7,
                ["ambient_wind_km_s"] = 400.0,
                ["start_distance_solar_radii"] = 21.5,
                ["target_distance_solar_radii"] = 215.0,
            });
}
