using FluentAssertions;
using Microsoft.ML;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Training.Adapters;

namespace SolarPipe.Tests.Unit;

// RULE-013: Sequential execution — MLContext shares native memory; parallel tests cause flakiness
[Collection("ML")]
[Trait("Category", "Unit")]
public class MlNetAdapterTests
{
    private readonly MlNetAdapter _adapter = new();

    // Helper: create a simple numeric IDataFrame for ML training
    // Features: speed (300–900 km/s range), density (5–30 per cm³)
    // Target: dst_estimate (linear combination — deterministic for small tests)
    private static InMemoryDataFrame MakeTrainingFrame(int rows = 80)
    {
        var schema = new DataSchema(new[]
        {
            new ColumnInfo("speed",   ColumnType.Float, true),
            new ColumnInfo("density", ColumnType.Float, true),
            new ColumnInfo("dst",     ColumnType.Float, true),
        }.ToList());

        // Use arithmetic progressions — no Random (RULE-003)
        var speed   = Enumerable.Range(0, rows).Select(i => 300f + i * 7.5f).ToArray();
        var density = Enumerable.Range(0, rows).Select(i => 5f  + i * 0.3f).ToArray();
        // dst ~ linear in speed and density (synthetic target)
        var dst     = Enumerable.Range(0, rows).Select(i => -(speed[i] * 0.1f + density[i] * 2f)).ToArray();

        return new InMemoryDataFrame(schema, [speed, density, dst]);
    }

    private static StageConfig MakeConfig(
        string modelType = "FastForest",
        IReadOnlyDictionary<string, object>? hyperparams = null) =>
        new StageConfig(
            Name: "test_stage",
            Framework: "mlnet",
            ModelType: modelType,
            DataSource: "test_src",
            Features: ["speed", "density"],
            Target: "dst",
            Hyperparameters: hyperparams);

    // Test 1: TrainAsync with FastForest produces an ITrainedModel with correct metadata
    [Fact]
    public async Task TrainAsync_FastForest_ReturnsTrainedModel()
    {
        var config = MakeConfig();
        using var df = MakeTrainingFrame();

        var model = await _adapter.TrainAsync(config, df, null, CancellationToken.None);

        model.Should().NotBeNull();
        model.ModelId.Should().Be("test_stage_FastForest");
        model.StageName.Should().Be("test_stage");
        // Without validation data, metrics are NaN sentinels
        double.IsNaN(model.Metrics.Rmse).Should().BeTrue();
    }

    // Test 2: Hyperparameter mapping — explicit hyperparams are applied; both configs train successfully
    // Ensures FeatureFraction=0.7 default doesn't break training (RULE-010 compliance is in source)
    [Fact]
    public async Task TrainAsync_HyperparameterOverrides_BothConfigsTrain()
    {
        var configDefault = MakeConfig();
        var configExplicit = MakeConfig(hyperparams: new Dictionary<string, object>
        {
            ["feature_fraction"] = 0.5,
            ["number_of_trees"] = 10,
            ["number_of_leaves"] = 8
        });

        using var df = MakeTrainingFrame();
        var m1 = await _adapter.TrainAsync(configDefault, df, null, CancellationToken.None);
        var m2 = await _adapter.TrainAsync(configExplicit, df, null, CancellationToken.None);

        m1.Should().NotBeNull();
        m2.Should().NotBeNull();
        m1.ModelId.Should().Be("test_stage_FastForest");
        m2.ModelId.Should().Be("test_stage_FastForest");

        // Both should produce predictions on the same input
        var r1 = await m1.PredictAsync(df, CancellationToken.None);
        var r2 = await m2.PredictAsync(df, CancellationToken.None);
        r1.Values.Should().HaveCount(df.RowCount);
        r2.Values.Should().HaveCount(df.RowCount);
    }

    // Test 3: Metrics collection — validation data produces finite RMSE/MAE/R²
    [Fact]
    public async Task TrainAsync_WithValidationData_ProducesFiniteMetrics()
    {
        var config = MakeConfig(hyperparams: new Dictionary<string, object>
        {
            ["number_of_trees"] = 20,
            ["number_of_leaves"] = 10
        });

        using var train = MakeTrainingFrame(rows: 60);
        using var validation = MakeTrainingFrame(rows: 20);

        var model = await _adapter.TrainAsync(config, train, validation, CancellationToken.None);

        model.Metrics.Rmse.Should().BeGreaterThan(0, "RMSE must be positive");
        model.Metrics.Mae.Should().BeGreaterThan(0, "MAE must be positive");
        // R² can be negative for poor models but should be finite
        double.IsNaN(model.Metrics.R2).Should().BeFalse("R² must be finite with validation data");
        double.IsInfinity(model.Metrics.R2).Should().BeFalse();
    }

    // Test 4: PredictAsync output shape matches input row count
    [Fact]
    public async Task PredictAsync_OutputShape_MatchesInputRows()
    {
        var config = MakeConfig(hyperparams: new Dictionary<string, object>
        {
            ["number_of_trees"] = 10
        });

        using var train = MakeTrainingFrame(rows: 60);
        using var predict = MakeTrainingFrame(rows: 15);

        var model = await _adapter.TrainAsync(config, train, null, CancellationToken.None);
        var result = await model.PredictAsync(predict, CancellationToken.None);

        result.Should().NotBeNull();
        result.Values.Should().HaveCount(15, "one prediction per input row");
        result.ModelId.Should().Be("test_stage_FastForest");
        result.Values.Should().OnlyContain(v => !float.IsNaN(v), "no NaN predictions for valid input");
    }

    // Test 5: Unsupported model type throws NotSupportedException with informative message
    [Fact]
    public async Task TrainAsync_UnsupportedModelType_ThrowsNotSupported()
    {
        var config = MakeConfig(modelType: "XGBoost");
        using var df = MakeTrainingFrame();

        var act = async () => await _adapter.TrainAsync(config, df, null, CancellationToken.None);

        await act.Should()
            .ThrowAsync<NotSupportedException>()
            .WithMessage("*XGBoost*not supported*MlNetAdapter*");
    }

    // Test 6: SaveAsync persists model file; LoadAsync on the model object throws NotSupportedException
    [Fact]
    public async Task SaveAsync_PersistsFile_LoadAsyncThrowsNotSupported()
    {
        var config = MakeConfig(hyperparams: new Dictionary<string, object>
        {
            ["number_of_trees"] = 5
        });

        using var df = MakeTrainingFrame(rows: 40);
        var model = await _adapter.TrainAsync(config, df, null, CancellationToken.None);

        var tempPath = Path.Combine(Path.GetTempPath(), $"mlnet_test_{Guid.NewGuid():N}.zip");
        try
        {
            await model.SaveAsync(tempPath, CancellationToken.None);
            File.Exists(tempPath).Should().BeTrue("model binary must be written to disk");
            new FileInfo(tempPath).Length.Should().BeGreaterThan(0, "saved model must be non-empty");

            var loadAct = async () => await model.LoadAsync(tempPath, CancellationToken.None);
            await loadAct.Should()
                .ThrowAsync<NotSupportedException>()
                .WithMessage("*FileSystemModelRegistry*");
        }
        finally
        {
            if (File.Exists(tempPath)) File.Delete(tempPath);
        }
    }
}
