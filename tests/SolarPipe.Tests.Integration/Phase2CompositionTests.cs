using FluentAssertions;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Prediction;
using SolarPipe.Training.Adapters;
using SolarPipe.Training.Physics;

namespace SolarPipe.Tests.Integration;

// Phase 2 integration tests: physics baseline + ML correction pipeline,
// and residual composition correctness.
// All use real adapters (no mocks) to verify end-to-end algebra.
[Trait("Category", "Integration")]
public sealed class Phase2CompositionTests
{
    // ─── helpers ──────────────────────────────────────────────────────────────────

    private static InMemoryDataFrame MakeTrainingFrame()
    {
        // 60 CME events — FastForest requires enough rows to form valid tree splits.
        // Speeds in [400, 2440] km/s covering slow, moderate, fast CMEs.
        int rows = 60;
        float[] speeds = Enumerable.Range(0, rows).Select(i => 400f + i * 34f).ToArray();

        // Compute analytical arrival times from drag model for ground truth
        float[] arrivalTimes = speeds.Select(v =>
        {
            var (_, hours) = DragBasedModel.RunOde(v, 400.0, 0.5e-7, 21.5, 215.0);
            return (float)hours;
        }).ToArray();

        // Add speed-proportional bias so RF correction has varied signal to learn.
        // Bias = 3 + 0.004*(speed - 400) so it varies from ~3h to ~10h across speed range.
        float[] biasedArrival = speeds.Zip(arrivalTimes, (v, t) => t + 3f + 0.004f * (v - 400f)).ToArray();

        var schema = new DataSchema([
            new ColumnInfo("radial_speed_km_s", ColumnType.Float, false),
            new ColumnInfo("arrival_time_hours", ColumnType.Float, false)
        ]);
        return new InMemoryDataFrame(schema, [speeds, biasedArrival]);
    }

    private static InMemoryDataFrame MakeSpeedOnlyFrame(float[] speeds)
    {
        var schema = new DataSchema([
            new ColumnInfo("radial_speed_km_s", ColumnType.Float, false)
        ]);
        return new InMemoryDataFrame(schema, [speeds]);
    }

    private static StageConfig MakeDragConfig() =>
        new StageConfig("drag_baseline", "Physics", "DragBased", "src",
            ["radial_speed_km_s"], "arrival_time_hours",
            new Dictionary<string, object>
            {
                ["gamma_km_inv"] = 0.5e-7,
                ["ambient_wind_km_s"] = 400.0,
                ["start_distance_solar_radii"] = 21.5,
                ["target_distance_solar_radii"] = 215.0
            });

    private static StageConfig MakeRfConfig() =>
        new StageConfig("rf_correction", "MlNet", "FastForest", "src",
            ["radial_speed_km_s", "residual_baseline_drag_baseline"],
            "arrival_time_hours",
            new Dictionary<string, object>
            {
                ["number_of_trees"] = 50,
                ["number_of_leaves"] = 20,
                ["feature_fraction"] = 0.7f
            });

    // ─── Test 1: Physics baseline alone produces finite arrival times ─────────────

    [Fact]
    public async Task PhysicsBaseline_ProducesFiniteArrivalTimes()
    {
        var physicsAdapter = new PhysicsAdapter();
        using var trainFrame = MakeTrainingFrame();

        var model = await physicsAdapter.TrainAsync(MakeDragConfig(), trainFrame, null, CancellationToken.None);

        using var predFrame = MakeSpeedOnlyFrame([800f, 1500f, 2000f]);
        var result = await model.PredictAsync(predFrame, CancellationToken.None);

        result.Values.Should().HaveCount(3);
        result.Values.Should().OnlyContain(v => !float.IsNaN(v) && v > 0f,
            "all physics predictions must be finite positive arrival times");
        // Faster CME arrives sooner
        result.Values[0].Should().BeGreaterThan(result.Values[1]);
        result.Values[1].Should().BeGreaterThan(result.Values[2]);
    }

    // ─── Test 2: Residual composition (drag_physics ^ rf_correction) trains ───────

    [Fact]
    public async Task ResidualComposition_PhysicsPlusRfCorrection_TrainsAndPredicts()
    {
        var physicsAdapter = new PhysicsAdapter();
        var mlAdapter = new MlNetAdapter();

        using var trainFrame = MakeTrainingFrame();

        // Train physics baseline
        var baselineModel = await physicsAdapter.TrainAsync(
            MakeDragConfig(), trainFrame, null, CancellationToken.None);

        // Build augmented training frame: add baseline predictions as feature for RF.
        // Column name must match what ResidualModel appends: "residual_baseline_{StageName}"
        var baselinePreds = await baselineModel.PredictAsync(trainFrame, CancellationToken.None);
        string residualColName = $"residual_baseline_{baselineModel.StageName}";

        // Augment training data with baseline predictions column
        using var augmentedTrain = trainFrame.AddColumn(residualColName, baselinePreds.Values);

        // Compute actual residuals (target - baseline) and store as target
        float[] arrivalTimes = trainFrame.GetColumn("arrival_time_hours");
        float[] residuals = arrivalTimes.Zip(baselinePreds.Values, (obs, pred) => obs - pred).ToArray();
        using var residualTarget = augmentedTrain.AddColumn("residual_target", residuals);

        // Train RF correction on residuals
        var rfConfig = new StageConfig("rf_correction", "MlNet", "FastForest", "src",
            ["radial_speed_km_s", residualColName],
            "residual_target",
            new Dictionary<string, object>
            {
                ["number_of_trees"] = 50,
                ["number_of_leaves"] = 20,
                ["feature_fraction"] = 0.7f
            });

        var correctionModel = await mlAdapter.TrainAsync(rfConfig, residualTarget, null, CancellationToken.None);

        // Build composed model
        var composedModel = new ResidualModel(baselineModel, correctionModel, "drag^rf");

        // Predict on unseen speeds
        using var testFrame = MakeSpeedOnlyFrame([750f, 1250f, 1800f]);
        var composedResult = await composedModel.PredictAsync(testFrame, CancellationToken.None);

        composedResult.Values.Should().HaveCount(3);
        composedResult.Values.Should().OnlyContain(v => !float.IsNaN(v),
            "composed model must produce finite predictions");
        // Faster CMEs should still arrive sooner
        composedResult.Values[0].Should().BeGreaterThan(composedResult.Values[1],
            "750 km/s CME must arrive after 1250 km/s CME");
        composedResult.Values[1].Should().BeGreaterThan(composedResult.Values[2],
            "1250 km/s CME must arrive after 1800 km/s CME");
    }

    // ─── Test 3: EnsembleModel with two physics models (different γ) ─────────────

    [Fact]
    public async Task EnsembleModel_TwoPhysicsModels_WeightedAverageIsConsistent()
    {
        var adapter = new PhysicsAdapter();
        using var trainFrame = MakeTrainingFrame();

        // Low drag (large CME)
        var lowDragConfig = new StageConfig("drag_low", "Physics", "DragBased", "src",
            ["radial_speed_km_s"], "arrival_time_hours",
            new Dictionary<string, object> { ["gamma_km_inv"] = 0.2e-7, ["ambient_wind_km_s"] = 400.0,
                ["start_distance_solar_radii"] = 21.5, ["target_distance_solar_radii"] = 215.0 });

        var highDragConfig = new StageConfig("drag_high", "Physics", "DragBased", "src",
            ["radial_speed_km_s"], "arrival_time_hours",
            new Dictionary<string, object> { ["gamma_km_inv"] = 2.0e-7, ["ambient_wind_km_s"] = 400.0,
                ["start_distance_solar_radii"] = 21.5, ["target_distance_solar_radii"] = 215.0 });

        var lowDragModel = await adapter.TrainAsync(lowDragConfig, trainFrame, null, CancellationToken.None);
        var highDragModel = await adapter.TrainAsync(highDragConfig, trainFrame, null, CancellationToken.None);

        var ensemble = new EnsembleModel([lowDragModel, highDragModel], name: "drag_ensemble");

        using var testFrame = MakeSpeedOnlyFrame([1000f]);
        var lowResult = await lowDragModel.PredictAsync(testFrame, CancellationToken.None);
        var highResult = await highDragModel.PredictAsync(testFrame, CancellationToken.None);
        var ensembleResult = await ensemble.PredictAsync(testFrame, CancellationToken.None);

        float expectedAvg = (lowResult.Values[0] + highResult.Values[0]) / 2f;
        ensembleResult.Values[0].Should().BeApproximately(expectedAvg, 0.01f,
            "equal-weight ensemble should equal arithmetic mean of both models");
    }

    // ─── Test 4: ChainedModel feeds physics output as feature into RF ─────────────

    [Fact]
    public async Task ChainedModel_PhysicsOutputFeedsIntoRf_ProducesFiniteResults()
    {
        var physicsAdapter = new PhysicsAdapter();
        var mlAdapter = new MlNetAdapter();

        using var trainFrame = MakeTrainingFrame();

        var baselineModel = await physicsAdapter.TrainAsync(MakeDragConfig(), trainFrame, null, CancellationToken.None);

        // For the chain test: RF sees (radial_speed_km_s + chained_{StageName}) columns.
        // Column name must match what ChainedModel appends: "chained_{StageName}".
        var physPreds = await baselineModel.PredictAsync(trainFrame, CancellationToken.None);
        string chainedColName = $"chained_{baselineModel.StageName}";
        using var augmented = trainFrame.AddColumn(chainedColName, physPreds.Values);

        var rfChainConfig = new StageConfig("rf_chain", "MlNet", "FastForest", "src",
            ["radial_speed_km_s", chainedColName],
            "arrival_time_hours",
            new Dictionary<string, object>
            {
                ["number_of_trees"] = 30,
                ["number_of_leaves"] = 15,
                ["feature_fraction"] = 0.7f
            });

        var rfModel = await mlAdapter.TrainAsync(rfChainConfig, augmented, null, CancellationToken.None);
        var chained = new ChainedModel(baselineModel, rfModel, "physics→rf");

        using var testFrame = MakeSpeedOnlyFrame([600f, 1000f, 1600f]);
        var result = await chained.PredictAsync(testFrame, CancellationToken.None);

        result.Values.Should().HaveCount(3);
        result.Values.Should().OnlyContain(v => !float.IsNaN(v) && v > 0f);
    }
}
