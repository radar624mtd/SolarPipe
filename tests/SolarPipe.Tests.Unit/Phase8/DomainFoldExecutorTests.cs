using FluentAssertions;
using NSubstitute;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Training.Sweep;

namespace SolarPipe.Tests.Unit.Phase8;

// Tests for the domain pipeline fold execution logic, exercised via DomainPipelineSweep
// (public surface) which wraps the internal DomainFoldExecutor.
//
// Mock strategy:
//   - MlNet adapter: returns constant 40f predictions for every train/predict call.
//   - Physics adapter: returns constant 40f predictions (drag + burton baselines).
[Trait("Category", "Unit")]
public sealed class DomainFoldExecutorTests
{
    // ── Helpers ───────────────────────────────────────────────────────────────

    private static IDataFrame MakeFrame(int rows)
    {
        var colNames = new[]
        {
            "launch_time",
            "transit_time_hours",
            "dst_min_nt",
            "storm_duration_hours",
            "cme_speed_kms",
            "sw_speed_ambient",
            "sw_density_ambient",
            "sw_bz_ambient",
            "sw_bt_ambient",
            "cme_latitude",
            "cme_longitude",
            "cme_half_angle_deg",
            "flare_class_numeric",
            "usflux",
            "meanshr",
            "totusjz",
            "totpot",
            "has_sharp_obs",
            "has_flare_obs",
            "has_bz_obs",
            "has_mass_obs",
            "arrival_speed_kms",
        };

        var columns = colNames.Select(n => new ColumnInfo(n, ColumnType.Float, true)).ToList();
        var schema  = new DataSchema(columns);

        var data = colNames.Select(name =>
        {
            if (name == "launch_time")
            {
                // Spread evenly so fold builder can split them
                float start = (float)new DateTimeOffset(2015, 1, 1, 0, 0, 0, TimeSpan.Zero).ToUnixTimeSeconds();
                float step  = (float)TimeSpan.FromDays(365.0 / rows).TotalSeconds;
                return Enumerable.Range(0, rows).Select(i => start + i * step).ToArray();
            }
            if (name == "transit_time_hours") return Enumerable.Repeat(48f, rows).ToArray();
            if (name == "dst_min_nt")         return Enumerable.Repeat(-50f, rows).ToArray();
            if (name == "storm_duration_hours") return Enumerable.Repeat(19.65f, rows).ToArray();
            if (name == "cme_speed_kms")      return Enumerable.Repeat(600f, rows).ToArray();
            if (name == "sw_speed_ambient")   return Enumerable.Repeat(400f, rows).ToArray();
            if (name == "sw_density_ambient") return Enumerable.Repeat(5f, rows).ToArray();
            if (name == "sw_bz_ambient")      return Enumerable.Repeat(-5f, rows).ToArray();
            if (name == "sw_bt_ambient")      return Enumerable.Repeat(5f, rows).ToArray();
            if (name == "cme_latitude")       return Enumerable.Repeat(10f, rows).ToArray();
            if (name == "cme_longitude")      return Enumerable.Repeat(5f, rows).ToArray();
            if (name == "cme_half_angle_deg") return Enumerable.Repeat(30f, rows).ToArray();
            return Enumerable.Repeat(1f, rows).ToArray();
        }).ToArray();

        return new InMemoryDataFrame(schema, data);
    }

    private static DomainPipelineRunConfig BuildRunConfig(int folds = 2)
    {
        static IReadOnlyDictionary<string, object> Hyper(params (string k, object v)[] kv) =>
            kv.ToDictionary(t => t.k, t => t.v);

        var stages = new Dictionary<string, StageConfig>(StringComparer.OrdinalIgnoreCase)
        {
            ["origination_rf"] = new StageConfig(
                "origination_rf", "MlNet", "FastForest", "",
                new[] { "cme_speed_kms", "cme_latitude", "cme_longitude" }, "arrival_speed_kms"),

            ["drag_baseline"] = new StageConfig(
                "drag_baseline", "Physics", "DragBased", "",
                new[] { "cme_speed_kms", "sw_speed_ambient" }, "transit_time_hours",
                Hyper(("drag_parameter", (object)"0.2e-7"),
                      ("background_speed_km_s", (object)"400"),
                      ("r_start_rs", (object)"21.5"),
                      ("r_stop_rs", (object)"215.0"))),

            ["drag_baseline_v2"] = new StageConfig(
                "drag_baseline_v2", "Physics", "DragBasedV2", "",
                new[] { "cme_speed_kms", "sw_speed_ambient" }, "arrival_speed_kms",
                Hyper(("drag_parameter", (object)"0.2e-7"),
                      ("background_speed_km_s", (object)"400"),
                      ("r_start_rs", (object)"21.5"),
                      ("r_stop_rs", (object)"215.0"))),

            ["transit_rf_residual"] = new StageConfig(
                "transit_rf_residual", "MlNet", "FastForest", "",
                new[] { "pred_arrival_speed_kms", "sw_speed_ambient" }, "transit_time_hours"),

            ["burton_ode"] = new StageConfig(
                "burton_ode", "Physics", "BurtonOde", "",
                new[] { "sw_bz_ambient", "sw_speed_ambient" }, "dst_min_nt"),

            ["impact_rf_residual"] = new StageConfig(
                "impact_rf_residual", "MlNet", "FastForest", "",
                new[] { "pred_transit_time_hours", "sw_bz_ambient" }, "dst_min_nt"),

            ["meta_arrival_rf"] = new StageConfig(
                "meta_arrival_rf", "MlNet", "FastForest", "",
                new[] { "pred_origination_arrival_speed_kms", "pred_transit_time_hours" }, "transit_time_hours"),

            ["meta_intensity_rf"] = new StageConfig(
                "meta_intensity_rf", "MlNet", "FastForest", "",
                new[] { "pred_origination_arrival_speed_kms", "pred_dst_min_nt" }, "dst_min_nt"),

            ["meta_duration_rf"] = new StageConfig(
                "meta_duration_rf", "MlNet", "FastForest", "",
                new[] { "pred_dst_min_nt", "pred_transit_time_hours" }, "storm_duration_hours"),
        };

        return new DomainPipelineRunConfig(
            Name:          "test_domain_pipeline",
            Folds:         folds,
            GapBufferDays: 0,
            MinTestEvents: 1,
            Origination:   new DomainGroupRunConfig("arrival_speed_kms", "origination_rf", "drag_baseline_v2", null),
            Transit:       new DomainGroupRunConfig("transit_time_hours", null, "drag_baseline", "transit_rf_residual"),
            Impact:        new DomainGroupRunConfig("dst_min_nt", null, "burton_ode", "impact_rf_residual"),
            MetaLearnerStages: new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arrival_time_hours"]   = "meta_arrival_rf",
                ["storm_intensity_nt"]   = "meta_intensity_rf",
                ["storm_duration_hours"] = "meta_duration_rf",
            },
            Stages: stages);
    }

    // Builds frame-size-aware mock adapters: PredictAsync returns an array sized to the input frame.
    private static IReadOnlyList<IFrameworkAdapter> BuildAdapters()
    {
        static ITrainedModel MakeSizeAwareModel(string id)
        {
            var model = Substitute.For<ITrainedModel>();
            model.PredictAsync(Arg.Any<IDataFrame>(), Arg.Any<CancellationToken>())
                .Returns(ci =>
                {
                    var frame = ci.Arg<IDataFrame>();
                    var preds = Enumerable.Repeat(40f, frame.RowCount).ToArray();
                    return Task.FromResult(new PredictionResult(preds, null, null, id, DateTime.UtcNow));
                });
            return model;
        }

        var mlNet = Substitute.For<IFrameworkAdapter>();
        mlNet.FrameworkType.Returns(FrameworkType.MlNet);
        mlNet.SupportedModels.Returns(new[] { "FastForest" });
        mlNet.TrainAsync(
            Arg.Any<StageConfig>(), Arg.Any<IDataFrame>(), Arg.Any<IDataFrame?>(), Arg.Any<CancellationToken>())
            .Returns(_ => Task.FromResult(MakeSizeAwareModel("mock_ml")));

        var physics = Substitute.For<IFrameworkAdapter>();
        physics.FrameworkType.Returns(FrameworkType.Physics);
        physics.SupportedModels.Returns(new[] { "DragBased", "DragBasedV2", "BurtonOde" });
        physics.TrainAsync(
            Arg.Any<StageConfig>(), Arg.Any<IDataFrame>(), Arg.Any<IDataFrame?>(), Arg.Any<CancellationToken>())
            .Returns(_ => Task.FromResult(MakeSizeAwareModel("mock_phys")));

        return new IFrameworkAdapter[] { mlNet, physics };
    }

    // ── DomainPipelineSweep.RunAsync — success ────────────────────────────────

    [Fact]
    public async Task RunAsync_ValidData_ReturnsResultWithFoldMetrics()
    {
        int rows   = 60;  // enough for 2-fold expanding window
        var data   = MakeFrame(rows);
        var config = BuildRunConfig(folds: 2);
        var sweep  = new DomainPipelineSweep(BuildAdapters());

        var result = await sweep.RunAsync(config, data, default);

        result.PipelineName.Should().Be("test_domain_pipeline");
        result.FoldMetrics.Count.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task RunAsync_ValidData_MaeValuesAreNonNegative()
    {
        int rows   = 60;
        var data   = MakeFrame(rows);
        var config = BuildRunConfig(folds: 2);
        var sweep  = new DomainPipelineSweep(BuildAdapters());

        var result = await sweep.RunAsync(config, data, default);

        result.MeanMaeTransit.Should().BeGreaterThanOrEqualTo(0.0);
        result.MeanMaeDst.Should().BeGreaterThanOrEqualTo(0.0);
        result.MeanMaeDuration.Should().BeGreaterThanOrEqualTo(0.0);
        result.MeanMaePhysicsBaseline.Should().BeGreaterThanOrEqualTo(0.0);
    }

    [Fact]
    public async Task RunAsync_MeanMaeMatchesFoldAverage()
    {
        int rows   = 60;
        var data   = MakeFrame(rows);
        var config = BuildRunConfig(folds: 2);
        var sweep  = new DomainPipelineSweep(BuildAdapters());

        var result = await sweep.RunAsync(config, data, default);

        double expectedTransit = result.FoldMetrics.Average(f => f.MaeTransit);
        result.MeanMaeTransit.Should().BeApproximately(expectedTransit, 1e-9);
    }

    // ── DomainPipelineSweep — validation ──────────────────────────────────────

    [Fact]
    public void Constructor_NullAdapters_Throws()
    {
        Action act = () => new DomainPipelineSweep(null!);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public async Task RunAsync_NoDateTimeColumn_ThrowsInvalidOperation()
    {
        // Frame without a DateTime or launch_time column
        var colNames = new[] { "cme_speed_kms", "transit_time_hours" };
        var schema   = new DataSchema(colNames.Select(n => new ColumnInfo(n, ColumnType.Float, true)).ToList());
        var data     = new InMemoryDataFrame(schema, colNames.Select(_ => new float[] { 600f }).ToArray());
        var config   = BuildRunConfig();
        var sweep    = new DomainPipelineSweep(BuildAdapters());

        await Assert.ThrowsAsync<InvalidOperationException>(
            () => sweep.RunAsync(config, data, default));
    }

    [Fact]
    public async Task RunAsync_TooFewRowsForMinTestEvents_ThrowsInvalidOperation()
    {
        // Only 2 rows, min_test_events=100 → no valid folds
        var config = BuildRunConfig(folds: 2) with { MinTestEvents = 100 };
        var data   = MakeFrame(2);
        var sweep  = new DomainPipelineSweep(BuildAdapters());

        await Assert.ThrowsAsync<InvalidOperationException>(
            () => sweep.RunAsync(config, data, default));
    }

    // ── DomainPipelineResult record ───────────────────────────────────────────

    [Fact]
    public void DomainFoldMetrics_ValueEquality()
    {
        var m1 = new DomainFoldMetrics(0, 10.0, 20.0, 5.0, 25.0);
        var m2 = new DomainFoldMetrics(0, 10.0, 20.0, 5.0, 25.0);
        m1.Should().Be(m2);
    }

    [Fact]
    public void DomainPipelineResult_AggregatedMaeIsAverage()
    {
        var folds = new[]
        {
            new DomainFoldMetrics(0, 10.0, 20.0, 5.0, 25.0),
            new DomainFoldMetrics(1, 20.0, 40.0, 15.0, 35.0),
        };
        var result = new DomainPipelineResult(
            "p", folds,
            folds.Average(f => f.MaeTransit),
            folds.Average(f => f.MaeDst),
            folds.Average(f => f.MaeDuration),
            folds.Average(f => f.MaePhysicsBaseline));

        result.MeanMaeTransit.Should().BeApproximately(15.0, 1e-9);
        result.MeanMaeDst.Should().BeApproximately(30.0, 1e-9);
    }
}
