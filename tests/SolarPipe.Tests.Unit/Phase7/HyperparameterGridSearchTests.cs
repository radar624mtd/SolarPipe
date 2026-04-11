using FluentAssertions;
using NSubstitute;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Training.Evaluation;
using SolarPipe.Training.Sweep;

namespace SolarPipe.Tests.Unit.Phase7;

[Trait("Category", "Unit")]
public sealed class HyperparameterGridSearchTests
{
    // ── Grid construction ─────────────────────────────────────────────────────

    [Fact]
    public async Task Search_SmallGrid_UsesFullGrid()
    {
        // 2x2 = 4 combinations — within 200-combo limit
        var grid = new Dictionary<string, IReadOnlyList<object>>
        {
            ["trees"]  = new object[] { 50, 100 },
            ["leaves"] = new object[] { 10, 20 },
        };

        var (sut, baseConfig, folds, dbm) = BuildTestFixture();

        var result = await sut.SearchAsync(baseConfig, grid, folds, dbm, CancellationToken.None);

        result.UsedLatinHypercube.Should().BeFalse();
        result.Entries.Should().HaveCount(4);
    }

    [Fact]
    public async Task Search_LargeGrid_UsesLatinHypercube()
    {
        // 5x5x4x4x5 = 2000 — exceeds 200 limit → LHS(100)
        var grid = new Dictionary<string, IReadOnlyList<object>>
        {
            ["a"] = new object[] { 1, 2, 3, 4, 5 },
            ["b"] = new object[] { 1, 2, 3, 4, 5 },
            ["c"] = new object[] { 1, 2, 3, 4 },
            ["d"] = new object[] { 1, 2, 3, 4 },
            ["e"] = new object[] { 1, 2, 3, 4, 5 },
        };

        var (sut, baseConfig, folds, dbm) = BuildTestFixture();

        var result = await sut.SearchAsync(baseConfig, grid, folds, dbm, CancellationToken.None);

        result.UsedLatinHypercube.Should().BeTrue();
        result.Entries.Should().HaveCount(100);
    }

    [Fact]
    public async Task Search_ResultsSortedByMaeAscending()
    {
        var grid = new Dictionary<string, IReadOnlyList<object>>
        {
            ["trees"] = new object[] { 50, 100 },
        };

        var (sut, baseConfig, folds, dbm) = BuildTestFixture();
        var result = await sut.SearchAsync(baseConfig, grid, folds, dbm, CancellationToken.None);

        var maes = result.Entries.Select(e => e.Aggregated.MaeMean).ToList();
        maes.Should().BeInAscendingOrder();
    }

    [Fact]
    public async Task Search_BestHyperparametersMatchBestEntry()
    {
        var grid = new Dictionary<string, IReadOnlyList<object>>
        {
            ["trees"] = new object[] { 50, 100, 150 },
        };

        var (sut, baseConfig, folds, dbm) = BuildTestFixture();
        var result = await sut.SearchAsync(baseConfig, grid, folds, dbm, CancellationToken.None);

        result.BestHyperparameters.Should().BeEquivalentTo(result.Entries[0].Hyperparameters);
    }

    [Fact]
    public async Task Search_EmptyFolds_Throws()
    {
        var grid = new Dictionary<string, IReadOnlyList<object>>
        {
            ["trees"] = new object[] { 50 },
        };
        var (sut, baseConfig, _, dbm) = BuildTestFixture();
        var emptyFolds = new List<(IDataFrame, IDataFrame)>();

        var act = async () => await sut.SearchAsync(baseConfig, grid, emptyFolds, dbm, CancellationToken.None);

        await act.Should().ThrowAsync<ArgumentException>();
    }

    // ── Fixture ───────────────────────────────────────────────────────────────

    private static (HyperparameterGridSearch sut,
                    StageConfig baseConfig,
                    IReadOnlyList<(IDataFrame Train, IDataFrame Test)> folds,
                    IReadOnlyList<double> dbm)
        BuildTestFixture()
    {
        var schema = new DataSchema(new[]
        {
            new ColumnInfo("feature", ColumnType.Float, false),
            new ColumnInfo("target",  ColumnType.Float, false),
        });
        var trainFrame = new InMemoryDataFrame(schema,
            new[] { new float[] { 1f, 2f, 3f }, new float[] { 10f, 20f, 30f } });
        var testFrame = new InMemoryDataFrame(schema,
            new[] { new float[] { 4f, 5f }, new float[] { 40f, 50f } });

        var model = Substitute.For<ITrainedModel>();
        model.PredictAsync(Arg.Any<IDataFrame>(), Arg.Any<CancellationToken>())
             .Returns(ci =>
             {
                 var f = (IDataFrame)ci[0];
                 return Task.FromResult(new PredictionResult(
                     Enumerable.Repeat(45f, f.RowCount).ToArray(),
                     null, null, "m", DateTime.UtcNow));
             });
        model.Metrics.Returns(new ModelMetrics(1, 1, 1));
        model.ModelId.Returns("m");
        model.StageName.Returns("s");

        var adapter = Substitute.For<IFrameworkAdapter>();
        adapter.FrameworkType.Returns(FrameworkType.MlNet);
        adapter.TrainAsync(Arg.Any<StageConfig>(), Arg.Any<IDataFrame>(),
            Arg.Any<IDataFrame?>(), Arg.Any<CancellationToken>())
            .Returns(Task.FromResult(model));

        var evaluator = new ComprehensiveMetricsEvaluator();
        var sut = new HyperparameterGridSearch(adapter, evaluator);

        var baseConfig = new StageConfig("test", "MlNet", "FastForest",
            string.Empty, new[] { "feature" }, "target");

        var folds = new List<(IDataFrame, IDataFrame)> { (trainFrame, testFrame) };
        var dbm = new List<double> { 12.0 };

        return (sut, baseConfig, folds, dbm);
    }
}
