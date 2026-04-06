using FluentAssertions;
using NSubstitute;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Training.MockData;

namespace SolarPipe.Tests.Unit;

[Trait("Category", "Unit")]
public sealed class MockDataStrategyTests
{
    // ─── helpers ────────────────────────────────────────────────────────────────

    private static InMemoryDataFrame MakeFrame(int rows, params string[] columns)
    {
        var infos = columns.Select(c => new ColumnInfo(c, ColumnType.Float, false)).ToList();
        var schema = new DataSchema(infos);
        var data = columns.Select(_ => Enumerable.Range(0, rows).Select(i => (float)i).ToArray()).ToArray();
        return new InMemoryDataFrame(schema, data);
    }

    private static ITrainedModel MakeModel(string id, float constant)
    {
        var m = Substitute.For<ITrainedModel>();
        m.ModelId.Returns(id);
        m.StageName.Returns(id);
        m.Metrics.Returns(new ModelMetrics(0, 0, 0));
        m.PredictAsync(Arg.Any<IDataFrame>(), Arg.Any<CancellationToken>())
            .Returns(ci =>
            {
                var frame = (IDataFrame)ci[0];
                return Task.FromResult(new PredictionResult(
                    Enumerable.Repeat(constant, frame.RowCount).ToArray(),
                    null, null, id, DateTime.UtcNow));
            });
        return m;
    }

    private static IFrameworkAdapter MakeAdapter(ITrainedModel model)
    {
        var a = Substitute.For<IFrameworkAdapter>();
        a.SupportedModels.Returns(new[] { "rf" });
        a.TrainAsync(Arg.Any<StageConfig>(), Arg.Any<IDataFrame>(), Arg.Any<IDataFrame?>(), Arg.Any<CancellationToken>())
            .Returns(Task.FromResult(model));
        return a;
    }

    private static StageConfig MakeStage(string target = "arrival_time") =>
        new("test_stage", "physics", "rf", "src",
            new[] { "speed", "density" }, target);

    // ─── MockDataStrategyFactory ─────────────────────────────────────────────────

    // 1. Factory creates the correct concrete strategy
    [Theory]
    [InlineData(MockDataStrategyType.PretrainThenFinetune)]
    [InlineData(MockDataStrategyType.MixedTraining)]
    [InlineData(MockDataStrategyType.ResidualCalibration)]
    public void Factory_Creates_CorrectStrategyType(MockDataStrategyType strategyType)
    {
        var adapter = MakeAdapter(MakeModel("m", 10f));
        var config = new MockDataConfig(strategyType, SyntheticWeight: 0.4f);

        var strategy = MockDataStrategyFactory.Create(config, adapter);

        strategy.Should().NotBeNull();
    }

    // 2. MixedTraining: merged frame has rows from both datasets
    [Fact]
    public async Task MixedTraining_MergesRows_FromBothDatasets()
    {
        var model = MakeModel("m", 5f);
        IDataFrame? capturedFrame = null;
        var adapter = Substitute.For<IFrameworkAdapter>();
        adapter.SupportedModels.Returns(new[] { "rf" });
        adapter.TrainAsync(Arg.Any<StageConfig>(), Arg.Any<IDataFrame>(), Arg.Any<IDataFrame?>(), Arg.Any<CancellationToken>())
            .Returns(ci =>
            {
                capturedFrame = (IDataFrame)ci[1];
                return Task.FromResult(model);
            });

        var synth = MakeFrame(10, "speed", "density", "arrival_time");
        var obs = MakeFrame(20, "speed", "density", "arrival_time");
        var stage = MakeStage();

        var config = new MockDataConfig(MockDataStrategyType.MixedTraining, SyntheticWeight: 0.5f);
        var sut = MockDataStrategyFactory.Create(config, adapter);

        await sut.TrainAsync(stage, synth, obs, CancellationToken.None);

        capturedFrame.Should().NotBeNull();
        capturedFrame!.RowCount.Should().BeGreaterThan(obs.RowCount, "merged > obs alone");
        capturedFrame.RowCount.Should().BeLessOrEqualTo(obs.RowCount + synth.RowCount);
    }

    // 3. MixedTraining: SyntheticWeight 0 or 1 throws ArgumentOutOfRangeException
    [Theory]
    [InlineData(0f)]
    [InlineData(1f)]
    public void MixedTraining_InvalidWeight_ThrowsArgumentOutOfRange(float weight)
    {
        var adapter = MakeAdapter(MakeModel("m", 0f));
        var config = new MockDataConfig(MockDataStrategyType.MixedTraining, SyntheticWeight: weight);

        var act = () => MockDataStrategyFactory.Create(config, adapter);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    // 4. PretrainThenFinetune: adapter is called twice
    [Fact]
    public async Task PretrainThenFinetune_CallsAdapter_Twice()
    {
        var model = MakeModel("m", 1f);
        var adapter = Substitute.For<IFrameworkAdapter>();
        adapter.SupportedModels.Returns(new[] { "rf" });
        adapter.TrainAsync(Arg.Any<StageConfig>(), Arg.Any<IDataFrame>(), Arg.Any<IDataFrame?>(), Arg.Any<CancellationToken>())
            .Returns(Task.FromResult(model));

        var synth = MakeFrame(10, "speed", "density", "arrival_time");
        var obs = MakeFrame(10, "speed", "density", "arrival_time");
        var stage = MakeStage();

        var config = new MockDataConfig(MockDataStrategyType.PretrainThenFinetune);
        var sut = MockDataStrategyFactory.Create(config, adapter);

        await sut.TrainAsync(stage, synth, obs, CancellationToken.None);

        await adapter.Received(2).TrainAsync(
            Arg.Any<StageConfig>(), Arg.Any<IDataFrame>(), Arg.Any<IDataFrame?>(), Arg.Any<CancellationToken>());
    }

    // 5. ResidualCalibration: final prediction = base + correction
    [Fact]
    public async Task ResidualCalibration_Prediction_IsSumOfBaseAndCorrection()
    {
        // base model returns 100f, correction model returns 5f → expect 105f
        var baseModel = MakeModel("base", 100f);
        var correctionModel = MakeModel("corr", 5f);

        int trainCallCount = 0;
        var adapter = Substitute.For<IFrameworkAdapter>();
        adapter.SupportedModels.Returns(new[] { "rf" });
        adapter.TrainAsync(Arg.Any<StageConfig>(), Arg.Any<IDataFrame>(), Arg.Any<IDataFrame?>(), Arg.Any<CancellationToken>())
            .Returns(ci =>
            {
                trainCallCount++;
                return Task.FromResult(trainCallCount == 1 ? baseModel : correctionModel);
            });

        var synth = MakeFrame(5, "speed", "density", "arrival_time");
        var obs = MakeFrame(5, "speed", "density", "arrival_time");
        var stage = MakeStage();

        var config = new MockDataConfig(MockDataStrategyType.ResidualCalibration);
        var sut = MockDataStrategyFactory.Create(config, adapter);

        var result = await sut.TrainAsync(stage, synth, obs, CancellationToken.None);
        var prediction = await result.PredictAsync(obs, CancellationToken.None);

        prediction.Values.Should().AllSatisfy(v => v.Should().BeApproximately(105f, 0.001f));
    }

    // 6. Empty syntheticData throws ArgumentException with stage context
    [Fact]
    public async Task AnyStrategy_EmptySyntheticData_ThrowsWithStageContext()
    {
        var adapter = MakeAdapter(MakeModel("m", 0f));
        var schema = new DataSchema(new[] { new ColumnInfo("speed", ColumnType.Float, false) });
        var emptySynth = new InMemoryDataFrame(schema, new[] { Array.Empty<float>() });
        var obs = MakeFrame(5, "speed", "density", "arrival_time");

        var config = new MockDataConfig(MockDataStrategyType.PretrainThenFinetune);
        var sut = MockDataStrategyFactory.Create(config, adapter);

        var act = async () => await sut.TrainAsync(MakeStage(), emptySynth, obs, CancellationToken.None);

        await act.Should().ThrowAsync<ArgumentException>()
            .WithMessage("*PretrainThenFinetune*");
    }
}
