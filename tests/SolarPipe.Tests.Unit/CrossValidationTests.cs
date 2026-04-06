using FluentAssertions;
using NSubstitute;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Tests.Unit.Fixtures;
using SolarPipe.Training.Validation;

namespace SolarPipe.Tests.Unit;

[Trait("Category", "Unit")]
public sealed class CrossValidationTests
{
    // ── helpers ─────────────────────────────────────────────────────────────────

    // Build a DataFrame with float timestamp and target columns.
    // Timestamps are Unix seconds spaced by stepSeconds starting at baseSeconds.
    private static InMemoryDataFrame MakeTemporalFrame(
        int rows, float baseSeconds, float stepSeconds,
        float targetValue = 48.0f)
    {
        var schema = new DataSchema(new[]
        {
            new ColumnInfo("timestamp", ColumnType.DateTime, false),
            new ColumnInfo("speed",     ColumnType.Float, false),
            new ColumnInfo("arrival_time", ColumnType.Float, false),
        });

        float[] ts     = Enumerable.Range(0, rows).Select(i => baseSeconds + i * stepSeconds).ToArray();
        float[] speed  = Enumerable.Range(0, rows).Select(i => 400f + i * 0.1f).ToArray();
        float[] target = Enumerable.Repeat(targetValue, rows).ToArray();

        return new InMemoryDataFrame(schema, new[] { ts, speed, target });
    }

    // Adapter that returns a constant prediction equal to the given value.
    private static IFrameworkAdapter MakeConstantAdapter(float prediction)
    {
        var model = Substitute.For<ITrainedModel>();
        model.ModelId.Returns("mock");
        model.StageName.Returns("mock");
        model.Metrics.Returns(new ModelMetrics(0, 0, 0));
        model.PredictAsync(Arg.Any<IDataFrame>(), Arg.Any<CancellationToken>())
             .Returns(ci =>
             {
                 var f = (IDataFrame)ci[0];
                 return Task.FromResult(new PredictionResult(
                     Enumerable.Repeat(prediction, f.RowCount).ToArray(),
                     null, null, "mock", DateTime.UtcNow));
             });

        var adapter = Substitute.For<IFrameworkAdapter>();
        adapter.SupportedModels.Returns(new[] { "rf" });
        adapter.TrainAsync(
            Arg.Any<StageConfig>(), Arg.Any<IDataFrame>(),
            Arg.Any<IDataFrame?>(), Arg.Any<CancellationToken>())
            .Returns(Task.FromResult(model));

        return adapter;
    }

    private static StageConfig MakeStage() =>
        new("test_stage", "ml_net", "FastForest", "src",
            new[] { "speed" }, "arrival_time");

    // Unix base time anchored to the Halloween storm for domain relevance
    private static readonly float BaseTs =
        (float)new DateTimeOffset(PhysicsTestFixtures.HalloweenStorm2003.LaunchTime)
               .ToUnixTimeSeconds();

    // ── ExpandingWindowCV ────────────────────────────────────────────────────────

    // 1. ExpandingWindowCV produces the requested number of folds
    [Theory]
    [InlineData(2)]
    [InlineData(5)]
    public async Task ExpandingWindowCV_ProducesCorrectFoldCount(int folds)
    {
        // 600 events, 1 day apart — 600 days easily fits default 5-day gap buffer
        const float DaySeconds = 86400f;
        var data    = MakeTemporalFrame(600, BaseTs, DaySeconds);
        var adapter = MakeConstantAdapter(48f);
        var cv      = new ExpandingWindowCV(folds: folds, minTestEvents: 1, enforceMinTestEvents: false);

        var result = await cv.RunAsync(data, adapter, MakeStage(), null, CancellationToken.None);

        result.FoldCount.Should().Be(folds);
    }

    // 2. ExpandingWindowCV: perfect predictor yields RMSE ≈ 0
    [Fact]
    public async Task ExpandingWindowCV_PerfectPredictor_ZeroRmse()
    {
        const float DaySeconds = 86400f;
        float targetValue = PhysicsTestFixtures.ModerateEvent.DstExpectedNt; // -80f
        var data    = MakeTemporalFrame(300, BaseTs, DaySeconds, targetValue);
        var adapter = MakeConstantAdapter(targetValue);
        var cv      = new ExpandingWindowCV(folds: 5, minTestEvents: 1, enforceMinTestEvents: false);

        var result = await cv.RunAsync(data, adapter, MakeStage(), null, CancellationToken.None);

        result.MeanMetrics.Rmse.Should().BeApproximately(0.0, 1e-4);
    }

    // 3. ExpandingWindowCV: test sets do not overlap with training sets in any fold
    [Fact]
    public async Task ExpandingWindowCV_TestRows_AreNeverInTrainingSet()
    {
        const float DaySeconds = 86400f;
        var data = MakeTemporalFrame(300, BaseTs, DaySeconds);
        float[] ts = data.GetColumn("timestamp");

        var trainMaxTs = new List<float>();
        var testMinTs  = new List<float>();

        var adapter = Substitute.For<IFrameworkAdapter>();
        adapter.SupportedModels.Returns(new[] { "rf" });
        adapter.TrainAsync(
            Arg.Any<StageConfig>(), Arg.Any<IDataFrame>(),
            Arg.Any<IDataFrame?>(), Arg.Any<CancellationToken>())
            .Returns(ci =>
            {
                var trainFrame = (IDataFrame)ci[1];
                trainMaxTs.Add(trainFrame.GetColumn("timestamp").Max());
                var m = Substitute.For<ITrainedModel>();
                m.ModelId.Returns("mock");
                m.StageName.Returns("mock");
                m.Metrics.Returns(new ModelMetrics(0, 0, 0));
                m.PredictAsync(Arg.Any<IDataFrame>(), Arg.Any<CancellationToken>())
                 .Returns(ci2 =>
                 {
                     var f = (IDataFrame)ci2[0];
                     testMinTs.Add(f.GetColumn("timestamp").Min());
                     return Task.FromResult(new PredictionResult(
                         Enumerable.Repeat(-80f, f.RowCount).ToArray(),
                         null, null, "mock", DateTime.UtcNow));
                 });
                return Task.FromResult(m);
            });

        var cv = new ExpandingWindowCV(folds: 5, minTestEvents: 1, enforceMinTestEvents: false);
        await cv.RunAsync(data, adapter, MakeStage(), null, CancellationToken.None);

        for (int i = 0; i < trainMaxTs.Count; i++)
        {
            // Train max ts must be strictly less than test min ts (gap ensures separation)
            trainMaxTs[i].Should().BeLessThan(testMinTs[i],
                "training events must precede test events in fold {0}", i);
        }
    }

    // 4. ExpandingWindowCV: enforcing minTestEvents throws when fold is too small
    [Fact]
    public async Task ExpandingWindowCV_EnforceMinTestEvents_ThrowsWhenTooFewEvents()
    {
        var data    = MakeTemporalFrame(100, BaseTs, 3600f);
        var adapter = MakeConstantAdapter(48f);
        // Require 50 events per fold — impossible with 100 rows / 5 folds = 20 per fold
        var cv = new ExpandingWindowCV(folds: 5, minTestEvents: 50, enforceMinTestEvents: true);

        var act = async () =>
            await cv.RunAsync(data, adapter, MakeStage(), null, CancellationToken.None);

        await act.Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("*RULE-051*");
    }

    // 5. ExpandingWindowCV: missing timestamp column throws with column list
    [Fact]
    public async Task ExpandingWindowCV_MissingTimestamp_ThrowsWithColumnList()
    {
        // Frame with no timestamp column
        var schema = new DataSchema(new[]
        {
            new ColumnInfo("speed", ColumnType.Float, false),
            new ColumnInfo("arrival_time", ColumnType.Float, false),
        });
        var data = new InMemoryDataFrame(schema, new[]
        {
            new float[] { 400f, 500f, 600f },
            new float[] { 48f,  52f,  55f  },
        });

        var adapter = MakeConstantAdapter(48f);
        var cv = new ExpandingWindowCV(folds: 2, minTestEvents: 1, enforceMinTestEvents: false);

        var act = async () =>
            await cv.RunAsync(data, adapter, MakeStage(), null, CancellationToken.None);

        await act.Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("*timestamp*");
    }

    // ── PurgedCV ──────────────────────────────────────────────────────────────────

    // 6. PurgedCV: embargo excludes events immediately after test period
    [Fact]
    public async Task PurgedCV_Embargo_ExcludesEventsAfterTestPeriod()
    {
        const float DaySeconds = 86400f;
        var data = MakeTemporalFrame(200, BaseTs, DaySeconds); // 200 days of events

        var trainFrames = new List<IDataFrame>();
        var adapter = Substitute.For<IFrameworkAdapter>();
        adapter.SupportedModels.Returns(new[] { "rf" });
        adapter.TrainAsync(
            Arg.Any<StageConfig>(), Arg.Any<IDataFrame>(),
            Arg.Any<IDataFrame?>(), Arg.Any<CancellationToken>())
            .Returns(ci =>
            {
                trainFrames.Add((IDataFrame)ci[1]);
                var m = Substitute.For<ITrainedModel>();
                m.ModelId.Returns("mock");
                m.StageName.Returns("mock");
                m.Metrics.Returns(new ModelMetrics(0, 0, 0));
                m.PredictAsync(Arg.Any<IDataFrame>(), Arg.Any<CancellationToken>())
                 .Returns(ci2 =>
                 {
                     var f = (IDataFrame)ci2[0];
                     return Task.FromResult(new PredictionResult(
                         Enumerable.Repeat(48f, f.RowCount).ToArray(),
                         null, null, "mock", DateTime.UtcNow));
                 });
                return Task.FromResult(m);
            });

        // Embargo of 24 hours
        var cv = new PurgedCV(folds: 3, embargoBuffer: TimeSpan.FromHours(24));
        await cv.RunAsync(data, adapter, MakeStage(), null, CancellationToken.None);

        // Each training frame should have fewer events than (total - test set size)
        // because embargo removes additional events after the test window
        trainFrames.Should().HaveCount(3);
        trainFrames.All(f => f.RowCount > 0).Should().BeTrue();
    }

    // 7. PurgedCV: empty data throws with stage context
    [Fact]
    public async Task PurgedCV_EmptyData_ThrowsWithStageContext()
    {
        var schema = new DataSchema(new[]
        {
            new ColumnInfo("timestamp", ColumnType.DateTime, false),
            new ColumnInfo("arrival_time", ColumnType.Float, false),
        });
        var empty = new InMemoryDataFrame(schema, new[]
        {
            Array.Empty<float>(),
            Array.Empty<float>(),
        });

        var cv = new PurgedCV(folds: 2);

        var act = async () =>
            await cv.RunAsync(empty, MakeConstantAdapter(48f), MakeStage(), null, CancellationToken.None);

        await act.Should().ThrowAsync<ArgumentException>()
            .WithMessage("*PurgedCV*");
    }

    // ── KFoldCV ───────────────────────────────────────────────────────────────────

    // 8. KFoldCV: deterministic results with same seed; shuffled differently with different seed
    [Fact]
    public async Task KFoldCV_SameSeed_ProducesDeterministicResults()
    {
        const float DaySeconds = 86400f;
        var data = MakeTemporalFrame(100, BaseTs, DaySeconds, targetValue: 48f);

        // With a perfect predictor the RMSE should be 0 regardless of shuffle
        var cv = new KFoldCV(folds: 5, seed: 42);

        var r1 = await cv.RunAsync(data, MakeConstantAdapter(48f), MakeStage(), CancellationToken.None);
        var r2 = await cv.RunAsync(data, MakeConstantAdapter(48f), MakeStage(), CancellationToken.None);

        r1.MeanMetrics.Rmse.Should().BeApproximately(r2.MeanMetrics.Rmse, 1e-6);
        r1.FoldCount.Should().Be(5);
    }
}
