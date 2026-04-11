using FluentAssertions;
using NSubstitute;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Prediction;

namespace SolarPipe.Tests.Unit.Phase7;

[Trait("Category", "Unit")]
public sealed class CompositionDecomposerTests
{
    private readonly CompositionDecomposer _sut = new();

    // ── Two-stage decomposition ───────────────────────────────────────────────

    [Fact]
    public async Task DecomposeAsync_TwoStages_WritesOutputFile()
    {
        var tmpDir = Path.Combine(Path.GetTempPath(), $"decomp_{Guid.NewGuid():N}");
        try
        {
            var (stages, data) = BuildFixture(
                s1Pred: new float[] { 50f, 60f, 40f },
                s2Pred: new float[] { 48f, 58f, 38f },
                obs:    new float[] { 48f, 58f, 38f });

            var result = await _sut.DecomposeAsync("H1", stages, data, "target", tmpDir, default);

            var expectedPath = Path.Combine(tmpDir, "composition_decomposition_H1.json");
            File.Exists(expectedPath).Should().BeTrue();
        }
        finally { if (Directory.Exists(tmpDir)) Directory.Delete(tmpDir, true); }
    }

    [Fact]
    public async Task DecomposeAsync_TwoStages_BothStagesInMetrics()
    {
        var tmpDir = Path.Combine(Path.GetTempPath(), $"decomp_{Guid.NewGuid():N}");
        try
        {
            var (stages, data) = BuildFixture(
                s1Pred: new float[] { 52f, 62f, 42f },
                s2Pred: new float[] { 49f, 59f, 39f },
                obs:    new float[] { 48f, 58f, 38f });

            var result = await _sut.DecomposeAsync("H2", stages, data, "target", tmpDir, default);

            result.StageMetrics.Keys.Should().Contain("baseline");
            result.StageMetrics.Keys.Should().Contain("correction");
        }
        finally { if (Directory.Exists(tmpDir)) Directory.Delete(tmpDir, true); }
    }

    [Fact]
    public async Task DecomposeAsync_SecondStagePerfect_PositiveReductionPct()
    {
        var tmpDir = Path.Combine(Path.GetTempPath(), $"decomp_{Guid.NewGuid():N}");
        try
        {
            var (stages, data) = BuildFixture(
                s1Pred: new float[] { 58f, 68f, 48f },    // mae = 10
                s2Pred: new float[] { 48f, 58f, 38f },    // perfect
                obs:    new float[] { 48f, 58f, 38f });

            var result = await _sut.DecomposeAsync("H3", stages, data, "target", tmpDir, default);

            var reductionEntry = result.ResidualReduction.Last();
            reductionEntry.ReductionPct.Should().BeGreaterThan(0);
        }
        finally { if (Directory.Exists(tmpDir)) Directory.Delete(tmpDir, true); }
    }

    [Fact]
    public async Task DecomposeAsync_CorrelationMatrix_DiagonalIsOne()
    {
        var tmpDir = Path.Combine(Path.GetTempPath(), $"decomp_{Guid.NewGuid():N}");
        try
        {
            var (stages, data) = BuildFixture(
                s1Pred: new float[] { 50f, 60f, 40f },
                s2Pred: new float[] { 48f, 58f, 38f },
                obs:    new float[] { 48f, 58f, 38f });

            var result = await _sut.DecomposeAsync("H4", stages, data, "target", tmpDir, default);

            result.CorrelationMatrix["baseline"]["baseline"].Should().BeApproximately(1.0, 1e-6);
            result.CorrelationMatrix["correction"]["correction"].Should().BeApproximately(1.0, 1e-6);
        }
        finally { if (Directory.Exists(tmpDir)) Directory.Delete(tmpDir, true); }
    }

    // ── Three-stage decomposition ─────────────────────────────────────────────

    [Fact]
    public async Task DecomposeAsync_ThreeStages_AllStagesPresent()
    {
        var tmpDir = Path.Combine(Path.GetTempPath(), $"decomp_{Guid.NewGuid():N}");
        try
        {
            var obs  = new float[] { 48f, 58f, 38f };
            var data = BuildData(obs);
            var stages = new List<(string, ITrainedModel)>
            {
                ("s1", MockModel(new float[] { 50f, 60f, 40f })),
                ("s2", MockModel(new float[] { 49f, 59f, 39f })),
                ("s3", MockModel(obs.ToArray())),
            };

            var result = await _sut.DecomposeAsync("H5", stages, data, "target", tmpDir, default);

            result.StageMetrics.Keys.Should().HaveCount(3);
        }
        finally { if (Directory.Exists(tmpDir)) Directory.Delete(tmpDir, true); }
    }

    // ── NaN propagation ───────────────────────────────────────────────────────

    [Fact]
    public async Task DecomposeAsync_NaNInPredictions_DoesNotThrow()
    {
        var tmpDir = Path.Combine(Path.GetTempPath(), $"decomp_{Guid.NewGuid():N}");
        try
        {
            var (stages, data) = BuildFixture(
                s1Pred: new float[] { float.NaN, 60f, 40f },
                s2Pred: new float[] { 48f, float.NaN, 38f },
                obs:    new float[] { 48f, 58f, 38f });

            var act = () => _sut.DecomposeAsync("HN", stages, data, "target", tmpDir, default);
            await act.Should().NotThrowAsync();
        }
        finally { if (Directory.Exists(tmpDir)) Directory.Delete(tmpDir, true); }
    }

    // ── Validation ────────────────────────────────────────────────────────────

    [Fact]
    public async Task DecomposeAsync_EmptyStages_Throws()
    {
        var data = BuildData(new float[] { 1f });
        var act  = () => _sut.DecomposeAsync("H0",
            new List<(string, ITrainedModel)>(), data, "target", "out", default);
        await act.Should().ThrowAsync<ArgumentException>();
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static (IReadOnlyList<(string Name, ITrainedModel Model)> stages, IDataFrame data) BuildFixture(
        float[] s1Pred, float[] s2Pred, float[] obs)
    {
        var data = BuildData(obs);
        var stages = new List<(string, ITrainedModel)>
        {
            ("baseline",   MockModel(s1Pred)),
            ("correction", MockModel(s2Pred)),
        };
        return (stages, data);
    }

    private static IDataFrame BuildData(float[] obs)
    {
        var schema = new DataSchema(new[]
        {
            new ColumnInfo("feature", ColumnType.Float, false),
            new ColumnInfo("target",  ColumnType.Float, false),
        });
        var features = Enumerable.Range(0, obs.Length).Select(i => (float)i).ToArray();
        return new InMemoryDataFrame(schema, new[] { features, obs });
    }

    private static ITrainedModel MockModel(float[] predictions)
    {
        var model = Substitute.For<ITrainedModel>();
        model.PredictAsync(Arg.Any<IDataFrame>(), Arg.Any<CancellationToken>())
             .Returns(Task.FromResult(new PredictionResult(
                 predictions, null, null, "m", DateTime.UtcNow)));
        model.ModelId.Returns("m");
        model.StageName.Returns("s");
        model.Metrics.Returns(new ModelMetrics(1, 1, 1));
        return model;
    }
}
