using FluentAssertions;
using NSubstitute;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Training.Adapters;
using SolarPipe.Training.Evaluation;
using SolarPipe.Training.Sweep;

namespace SolarPipe.Tests.Unit.Phase7;

[Trait("Category", "Unit")]
public sealed class ModelSweepPreFlightTests
{
    // ── All checks pass ───────────────────────────────────────────────────────

    [Fact]
    public async Task PreFlight_AllChecksPass_ReturnsSuccess()
    {
        var (sut, config, dataFrames) = BuildFixture();

        var result = await sut.ValidatePreFlightAsync(config, dataFrames, default);

        result.IsSuccess.Should().BeTrue();
        result.Failures.Should().BeEmpty();
    }

    // ── Data source failures ──────────────────────────────────────────────────

    [Fact]
    public async Task PreFlight_EmptyDataFrame_FailsWithDataSourceComponent()
    {
        var (sut, config, _) = BuildFixture();

        var emptySchema = new DataSchema(new[] { new ColumnInfo("x", ColumnType.Float, false) });
        var emptyFrame = new InMemoryDataFrame(emptySchema, new[] { Array.Empty<float>() });
        var dataFrames = new Dictionary<string, IDataFrame> { ["cme"] = emptyFrame };

        var result = await sut.ValidatePreFlightAsync(config, dataFrames, default);

        result.HasFailures.Should().BeTrue();
        result.Failures.Should().Contain(f => f.Component == "data_source");
    }

    // ── Adapter failures ──────────────────────────────────────────────────────

    [Fact]
    public async Task PreFlight_MissingAdapter_FailsWithAdapterComponent()
    {
        var (_, config, dataFrames) = BuildFixture();

        var noAdapters = Array.Empty<IFrameworkAdapter>();
        var sut = BuildSweep(noAdapters, sidecar: null);

        var result = await sut.ValidatePreFlightAsync(config, dataFrames, default);

        result.HasFailures.Should().BeTrue();
        result.Failures.Should().Contain(f => f.Component == "adapter");
    }

    // ── Config reference failures ─────────────────────────────────────────────

    [Fact]
    public async Task PreFlight_HypothesisReferencesUnknownStage_FailsWithConfigComponent()
    {
        var (sut, config, dataFrames) = BuildFixture();

        // Add hypothesis with bad stage ref — create new config with the bad entry
        var badHypotheses = config.Hypotheses.ToList();
        badHypotheses.Add(new SweepHypothesis("H_BAD", "unknown_stage", new[] { "nonexistent_stage" }));
        var badConfig = config with { Hypotheses = badHypotheses };

        var result = await sut.ValidatePreFlightAsync(badConfig, dataFrames, default);

        result.HasFailures.Should().BeTrue();
        result.Failures.Should().Contain(f => f.Component == "config");
    }

    // ── Write permission failures ─────────────────────────────────────────────

    [Fact]
    public async Task PreFlight_ReadOnlyRegistryPath_FailsWithRegistryOrCacheComponent()
    {
        var (_, config, dataFrames) = BuildFixture();

        // Use a path that is a file (cannot be used as directory)
        var tmpFile = Path.GetTempFileName();
        try
        {
            var sut = new ModelSweep(
                BuildAdapters(),
                new ComprehensiveMetricsEvaluator(),
                new NnlsEnsembleOptimizer(),
                cacheRoot: Path.GetTempPath(),
                registryRoot: tmpFile,  // file, not directory
                sidecarAdapter: null);

            var result = await sut.ValidatePreFlightAsync(config, dataFrames, default);
            result.HasFailures.Should().BeTrue();
            result.Failures.Should().Contain(f =>
                f.Component == "registry" || f.Component == "cache");
        }
        finally { File.Delete(tmpFile); }
    }

    // ── SweepId computation ───────────────────────────────────────────────────

    [Fact]
    public void ComputeSweepId_SameConfig_ProducesSameId()
    {
        var config = BuildRunConfig();
        var id1 = ModelSweep.ComputeSweepId(config);
        var id2 = ModelSweep.ComputeSweepId(config);
        id1.Should().Be(id2);
    }

    [Fact]
    public void ComputeSweepId_DifferentConfigName_ProducesDifferentId()
    {
        var config1 = BuildRunConfig("sweep_a");
        var config2 = BuildRunConfig("sweep_b");
        ModelSweep.ComputeSweepId(config1).Should().NotBe(ModelSweep.ComputeSweepId(config2));
    }

    [Fact]
    public void ComputeSweepId_StartsWithSweepPrefix()
    {
        var config = BuildRunConfig();
        ModelSweep.ComputeSweepId(config).Should().StartWith("sweep_");
    }

    // ── Fixture helpers ───────────────────────────────────────────────────────

    private static (ModelSweep sut, SweepRunConfig config, IReadOnlyDictionary<string, IDataFrame> dataFrames)
        BuildFixture()
    {
        var config = BuildRunConfig();
        var sut    = BuildSweep(BuildAdapters(), sidecar: null);

        var schema = new DataSchema(new[]
        {
            new ColumnInfo("feature", ColumnType.Float, false),
            new ColumnInfo("target",  ColumnType.Float, false),
        });
        var frame = new InMemoryDataFrame(schema,
            new[] { new float[] { 1f, 2f, 3f }, new float[] { 10f, 20f, 30f } });

        var dataFrames = new Dictionary<string, IDataFrame> { ["cme"] = frame };
        return (sut, config, dataFrames);
    }

    private static ModelSweep BuildSweep(
        IReadOnlyList<IFrameworkAdapter> adapters,
        GrpcSidecarAdapter? sidecar)
    {
        return new ModelSweep(
            adapters,
            new ComprehensiveMetricsEvaluator(),
            new NnlsEnsembleOptimizer(),
            cacheRoot: Path.Combine(Path.GetTempPath(), $"sweep_cache_{Guid.NewGuid():N}"),
            registryRoot: Path.Combine(Path.GetTempPath(), $"sweep_reg_{Guid.NewGuid():N}"),
            sidecarAdapter: sidecar);
    }

    private static IReadOnlyList<IFrameworkAdapter> BuildAdapters()
    {
        var adapter = Substitute.For<IFrameworkAdapter>();
        adapter.FrameworkType.Returns(FrameworkType.MlNet);
        return new[] { adapter };
    }

    private static SweepRunConfig BuildRunConfig(string name = "test_sweep")
    {
        var stages = new Dictionary<string, StageConfig>(StringComparer.OrdinalIgnoreCase)
        {
            ["s1"] = new StageConfig("s1", "MlNet", "FastForest", string.Empty,
                new[] { "f1" }, "target"),
            ["s2"] = new StageConfig("s2", "MlNet", "FastForest", string.Empty,
                new[] { "f1" }, "target"),
        };

        return new SweepRunConfig(
            Name: name,
            Parallel: false,
            Folds: 2,
            GapBufferDays: 1,
            MinTestEvents: 1,
            Hypotheses: new[]
            {
                new SweepHypothesis("H1", "s1 ^ s2", new[] { "s1", "s2" }),
            },
            Stages: stages);
    }
}
