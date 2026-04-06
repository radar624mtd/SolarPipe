using FluentAssertions;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Training.Adapters;
using SolarPipe.Training.Registry;

namespace SolarPipe.Tests.Unit;

// RULE-013: Sequential execution — ML.NET native memory; parallel registry tests may collide on temp dirs
[Collection("ML")]
[Trait("Category", "Unit")]
public class FileSystemModelRegistryTests : IDisposable
{
    private readonly string _tempDir;
    private readonly FileSystemModelRegistry _registry;
    private readonly MlNetAdapter _adapter = new();

    public FileSystemModelRegistryTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"solarpipe_reg_{Guid.NewGuid():N}");
        _registry = new FileSystemModelRegistry(_tempDir);
    }

    public void Dispose()
    {
        _registry.Dispose();
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
    }

    private static InMemoryDataFrame MakeTrainingFrame(int rows = 60)
    {
        var schema = new DataSchema(new[]
        {
            new ColumnInfo("speed",   ColumnType.Float, true),
            new ColumnInfo("density", ColumnType.Float, true),
            new ColumnInfo("dst",     ColumnType.Float, true),
        }.ToList());

        var speed   = Enumerable.Range(0, rows).Select(i => 300f + i * 7.5f).ToArray();
        var density = Enumerable.Range(0, rows).Select(i => 5f  + i * 0.3f).ToArray();
        var dst     = Enumerable.Range(0, rows).Select(i => -(speed[i] * 0.1f + density[i] * 2f)).ToArray();

        return new InMemoryDataFrame(schema, [speed, density, dst]);
    }

    private static StageConfig MakeConfig(string stageName = "test_stage") =>
        new StageConfig(
            Name: stageName,
            Framework: "mlnet",
            ModelType: "FastForest",
            DataSource: "test_src",
            Features: ["speed", "density"],
            Target: "dst",
            Hyperparameters: new Dictionary<string, object> { ["number_of_trees"] = 5 });

    private async Task<(Core.Interfaces.ITrainedModel model, ModelArtifact artifact)> TrainAndBuildArtifact(
        string stageName = "test_stage",
        string version = "1.0.0")
    {
        using var df = MakeTrainingFrame();
        var config = MakeConfig(stageName);
        var model = await _adapter.TrainAsync(config, df, null, CancellationToken.None);

        var artifact = new ModelArtifact
        {
            ModelId = model.ModelId,
            Version = version,
            StageName = model.StageName,
            Config = config,
            Metrics = model.Metrics,
            DataFingerprint = "placeholder",
            TrainedAt = DateTime.UtcNow,
            ArtifactPath = Path.Combine(_tempDir, model.ModelId, version, "model.bin"),
        };

        return (model, artifact);
    }

    // Test 1: RegisterAsync saves model binary and metadata; both files must exist after registration
    [Fact]
    public async Task RegisterAsync_SavesModelAndMetadata()
    {
        var (model, artifact) = await TrainAndBuildArtifact();

        await _registry.RegisterAsync(artifact, model, CancellationToken.None);

        var modelDir = Path.Combine(_tempDir, artifact.ModelId, "1.0.0");
        File.Exists(Path.Combine(modelDir, "model.bin")).Should().BeTrue("model binary must be written");
        File.Exists(Path.Combine(modelDir, "metadata.json")).Should().BeTrue("metadata must be written");
    }

    // Test 2: LoadAsync restores a model that can produce predictions
    [Fact]
    public async Task LoadAsync_RestoresTrainedModel_CanPredict()
    {
        var (model, artifact) = await TrainAndBuildArtifact();
        await _registry.RegisterAsync(artifact, model, CancellationToken.None);

        var loaded = await _registry.LoadAsync(artifact.ModelId, "1.0.0", CancellationToken.None);

        loaded.Should().NotBeNull();
        loaded.ModelId.Should().Be(artifact.ModelId);

        using var input = MakeTrainingFrame(rows: 10);
        var result = await loaded.PredictAsync(input, CancellationToken.None);
        result.Values.Should().HaveCount(10, "loaded model must produce one prediction per row");
        result.Values.Should().OnlyContain(v => !float.IsNaN(v), "predictions must be finite");
    }

    // Test 3: ListAsync returns all registered models; optional stage filter works correctly
    [Fact]
    public async Task ListAsync_ReturnsAllModels_FilterByStage()
    {
        var (m1, a1) = await TrainAndBuildArtifact(stageName: "stage_a");
        var (m2, a2) = await TrainAndBuildArtifact(stageName: "stage_b");

        await _registry.RegisterAsync(a1, m1, CancellationToken.None);
        await _registry.RegisterAsync(a2, m2, CancellationToken.None);

        var all = await _registry.ListAsync(stageName: null, CancellationToken.None);
        all.Should().HaveCount(2, "both registered models must appear");

        var filtered = await _registry.ListAsync(stageName: "stage_a", CancellationToken.None);
        filtered.Should().HaveCount(1);
        filtered[0].StageName.Should().Be("stage_a");
    }

    // Test 4: Version auto-increment — registering the same model ID twice increments to 1.0.1
    [Fact]
    public async Task RegisterAsync_DuplicateVersion_AutoIncrementsToNextPatch()
    {
        var (m1, a1) = await TrainAndBuildArtifact(version: "1.0.0");
        var (m2, a2) = await TrainAndBuildArtifact(version: "1.0.0");

        await _registry.RegisterAsync(a1, m1, CancellationToken.None);
        await _registry.RegisterAsync(a2, m2, CancellationToken.None);

        var modelDir = Path.Combine(_tempDir, a1.ModelId);
        var versions = Directory.EnumerateDirectories(modelDir)
            .Select(Path.GetFileName)
            .Order()
            .ToList();

        versions.Should().BeEquivalentTo(["1.0.0", "1.0.1"],
            "second registration must auto-increment patch version");
    }

    // Test 5: LoadAsync with corrupted metadata throws InvalidOperationException with informative message
    [Fact]
    public async Task LoadAsync_CorruptedMetadata_ThrowsInvalidOperation()
    {
        var (model, artifact) = await TrainAndBuildArtifact();
        await _registry.RegisterAsync(artifact, model, CancellationToken.None);

        // Corrupt the metadata file
        var metaPath = Path.Combine(_tempDir, artifact.ModelId, "1.0.0", "metadata.json");
        await File.WriteAllTextAsync(metaPath, "{ not valid json !!!");

        var act = async () => await _registry.LoadAsync(artifact.ModelId, "1.0.0", CancellationToken.None);

        await act.Should()
            .ThrowAsync<InvalidOperationException>()
            .WithMessage($"*{artifact.ModelId}*");
    }
}
