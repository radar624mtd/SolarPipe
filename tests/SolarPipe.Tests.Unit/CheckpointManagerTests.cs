using FluentAssertions;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Training.Checkpoint;

namespace SolarPipe.Tests.Unit;

[Trait("Category", "Unit")]
public sealed class CheckpointManagerTests : IDisposable
{
    private readonly string _cacheRoot;
    private readonly CheckpointManager _sut;

    public CheckpointManagerTests()
    {
        _cacheRoot = Path.Combine(Path.GetTempPath(), $"sp_ckpt_{Guid.NewGuid():N}");
        _sut = new CheckpointManager(_cacheRoot);
    }

    public void Dispose()
    {
        if (Directory.Exists(_cacheRoot))
            Directory.Delete(_cacheRoot, recursive: true);
    }

    private static InMemoryDataFrame MakeFrame(int rows = 10)
    {
        var schema = new DataSchema([
            new ColumnInfo("speed", ColumnType.Float, IsNullable: false),
            new ColumnInfo("transit", ColumnType.Float, IsNullable: false),
        ]);
        var speed = Enumerable.Range(0, rows).Select(i => (float)(400 + i * 10)).ToArray();
        var transit = Enumerable.Range(0, rows).Select(i => (float)(40 + i)).ToArray();
        return new InMemoryDataFrame(schema, [speed, transit]);
    }

    private static StageConfig MakeConfig(string name = "drag_baseline") => new(
        Name: name,
        Framework: "Physics",
        ModelType: "DragBased",
        DataSource: "catalog",
        Features: ["speed"],
        Target: "transit",
        Hyperparameters: new Dictionary<string, object>());

    // 1. Write then TryRead returns identical data
    [Fact]
    public async Task WriteAndRead_RoundTrip_ReturnsMatchingFrame()
    {
        var frame = MakeFrame(20);
        var config = MakeConfig();

        await _sut.WriteAsync("pipeline1", "stage1", frame, config, "fp1", CancellationToken.None);
        var result = await _sut.TryReadAsync("pipeline1", "stage1", config, "fp1", CancellationToken.None);

        result.Should().NotBeNull();
        result!.RowCount.Should().Be(20);
        result.Schema.Columns.Select(c => c.Name).Should().Equal("speed", "transit");
        result.GetColumn("speed")[0].Should().BeApproximately(400f, 0.001f);
        result.GetColumn("speed")[19].Should().BeApproximately(590f, 0.001f);
        result.Dispose();
    }

    // 2. Config change invalidates checkpoint
    [Fact]
    public async Task TryRead_ConfigChanged_ReturnsNull()
    {
        var frame = MakeFrame();
        var configV1 = MakeConfig();

        await _sut.WriteAsync("p", "s", frame, configV1, "fp", CancellationToken.None);

        var configV2 = new StageConfig(
            Name: "drag_baseline",
            Framework: "Physics",
            ModelType: "DragBased",
            DataSource: "catalog",
            Features: ["speed"],
            Target: "transit",
            Hyperparameters: new Dictionary<string, object> { ["drag_parameter"] = 0.5e-7 });

        var result = await _sut.TryReadAsync("p", "s", configV2, "fp", CancellationToken.None);

        result.Should().BeNull("config fingerprint changed — checkpoint must be invalidated");
        // Checkpoint files should be deleted
        File.Exists(_sut.GetCheckpointPath("p", "s")).Should().BeFalse();
    }

    // 3. Input fingerprint change invalidates checkpoint
    [Fact]
    public async Task TryRead_InputFingerprintChanged_ReturnsNull()
    {
        var frame = MakeFrame();
        var config = MakeConfig();

        await _sut.WriteAsync("p", "s", frame, config, "fingerprint_v1", CancellationToken.None);
        var result = await _sut.TryReadAsync("p", "s", config, "fingerprint_v2", CancellationToken.None);

        result.Should().BeNull("input data fingerprint changed — checkpoint must be invalidated");
    }

    // 4. Missing checkpoint returns null without throwing
    [Fact]
    public async Task TryRead_NoCheckpointExists_ReturnsNull()
    {
        var config = MakeConfig();
        var result = await _sut.TryReadAsync("p", "nonexistent", config, "fp", CancellationToken.None);
        result.Should().BeNull();
    }

    // 5. Corrupt meta JSON returns null without throwing
    [Fact]
    public async Task TryRead_CorruptMeta_ReturnsNull()
    {
        var frame = MakeFrame();
        var config = MakeConfig();

        await _sut.WriteAsync("p", "s", frame, config, "fp", CancellationToken.None);
        // Corrupt the meta file
        await File.WriteAllTextAsync(_sut.GetMetaPath("p", "s"), "{ not valid json {{{{");

        var result = await _sut.TryReadAsync("p", "s", config, "fp", CancellationToken.None);

        result.Should().BeNull("corrupt meta must be handled gracefully");
    }

    // 6. ClearAsync removes all checkpoints for that pipeline
    [Fact]
    public async Task ClearAsync_RemovesAllCheckpointsForPipeline()
    {
        var frame = MakeFrame();
        var config = MakeConfig();

        await _sut.WriteAsync("pipeline_a", "stage1", frame, config, "fp", CancellationToken.None);
        await _sut.WriteAsync("pipeline_a", "stage2", frame, config, "fp", CancellationToken.None);

        await _sut.ClearAsync("pipeline_a", CancellationToken.None);

        File.Exists(_sut.GetCheckpointPath("pipeline_a", "stage1")).Should().BeFalse();
        File.Exists(_sut.GetCheckpointPath("pipeline_a", "stage2")).Should().BeFalse();
    }

    // 7. ClearAsync does not affect other pipelines
    [Fact]
    public async Task ClearAsync_DoesNotAffectOtherPipelines()
    {
        var frame = MakeFrame();
        var config = MakeConfig();

        await _sut.WriteAsync("pipeline_a", "stage1", frame, config, "fp", CancellationToken.None);
        await _sut.WriteAsync("pipeline_b", "stage1", frame, config, "fp", CancellationToken.None);

        await _sut.ClearAsync("pipeline_a", CancellationToken.None);

        // pipeline_b checkpoint must survive
        var result = await _sut.TryReadAsync("pipeline_b", "stage1", config, "fp", CancellationToken.None);
        result.Should().NotBeNull();
        result!.Dispose();
    }

    // 8. Empty IDataFrame round-trips without error
    [Fact]
    public async Task WriteAndRead_EmptyFrame_Succeeds()
    {
        var schema = new DataSchema([new ColumnInfo("x", ColumnType.Float, IsNullable: false)]);
        using var emptyFrame = new InMemoryDataFrame(schema, [Array.Empty<float>()]);
        var config = MakeConfig();

        await _sut.WriteAsync("p", "s", emptyFrame, config, "fp", CancellationToken.None);
        var result = await _sut.TryReadAsync("p", "s", config, "fp", CancellationToken.None);

        result.Should().NotBeNull();
        result!.RowCount.Should().Be(0);
        result.Dispose();
    }

    // 9. Checkpoint path is deterministic
    [Fact]
    public void GetCheckpointPath_IsDeterministic()
    {
        var path1 = _sut.GetCheckpointPath("pipe", "stage");
        var path2 = _sut.GetCheckpointPath("pipe", "stage");
        path1.Should().Be(path2);
        path1.Should().EndWith("stage.checkpoint");
    }

    // 10. Meta JSON written contains expected fields
    [Fact]
    public async Task WriteAsync_MetaContainsRequiredFields()
    {
        var frame = MakeFrame(5);
        var config = MakeConfig();

        await _sut.WriteAsync("p", "s", frame, config, "myfp", CancellationToken.None);

        var metaJson = await File.ReadAllTextAsync(_sut.GetMetaPath("p", "s"));
        metaJson.Should().Contain("\"StageName\"");
        metaJson.Should().Contain("\"ConfigFingerprint\"");
        metaJson.Should().Contain("\"InputFingerprint\"");
        metaJson.Should().Contain("\"myfp\"");
        metaJson.Should().Contain("\"RowCount\"");
        metaJson.Should().Contain("5");
    }
}
