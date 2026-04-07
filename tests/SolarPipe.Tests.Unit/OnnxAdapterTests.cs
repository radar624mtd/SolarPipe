using FluentAssertions;
using SolarPipe.Core.Models;
using SolarPipe.Training.Adapters;

namespace SolarPipe.Tests.Unit;

[Trait("Category", "Unit")]
public class OnnxAdapterTests
{
    private readonly OnnxAdapter _adapter = new();

    private static StageConfig MakeConfig(
        string modelType = "Standard",
        IReadOnlyDictionary<string, object>? hp = null) =>
        new StageConfig(
            Name: "onnx_stage",
            Framework: "Onnx",
            ModelType: modelType,
            DataSource: "src",
            Features: ["feature_a"],
            Target: "target",
            Hyperparameters: hp);

    [Fact]
    public void FrameworkType_IsOnnx()
    {
        _adapter.FrameworkType.Should().Be(FrameworkType.Onnx);
    }

    [Fact]
    public void SupportedModels_ContainsStandardAndNeuralOde()
    {
        _adapter.SupportedModels.Should().Contain("Standard");
        _adapter.SupportedModels.Should().Contain("NeuralOde");
    }

    [Fact]
    public async Task TrainAsync_MissingModelPath_ThrowsArgumentException()
    {
        var config = MakeConfig(hp: new Dictionary<string, object>());

        var act = async () => await _adapter.TrainAsync(config, null!, null, CancellationToken.None);

        await act.Should().ThrowAsync<ArgumentException>()
            .WithMessage("*model_path*");
    }

    [Fact]
    public async Task TrainAsync_NonexistentFile_ThrowsFileNotFoundException()
    {
        var config = MakeConfig(hp: new Dictionary<string, object>
        {
            ["model_path"] = "/tmp/nonexistent_model_xyzabc.onnx"
        });

        var act = async () => await _adapter.TrainAsync(config, null!, null, CancellationToken.None);

        await act.Should().ThrowAsync<FileNotFoundException>()
            .WithMessage("*ONNX model file not found*");
    }
}
