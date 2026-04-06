using System.Text.Json;
using FluentAssertions;
using SolarPipe.Config;
using SolarPipe.Core.Interfaces;
using SolarPipe.Data;
using SolarPipe.Data.Providers;
using SolarPipe.Host;
using SolarPipe.Host.Commands;
using SolarPipe.Training.Adapters;
using SolarPipe.Training.Registry;

namespace SolarPipe.Tests.Integration;

[Trait("Category", "Integration")]
public sealed class CliCommandTests : IDisposable
{
    private readonly string _workDir;
    private readonly string _configPath;
    private readonly string _csvPath;
    private readonly string _registryPath;

    private readonly PipelineConfigLoader _loader;
    private readonly DataSourceRegistry _dataRegistry;
    private readonly IReadOnlyList<IFrameworkAdapter> _adapters;
    private readonly IModelRegistry _modelRegistry;

    public CliCommandTests()
    {
        _workDir = Path.Combine(Path.GetTempPath(), $"solarpipe_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_workDir);

        _csvPath = Path.Combine(_workDir, "data.csv");
        _configPath = Path.Combine(_workDir, "pipeline.yaml");
        _registryPath = Path.Combine(_workDir, "registry");

        WriteSolarWindCsv(_csvPath);
        WriteYamlConfig(_configPath, _csvPath);

        var csvProvider = new CsvProvider();
        var sqliteProvider = new SqliteProvider();
        _dataRegistry = new DataSourceRegistry();
        _dataRegistry.RegisterProvider(csvProvider);
        _dataRegistry.RegisterProvider(sqliteProvider);

        _adapters = new IFrameworkAdapter[] { new MlNetAdapter() };
        _modelRegistry = new FileSystemModelRegistry(_registryPath);
        _loader = new PipelineConfigLoader();
    }

    public void Dispose()
    {
        if (Directory.Exists(_workDir))
            Directory.Delete(_workDir, recursive: true);
    }

    // 1. ValidateCommand accepts a well-formed config and returns ExitCodes.Success
    [Fact]
    public async Task ValidateCommand_ValidConfig_ReturnsSuccess()
    {
        var sut = new ValidateCommand(_loader, _dataRegistry);

        var exit = await sut.ExecuteAsync(["--config", _configPath], CancellationToken.None);

        exit.Should().Be(ExitCodes.Success);
    }

    // 2. ValidateCommand rejects a missing config file with ValidationFailed
    [Fact]
    public async Task ValidateCommand_MissingFile_ReturnsValidationFailed()
    {
        var sut = new ValidateCommand(_loader, _dataRegistry);

        var exit = await sut.ExecuteAsync(["--config", "/nonexistent/path.yaml"], CancellationToken.None);

        exit.Should().Be(ExitCodes.ValidationFailed);
    }

    // 3. TrainCommand trains a FastForest stage and registers the model
    [Fact]
    public async Task TrainCommand_SingleStage_RegistersModel()
    {
        var sut = new TrainCommand(_loader, _dataRegistry, _adapters, _modelRegistry);

        var exit = await sut.ExecuteAsync(["--config", _configPath], CancellationToken.None);

        exit.Should().Be(ExitCodes.Success);

        var artifacts = await _modelRegistry.ListAsync(ct: CancellationToken.None);
        artifacts.Should().HaveCount(1);
        artifacts[0].StageName.Should().Be("arrival_predictor");
        artifacts[0].Metrics.Rmse.Should().BeGreaterThan(0f);
    }

    // 4. PredictCommand writes valid JSON after training
    [Fact]
    public async Task PredictCommand_AfterTrain_WritesJsonOutput()
    {
        // Arrange — train first so registry has a model
        var trainCmd = new TrainCommand(_loader, _dataRegistry, _adapters, _modelRegistry);
        var trainExit = await trainCmd.ExecuteAsync(["--config", _configPath], CancellationToken.None);
        trainExit.Should().Be(ExitCodes.Success, "train must succeed before predict");

        var outputPath = Path.Combine(_workDir, "predictions.json");
        var predictCmd = new PredictCommand(_loader, _dataRegistry, _modelRegistry);

        // Act
        var exit = await predictCmd.ExecuteAsync(
            ["--config", _configPath, "--input", _csvPath, "--output", outputPath],
            CancellationToken.None);

        // Assert
        exit.Should().Be(ExitCodes.Success);
        File.Exists(outputPath).Should().BeTrue("output file should be written");

        var json = await File.ReadAllTextAsync(outputPath);
        var doc = JsonDocument.Parse(json);
        doc.RootElement.ValueKind.Should().Be(JsonValueKind.Array);

        var first = doc.RootElement[0];
        first.GetProperty("stage").GetString().Should().Be("arrival_predictor");
        first.GetProperty("values").GetArrayLength().Should().BeGreaterThan(0);
    }

    // --- helpers ---

    private static void WriteSolarWindCsv(string path)
    {
        // 60 rows — enough for an 80/20 train split (48 train, 12 validation) with FastForest
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("speed,density,bz_gsm,transit_hours");
        var rng = new Random(42);
        for (int i = 0; i < 60; i++)
        {
            float speed = 400f + (i * 3f);        // 400–577 km/s
            float density = 5f + (i * 0.2f);      // 5–17 n/cc
            float bz = -10f + (i % 10);            // -10 to -1 nT (GSM-frame RULE-031)
            float hours = 48f - (speed - 400f) * 0.05f; // synthetic target
            sb.AppendLine($"{speed:F1},{density:F2},{bz:F1},{hours:F2}");
        }
        File.WriteAllText(path, sb.ToString());
    }

    private static void WriteYamlConfig(string configPath, string csvPath)
    {
        // Forward slashes work on Windows in YAML strings
        var csvUri = csvPath.Replace('\\', '/');
        var yaml = $@"name: solar_wind_integration_test

data_sources:
  solar_wind:
    provider: csv
    connection_string: {csvUri}

stages:
  arrival_predictor:
    framework: MlNet
    model_type: FastForest
    data_source: solar_wind
    features:
      - speed
      - density
      - bz_gsm
    target: transit_hours
    hyperparameters:
      NumberOfTrees: 20
      NumberOfLeaves: 4
      FeatureFraction: 0.7
";
        File.WriteAllText(configPath, yaml);
    }
}
