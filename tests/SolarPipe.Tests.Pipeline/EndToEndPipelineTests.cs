using System.Diagnostics;
using System.Text.Json;
using FluentAssertions;

namespace SolarPipe.Tests.Pipeline;

[Trait("Category", "Pipeline")]
public sealed class EndToEndPipelineTests : IDisposable
{
    private readonly string _workDir;
    private readonly string _csvPath;
    private readonly string _configPath;
    private readonly string _registryPath;
    private readonly string _hostDll;

    public EndToEndPipelineTests()
    {
        _workDir = Path.Combine(Path.GetTempPath(), $"solarpipe_e2e_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_workDir);

        _csvPath = Path.Combine(_workDir, "solar_wind.csv");
        _configPath = Path.Combine(_workDir, "pipeline.yaml");
        _registryPath = Path.Combine(_workDir, "registry");

        WriteSolarWindCsv(_csvPath);
        WriteYamlConfig(_configPath, _csvPath);

        // Navigate from the test output dir up to the solution root, then down to the
        // host project's own output dir. This avoids the MSB3277 conflict where the
        // test output carries Microsoft.Extensions.ML's DI.Abstractions 2.1.0 instead
        // of the 8.0.0 required by SolarPipe.Host.deps.json.
        //
        // Test output:  <root>/tests/SolarPipe.Tests.Pipeline/bin/Debug/net8.0/
        // Host output:  <root>/src/SolarPipe.Host/bin/Debug/net8.0/
        // Navigate: up 6 dirs → solution root → src/SolarPipe.Host/bin/Debug/net8.0
        var testBinDir = AppContext.BaseDirectory;          // …/tests/SolarPipe.Tests.Pipeline/bin/Debug/net8.0/
        // 5 levels up: net8.0 → Debug → bin → SolarPipe.Tests.Pipeline → tests → solution root
        var solutionRoot = Path.GetFullPath(
            Path.Combine(testBinDir, "..", "..", "..", "..", ".."));
        _hostDll = Path.Combine(
            solutionRoot,
            "src", "SolarPipe.Host", "bin", "Debug", "net8.0",
            "SolarPipe.Host.dll");
    }

    public void Dispose()
    {
        if (Directory.Exists(_workDir))
            Directory.Delete(_workDir, recursive: true);
    }

    // 1. CLI validate command exits 0 for a valid config
    [Fact]
    public async Task Validate_ValidConfig_ExitsZero()
    {
        var (exitCode, stdout, stderr) = await RunCliAsync(
            ["validate", "--config", _configPath]);

        exitCode.Should().Be(0, $"stderr: {stderr}\nstdout: {stdout}");
    }

    // 2. CLI train command exits 0 and writes registry artifacts
    [Fact]
    public async Task Train_SingleStage_ExitsZeroAndWritesArtifacts()
    {
        var (exitCode, stdout, stderr) = await RunCliAsync(
            ["train", "--config", _configPath]);

        exitCode.Should().Be(0, $"stderr: {stderr}\nstdout: {stdout}");

        // Registry should have at least one model directory
        Directory.Exists(_registryPath).Should().BeTrue("registry directory must be created");
        var modelDirs = Directory.GetDirectories(_registryPath, "*", SearchOption.AllDirectories);
        modelDirs.Should().NotBeEmpty("at least one model artifact directory must exist");
    }

    // 3. Full pipeline: validate → train → predict produces valid JSON output
    [Fact]
    public async Task FullPipeline_ValidateThenTrainThenPredict_ProducesValidJson()
    {
        // Step 1: validate
        var (validateExit, _, validateErr) = await RunCliAsync(
            ["validate", "--config", _configPath]);
        validateExit.Should().Be(0, $"validate failed: {validateErr}");

        // Step 2: train
        var (trainExit, _, trainErr) = await RunCliAsync(
            ["train", "--config", _configPath]);
        trainExit.Should().Be(0, $"train failed: {trainErr}");

        // Step 3: predict
        var outputPath = Path.Combine(_workDir, "predictions.json");
        var (predictExit, _, predictErr) = await RunCliAsync(
            ["predict", "--config", _configPath, "--input", _csvPath, "--output", outputPath]);
        predictExit.Should().Be(0, $"predict failed: {predictErr}");

        // Verify output file exists and is valid JSON
        File.Exists(outputPath).Should().BeTrue("predictions.json must be written");

        var json = await File.ReadAllTextAsync(outputPath);
        var doc = JsonDocument.Parse(json);
        doc.RootElement.ValueKind.Should().Be(JsonValueKind.Array,
            "output must be a JSON array of prediction results");

        var first = doc.RootElement[0];
        first.GetProperty("stage").GetString().Should().Be("arrival_predictor");
        first.GetProperty("values").GetArrayLength().Should().BeGreaterThan(0,
            "predictions array must have at least one value");
    }

    // --- helpers ---

    private async Task<(int ExitCode, string Stdout, string Stderr)> RunCliAsync(
        string[] args,
        int timeoutMs = 60_000)
    {
        File.Exists(_hostDll).Should().BeTrue(
            $"SolarPipe.Host.dll not found at {_hostDll}. " +
            "The Tests.Pipeline project must reference SolarPipe.Host.");

        var psi = new ProcessStartInfo
        {
            FileName = "dotnet",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        // Build argument list: exec <dll> <command args...>
        psi.ArgumentList.Add("exec");
        psi.ArgumentList.Add(_hostDll);
        foreach (var arg in args)
            psi.ArgumentList.Add(arg);

        // Point registry at the isolated temp dir for this test run
        psi.Environment["SOLARPIPE_REGISTRY"] = _registryPath;

        using var proc = new Process { StartInfo = psi };

        var stdoutTask = new TaskCompletionSource<string>();
        var stderrTask = new TaskCompletionSource<string>();

        proc.Start();

        var stdoutCapture = proc.StandardOutput.ReadToEndAsync();
        var stderrCapture = proc.StandardError.ReadToEndAsync();

        using var cts = new CancellationTokenSource(timeoutMs);
        await proc.WaitForExitAsync(cts.Token);

        var stdout = await stdoutCapture;
        var stderr = await stderrCapture;

        return (proc.ExitCode, stdout, stderr);
    }

    private static void WriteSolarWindCsv(string path)
    {
        // 60 rows — sufficient for an 80/20 FastForest split (48 train, 12 validation)
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("speed,density,bz_gsm,transit_hours");
        for (int i = 0; i < 60; i++)
        {
            float speed = 400f + (i * 3f);           // 400–577 km/s
            float density = 5f + (i * 0.2f);         // 5–17 n/cc
            float bz = -10f + (i % 10);               // -10 to -1 nT (GSM-frame, RULE-031)
            float hours = 48f - (speed - 400f) * 0.05f; // synthetic transit time target
            sb.AppendLine($"{speed:F1},{density:F2},{bz:F1},{hours:F2}");
        }
        File.WriteAllText(path, sb.ToString());
    }

    private static void WriteYamlConfig(string configPath, string csvPath)
    {
        var csvUri = csvPath.Replace('\\', '/');
        var yaml = $@"name: e2e_pipeline_test

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
      number_of_trees: 20
      number_of_leaves: 4
      feature_fraction: 0.7
";
        File.WriteAllText(configPath, yaml);
    }
}
