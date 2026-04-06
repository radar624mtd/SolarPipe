using System.Text.Json;
using SolarPipe.Config;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data;

namespace SolarPipe.Host.Commands;

public sealed class PredictCommand : ICommand
{
    private readonly PipelineConfigLoader _loader;
    private readonly DataSourceRegistry _dataRegistry;
    private readonly IModelRegistry _modelRegistry;

    public PredictCommand(
        PipelineConfigLoader loader,
        DataSourceRegistry dataRegistry,
        IModelRegistry modelRegistry)
    {
        _loader = loader;
        _dataRegistry = dataRegistry;
        _modelRegistry = modelRegistry;
    }

    public async Task<int> ExecuteAsync(string[] args, CancellationToken ct)
    {
        var configPath = ArgParser.Require(args, "--config");
        var inputPath = ArgParser.Require(args, "--input");
        var outputPath = ArgParser.Require(args, "--output");
        var stageFilter = ArgParser.Get(args, "--stage");

        try
        {
            var config = await _loader.LoadAsync(configPath, ct);

            // Load input via inline CSV data source
            var inputSource = new DataSourceConfig("__predict_input__", "csv", inputPath);
            _dataRegistry.RegisterSource(inputSource);
            var input = await _dataRegistry.LoadAsync("__predict_input__", new DataQuery(), ct);

            var results = new List<object>();

            foreach (var (stageName, _) in config.Stages)
            {
                if (stageFilter != null && !string.Equals(stageName, stageFilter, StringComparison.OrdinalIgnoreCase))
                    continue;

                // Load latest model for this stage
                var artifacts = await _modelRegistry.ListAsync(stageName, ct);
                if (artifacts.Count == 0)
                {
                    Console.Error.WriteLine(
                        $"PREDICT_ERROR stage={stageName} type=NoModel " +
                        $"message=\"No trained model found for stage '{stageName}'. Run train first.\"");
                    return ExitCodes.PredictionFailed;
                }

                var latest = artifacts.OrderByDescending(a => a.TrainedAt).First();

                // RULE-151: timeout prediction at 60s
                using var stageCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
                stageCts.CancelAfter(TimeSpan.FromSeconds(60));

                ITrainedModel model;
                try
                {
                    model = await _modelRegistry.LoadAsync(latest.ModelId, latest.Version, stageCts.Token);
                }
                catch (OperationCanceledException) when (!ct.IsCancellationRequested)
                {
                    Console.Error.WriteLine(
                        $"PREDICT_ERROR stage={stageName} type=Timeout " +
                        $"message=\"Model load exceeded 60s. Check logs/dotnet_latest.json.\"");
                    return ExitCodes.PredictionFailed;
                }

                var result = await model.PredictAsync(input, stageCts.Token);
                results.Add(new
                {
                    stage = stageName,
                    model_id = result.ModelId,
                    generated_at = result.GeneratedAt,
                    values = result.Values,
                    lower_bound = result.LowerBound,
                    upper_bound = result.UpperBound,
                });
            }

            var json = JsonSerializer.Serialize(results, new JsonSerializerOptions { WriteIndented = true });

            // Atomic write (RULE-040)
            var tempPath = outputPath + $".tmp_{Guid.NewGuid():N}";
            await File.WriteAllTextAsync(tempPath, json, ct);
            File.Move(tempPath, outputPath, overwrite: true);

            Console.WriteLine($"OK output={outputPath} stages={results.Count}");
            return ExitCodes.Success;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(
                $"PREDICT_ERROR type={ex.GetType().Name} message=\"{ex.Message}\"");
            return ExitCodes.PredictionFailed;
        }
    }
}
