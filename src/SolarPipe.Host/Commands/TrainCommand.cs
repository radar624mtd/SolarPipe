using SolarPipe.Config;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data;

namespace SolarPipe.Host.Commands;

public sealed class TrainCommand : ICommand
{
    private readonly PipelineConfigLoader _loader;
    private readonly DataSourceRegistry _dataRegistry;
    private readonly IReadOnlyList<IFrameworkAdapter> _adapters;
    private readonly IModelRegistry _modelRegistry;

    public TrainCommand(
        PipelineConfigLoader loader,
        DataSourceRegistry dataRegistry,
        IReadOnlyList<IFrameworkAdapter> adapters,
        IModelRegistry modelRegistry)
    {
        _loader = loader;
        _dataRegistry = dataRegistry;
        _adapters = adapters;
        _modelRegistry = modelRegistry;
    }

    public async Task<int> ExecuteAsync(string[] args, CancellationToken ct)
    {
        string configPath;
        try { configPath = ArgParser.Require(args, "--config"); }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine($"TRAIN_ERROR type=MissingArguments message=\"{ex.Message}\"");
            return ExitCodes.MissingArguments;
        }
        var stageFilter = ArgParser.Get(args, "--stage");

        try
        {
            var config = await _loader.LoadAsync(configPath, ct);

            foreach (var (key, _) in config.DataSources)
                _dataRegistry.RegisterSource(config.ToDataSourceConfig(key));

            foreach (var (stageName, stageYaml) in config.Stages)
            {
                if (stageFilter != null && !string.Equals(stageName, stageFilter, StringComparison.OrdinalIgnoreCase))
                    continue;

                var stageConfig = config.ToStageConfig(stageName);
                var adapter = ResolveAdapter(stageConfig);

                var data = await _dataRegistry.LoadAsync(stageYaml.DataSource, new DataQuery(), ct);
                var (train, validation) = SplitTrainValidation(data, trainFraction: 0.8);

                // RULE-150: LongRunning for CPU-bound ML training
                ITrainedModel model;
                using var stageCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
                stageCts.CancelAfter(TimeSpan.FromMinutes(30)); // RULE-151

                try
                {
                    // RULE-150: MlNetAdapter.TrainAsync dispatches LongRunning internally
                    model = await adapter.TrainAsync(stageConfig, train, validation, stageCts.Token);
                }
                catch (OperationCanceledException) when (!ct.IsCancellationRequested)
                {
                    Console.Error.WriteLine(
                        $"TRAIN_ERROR stage={stageName} type=Timeout " +
                        $"message=\"Stage exceeded 30 minutes. Check logs/dotnet_latest.json for last activity.\"");
                    return ExitCodes.TrainingFailed;
                }

                var artifact = new ModelArtifact
                {
                    ModelId = model.ModelId,
                    Version = "1.0.0",
                    StageName = stageName,
                    Config = stageConfig,
                    Metrics = model.Metrics,
                    DataFingerprint = string.Empty,
                    TrainedAt = DateTime.UtcNow,
                    ArtifactPath = string.Empty,
                };

                await _modelRegistry.RegisterAsync(artifact, model, ct);
                Console.WriteLine($"OK stage={stageName} model_id={model.ModelId} rmse={model.Metrics.Rmse:F4}");
            }

            return ExitCodes.Success;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(
                $"TRAIN_ERROR stage={stageFilter ?? "all"} type={ex.GetType().Name} message=\"{ex.Message}\"");
            return ExitCodes.TrainingFailed;
        }
    }

    private IFrameworkAdapter ResolveAdapter(StageConfig config)
    {
        var adapter = _adapters.FirstOrDefault(a =>
            string.Equals(a.FrameworkType.ToString(), config.Framework, StringComparison.OrdinalIgnoreCase));

        if (adapter is null)
            throw new InvalidOperationException(
                $"TrainCommand: no adapter for framework='{config.Framework}' stage='{config.Name}'. " +
                $"Registered: [{string.Join(", ", _adapters.Select(a => a.FrameworkType))}].");

        return adapter;
    }

    private static (IDataFrame Train, IDataFrame Validation) SplitTrainValidation(
        IDataFrame data, double trainFraction)
    {
        // Temporal split — no random k-fold (RULE-051)
        int totalRows = data.RowCount;
        int trainRows = (int)(totalRows * trainFraction);

        var train = data.Slice(0, trainRows);
        var validation = data.Slice(trainRows, totalRows - trainRows);
        return (train, validation);
    }
}
