using SolarPipe.Config;
using SolarPipe.Data;

namespace SolarPipe.Host.Commands;

public sealed class ValidateCommand : ICommand
{
    private readonly PipelineConfigLoader _loader;
    private readonly DataSourceRegistry _dataRegistry;

    public ValidateCommand(PipelineConfigLoader loader, DataSourceRegistry dataRegistry)
    {
        _loader = loader;
        _dataRegistry = dataRegistry;
    }

    public async Task<int> ExecuteAsync(string[] args, CancellationToken ct)
    {
        var configPath = ArgParser.Require(args, "--config");

        try
        {
            var config = await _loader.LoadAsync(configPath, ct);

            // Register declared data sources so schema discovery can run
            foreach (var (key, _) in config.DataSources)
                _dataRegistry.RegisterSource(config.ToDataSourceConfig(key));

            // Validate schema reachability for each stage's data source
            foreach (var (stageName, stage) in config.Stages)
            {
                try
                {
                    await _dataRegistry.DiscoverSchemaAsync(stage.DataSource, ct);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine(
                        $"VALIDATE_ERROR stage={stageName} source={stage.DataSource} " +
                        $"message=\"{ex.Message}\"");
                    return ExitCodes.ValidationFailed;
                }
            }

            Console.WriteLine($"OK config={configPath} stages={config.Stages.Count} sources={config.DataSources.Count}");
            return ExitCodes.Success;
        }
        catch (FileNotFoundException ex)
        {
            Console.Error.WriteLine($"VALIDATE_ERROR type=FileNotFound path=\"{configPath}\" message=\"{ex.Message}\"");
            return ExitCodes.ValidationFailed;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"VALIDATE_ERROR type={ex.GetType().Name} message=\"{ex.Message}\"");
            return ExitCodes.ValidationFailed;
        }
    }
}
