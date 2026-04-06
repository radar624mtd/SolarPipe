using Microsoft.Extensions.DependencyInjection;
using SolarPipe.Config;
using SolarPipe.Core.Interfaces;
using SolarPipe.Data;
using SolarPipe.Data.Providers;
using SolarPipe.Host.Commands;
using SolarPipe.Training.Adapters;
using SolarPipe.Training.Registry;

namespace SolarPipe.Host;

internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        if (args.Length == 0)
        {
            PrintUsage();
            return ExitCodes.MissingArguments;
        }

        var command = args[0];
        var commandArgs = args.Skip(1).ToArray();

        using var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = true;
            cts.Cancel();
        };

        var services = BuildServices();
        return command switch
        {
            "validate" => await services.GetRequiredService<ValidateCommand>().ExecuteAsync(commandArgs, cts.Token),
            "train"    => await services.GetRequiredService<TrainCommand>().ExecuteAsync(commandArgs, cts.Token),
            "predict"  => await services.GetRequiredService<PredictCommand>().ExecuteAsync(commandArgs, cts.Token),
            "inspect"  => await services.GetRequiredService<InspectCommand>().ExecuteAsync(commandArgs, cts.Token),
            _ => UnknownCommand(command),
        };
    }

    private static ServiceProvider BuildServices()
    {
        var registryPath = Environment.GetEnvironmentVariable("SOLARPIPE_REGISTRY")
            ?? "./models/registry";

        var csvProvider = new CsvProvider();
        var sqliteProvider = new SqliteProvider();
        var dataRegistry = new DataSourceRegistry();
        dataRegistry.RegisterProvider(csvProvider);
        dataRegistry.RegisterProvider(sqliteProvider);

        var mlNetAdapter = new MlNetAdapter();
        var adapters = new IFrameworkAdapter[] { mlNetAdapter };

        var modelRegistry = new FileSystemModelRegistry(registryPath);
        var configLoader = new PipelineConfigLoader();

        return new ServiceCollection()
            .AddSingleton(configLoader)
            .AddSingleton(dataRegistry)
            .AddSingleton<IReadOnlyList<IFrameworkAdapter>>(adapters)
            .AddSingleton<IModelRegistry>(modelRegistry)
            .AddSingleton<ValidateCommand>()
            .AddSingleton<TrainCommand>()
            .AddSingleton<PredictCommand>()
            .AddSingleton<InspectCommand>()
            .BuildServiceProvider();
    }

    private static int UnknownCommand(string command)
    {
        Console.Error.WriteLine(
            $"ERROR type=UnknownCommand command=\"{command}\" " +
            $"valid_commands=\"validate, train, predict, inspect\"");
        PrintUsage();
        return ExitCodes.UnknownCommand;
    }

    private static void PrintUsage()
    {
        Console.WriteLine("Usage: solarpipe <command> [options]");
        Console.WriteLine("  validate --config <path>");
        Console.WriteLine("  train    --config <path> [--stage <name>]");
        Console.WriteLine("  predict  --config <path> --input <csv> --output <json> [--stage <name>]");
        Console.WriteLine("  inspect  [--stage <name>]");
    }
}
