using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using SolarPipe.Config;
using SolarPipe.Core.Interfaces;
using SolarPipe.Data;
using SolarPipe.Data.Providers;
using SolarPipe.Host.Commands;
using SolarPipe.Training.Adapters;
using SolarPipe.Training.Checkpoint;
using SolarPipe.Training.Evaluation;
using SolarPipe.Training.Registry;
using SolarPipe.Training.Features;
using SolarPipe.Training.Sweep;

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

        await using var services = BuildServices();
        return command switch
        {
            "validate"        => await services.GetRequiredService<ValidateCommand>().ExecuteAsync(commandArgs, cts.Token),
            "train"           => await services.GetRequiredService<TrainCommand>().ExecuteAsync(commandArgs, cts.Token),
            "predict"         => await services.GetRequiredService<PredictCommand>().ExecuteAsync(commandArgs, cts.Token),
            "inspect"         => await services.GetRequiredService<InspectCommand>().ExecuteAsync(commandArgs, cts.Token),
            "validate-events" => await services.GetRequiredService<ValidateEventsCommand>().ExecuteAsync(commandArgs, cts.Token),
            "sweep"           => await services.GetRequiredService<SweepCommand>().ExecuteAsync(commandArgs, cts.Token),
            "domain-sweep"    => await services.GetRequiredService<DomainSweepCommand>().ExecuteAsync(commandArgs, cts.Token),
            "predict-progressive"  => await services.GetRequiredService<PredictProgressiveCommand>().ExecuteAsync(commandArgs, cts.Token),
            "fetch-synoptic-map"          => await new FetchSynopticMapCommand().ExecuteAsync(commandArgs, cts.Token),
            "compute-pfss-topology"       => await new ComputePfssTopologyCommand().ExecuteAsync(commandArgs, cts.Token),
            "compute-pfss-mmap"           => await new ComputePfssMmapCommand().ExecuteAsync(commandArgs, cts.Token),
            "read-mmap"                   => await services.GetRequiredService<ReadMmapCommand>().ExecuteAsync(commandArgs, cts.Token),
            _ => UnknownCommand(command),
        };
    }

    private static ServiceProvider BuildServices()
    {
        var registryPath = Environment.GetEnvironmentVariable("SOLARPIPE_REGISTRY")
            ?? "./models/registry";

        var csvProvider = new CsvProvider();
        var sqliteProvider = new SqliteProvider();
        var parquetProvider = new ParquetProvider();
        var dataRegistry = new DataSourceRegistry();
        dataRegistry.RegisterProvider(csvProvider);
        dataRegistry.RegisterProvider(sqliteProvider);
        dataRegistry.RegisterProvider(parquetProvider);

        var mlNetAdapter     = new MlNetAdapter();
        var physicsAdapter   = new PhysicsAdapter();
        var onnxAdapter      = new OnnxAdapter();

        var modelRegistry = new FileSystemModelRegistry(registryPath);
        var configLoader = new PipelineConfigLoader();

        var cacheRoot = Environment.GetEnvironmentVariable("SOLARPIPE_CACHE") ?? "./cache";
        var checkpointManager = new CheckpointManager(cacheRoot);

        var metricsEvaluator = new ComprehensiveMetricsEvaluator();
        var nnlsOptimizer    = new NnlsEnsembleOptimizer();
        var sweepLoader       = new SweepConfigLoader();
        var domainSweepLoader = new DomainSweepConfigLoader();

        var sidecarAddress = Environment.GetEnvironmentVariable("SOLARPIPE_SIDECAR_ADDRESS")
            ?? "http://localhost:50051";
        GrpcSidecarAdapter? sidecarAdapter = null;
        try { sidecarAdapter = new GrpcSidecarAdapter(sidecarAddress); }
        catch { /* sidecar optional — pre-flight will fail if H7 is needed */ }

        // Build SidecarLifecycleService so SweepCommand can start/stop the Python process.
        // Uses NullApplicationLifetime since the CLI host does not use IGenericHost.
        var sidecarOptions = SidecarOptions.FromEnvironment();
        SidecarLifecycleService? sidecarLifecycle = sidecarOptions.Enabled
            ? new SidecarLifecycleService(
                sidecarOptions,
                new NullApplicationLifetime(),
                NullLogger<SidecarLifecycleService>.Instance)
            : null;

        IReadOnlyList<IFrameworkAdapter> allAdapters = sidecarAdapter is not null
            ? new IFrameworkAdapter[] { mlNetAdapter, physicsAdapter, onnxAdapter, sidecarAdapter }
            : new IFrameworkAdapter[] { mlNetAdapter, physicsAdapter, onnxAdapter };

        var modelSweep  = new ModelSweep(allAdapters, metricsEvaluator, nnlsOptimizer,
            cacheRoot, registryPath, sidecarAdapter);
        var domainSweep = new DomainPipelineSweep(allAdapters);

        return new ServiceCollection()
            .AddSingleton(configLoader)
            .AddSingleton(dataRegistry)
            .AddSingleton<IReadOnlyList<IFrameworkAdapter>>(allAdapters)
            .AddSingleton<IModelRegistry>(modelRegistry)
            .AddSingleton(checkpointManager)
            .AddSingleton<ValidateCommand>()
            .AddSingleton<TrainCommand>()
            .AddSingleton<PredictCommand>()
            .AddSingleton<InspectCommand>()
            .AddSingleton(sp => new ValidateEventsCommand(
                sp.GetRequiredService<PipelineConfigLoader>(),
                sp.GetRequiredService<IModelRegistry>(),
                sp.GetRequiredService<DataSourceRegistry>()))
            .AddSingleton(sp => new SweepCommand(
                sweepLoader,
                sp.GetRequiredService<DataSourceRegistry>(),
                sp.GetRequiredService<IReadOnlyList<IFrameworkAdapter>>(),
                modelSweep,
                cacheRoot,
                sidecarLifecycle))
            .AddSingleton(sp => new DomainSweepCommand(
                domainSweepLoader,
                sp.GetRequiredService<DataSourceRegistry>(),
                sp.GetRequiredService<IReadOnlyList<IFrameworkAdapter>>(),
                domainSweep))
            .AddSingleton(sp => new PredictProgressiveCommand(domainSweepLoader))
            .AddSingleton<ReadMmapCommand>()
            .BuildServiceProvider();
    }

    private static int UnknownCommand(string command)
    {
        Console.Error.WriteLine(
            $"ERROR type=UnknownCommand command=\"{command}\" " +
            $"valid_commands=\"validate, train, predict, inspect, validate-events, sweep, domain-sweep, predict-progressive\"");
        PrintUsage();
        return ExitCodes.UnknownCommand;
    }

    private static void PrintUsage()
    {
        Console.WriteLine("Usage: solarpipe <command> [options]");
        Console.WriteLine("  validate         --config <path>");
        Console.WriteLine("  train            --config <path> [--stage <name>] [--resume-from-stage <name>] [--no-cache]");
        Console.WriteLine("  predict          --config <path> --input <csv> --output <json> [--stage <name>]");
        Console.WriteLine("  inspect          [--stage <name>]");
        Console.WriteLine("  validate-events  --config <path> [--output <path>]");
        Console.WriteLine("  sweep            --config <path> [--resume|--fresh] [--output <dir>]");
        Console.WriteLine("  domain-sweep     --config <path> [--output <dir>]");
        Console.WriteLine("  predict-progressive --config <path> [--event <iso> --until <iso> | --backtest]");
        Console.WriteLine("                      [--mode density|static] [--n-ref <cm-3>] [--output <dir>]");
        Console.WriteLine("                      [--omni-db <conn>] [--allow-future]");
        Console.WriteLine("  fetch-synoptic-map      --cr <number> --output <path> [--cache <dir>]");
        Console.WriteLine("  compute-pfss-topology   --db <path> [--workers N] [--cr-cache <dir>] [--force] [--max-events N]");
        Console.WriteLine("  read-mmap               --map-name <name> --rows <num_rows> --cols <num_cols>");
    }
}
