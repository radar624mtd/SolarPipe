using System.Text.Json;
using SolarPipe.Config;
using SolarPipe.Config.Models;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data;
using SolarPipe.Training.Sweep;

namespace SolarPipe.Host.Commands;

// CLI entry point for Phase 8 domain-decomposed pipeline sweep.
// Usage:
//   domain-sweep --config configs/phase8_domain_sweep.yaml
//   domain-sweep --config configs/phase8_domain_sweep.yaml --output output/
//
// Flow: load config → load staging data → run DomainPipelineSweep → write results JSON
public sealed class DomainSweepCommand : ICommand
{
    private readonly DomainSweepConfigLoader _loader;
    private readonly DataSourceRegistry _dataRegistry;
    private readonly IReadOnlyList<IFrameworkAdapter> _adapters;
    private readonly DomainPipelineSweep _sweep;

    public DomainSweepCommand(
        DomainSweepConfigLoader loader,
        DataSourceRegistry dataRegistry,
        IReadOnlyList<IFrameworkAdapter> adapters,
        DomainPipelineSweep sweep)
    {
        _loader       = loader       ?? throw new ArgumentNullException(nameof(loader));
        _dataRegistry = dataRegistry ?? throw new ArgumentNullException(nameof(dataRegistry));
        _adapters     = adapters     ?? throw new ArgumentNullException(nameof(adapters));
        _sweep        = sweep        ?? throw new ArgumentNullException(nameof(sweep));
    }

    public async Task<int> ExecuteAsync(string[] args, CancellationToken ct)
    {
        string configPath;
        try { configPath = ArgParser.Require(args, "--config"); }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine($"DOMAIN_SWEEP_ERROR type=MissingArguments message=\"{ex.Message}\"");
            return ExitCodes.MissingArguments;
        }

        string outputDir = ArgParser.Get(args, "--output") ?? "output";

        try
        {
            var config = await _loader.LoadAsync(configPath, ct);
            Console.WriteLine($"DOMAIN_SWEEP_CONFIG loaded={configPath} name={config.DomainPipeline.Name}");

            // Load and register staging data source
            var data = await LoadStagingDataAsync(config, ct);
            Console.WriteLine($"DOMAIN_SWEEP_DATA rows={data.RowCount} cols={data.Schema.Columns.Count}");

            // Convert YAML config → run config
            var runConfig = ToRunConfig(config, _loader);

            // Run sweep
            var result = await _sweep.RunAsync(runConfig, data, ct);

            // Write results JSON
            Directory.CreateDirectory(outputDir);
            var outPath = Path.Combine(outputDir, "phase8_domain_results.json");
            await WriteResultsAsync(result, outPath, ct);
            Console.WriteLine($"DOMAIN_SWEEP_RESULTS written={outPath}");

            Console.WriteLine(
                $"DOMAIN_SWEEP_RESULT " +
                $"transit_mae={result.MeanMaeTransit:F3} " +
                $"dst_mae={result.MeanMaeDst:F3} " +
                $"duration_mae={result.MeanMaeDuration:F3}");

            Console.WriteLine(
                $"DOMAIN_SWEEP_BASELINE physics_mae={result.MeanMaePhysicsBaseline:F3} " +
                $"(DragBased-only on transit_time_hours; Phase 7 H1 ref=20.259)");

            // Print CCMC event comparison if any per-event predictions are available
            var allPreds = result.FoldMetrics
                .Where(f => f.EventPredictions != null)
                .SelectMany(f => f.EventPredictions!)
                .ToList();

            if (allPreds.Count > 0)
            {
                Console.WriteLine("\nCCMC 4-Event Benchmark Comparison:");
                Console.WriteLine($"{"Event",-32} {"Obs":>6} {"Ours":>7} {"Err":>7} {"CCMC":>7} {"Model",-10}");
                Console.WriteLine(new string('-', 75));
                double sumErr = 0; int n = 0;
                foreach (var (key, bench) in CcmcBenchmarks)
                {
                    float launchTs = (float)DateTimeOffset.Parse(key).ToUnixTimeSeconds();
                    var match = allPreds.MinBy(e => Math.Abs(e.LaunchTimeUnix - launchTs));
                    if (match == null) continue;
                    float err = Math.Abs(match.PredTransit - bench.ObsTransit);
                    sumErr += err; n++;
                    Console.WriteLine(
                        $"{key,-32} {bench.ObsTransit,6:F1}h {match.PredTransit,6:F1}h " +
                        $"{match.PredTransit - bench.ObsTransit,+7:F1}h {bench.CcmcMae,6:F1}h  {bench.Model,-10}");
                }
                if (n > 0)
                    Console.WriteLine($"{"CCMC 4-event MAE",-32} {"":>6}  {"":>6}  {sumErr/n,6:F1}h {"":>6}  (ours vs CCMC avg)");
            }

            return ExitCodes.Success;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"DOMAIN_SWEEP_ERROR type={ex.GetType().Name} message=\"{ex.Message}\"");
            return ExitCodes.SweepFailed;
        }
    }

    // ── Config conversion ─────────────────────────────────────────────────────

    private static DomainPipelineRunConfig ToRunConfig(DomainSweepConfig config, DomainSweepConfigLoader loader)
    {
        var stages = config.Stages.Keys.ToDictionary(
            k => k,
            k => loader.ToStageConfig(config, k),
            StringComparer.OrdinalIgnoreCase);

        var meta = config.DomainPipeline;

        var origination = new DomainGroupRunConfig(
            meta.Origination.Target,
            meta.Origination.Stage,
            null, null);

        var transit = new DomainGroupRunConfig(
            meta.Transit.Target,
            null,
            meta.Transit.PhysicsStage,
            meta.Transit.ResidualStage);

        var impact = new DomainGroupRunConfig(
            meta.Impact.Target,
            null,
            meta.Impact.PhysicsStage,
            meta.Impact.ResidualStage);

        return new DomainPipelineRunConfig(
            Name:              meta.Name,
            Folds:             meta.Cv.Folds,
            GapBufferDays:     meta.Cv.GapBufferDays,
            MinTestEvents:     meta.Cv.MinTestEvents,
            Origination:       origination,
            Transit:           transit,
            Impact:            impact,
            MetaLearnerStages: meta.MetaLearner.Stages,
            Stages:            stages,
            HeldOutAfter:      meta.Cv.HeldOutAfter);
    }

    // ── Data loading ──────────────────────────────────────────────────────────

    private async Task<IDataFrame> LoadStagingDataAsync(DomainSweepConfig config, CancellationToken ct)
    {
        var ds = config.DomainPipeline.DataSource;

        var sourceConfig = new DataSourceConfig(
            Name:             "staging_db",
            Provider:         ds.Provider,
            ConnectionString: ds.ConnectionString,
            Options: new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["table"] = ds.Table,
            });
        _dataRegistry.RegisterSource(sourceConfig);

        var filter = string.IsNullOrWhiteSpace(ds.Filter)
            ? string.Empty
            : $" WHERE {ds.Filter.Trim()}";

        var query = new DataQuery(
            Sql: $"SELECT * FROM \"{ds.Table}\"{filter} ORDER BY launch_time");

        try
        {
            return await _dataRegistry.LoadAsync("staging_db", query, ct);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"DomainSweepCommand: failed to load staging data: {ex.Message}", ex);
        }
    }

    // ── Output ────────────────────────────────────────────────────────────────

    // Known CCMC benchmark events: launch_time UTC → (observed_transit_h, ccmc_mae_h, model_name)
    private static readonly IReadOnlyDictionary<string, (float ObsTransit, float CcmcMae, string Model)> CcmcBenchmarks =
        new Dictionary<string, (float, float, string)>
        {
            ["2026-01-18T18:09:00+00:00"] = (25.5f,  7.8f,  "ELEvo"),
            ["2026-03-18T09:23:00+00:00"] = (59.9f,  3.2f,  "SIDC"),
            ["2026-03-30T03:24:00+00:00"] = (56.1f,  19.9f, "ELEvo"),
            ["2026-04-01T23:45:00+00:00"] = (40.1f,  8.6f,  "Median"),
        };

    private static async Task WriteResultsAsync(
        DomainPipelineResult result, string path, CancellationToken ct)
    {
        // Build per-event list from all fold predictions (typically one fold in held-out mode)
        var allEvents = result.FoldMetrics
            .Where(f => f.EventPredictions != null)
            .SelectMany(f => f.EventPredictions!)
            .Select(e => new
            {
                launch_time_unix = e.LaunchTimeUnix,
                launch_time_utc  = DateTimeOffset.FromUnixTimeSeconds((long)e.LaunchTimeUnix).ToString("o"),
                obs_transit_h    = e.ObsTransit,
                pred_transit_h   = e.PredTransit,
                error_transit_h  = e.PredTransit - e.ObsTransit,
                abs_error_h      = Math.Abs(e.PredTransit - e.ObsTransit),
                physics_transit_h = e.PhysicsTransit,
                obs_dst_nt       = e.ObsDst,
                pred_dst_nt      = e.PredDst,
            })
            .OrderBy(e => e.launch_time_unix)
            .ToList();

        // CCMC comparison: match our 4 benchmark events by launch_time
        var ccmcComparison = CcmcBenchmarks.Select(kv =>
        {
            var launchTs = (float)DateTimeOffset.Parse(kv.Key).ToUnixTimeSeconds();
            var match = allEvents.MinBy(e => Math.Abs(e.launch_time_unix - launchTs));
            float ourError = match != null ? Math.Abs(match.pred_transit_h - kv.Value.ObsTransit) : float.NaN;
            return new
            {
                cme_id          = kv.Key,
                obs_transit_h   = kv.Value.ObsTransit,
                ccmc_model      = kv.Value.Model,
                ccmc_mae_h      = kv.Value.CcmcMae,
                our_pred_h      = match?.pred_transit_h,
                our_error_h     = ourError,
                our_vs_ccmc     = float.IsNaN(ourError) ? float.NaN : ourError - kv.Value.CcmcMae,
            };
        }).ToList();

        double ccmcMeanMae = ccmcComparison.Where(c => !float.IsNaN(c.our_error_h))
                                            .Average(c => (double)c.our_error_h);

        var output = new
        {
            pipeline_name             = result.PipelineName,
            generated_at              = DateTime.UtcNow,
            phase7_h1_reference       = 20.259,
            mean_mae_transit          = result.MeanMaeTransit,
            mean_mae_dst              = result.MeanMaeDst,
            mean_mae_duration         = result.MeanMaeDuration,
            mean_mae_physics_baseline = result.MeanMaePhysicsBaseline,
            ccmc_4event_mae           = ccmcMeanMae,
            folds = result.FoldMetrics.Select(f => new
            {
                fold                 = f.FoldIndex,
                mae_transit          = f.MaeTransit,
                mae_dst              = f.MaeDst,
                mae_duration         = f.MaeDuration,
                mae_physics_baseline = f.MaePhysicsBaseline,
            }),
            ccmc_comparison = ccmcComparison,
            per_event       = allEvents,
        };

        var json = JsonSerializer.Serialize(output, new JsonSerializerOptions
        {
            WriteIndented = true,
            NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
        });
        await File.WriteAllTextAsync(path, json, ct);
    }
}
