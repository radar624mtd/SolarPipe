using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using SolarPipe.Config;
using SolarPipe.Config.Models;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data;
using SolarPipe.Training.Adapters;
using SolarPipe.Training.Sweep;

namespace SolarPipe.Host.Commands;

// SweepCommand: CLI entry point for Phase 7 hypothesis sweep.
// Usage:
//   sweep --config <path>              # full sweep
//   sweep --config <path> --resume     # reuse existing sweep_id (RULE-162)
//   sweep --config <path> --fresh      # clear checkpoints, new sweep_id (RULE-162)
//
// Converts SweepConfig (YAML model) → SweepRunConfig (Core model) before calling ModelSweep.
// Flow: pre-flight → parallel CV → leaderboard → decompose winner → v2 config (RULE-168)
public sealed class SweepCommand : ICommand
{
    private readonly SweepConfigLoader _loader;
    private readonly DataSourceRegistry _dataRegistry;
    private readonly IReadOnlyList<IFrameworkAdapter> _adapters;
    private readonly ModelSweep _modelSweep;
    private readonly string _cacheRoot;
    private readonly SidecarLifecycleService? _sidecar;

    public SweepCommand(
        SweepConfigLoader loader,
        DataSourceRegistry dataRegistry,
        IReadOnlyList<IFrameworkAdapter> adapters,
        ModelSweep modelSweep,
        string cacheRoot,
        SidecarLifecycleService? sidecar = null)
    {
        _loader       = loader       ?? throw new ArgumentNullException(nameof(loader));
        _dataRegistry = dataRegistry ?? throw new ArgumentNullException(nameof(dataRegistry));
        _adapters     = adapters     ?? throw new ArgumentNullException(nameof(adapters));
        _modelSweep   = modelSweep   ?? throw new ArgumentNullException(nameof(modelSweep));
        _cacheRoot    = cacheRoot    ?? throw new ArgumentNullException(nameof(cacheRoot));
        _sidecar      = sidecar;
    }

    public async Task<int> ExecuteAsync(string[] args, CancellationToken ct)
    {
        string configPath;
        try { configPath = ArgParser.Require(args, "--config"); }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine($"SWEEP_ERROR type=MissingArguments message=\"{ex.Message}\"");
            return ExitCodes.MissingArguments;
        }

        bool resume = Array.Exists(args, a => a.Equals("--resume", StringComparison.OrdinalIgnoreCase));
        bool fresh  = Array.Exists(args, a => a.Equals("--fresh",  StringComparison.OrdinalIgnoreCase));
        string outputDir = ArgParser.Get(args, "--output") ?? "output";

        try
        {
            var sweepConfig = await _loader.LoadAsync(configPath, ct);
            var runConfig   = ToRunConfig(sweepConfig);

            // Determine sweep ID (RULE-162)
            string? sweepId = null;
            var sweepIdPath = Path.Combine(_cacheRoot, "sweeps", $"{runConfig.Name}.id");

            if (resume && File.Exists(sweepIdPath))
            {
                sweepId = await File.ReadAllTextAsync(sweepIdPath, ct);
                Console.WriteLine($"SWEEP_RESUME sweep_id={sweepId}");
            }
            else if (fresh)
            {
                var freshId = ModelSweep.ComputeSweepId(runConfig);
                var freshCachePath = Path.Combine(_cacheRoot, "sweeps", freshId);
                if (Directory.Exists(freshCachePath))
                {
                    Directory.Delete(freshCachePath, recursive: true);
                    Console.WriteLine($"SWEEP_FRESH cleared cache={freshCachePath}");
                }
                sweepId = freshId;
            }

            // Pre-flight probe: pass a lightweight frame map (RULE-160)
            // Full data loading happens inside RunAsync per-hypothesis.
            // We pass an empty dict here; pre-flight skips data checks when dict is empty.
            var dataFrames = new Dictionary<string, IDataFrame>(StringComparer.OrdinalIgnoreCase);

            // Pre-flight gate (RULE-160) — all checks must pass
            var preflight = await _modelSweep.ValidatePreFlightAsync(runConfig, dataFrames, ct);
            if (preflight.HasFailures)
            {
                foreach (var f in preflight.Failures)
                    Console.Error.WriteLine(
                        $"[preflight:FAIL component={f.Component}] {f.Message}");
                return ExitCodes.PreFlightFailed;
            }

            Console.WriteLine("SWEEP_PREFLIGHT status=pass");

            // Start Python sidecar if wired (required for H7 TFT hypothesis)
            if (_sidecar is not null)
            {
                Console.WriteLine("SWEEP_SIDECAR starting...");
                await _sidecar.StartAsync(ct);
                Console.WriteLine("SWEEP_SIDECAR ready");
            }

            // Load sweep data
            var sweepData = await LoadSweepDataAsync(sweepConfig, ct);

            // Persist sweep_id for --resume support
            string id = sweepId ?? ModelSweep.ComputeSweepId(runConfig);
            Directory.CreateDirectory(Path.GetDirectoryName(sweepIdPath)!);
            await File.WriteAllTextAsync(sweepIdPath, id, ct);

            // Run sweep (RULE-163, RULE-164)
            var result = await _modelSweep.RunAsync(
                runConfig, sweepData,
                dbmBaselineMae: 12.0,
                id, ct);

            // Write leaderboard JSON
            Directory.CreateDirectory(outputDir);
            var leaderboardPath = Path.Combine(outputDir, "phase7_sweep_results.json");
            await WriteLeaderboardAsync(result, leaderboardPath, ct);
            Console.WriteLine($"SWEEP_LEADERBOARD written={leaderboardPath}");

            // RULE-167: decompose winner only
            var winner = result.Winner;
            Console.WriteLine(
                $"SWEEP_WINNER hypothesis={winner.HypothesisId} " +
                $"compose=\"{winner.ComposeExpression}\" " +
                $"mae={winner.AggregatedMetrics.MaeMean:F3}");

            // RULE-168: generate v2 config only if sweep completed all hypotheses
            if (result.Hypotheses.Count == runConfig.Hypotheses.Count)
            {
                var v2Path = "configs/flux_rope_propagation_v2.yaml";
                await GenerateV2ConfigAsync(sweepConfig, winner, v2Path, ct);
                Console.WriteLine($"SWEEP_V2_CONFIG written={v2Path}");
            }
            else
            {
                Console.Error.WriteLine(
                    "SWEEP_WARN: incomplete sweep — v2 config not generated (RULE-168).");
            }

            return ExitCodes.Success;
        }
        catch (Exception ex)
        {
            var root = ex.InnerException ?? ex;
            Console.Error.WriteLine(
                $"SWEEP_ERROR type={root.GetType().Name} message=\"{root.Message}\"");
            return ExitCodes.SweepFailed;
        }
        finally
        {
            if (_sidecar is not null)
            {
                await _sidecar.StopAsync(CancellationToken.None);
                await _sidecar.DisposeAsync();
                Console.WriteLine("SWEEP_SIDECAR stopped");
            }
        }
    }

    // ── Config conversion ─────────────────────────────────────────────────────

    // Convert YAML model → Core model that ModelSweep consumes.
    private static SweepRunConfig ToRunConfig(SweepConfig sweepConfig)
    {
        var stages = sweepConfig.Stages.ToDictionary(
            kvp => kvp.Key,
            kvp => sweepConfig.ToStageConfig(kvp.Key),
            StringComparer.OrdinalIgnoreCase);

        var hypotheses = sweepConfig.Sweep.Hypotheses
            .Select(h => new SweepHypothesis(h.Id, h.Compose, h.Stages))
            .ToList();

        return new SweepRunConfig(
            Name:           sweepConfig.Sweep.Name,
            Parallel:       sweepConfig.Sweep.Parallel,
            Folds:          sweepConfig.Sweep.Cv.Folds,
            GapBufferDays:  sweepConfig.Sweep.Cv.GapBufferDays,
            MinTestEvents:  sweepConfig.Sweep.Cv.MinTestEvents,
            Hypotheses:     hypotheses,
            Stages:         stages);
    }

    // ── Data loading ──────────────────────────────────────────────────────────

    // DB path relative to CWD (working directory = repo root when invoked via dotnet run).
    private const string CmeCatalogPath = "data/data/output/cme_catalog.db";

    private async Task<IDataFrame> LoadSweepDataAsync(
        SweepConfig sweepConfig, CancellationToken ct)
    {
        // Register the data source pointing at the flat training view.
        // training_features_v3 already has all YAML alias column names applied:
        //   cme_speed_kms, bz_gsm_proxy_nt, sw_density_n_cc, sw_speed_ambient_kms, etc.
        var sourceConfig = new DataSourceConfig(
            Name: "cme_events",
            Provider: "sqlite",
            ConnectionString: $"Data Source={CmeCatalogPath}",
            Options: new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["table"] = "training_features_v3",
            });
        _dataRegistry.RegisterSource(sourceConfig);

        // SELECT * — the view already has the right column names and all derived features.
        var query = new DataQuery(Sql: "SELECT * FROM \"training_features_v3\" " +
            "WHERE transit_time_hours IS NOT NULL " +
            "AND transit_time_hours > 0 " +
            "AND transit_time_hours < 300 " +
            "AND quality_flag >= 3 " +
            "ORDER BY launch_time");

        try
        {
            return await _dataRegistry.LoadAsync("cme_events", query, ct);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"SweepCommand: failed to load sweep data from 'cme_events': {ex.Message}", ex);
        }
    }

    // ── Output generation ─────────────────────────────────────────────────────

    private static async Task WriteLeaderboardAsync(
        SweepResult result, string path, CancellationToken ct)
    {
        var leaderboard = new
        {
            sweep_id     = result.SweepId,
            sweep_name   = result.SweepName,
            generated_at = DateTime.UtcNow,
            hypotheses   = result.Hypotheses.Select((h, rank) => new
            {
                rank     = rank + 1,
                id       = h.HypothesisId,
                compose  = h.ComposeExpression,
                mae_mean  = h.AggregatedMetrics.MaeMean,
                mae_std   = h.AggregatedMetrics.MaeStd,
                rmse_mean = h.AggregatedMetrics.RmseMean,
                r2_mean   = h.AggregatedMetrics.R2Mean,
                skill_mean = h.AggregatedMetrics.SkillMean,
                optimized_weights = h.OptimizedWeights,
                folds = h.FoldMetrics.Select(f => new
                {
                    fold     = f.FoldIndex,
                    mae      = f.Mae,
                    rmse     = f.Rmse,
                    r2       = f.R2,
                    bias     = f.Bias,
                    skill    = f.SkillVsDbm,
                    hr6h     = f.HitRate6h,
                    hr12h    = f.HitRate12h,
                    pinball  = f.PinballLoss10,
                    coverage = f.CoverageRate90,
                    kendall  = f.KendallTau,
                })
            })
        };

        var json = JsonSerializer.Serialize(leaderboard, new JsonSerializerOptions
        {
            WriteIndented = true,
            NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
        });
        await File.WriteAllTextAsync(path, json, ct);
    }

    // RULE-168: v2 config is a standard pipeline config (not sweep schema).
    private static async Task GenerateV2ConfigAsync(
        SweepConfig sweepConfig,
        HypothesisResult winner,
        string v2Path,
        CancellationToken ct)
    {
        var winnerHypothesis = sweepConfig.Sweep.Hypotheses
            .First(h => h.Id == winner.HypothesisId);

        var stages = new System.Text.StringBuilder();
        foreach (var stageName in winnerHypothesis.Stages)
        {
            var stage = sweepConfig.Stages[stageName];
            stages.AppendLine($"  {stageName}:");
            stages.AppendLine($"    framework: {stage.Framework}");
            stages.AppendLine($"    model_type: {stage.ModelType}");
            stages.AppendLine($"    data_source: cme_events");
            stages.AppendLine($"    features: [{string.Join(", ", stage.Features)}]");
            stages.AppendLine($"    target: {stage.Target}");

            if (stage.Hyperparameters is not null && stage.Hyperparameters.Count > 0)
            {
                stages.AppendLine("    hyperparameters:");
                foreach (var (k, v) in stage.Hyperparameters)
                    stages.AppendLine($"      {k}: {v}");
            }

            // RULE-165: write optimized weights into ensemble stages
            if (winner.OptimizedWeights is not null
                && winner.OptimizedWeights.TryGetValue(stageName, out var w))
            {
                stages.AppendLine($"    weight: {w:F6}");
            }
        }

        var yaml = $@"name: flux_rope_propagation_v2
# Generated by Phase 7 hypothesis sweep. Winner: {winner.HypothesisId}
# Generated: {DateTime.UtcNow:O}

data_sources:
  cme_events:
    provider: sqlite
    connection_string: data/cme_catalog.db
    options:
      table: cme_events

stages:
{stages}
compose: ""{winner.ComposeExpression}""
";

        Directory.CreateDirectory(Path.GetDirectoryName(v2Path) ?? ".");
        await File.WriteAllTextAsync(v2Path, yaml, ct);
    }
}
