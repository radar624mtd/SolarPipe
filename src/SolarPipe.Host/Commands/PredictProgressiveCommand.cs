using System.Globalization;
using Microsoft.Data.Sqlite;
using SolarPipe.Config;
using SolarPipe.Config.Models;
using SolarPipe.Data;
using SolarPipe.Training.Physics;

namespace SolarPipe.Host.Commands;

// Phase 9 M3: predict-progressive CLI command.
//
// Runs ProgressiveDragPropagator (M1) with DensityModulatedDrag (M2) backed by
// L1ObservationStream. γ₀, ambient wind, r_start, r_stop are read from the
// drag_baseline stage hyperparameters in a DomainSweepConfig-style YAML
// (Phase 8 live eval config), honouring OrdinalIgnoreCase key lookup.
//
// Modes:
//   --event <iso>   single-event (prints one trajectory)
//   --backtest      iterate every event in feature_vectors matching the held-out filter
//
// No-leakage contract:
//   L1ObservationStream caps lookahead to min(72h, observed transit * 1.1).
//   Command additionally refuses to consume any row past t0 + maxHours and
//   fails fast if --allow-future is not set and a future-label leak is detected.
public sealed class PredictProgressiveCommand : ICommand
{
    private const double DefaultNReference = DensityModulatedDrag.DefaultNReference;
    private const double MaxLookaheadHours = L1ObservationStream.MaxWindowHours;

    private readonly DomainSweepConfigLoader _loader;

    public PredictProgressiveCommand(DomainSweepConfigLoader loader)
    {
        _loader = loader;
    }

    public async Task<int> ExecuteAsync(string[] args, CancellationToken ct)
    {
        string configPath;
        try { configPath = ArgParser.Require(args, "--config"); }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine($"PREDICT_PROGRESSIVE_ERROR type=MissingArguments message=\"{ex.Message}\"");
            return ExitCodes.MissingArguments;
        }

        string mode                = ArgParser.Get(args, "--mode") ?? "density";
        string? eventIso           = ArgParser.Get(args, "--event");
        string? untilIso           = ArgParser.Get(args, "--until");
        string? backtest           = ArgParser.Get(args, "--backtest");
        string? nRefArg            = ArgParser.Get(args, "--n-ref");
        string? preLaunchHoursArg  = ArgParser.Get(args, "--pre-launch-hours");
        string? speedThresholdArg  = ArgParser.Get(args, "--speed-threshold");
        string outputDir           = ArgParser.Get(args, "--output") ?? "output/progressive";
        string? omniDbOverride     = ArgParser.Get(args, "--omni-db");
        bool allowFuture           = args.Any(a => string.Equals(a, "--allow-future", StringComparison.OrdinalIgnoreCase));

        if (!string.Equals(mode, "density", StringComparison.OrdinalIgnoreCase) &&
            !string.Equals(mode, "static",  StringComparison.OrdinalIgnoreCase))
        {
            Console.Error.WriteLine($"PREDICT_PROGRESSIVE_ERROR type=BadMode message=\"--mode must be 'density' or 'static', got '{mode}'\"");
            return ExitCodes.MissingArguments;
        }

        if (allowFuture)
        {
            Console.Error.WriteLine(
                "PREDICT_PROGRESSIVE_WARN type=LeakageBypass " +
                "message=\"--allow-future is set; no-leakage guard is DISABLED. " +
                "This is only valid for hindcast replay, NEVER for production predictions.\"");
        }

        bool isExplicitNRef = nRefArg != null;
        double nRef = DefaultNReference;
        if (nRefArg != null && !double.TryParse(nRefArg, NumberStyles.Float, CultureInfo.InvariantCulture, out nRef))
        {
            Console.Error.WriteLine($"PREDICT_PROGRESSIVE_ERROR type=BadNRef value=\"{nRefArg}\"");
            return ExitCodes.MissingArguments;
        }

        double preLaunchHours = L1ObservationStream.PreLaunchLookbackHours;
        if (preLaunchHoursArg != null && !double.TryParse(preLaunchHoursArg, NumberStyles.Float, CultureInfo.InvariantCulture, out preLaunchHours))
        {
            Console.Error.WriteLine($"PREDICT_PROGRESSIVE_ERROR type=BadPreLaunchHours value=\"{preLaunchHoursArg}\"");
            return ExitCodes.MissingArguments;
        }

        // --speed-threshold: CMEs with v0 below this use static mode; at or above use density mode.
        // Only active when --mode density is set. Default 540 km/s (empirical gap in 71-event set).
        double? speedThreshold = null;
        if (speedThresholdArg != null)
        {
            if (!double.TryParse(speedThresholdArg, NumberStyles.Float, CultureInfo.InvariantCulture, out double st))
            {
                Console.Error.WriteLine($"PREDICT_PROGRESSIVE_ERROR type=BadSpeedThreshold value=\"{speedThresholdArg}\"");
                return ExitCodes.MissingArguments;
            }
            speedThreshold = st;
        }

        var traceId = Guid.NewGuid().ToString("N")[..12];
        try
        {
            var config = await _loader.LoadAsync(configPath, ct);
            var dragParams = ExtractDragParameters(config);
            var staging = config.DomainPipeline.DataSource.ConnectionString;
            var omniDb  = omniDbOverride ?? ResolveOmniConnectionString();

            Directory.CreateDirectory(outputDir);

            var events = !string.IsNullOrWhiteSpace(backtest)
                ? await LoadEventsFromFeatureVectorsAsync(staging, config.DomainPipeline.Cv.HeldOutAfter, config.DomainPipeline.DataSource.Filter, ct)
                : new List<EventRow> { await LookupEventAsync(staging, eventIso, untilIso, ct) };

            string nRefMode = string.Equals(mode, "static", StringComparison.OrdinalIgnoreCase) ? "static"
                            : isExplicitNRef ? "cli_override" : "derived";
            Console.WriteLine($"PREDICT_PROGRESSIVE_START trace={traceId} events={events.Count} mode={mode} n_ref_mode={nRefMode} n_ref={nRef:F2} pre_launch_hours={preLaunchHours:F0}" +
                (speedThreshold.HasValue ? $" speed_threshold={speedThreshold:F0}" : ""));
            PredictProgressiveOutput.StructuredLog(new
            {
                kind = "start",
                trace_id = traceId,
                mode,
                n_ref_mode = nRefMode,
                n_ref = nRef,
                pre_launch_hours = preLaunchHours,
                speed_threshold = speedThreshold,
                config = configPath,
                events = events.Count,
                allow_future = allowFuture,
            });

            var results = new List<PredictProgressiveEventResult>();
            foreach (var ev in events)
            {
                ct.ThrowIfCancellationRequested();
                try
                {
                    var r = RunEvent(ev, dragParams, omniDb, mode, nRef, isExplicitNRef, preLaunchHours, speedThreshold, allowFuture, outputDir, traceId);
                    results.Add(r);
                    PredictProgressiveOutput.WriteEventJson(outputDir, r);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine(
                        $"PREDICT_PROGRESSIVE_WARN event={ev.ActivityId} type={ex.GetType().Name} message=\"{ex.Message}\"");
                    PredictProgressiveOutput.StructuredLog(new
                    {
                        kind = "event_error",
                        trace_id = traceId,
                        activity_id = ev.ActivityId,
                        error_type = ex.GetType().Name,
                        message = ex.Message,
                    });
                }
            }

            PredictProgressiveOutput.WriteResultsCsv(outputDir, results);
            PredictProgressiveOutput.PrintLeaderboard(results);

            Console.WriteLine(
                $"PREDICT_PROGRESSIVE_OK trace={traceId} ok={results.Count}/{events.Count} output={outputDir}");
            return results.Count == 0 ? ExitCodes.PredictionFailed : ExitCodes.Success;
        }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine($"PREDICT_PROGRESSIVE_ERROR type=MissingArguments message=\"{ex.Message}\"");
            return ExitCodes.MissingArguments;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(
                $"PREDICT_PROGRESSIVE_ERROR trace={traceId} type={ex.GetType().Name} message=\"{ex.Message}\"");
            return ExitCodes.PredictionFailed;
        }
    }

    private static DragParameters ExtractDragParameters(DomainSweepConfig config)
    {
        var stageName = config.DomainPipeline.Transit.PhysicsStage
            ?? throw new InvalidOperationException("domain_pipeline.transit.physics_stage is required.");
        if (!config.Stages.TryGetValue(stageName, out var stage))
            throw new InvalidOperationException($"Stage '{stageName}' referenced by transit.physics_stage not found.");
        if (!string.Equals(stage.Framework, "Physics", StringComparison.OrdinalIgnoreCase) ||
            !string.Equals(stage.ModelType, "DragBased", StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException(
                $"Stage '{stageName}' must be framework=Physics model_type=DragBased, got {stage.Framework}/{stage.ModelType}.");

        var hp = new Dictionary<string, object>(stage.Hyperparameters ?? new(), StringComparer.OrdinalIgnoreCase);
        double gamma0         = GetDouble(hp, "drag_parameter",       0.2e-7);
        double ambientWind    = GetDouble(hp, "background_speed_km_s", 400.0);
        double rStart         = GetDouble(hp, "r_start_rs",            21.5);
        double rStop          = GetDouble(hp, "r_stop_rs",             215.0);
        return new DragParameters(gamma0, ambientWind, rStart, rStop);
    }

    private static double GetDouble(Dictionary<string, object> hp, string key, double fallback)
    {
        if (!hp.TryGetValue(key, out var v) || v is null) return fallback;
        return v switch
        {
            double d => d,
            float f  => f,
            int i    => i,
            long l   => l,
            string s => double.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture),
            _        => Convert.ToDouble(v, CultureInfo.InvariantCulture),
        };
    }

    private static string ResolveOmniConnectionString()
    {
        // Phase 9 §3.5: omni_hourly lives in solar_data.db, which sits at the
        // repository root (authoritative path per docs/DATA_SCHEMA_REFERENCE.md
        // §1 — "solar_data.db — 11 GB (root of project)"). --omni-db remains
        // available as an explicit override for out-of-tree data.
        return "Data Source=solar_data.db";
    }

    private static async Task<EventRow> LookupEventAsync(
        string stagingConn, string? eventIso, string? untilIso, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(eventIso))
            throw new ArgumentException("--event <iso-launch-time> is required when --backtest is not set.");
        var launch = DateTime.Parse(eventIso, CultureInfo.InvariantCulture,
            DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal);

        using var conn = new SqliteConnection(stagingConn);
        await conn.OpenAsync(ct);
        using var cmd = conn.CreateCommand();
        // feature_vectors.launch_time is written in one of two shapes depending
        // on producer: "yyyy-MM-dd HH:mm:ss" (test fixtures, older staging) or
        // "yyyy-MM-ddTHH:mm:ss+00:00" (current pandas ISO export). Match both
        // by scanning a per-minute prefix and parsing on the client.
        cmd.CommandText =
            @"SELECT activity_id, launch_time, cme_speed_kms, transit_time_hours, icme_arrival_time
              FROM feature_vectors
              WHERE launch_time LIKE @space_prefix OR launch_time LIKE @iso_prefix
              ORDER BY launch_time ASC";
        cmd.Parameters.AddWithValue("@space_prefix", launch.ToString("yyyy-MM-dd HH:mm") + "%");
        cmd.Parameters.AddWithValue("@iso_prefix",   launch.ToString("yyyy-MM-ddTHH:mm") + "%");
        using var r = await cmd.ExecuteReaderAsync(ct);
        if (!await r.ReadAsync(ct))
            throw new InvalidOperationException($"No feature_vectors row for launch_time={launch:O}.");

        string activityId = r.GetString(0);
        double v0 = r.GetDouble(2);
        double? observedTransit = r.IsDBNull(3) ? null : r.GetDouble(3);

        double until = untilIso is null
            ? MaxLookaheadHours
            : Math.Min(MaxLookaheadHours,
                (DateTime.Parse(untilIso, CultureInfo.InvariantCulture,
                    DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal) - launch).TotalHours);

        return new EventRow(activityId, launch, v0, observedTransit, until);
    }

    private static async Task<List<EventRow>> LoadEventsFromFeatureVectorsAsync(
        string stagingConn, string? heldOutAfter, string? filter, CancellationToken ct)
    {
        var list = new List<EventRow>();
        using var conn = new SqliteConnection(stagingConn);
        await conn.OpenAsync(ct);
        using var cmd = conn.CreateCommand();
        var whereParts = new List<string>();
        if (!string.IsNullOrWhiteSpace(heldOutAfter))
            whereParts.Add("launch_time >= @held");
        whereParts.Add("cme_speed_kms IS NOT NULL");
        whereParts.Add("transit_time_hours IS NOT NULL");
        whereParts.Add("transit_time_hours > 0");
        whereParts.Add("transit_time_hours < 300");
        // The YAML `filter` is a trusted config value (same pattern Phase 8
        // DomainPipelineSweep already uses for feature_vectors filtering). It
        // cannot be user-supplied; any attacker able to write the YAML already
        // owns the pipeline. Still concatenated rather than parameterised
        // because SQLite does not support parameterising arbitrary WHERE clauses.
        if (!string.IsNullOrWhiteSpace(filter))
            whereParts.Add("(" + filter + ")");

        cmd.CommandText =
            "SELECT activity_id, launch_time, cme_speed_kms, transit_time_hours " +
            "FROM feature_vectors WHERE " + string.Join(" AND ", whereParts) +
            " ORDER BY launch_time ASC";
        if (!string.IsNullOrWhiteSpace(heldOutAfter))
        {
            var heldDate = DateTime.Parse(heldOutAfter, CultureInfo.InvariantCulture,
                DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal);
            cmd.Parameters.AddWithValue("@held", heldDate.ToString("yyyy-MM-dd HH:mm:ss"));
        }

        using var r = await cmd.ExecuteReaderAsync(ct);
        while (await r.ReadAsync(ct))
        {
            var launch = DateTime.Parse(r.GetString(1), CultureInfo.InvariantCulture,
                DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal);
            double v0 = r.GetDouble(2);
            double transit = r.GetDouble(3);
            // M6.5: fixed 72h cap, independent of the label. Previously this was
            // min(72h, transit * 1.1) — that used the label to bound the window,
            // which is a mild leak AND double-bound the propagator runway so fast
            // events (Mar-30, Apr-01) could time out before hitting 215 Rs. The
            // propagator terminates on target-reached, shock-detected, or 72h —
            // whichever fires first — so a fixed 72h cap is safe for all events.
            double until = MaxLookaheadHours;
            list.Add(new EventRow(r.GetString(0), launch, v0, transit, until));
        }
        return list;
    }

    private static PredictProgressiveEventResult RunEvent(
        EventRow ev, DragParameters drag, string omniConn,
        string mode, double nRef, bool isExplicitNRef, double preLaunchHours,
        double? speedThreshold, bool allowFuture, string outputDir, string traceId)
    {
        var stream = L1ObservationStream.LoadFromSqlite(omniConn, ev.LaunchTime, ev.MaxHours);

        // No-leakage assertion: L1ObservationStream already caps the window, but
        // reconfirm that WindowEnd - LaunchTime does not exceed maxHours unless allowed.
        double actualWindowHours = (stream.WindowEnd - stream.LaunchTime).TotalHours;
        if (!allowFuture && actualWindowHours > ev.MaxHours + 1e-6)
            throw new InvalidOperationException(
                $"Leakage guard: window {actualWindowHours:F2}h exceeds maxHours {ev.MaxHours:F2}h (use --allow-future to bypass).");

        // Per-event model routing: if speed threshold is set and this CME is slow,
        // fall back to static regardless of the global --mode flag. Density modulation
        // causes timeouts and regressions for slow CMEs (v0 < threshold) because γ_eff
        // with real density data adds noise around the already-calibrated static γ₀.
        bool globalStatic = string.Equals(mode, "static", StringComparison.OrdinalIgnoreCase);
        bool routedToStatic = speedThreshold.HasValue && ev.InitialSpeedKmPerSec < speedThreshold.Value;
        bool isStatic = globalStatic || routedToStatic;
        string modeEffective = isStatic ? "static" : "density";

        // Derive n_ref from the pre-launch OMNI window unless caller provided an explicit override.
        double nRefDerived = isStatic || isExplicitNRef
            ? double.NaN
            : L1ObservationStream.ComputePreLaunchDensityMedian(omniConn, ev.LaunchTime, preLaunchHours);
        double nRefEffective = (!isStatic && !isExplicitNRef && !double.IsNaN(nRefDerived))
            ? nRefDerived
            : nRef;
        string nRefMode = isStatic ? "static" : isExplicitNRef ? "cli_override" : "derived";

        IDragCoefficientProvider gammaProvider = isStatic
            ? new StaticDragCoefficient(drag.Gamma0KmInv)
            : new DensityModulatedDrag(drag.Gamma0KmInv, stream, nRefEffective);
        // Static mode MUST reproduce Phase 8 DragBasedModel.RunOde exactly — no L1
        // assimilation of any kind, including shock detection. Density mode opts in.
        IShockDetector? shock = isStatic ? null : new L1ShockDetector(stream);

        var propagator = new ProgressiveDragPropagator(
            gammaProvider,
            drag.AmbientWindKmPerSec,
            drag.RStartSolarRadii,
            drag.RStopSolarRadii,
            ev.MaxHours);

        var traj = propagator.Propagate(ev.InitialSpeedKmPerSec, shock);

        double? errorHours = (ev.ObservedTransitHours.HasValue && traj.ArrivalTimeHours.HasValue)
            ? traj.ArrivalTimeHours.Value - ev.ObservedTransitHours.Value
            : (double?)null;

        double gammaEffMean = traj.Steps.Count == 0 ? double.NaN :
            traj.Steps.Average(s => s.GammaKmInv);
        double gammaEffFinal = traj.Steps.Count == 0 ? double.NaN :
            traj.Steps[^1].GammaKmInv;

        PredictProgressiveOutput.WriteTrajectoryCsv(outputDir, ev.ActivityId, traj);

        PredictProgressiveOutput.StructuredLog(new
        {
            kind = "event_result",
            trace_id = traceId,
            activity_id = ev.ActivityId,
            launch_time = ev.LaunchTime.ToString("O"),
            initial_speed_kms = ev.InitialSpeedKmPerSec,
            mode,
            mode_effective = modeEffective,
            routed_to_static = routedToStatic,
            n_ref_mode = nRefMode,
            n_ref_derived = double.IsNaN(nRefDerived) ? (double?)null : nRefDerived,
            n_ref_effective = nRefEffective,
            arrival_hours = traj.ArrivalTimeHours,
            arrival_speed_kms = traj.ArrivalSpeedKmPerSec,
            termination = traj.TerminationReason,
            shock_arrived = traj.ShockArrived,
            n_missing_hours = traj.NMissingHours,
            density_coverage = stream.DensityCoverage,
            observed_transit_hours = ev.ObservedTransitHours,
            error_hours = errorHours,
            gamma_eff_mean = gammaEffMean,
            gamma_eff_final = gammaEffFinal,
            steps = traj.Steps.Count,
        });

        return new PredictProgressiveEventResult(
            ev.ActivityId, ev.LaunchTime, ev.InitialSpeedKmPerSec,
            ev.ObservedTransitHours, traj.ArrivalTimeHours, errorHours,
            traj.TerminationReason, traj.ShockArrived, traj.NMissingHours,
            stream.DensityCoverage, gammaEffMean, gammaEffFinal, traj);
    }

    private readonly record struct DragParameters(
        double Gamma0KmInv, double AmbientWindKmPerSec, double RStartSolarRadii, double RStopSolarRadii);

    private sealed record EventRow(
        string ActivityId, DateTime LaunchTime, double InitialSpeedKmPerSec,
        double? ObservedTransitHours, double MaxHours);
}
