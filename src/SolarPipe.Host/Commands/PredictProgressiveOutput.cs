using System.Globalization;
using System.Text;
using System.Text.Json;
using SolarPipe.Training.Physics;

namespace SolarPipe.Host.Commands;

// Phase 9 M3: output writers for predict-progressive.
// Split from PredictProgressiveCommand so the command file stays under 400 lines.
// Pure IO — no command-logic coupling. Reused by the single-event and backtest paths.
internal static class PredictProgressiveOutput
{
    public static void WriteTrajectoryCsv(string outputDir, string activityId, ProgressiveTrajectory traj)
    {
        var safeId = SanitizeId(activityId);
        var path = Path.Combine(outputDir, $"trajectory_{safeId}.csv");
        var sb = new StringBuilder();
        sb.AppendLine("hour_index,time_hours,r_solar_radii,speed_km_s,gamma_km_inv,n_obs,v_obs,gamma_fellback");
        foreach (var s in traj.Steps)
        {
            sb.Append(s.HourIndex).Append(',')
              .Append(s.TimeHours.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
              .Append(s.RSolarRadii.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
              .Append(s.SpeedKmPerSec.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
              .Append(s.GammaKmInv.ToString("E6", CultureInfo.InvariantCulture)).Append(',')
              .Append(s.NObs?.ToString("F3", CultureInfo.InvariantCulture) ?? "").Append(',')
              .Append(s.VObs?.ToString("F3", CultureInfo.InvariantCulture) ?? "").Append(',')
              .Append(s.GammaFellBack ? "1" : "0").AppendLine();
        }
        AtomicWrite(path, sb.ToString());
    }

    public static void WriteEventJson(string outputDir, PredictProgressiveEventResult r)
    {
        var safeId = SanitizeId(r.ActivityId);
        var path = Path.Combine(outputDir, $"progressive_{safeId}.json");
        var payload = new
        {
            activity_id = r.ActivityId,
            launch_time = r.LaunchTime.ToString("O"),
            initial_speed_kms = r.InitialSpeedKmPerSec,
            observed_transit_hours = r.ObservedTransitHours,
            arrival_time_hours = r.ArrivalTimeHours,
            error_hours = r.ErrorHours,
            termination_reason = r.TerminationReason,
            shock_arrived = r.ShockArrived,
            n_missing_hours = r.NMissingHours,
            density_coverage = r.DensityCoverage,
            gamma_eff_mean = r.GammaEffMean,
            gamma_eff_final = r.GammaEffFinal,
            trajectory = r.Trajectory.Steps.Select(s => new
            {
                hour_index = s.HourIndex,
                time_hours = s.TimeHours,
                r_solar_radii = s.RSolarRadii,
                speed_km_s = s.SpeedKmPerSec,
                gamma_km_inv = s.GammaKmInv,
                n_obs = s.NObs,
                v_obs = s.VObs,
                gamma_fellback = s.GammaFellBack,
            }),
        };
        var json = JsonSerializer.Serialize(payload, new JsonSerializerOptions
        {
            WriteIndented = true,
            NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
        });
        AtomicWrite(path, json);
    }

    public static void WriteResultsCsv(string outputDir, IReadOnlyList<PredictProgressiveEventResult> results)
    {
        var path = Path.Combine(outputDir, "progressive_results.csv");
        var sb = new StringBuilder();
        sb.AppendLine("activity_id,launch_time,initial_speed_kms,observed_transit_hours,arrival_time_hours," +
                      "error_hours,termination_reason,shock_arrived,n_missing_hours,density_coverage," +
                      "gamma_eff_mean,gamma_eff_final");
        foreach (var r in results)
        {
            sb.Append(r.ActivityId).Append(',')
              .Append(r.LaunchTime.ToString("O")).Append(',')
              .Append(r.InitialSpeedKmPerSec.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
              .Append(r.ObservedTransitHours?.ToString("F3", CultureInfo.InvariantCulture) ?? "").Append(',')
              .Append(r.ArrivalTimeHours?.ToString("F3", CultureInfo.InvariantCulture) ?? "").Append(',')
              .Append(r.ErrorHours?.ToString("F3", CultureInfo.InvariantCulture) ?? "").Append(',')
              .Append(r.TerminationReason).Append(',')
              .Append(r.ShockArrived ? "1" : "0").Append(',')
              .Append(r.NMissingHours).Append(',')
              .Append(r.DensityCoverage.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
              .Append(r.GammaEffMean.ToString("E6", CultureInfo.InvariantCulture)).Append(',')
              .Append(r.GammaEffFinal.ToString("E6", CultureInfo.InvariantCulture)).AppendLine();
        }
        AtomicWrite(path, sb.ToString());
    }

    public static void PrintLeaderboard(IReadOnlyList<PredictProgressiveEventResult> results)
    {
        var withError = results.Where(r => r.ErrorHours.HasValue).ToList();
        if (withError.Count == 0)
        {
            Console.WriteLine("PREDICT_PROGRESSIVE_LEADERBOARD n=0 (no labeled events)");
            return;
        }
        double mae    = withError.Average(r => Math.Abs(r.ErrorHours!.Value));
        double rmse   = Math.Sqrt(withError.Average(r => r.ErrorHours!.Value * r.ErrorHours!.Value));
        double bias   = withError.Average(r => r.ErrorHours!.Value);
        int shockN    = results.Count(r => r.ShockArrived);
        int fallbacks = results.Count(r => r.DensityCoverage < 0.8);
        var inv = CultureInfo.InvariantCulture;
        Console.WriteLine(
            $"PREDICT_PROGRESSIVE_LEADERBOARD n={withError.Count} " +
            $"mae={mae.ToString("F2", inv)}h rmse={rmse.ToString("F2", inv)}h " +
            $"bias={bias.ToString("+0.00;-0.00", inv)}h " +
            $"shock_detected={shockN} low_coverage={fallbacks}");
    }

    public static void StructuredLog(object entry)
    {
        try
        {
            Directory.CreateDirectory("logs");
            var line = JsonSerializer.Serialize(entry, new JsonSerializerOptions
            {
                NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
            });
            File.AppendAllText("logs/dotnet_latest.json", line + Environment.NewLine);
        }
        catch { /* logging must never break the command */ }
    }

    public static void AtomicWrite(string path, string content)
    {
        var tmp = path + $".tmp_{Guid.NewGuid():N}";
        File.WriteAllText(tmp, content);
        File.Move(tmp, path, overwrite: true);
    }

    private static string SanitizeId(string id) =>
        string.Concat(id.Select(c => char.IsLetterOrDigit(c) || c == '-' || c == '_' ? c : '_'));
}
