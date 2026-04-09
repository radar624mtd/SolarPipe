using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using CsvHelper;
using CsvHelper.Configuration;

namespace SolarPipe.Training.Validation;

// ── CSV record types ─────────────────────────────────────────────────────────

public sealed class ObservedTransitRecord
{
    public string CmeId { get; set; } = "";
    public string DepartureUtc { get; set; } = "";
    public string ArrivalUtc { get; set; } = "";
    public double TransitHoursObserved { get; set; }
    public double CmeSpeedKms { get; set; }
    public string BzGsmMinNt { get; set; } = "";   // string — may be empty (Bz gap)
    public double ArrivalSpeedKms { get; set; }
    public string StormLevel { get; set; } = "";
    public string Notes { get; set; } = "";
}

public sealed class ObservedTransitMap : ClassMap<ObservedTransitRecord>
{
    public ObservedTransitMap()
    {
        Map(m => m.CmeId).Name("cme_id");
        Map(m => m.DepartureUtc).Name("departure_utc");
        Map(m => m.ArrivalUtc).Name("arrival_utc");
        Map(m => m.TransitHoursObserved).Name("transit_hours_observed");
        Map(m => m.CmeSpeedKms).Name("cme_speed_kms");
        Map(m => m.BzGsmMinNt).Name("bz_gsm_min_nt");
        Map(m => m.ArrivalSpeedKms).Name("arrival_speed_kms");
        Map(m => m.StormLevel).Name("storm_level");
        Map(m => m.Notes).Name("notes");
    }
}

public sealed class ScoreboardRecord
{
    public string CmeId { get; set; } = "";
    public string Model { get; set; } = "";
    public string PredictedArrivalUtc { get; set; } = "";
    public string ActualArrivalUtc { get; set; } = "";
    public string ErrorHours { get; set; } = "";    // string — may be empty
    public string MaePublished { get; set; } = "";  // string — may be empty
}

public sealed class ScoreboardMap : ClassMap<ScoreboardRecord>
{
    public ScoreboardMap()
    {
        Map(m => m.CmeId).Name("cme_id");
        Map(m => m.Model).Name("model");
        Map(m => m.PredictedArrivalUtc).Name("predicted_arrival_utc");
        Map(m => m.ActualArrivalUtc).Name("actual_arrival_utc");
        Map(m => m.ErrorHours).Name("error_hours");
        Map(m => m.MaePublished).Name("mae_published");
    }
}

// ── Domain models ────────────────────────────────────────────────────────────

public sealed record ObservedTransit(
    string CmeId,
    DateTime DepartureUtc,
    DateTime ArrivalUtc,
    double TransitHoursObserved,
    double CmeSpeedKms,
    double BzGsmMinNt,       // NaN when data not available (RULE-120)
    double ArrivalSpeedKms,
    string StormLevel,
    string Notes);

public sealed record ScoreboardEntry(
    string CmeId,
    string Model,
    double MaePublished);   // NaN when not published

public sealed record EventPrediction(
    string CmeId,
    double PredictedTransitHours,
    double LowerBound = double.NaN,
    double UpperBound = double.NaN);

public sealed record ValidationMetrics(
    double Mae,
    double Rmse,
    double Bias,
    int HitRate6h,
    int HitRate12h,
    int TotalEvents,
    double SkillVsDbm,
    double IntervalWidth90 = double.NaN,
    double CoverageRate90 = double.NaN)
{
    public static readonly ValidationMetrics Empty =
        new(double.NaN, double.NaN, double.NaN, 0, 0, 0, double.NaN);
}

public sealed record EventRowReport(
    string CmeId,
    string EventLabel,
    double ActualHours,
    double PredictedHours,
    double ErrorHours,
    double BestScoreboardMae,
    string BestScoreboardModel,
    double DbmMae,
    double IntervalLower = double.NaN,
    double IntervalUpper = double.NaN);

public sealed record ValidationReport(
    ValidationMetrics Metrics,
    IReadOnlyList<EventRowReport> EventRows,
    string AsciiTable,
    DateTime GeneratedAt);

// ── Validator ────────────────────────────────────────────────────────────────

public sealed class EventValidator
{
    private static readonly CsvConfiguration _csvConfig = new(CultureInfo.InvariantCulture)
    {
        HasHeaderRecord = true,
        MissingFieldFound = null,
        BadDataFound = null,
    };

    private static readonly JsonSerializerOptions _jsonOpts = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };

    public static async Task<List<ObservedTransit>> LoadObservedTransitsAsync(
        string csvPath, CancellationToken ct)
    {
        using var reader = new StreamReader(csvPath);
        using var csv = new CsvReader(reader, _csvConfig);
        csv.Context.RegisterClassMap<ObservedTransitMap>();

        var records = new List<ObservedTransit>();
        await foreach (var r in csv.GetRecordsAsync<ObservedTransitRecord>(ct))
        {
            records.Add(new ObservedTransit(
                CmeId:                r.CmeId,
                DepartureUtc:         ParseUtcIso(r.DepartureUtc),
                ArrivalUtc:           ParseUtcIso(r.ArrivalUtc),
                TransitHoursObserved: r.TransitHoursObserved,
                CmeSpeedKms:          r.CmeSpeedKms,
                BzGsmMinNt:           ParseOptionalDouble(r.BzGsmMinNt), // NaN → gap (RULE-120)
                ArrivalSpeedKms:      r.ArrivalSpeedKms,
                StormLevel:           r.StormLevel,
                Notes:                r.Notes));
        }
        return records;
    }

    public static async Task<List<ScoreboardEntry>> LoadScoreboardAsync(
        string csvPath, CancellationToken ct)
    {
        using var reader = new StreamReader(csvPath);
        using var csv = new CsvReader(reader, _csvConfig);
        csv.Context.RegisterClassMap<ScoreboardMap>();

        var records = new List<ScoreboardEntry>();
        await foreach (var r in csv.GetRecordsAsync<ScoreboardRecord>(ct))
        {
            double mae = ParseOptionalDouble(r.MaePublished);
            records.Add(new ScoreboardEntry(r.CmeId, r.Model, mae));
        }
        return records;
    }

    public static ValidationMetrics ComputeMetrics(
        IReadOnlyList<double> predicted,
        IReadOnlyList<double> observed,
        double dbmBaselineMae)
    {
        if (predicted.Count == 0 || predicted.Count != observed.Count)
            return ValidationMetrics.Empty;

        double sumAbs = 0, sumSq = 0, sumErr = 0;
        int hit6 = 0, hit12 = 0;

        for (int i = 0; i < predicted.Count; i++)
        {
            double err = predicted[i] - observed[i];
            sumAbs += Math.Abs(err);
            sumSq  += err * err;
            sumErr += err;
            if (Math.Abs(err) <= 6.0)  hit6++;
            if (Math.Abs(err) <= 12.0) hit12++;
        }

        double n     = predicted.Count;
        double mae   = sumAbs / n;
        double rmse  = Math.Sqrt(sumSq / n);
        double bias  = sumErr / n;
        double skill = double.IsNaN(dbmBaselineMae) || dbmBaselineMae == 0.0
            ? double.NaN
            : 1.0 - mae / dbmBaselineMae;

        return new ValidationMetrics(mae, rmse, bias, hit6, hit12, (int)n, skill);
    }

    public static ValidationReport BuildReport(
        IReadOnlyList<ObservedTransit> observed,
        IReadOnlyList<ScoreboardEntry> scoreboard,
        IReadOnlyList<EventPrediction> predictions)
    {
        var dbmEntries = scoreboard
            .Where(e => e.Model.Equals("DBM_baseline", StringComparison.OrdinalIgnoreCase)
                     && !double.IsNaN(e.MaePublished))
            .Select(e => e.MaePublished)
            .ToList();
        double dbmMae = dbmEntries.Count > 0 ? dbmEntries.Average() : double.NaN;

        var predMap = predictions.ToDictionary(p => p.CmeId, p => p.PredictedTransitHours);

        var rows     = new List<EventRowReport>(observed.Count);
        var predList = new List<double>();
        var obsList  = new List<double>();

        foreach (var transit in observed)
        {
            var boardEntries = scoreboard
                .Where(e => e.CmeId == transit.CmeId && !double.IsNaN(e.MaePublished))
                .OrderBy(e => e.MaePublished)
                .ToList();

            double bestMae   = boardEntries.Count > 0 ? boardEntries[0].MaePublished : double.NaN;
            string bestModel = boardEntries.Count > 0 ? boardEntries[0].Model : "—";

            double eventDbm = scoreboard
                .Where(e => e.CmeId == transit.CmeId
                         && e.Model.Equals("DBM_baseline", StringComparison.OrdinalIgnoreCase)
                         && !double.IsNaN(e.MaePublished))
                .Select(e => (double?)e.MaePublished)
                .FirstOrDefault() ?? dbmMae;

            var predEntry = predictions.FirstOrDefault(p => p.CmeId == transit.CmeId);
            double pred = predEntry?.PredictedTransitHours ?? double.NaN;
            if (pred == 0.0) pred = double.NaN;

            double err = double.IsNaN(pred) ? double.NaN : pred - transit.TransitHoursObserved;

            double lo = predEntry is not null ? predEntry.LowerBound : double.NaN;
            double hi = predEntry is not null ? predEntry.UpperBound : double.NaN;

            rows.Add(new EventRowReport(
                CmeId:               transit.CmeId,
                EventLabel:          BuildEventLabel(transit),
                ActualHours:         transit.TransitHoursObserved,
                PredictedHours:      pred,
                ErrorHours:          err,
                BestScoreboardMae:   bestMae,
                BestScoreboardModel: bestModel,
                DbmMae:              eventDbm,
                IntervalLower:       lo,
                IntervalUpper:       hi));

            if (!double.IsNaN(pred))
            {
                predList.Add(pred);
                obsList.Add(transit.TransitHoursObserved);
            }
        }

        // Compute conformal interval coverage if intervals are present
        double intervalWidth = double.NaN;
        double coverageRate = double.NaN;
        var rowsWithIntervals = rows.Where(r =>
            !double.IsNaN(r.IntervalLower) && !double.IsNaN(r.IntervalUpper)).ToList();
        if (rowsWithIntervals.Count > 0)
        {
            intervalWidth = rowsWithIntervals.Average(r => r.IntervalUpper - r.IntervalLower);
            int covered = rowsWithIntervals.Count(r =>
                r.ActualHours >= r.IntervalLower && r.ActualHours <= r.IntervalUpper);
            coverageRate = (double)covered / rowsWithIntervals.Count;
        }

        var baseMetrics = ComputeMetrics(predList, obsList, dbmMae);
        var metrics = baseMetrics with
        {
            IntervalWidth90 = intervalWidth,
            CoverageRate90 = coverageRate,
        };
        string table = BuildAsciiTable(rows, metrics, dbmMae);

        return new ValidationReport(metrics, rows, table, DateTime.UtcNow);
    }

    public static string SerializeReport(ValidationReport report) =>
        JsonSerializer.Serialize(report, _jsonOpts);

    // ── helpers ──────────────────────────────────────────────────────────────

    private static string BuildEventLabel(ObservedTransit t)
    {
        var d = t.DepartureUtc;
        return $"{d:yyyy-MM-dd} {t.StormLevel} {t.Notes.Split(';')[0].Trim()}";
    }

    private static string BuildAsciiTable(
        IReadOnlyList<EventRowReport> rows,
        ValidationMetrics m,
        double dbmMae)
    {
        var sb  = new StringBuilder();
        string bar = new('═', 72);

        sb.AppendLine("SolarPipe 2026 Validation Results");
        sb.AppendLine(bar);
        sb.AppendLine(
            $"{"Event",-30} {"Actual(h)",8} {"SP(h)",8} {"Err(h)",8} {"Best",14} {"DBM(h)",7}");
        sb.AppendLine(bar);

        foreach (var r in rows)
        {
            string sp  = double.IsNaN(r.PredictedHours) ? "—" : r.PredictedHours.ToString("F1");
            string err = double.IsNaN(r.ErrorHours)     ? "—" : r.ErrorHours.ToString("+0.0;-0.0");
            string best = double.IsNaN(r.BestScoreboardMae)
                ? "—"
                : $"{r.BestScoreboardMae:F1}h ({r.BestScoreboardModel})";
            string dbm = double.IsNaN(r.DbmMae) ? "—" : r.DbmMae.ToString("F1");

            string label = r.EventLabel.Length > 29
                ? r.EventLabel[..29]
                : r.EventLabel;

            sb.AppendLine(
                $"{label,-30} {r.ActualHours,8:F1} {sp,8} {err,8} {best,14} {dbm,7}");
        }

        sb.AppendLine(bar);

        string maeStr   = double.IsNaN(m.Mae)        ? "—.—" : m.Mae.ToString("F1");
        string skillStr = double.IsNaN(m.SkillVsDbm) ? "—.—" : m.SkillVsDbm.ToString("F2");
        string dbmStr   = double.IsNaN(dbmMae)       ? "—.—" : dbmMae.ToString("F1");

        sb.AppendLine(
            $"SolarPipe MAE: {maeStr} h    Skill vs DBM: {skillStr}    Hit Rate ±12h: {m.HitRate12h}/{m.TotalEvents}");
        sb.Append($"Published DBM MAE: {dbmStr} h");

        if (!double.IsNaN(m.IntervalWidth90))
        {
            sb.AppendLine();
            string widthStr = m.IntervalWidth90.ToString("F1");
            string covStr = double.IsNaN(m.CoverageRate90) ? "—" : (m.CoverageRate90 * 100).ToString("F0");
            sb.Append($"90% Prediction Interval: ±{(m.IntervalWidth90 / 2):F1} h    Coverage: {covStr}%");
        }

        return sb.ToString();
    }

    private static double ParseOptionalDouble(string s)
    {
        s = s.Trim();
        if (string.IsNullOrEmpty(s)) return double.NaN;
        return double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out double v)
            ? v
            : double.NaN;
    }

    private static DateTime ParseUtcIso(string s)
    {
        if (DateTime.TryParse(s.Trim(), CultureInfo.InvariantCulture,
                DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var dt))
            return dt;
        return DateTime.MinValue;
    }
}
