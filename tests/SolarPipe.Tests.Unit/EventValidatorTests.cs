using FluentAssertions;
using SolarPipe.Training.Validation;

namespace SolarPipe.Tests.Unit;

[Trait("Category", "Unit")]
public sealed class EventValidatorTests : IDisposable
{
    // Minimal valid CSVs written to temp files for each test
    private readonly string _observedCsv;
    private readonly string _scoreboardCsv;

    private const string ObservedHeader =
        "cme_id,departure_utc,arrival_utc,transit_hours_observed," +
        "cme_speed_kms,bz_gsm_min_nt,arrival_speed_kms,storm_level,notes";

    private const string ScoreboardHeader =
        "cme_id,model,predicted_arrival_utc,actual_arrival_utc,error_hours,mae_published";

    public EventValidatorTests()
    {
        _observedCsv   = Path.GetTempFileName();
        _scoreboardCsv = Path.GetTempFileName();
    }

    public void Dispose()
    {
        if (File.Exists(_observedCsv))   File.Delete(_observedCsv);
        if (File.Exists(_scoreboardCsv)) File.Delete(_scoreboardCsv);
    }

    // ── 1. MAE computed correctly ────────────────────────────────────────────

    [Fact]
    public void ComputeMetrics_Mae_IsAbsoluteMeanError()
    {
        var predicted = new double[] { 30.0, 50.0 };
        var observed  = new double[] { 25.0, 60.0 };   // errors: +5, -10 → MAE = 7.5

        var m = EventValidator.ComputeMetrics(predicted, observed, dbmBaselineMae: 14.8);

        m.Mae.Should().BeApproximately(7.5, 1e-9);
    }

    // ── 2. RMSE computed correctly ───────────────────────────────────────────

    [Fact]
    public void ComputeMetrics_Rmse_IsRootMeanSquaredError()
    {
        var predicted = new double[] { 30.0, 50.0 };
        var observed  = new double[] { 25.0, 60.0 };   // sq errors: 25, 100 → RMSE = √62.5

        var m = EventValidator.ComputeMetrics(predicted, observed, dbmBaselineMae: 14.8);

        m.Rmse.Should().BeApproximately(Math.Sqrt(62.5), 1e-9);
    }

    // ── 3. Skill score vs DBM baseline ──────────────────────────────────────

    [Fact]
    public void ComputeMetrics_SkillScore_IsOneMinusMaeOverDbm()
    {
        // MAE = 7.5, DBM = 14.8  →  skill = 1 - 7.5/14.8
        var predicted = new double[] { 30.0, 50.0 };
        var observed  = new double[] { 25.0, 60.0 };

        var m = EventValidator.ComputeMetrics(predicted, observed, dbmBaselineMae: 14.8);

        double expected = 1.0 - 7.5 / 14.8;
        m.SkillVsDbm.Should().BeApproximately(expected, 1e-9);
    }

    // ── 4. Skill score undefined when DBM = 0 ───────────────────────────────

    [Fact]
    public void ComputeMetrics_SkillScore_IsNaN_WhenDbmIsZero()
    {
        var m = EventValidator.ComputeMetrics(
            new double[] { 30.0 }, new double[] { 25.0 }, dbmBaselineMae: 0.0);

        double.IsNaN(m.SkillVsDbm).Should().BeTrue();
    }

    // ── 5. Hit rate ±6h / ±12h counts correct ───────────────────────────────

    [Fact]
    public void ComputeMetrics_HitRates_CountWithinWindowCorrectly()
    {
        // errors: 4, 8, 14  →  ±6h: 1, ±12h: 2
        var predicted = new double[] { 29.0, 33.0, 39.0 };
        var observed  = new double[] { 25.0, 25.0, 25.0 };

        var m = EventValidator.ComputeMetrics(predicted, observed, dbmBaselineMae: 14.8);

        m.HitRate6h.Should().Be(1);
        m.HitRate12h.Should().Be(2);
        m.TotalEvents.Should().Be(3);
    }

    // ── 6. CSV parsing — observed transits loaded correctly ──────────────────

    [Fact]
    public async Task LoadObservedTransitsAsync_ParsesRowsCorrectly()
    {
        await File.WriteAllTextAsync(_observedCsv,
            ObservedHeader + "\n" +
            "2026-01-18T18:09:00-CME-001,2026-01-18T18:09Z,2026-01-19T18:55Z," +
            "25.5,1473,-21.8,1100,G4,X1.9 flare AR14341\n");

        var rows = await EventValidator.LoadObservedTransitsAsync(_observedCsv, CancellationToken.None);

        rows.Should().HaveCount(1);
        rows[0].CmeId.Should().Be("2026-01-18T18:09:00-CME-001");
        rows[0].TransitHoursObserved.Should().BeApproximately(25.5, 1e-6);
        rows[0].BzGsmMinNt.Should().BeApproximately(-21.8, 1e-6);
        rows[0].StormLevel.Should().Be("G4");
    }

    // ── 7. Empty Bz field maps to NaN (RULE-120) ────────────────────────────

    [Fact]
    public async Task LoadObservedTransitsAsync_EmptyBz_MapsToNaN()
    {
        await File.WriteAllTextAsync(_observedCsv,
            ObservedHeader + "\n" +
            "2026-03-18T09:23:00-CME-001,2026-03-18T09:23Z,2026-03-20T20:17Z," +
            "59.9,731,,567,G3,M2.7 flare AR4392\n");   // bz_gsm_min_nt left empty

        var rows = await EventValidator.LoadObservedTransitsAsync(_observedCsv, CancellationToken.None);

        double.IsNaN(rows[0].BzGsmMinNt).Should().BeTrue(
            "empty Bz field must become NaN per RULE-120, not 0 or a sentinel");
    }

    // ── 8. BuildReport with predictions shows predicted values ─────────────

    [Fact]
    public void BuildReport_WithPredictions_PopulatesSpColumn()
    {
        var transit = new ObservedTransit(
            "CME-001", DateTime.UtcNow.AddDays(-2), DateTime.UtcNow.AddDays(-1),
            25.5, 1473, -21.8, 1100, "G4", "test");

        var prediction = new EventPrediction("CME-001", 28.3);

        var report = EventValidator.BuildReport(
            new[] { transit },
            Array.Empty<ScoreboardEntry>(),
            new[] { prediction });

        report.EventRows[0].PredictedHours.Should().BeApproximately(28.3, 0.01,
            "Prediction value must flow through to report");
        report.EventRows[0].ErrorHours.Should().BeApproximately(2.8, 0.01);
        report.Metrics.TotalEvents.Should().Be(1);
        double.IsNaN(report.Metrics.Mae).Should().BeFalse();
    }

    // ── 9. JSON report schema has required top-level keys ───────────────────

    [Fact]
    public void SerializeReport_JsonContainsRequiredKeys()
    {
        var transit = new ObservedTransit(
            "CME-001", DateTime.UtcNow.AddDays(-2), DateTime.UtcNow.AddDays(-1),
            25.5, 1473, -21.8, 1100, "G4", "test");

        var report = EventValidator.BuildReport(
            new[] { transit },
            Array.Empty<ScoreboardEntry>(),
            Array.Empty<EventPrediction>());

        var json = EventValidator.SerializeReport(report);

        json.Should().Contain("\"Metrics\"");
        json.Should().Contain("\"EventRows\"");
        json.Should().Contain("\"AsciiTable\"");
        json.Should().Contain("\"GeneratedAt\"");
    }

    // ── 10. EventPrediction with intervals propagates to report ─────────────

    [Fact]
    public void BuildReport_WithIntervals_PopulatesIntervalColumns()
    {
        var transit = new ObservedTransit(
            "CME-001", DateTime.UtcNow.AddDays(-2), DateTime.UtcNow.AddDays(-1),
            25.5, 1473, -21.8, 1100, "G4", "test");

        var prediction = new EventPrediction("CME-001", 28.3, LowerBound: 20.0, UpperBound: 36.0);

        var report = EventValidator.BuildReport(
            new[] { transit },
            Array.Empty<ScoreboardEntry>(),
            new[] { prediction });

        report.EventRows[0].IntervalLower.Should().BeApproximately(20.0, 0.01);
        report.EventRows[0].IntervalUpper.Should().BeApproximately(36.0, 0.01);
    }

    // ── 11. Coverage rate computed from interval bounds ──────────────────────

    [Fact]
    public void BuildReport_CoverageRate_ComputedFromIntervals()
    {
        var transits = new[]
        {
            new ObservedTransit("CME-001", DateTime.UtcNow.AddDays(-4), DateTime.UtcNow.AddDays(-3),
                25.5, 1473, -21.8, 1100, "G4", "test"),
            new ObservedTransit("CME-002", DateTime.UtcNow.AddDays(-2), DateTime.UtcNow.AddDays(-1),
                50.0, 731, -28.0, 567, "G3", "test"),
        };

        var predictions = new[]
        {
            // Actual=25.5, interval [20, 30] → covered
            new EventPrediction("CME-001", 26.0, LowerBound: 20.0, UpperBound: 30.0),
            // Actual=50.0, interval [40, 45] → NOT covered
            new EventPrediction("CME-002", 42.0, LowerBound: 40.0, UpperBound: 45.0),
        };

        var report = EventValidator.BuildReport(transits, Array.Empty<ScoreboardEntry>(), predictions);

        report.Metrics.CoverageRate90.Should().BeApproximately(0.5, 1e-9,
            "1 of 2 events falls within its interval");
        double.IsNaN(report.Metrics.IntervalWidth90).Should().BeFalse();
    }

    // ── 12. IntervalWidth90 is NaN when no intervals present ────────────────

    [Fact]
    public void BuildReport_NoIntervals_IntervalWidthIsNaN()
    {
        var transit = new ObservedTransit(
            "CME-001", DateTime.UtcNow.AddDays(-2), DateTime.UtcNow.AddDays(-1),
            25.5, 1473, -21.8, 1100, "G4", "test");

        var prediction = new EventPrediction("CME-001", 28.3); // no interval bounds

        var report = EventValidator.BuildReport(
            new[] { transit },
            Array.Empty<ScoreboardEntry>(),
            new[] { prediction });

        double.IsNaN(report.Metrics.IntervalWidth90).Should().BeTrue(
            "no intervals → IntervalWidth90 should be NaN");
        double.IsNaN(report.Metrics.CoverageRate90).Should().BeTrue();
    }

    // ── 13. ASCII table shows interval line when intervals present ──────────

    [Fact]
    public void BuildReport_WithIntervals_AsciiTableContainsIntervalLine()
    {
        var transit = new ObservedTransit(
            "CME-001", DateTime.UtcNow.AddDays(-2), DateTime.UtcNow.AddDays(-1),
            25.5, 1473, -21.8, 1100, "G4", "test");

        var prediction = new EventPrediction("CME-001", 28.3, LowerBound: 20.0, UpperBound: 36.0);

        var report = EventValidator.BuildReport(
            new[] { transit },
            Array.Empty<ScoreboardEntry>(),
            new[] { prediction });

        report.AsciiTable.Should().Contain("90% Prediction Interval");
        report.AsciiTable.Should().Contain("Coverage:");
    }
}
