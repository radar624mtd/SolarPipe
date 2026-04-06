using FluentAssertions;
using NSubstitute;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Tests.Unit.Fixtures;
using SolarPipe.Training.Validation;

namespace SolarPipe.Tests.Unit;

[Trait("Category", "Unit")]
public sealed class UncertaintyQuantificationTests
{
    // ── SplitConformalPredictor ──────────────────────────────────────────────────

    // 1. SplitConformalPredictor: 90% coverage — interval contains at least 90% of test points
    [Fact]
    public void SplitConformalPredictor_90Coverage_HoldsOnCalibrationData()
    {
        // Calibrate on residuals from moderate-event domain values
        float[] actuals   = BuildDstArray(200, PhysicsTestFixtures.ModerateEvent.DstExpectedNt);
        float[] predicted = BuildBiasedArray(actuals, biasPerRow: 0.5f);

        var scp = new SplitConformalPredictor();
        scp.Calibrate(actuals, predicted);

        float width = scp.GetIntervalWidth(alpha: 0.1f);

        // Coverage: fraction of calibration points within ±width
        int covered = actuals.Zip(predicted).Count(p => MathF.Abs(p.First - p.Second) <= width);
        double coverage = (double)covered / actuals.Length;

        coverage.Should().BeGreaterOrEqualTo(0.89,
            "split conformal guarantees ≥ 90% coverage at alpha=0.1");
    }

    // 2. SplitConformalPredictor: wider interval at lower alpha (higher coverage)
    [Fact]
    public void SplitConformalPredictor_LowerAlpha_WiderInterval()
    {
        float[] actuals   = BuildDstArray(100, PhysicsTestFixtures.ModerateEvent.DstExpectedNt);
        float[] predicted = BuildBiasedArray(actuals, biasPerRow: 2.0f);

        var scp = new SplitConformalPredictor();
        scp.Calibrate(actuals, predicted);

        float width90 = scp.GetIntervalWidth(alpha: 0.10f);  // 90% coverage
        float width95 = scp.GetIntervalWidth(alpha: 0.05f);  // 95% coverage

        width95.Should().BeGreaterOrEqualTo(width90,
            "95% coverage interval must be at least as wide as 90%");
    }

    // 3. SplitConformalPredictor: throws when not calibrated
    [Fact]
    public void SplitConformalPredictor_NotCalibrated_Throws()
    {
        var scp = new SplitConformalPredictor();
        var act = () => scp.GetIntervalWidth(0.1f);
        act.Should().Throw<InvalidOperationException>()
           .WithMessage("*Calibrate*");
    }

    // ── EnbPiPredictor ───────────────────────────────────────────────────────────

    // 4. EnbPI: interval width narrows after residuals in a new phase shrink
    //    Simulates transition from active → quiet solar wind (residuals drop from 30h to 5h)
    [Fact]
    public void EnbPiPredictor_WindowAdapts_AfterDistributionShift()
    {
        var enbpi = new EnbPiPredictor(windowSize: 50);

        // Phase 1: large residuals (active period, large errors)
        for (int i = 0; i < 50; i++)
            enbpi.UpdateResidual(actual: 48f, predicted: 18f); // 30h residual

        float widthAfterActive = enbpi.GetIntervalWidth(alpha: 0.1f);

        // Phase 2: inject small residuals (quiet period) — fills the window
        for (int i = 0; i < 50; i++)
            enbpi.UpdateResidual(actual: 48f, predicted: 43f); // 5h residual

        float widthAfterQuiet = enbpi.GetIntervalWidth(alpha: 0.1f);

        widthAfterQuiet.Should().BeLessThan(widthAfterActive,
            "EnbPI interval must adapt (narrow) after quiet-period residuals replace active-period residuals");
    }

    // 5. EnbPI: Reset clears window; subsequent query before re-calibration throws
    [Fact]
    public void EnbPiPredictor_Reset_ClearsWindow()
    {
        var enbpi = new EnbPiPredictor(windowSize: 20);
        enbpi.UpdateResidual(48f, 40f);
        enbpi.GetIntervalWidth(0.1f); // should not throw

        enbpi.Reset();

        var act = () => enbpi.GetIntervalWidth(0.1f);
        act.Should().Throw<InvalidOperationException>();
    }

    // ── FeatureImportanceAnalyzer ─────────────────────────────────────────────────

    // 6. FeatureImportanceAnalyzer: signal feature ranks above noise feature
    [Fact]
    public async Task FeatureImportanceAnalyzer_SignalFeature_RanksAboveNoise()
    {
        // Build test data: "speed" strongly correlates with arrival_time; "noise" does not.
        // arrival_time ≈ 2 * speed (deterministic mapping used for test predictor)
        const int rows = 60;
        var schema = new DataSchema(new[]
        {
            new ColumnInfo("speed",        ColumnType.Float, false),
            new ColumnInfo("noise",        ColumnType.Float, false),
            new ColumnInfo("arrival_time", ColumnType.Float, false),
        });

        // Speed values from 200 to 2600 km/s in equal steps — domain-valid range
        float[] speed  = Enumerable.Range(0, rows).Select(i => 200f + i * (2400f / rows)).ToArray();
        float[] noise  = Enumerable.Range(0, rows).Select(i => (float)(i % 7) * 3f).ToArray();
        float[] target = speed.Select(s => s / 50f).ToArray(); // arrival_time proportional to speed

        var data = new InMemoryDataFrame(schema, new[] { speed, noise, target });

        // Model predicts arrival_time = speed/50 (uses only signal feature)
        var model = Substitute.For<ITrainedModel>();
        model.ModelId.Returns("mock");
        model.StageName.Returns("test_stage");
        model.Metrics.Returns(new ModelMetrics(0, 0, 0));
        model.PredictAsync(Arg.Any<IDataFrame>(), Arg.Any<CancellationToken>())
             .Returns(ci =>
             {
                 var f = (IDataFrame)ci[0];
                 float[] s = f.GetColumn("speed");
                 float[] vals = s.Select(v => v / 50f).ToArray();
                 return Task.FromResult(new PredictionResult(vals, null, null, "mock", DateTime.UtcNow));
             });

        var analyzer = new FeatureImportanceAnalyzer(permutations: 3, seed: 42);
        var results = await analyzer.ComputeAsync(
            model, data, "arrival_time",
            new[] { "speed", "noise" },
            CancellationToken.None);

        results.Should().HaveCount(2);
        var speedResult = results.First(r => r.FeatureName == "speed");
        var noiseResult = results.First(r => r.FeatureName == "noise");

        speedResult.MeanImportance.Should().BeGreaterThan(noiseResult.MeanImportance,
            "speed has a deterministic signal; noise column does not affect predictions");
    }

    // ── helpers ──────────────────────────────────────────────────────────────────

    // Build Dst array with small deterministic offsets from a base value
    private static float[] BuildDstArray(int n, float baseValue)
        => Enumerable.Range(0, n).Select(i => baseValue + (i % 10) * 0.5f).ToArray();

    // Add a deterministic per-row bias to simulate model predictions
    private static float[] BuildBiasedArray(float[] actuals, float biasPerRow)
        => actuals.Select((a, i) => a + (i % 5) * biasPerRow - biasPerRow).ToArray();
}
