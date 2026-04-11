using FluentAssertions;
using SolarPipe.Training.Evaluation;

namespace SolarPipe.Tests.Unit.Phase7;

[Trait("Category", "Unit")]
public sealed class ComprehensiveMetricsEvaluatorTests
{
    private readonly ComprehensiveMetricsEvaluator _sut = new();

    // Helper: perfect predictions
    private static (float[] obs, float[] pred) Perfect(int n, float value = 48f)
    {
        var obs  = Enumerable.Repeat(value, n).Select(v => v).ToArray();
        var pred = obs.ToArray();
        return (obs, pred);
    }

    // ── MAE ──────────────────────────────────────────────────────────────────

    [Fact]
    public void EvaluateFold_PerfectPredictions_ZeroMae()
    {
        var (obs, pred) = Perfect(10);
        var result = _sut.EvaluateFold(0, obs, pred, 12.0);
        result.Mae.Should().BeApproximately(0.0, 1e-6);
    }

    [Fact]
    public void EvaluateFold_ConstantOffset_CorrectMae()
    {
        var obs  = new float[] { 10f, 20f, 30f };
        var pred = new float[] { 12f, 22f, 32f };
        var result = _sut.EvaluateFold(0, obs, pred, 12.0);
        result.Mae.Should().BeApproximately(2.0, 1e-6);
    }

    // ── RMSE ─────────────────────────────────────────────────────────────────

    [Fact]
    public void EvaluateFold_PerfectPredictions_ZeroRmse()
    {
        var (obs, pred) = Perfect(10);
        var result = _sut.EvaluateFold(0, obs, pred, 12.0);
        result.Rmse.Should().BeApproximately(0.0, 1e-6);
    }

    [Fact]
    public void EvaluateFold_KnownErrors_CorrectRmse()
    {
        // errors are [3, -3] → rmse = 3.0
        var obs  = new float[] { 10f, 20f };
        var pred = new float[] { 13f, 17f };
        var result = _sut.EvaluateFold(0, obs, pred, 12.0);
        result.Rmse.Should().BeApproximately(3.0, 1e-6);
    }

    // ── R² ───────────────────────────────────────────────────────────────────

    [Fact]
    public void EvaluateFold_PerfectPredictions_R2IsOne()
    {
        // Use varied values so ssTot > 0, otherwise R² is undefined when variance = 0
        var obs  = new float[] { 40f, 50f, 60f, 70f, 80f };
        var pred = obs.ToArray();  // perfect
        var result = _sut.EvaluateFold(0, obs, pred, 12.0);
        result.R2.Should().BeApproximately(1.0, 1e-6);
    }

    // ── Bias ─────────────────────────────────────────────────────────────────

    [Fact]
    public void EvaluateFold_PredictionsHigherThanObserved_PositiveBias()
    {
        var obs  = new float[] { 48f, 48f, 48f };
        var pred = new float[] { 50f, 50f, 50f };
        var result = _sut.EvaluateFold(0, obs, pred, 12.0);
        result.Bias.Should().BeApproximately(2.0, 1e-6);
    }

    // ── Skill vs DBM ─────────────────────────────────────────────────────────

    [Fact]
    public void EvaluateFold_ZeroMae_SkillIsOne()
    {
        var (obs, pred) = Perfect(10);
        var result = _sut.EvaluateFold(0, obs, pred, dbmBaselineMae: 5.0);
        result.SkillVsDbm.Should().BeApproximately(1.0, 1e-6);
    }

    [Fact]
    public void EvaluateFold_MaeEqualsDbm_SkillIsZero()
    {
        var obs  = new float[] { 10f, 20f };
        var pred = new float[] { 16f, 14f }; // mae = 6
        var result = _sut.EvaluateFold(0, obs, pred, dbmBaselineMae: 6.0);
        result.SkillVsDbm.Should().BeApproximately(0.0, 1e-6);
    }

    // ── Hit rates ────────────────────────────────────────────────────────────

    [Fact]
    public void EvaluateFold_AllWithin6h_HitRate6hIsOne()
    {
        var obs  = new float[] { 48f, 48f, 48f };
        var pred = new float[] { 50f, 44f, 48f }; // errors: 2, 4, 0 — all ≤6
        var result = _sut.EvaluateFold(0, obs, pred, 12.0);
        result.HitRate6h.Should().BeApproximately(1.0, 1e-6);
    }

    [Fact]
    public void EvaluateFold_NoneWithin6h_HitRate6hIsZero()
    {
        var obs  = new float[] { 48f, 48f };
        var pred = new float[] { 62f, 62f }; // errors: 14 — none ≤6
        var result = _sut.EvaluateFold(0, obs, pred, 12.0);
        result.HitRate6h.Should().BeApproximately(0.0, 1e-6);
    }

    // ── Pinball loss ─────────────────────────────────────────────────────────

    [Fact]
    public void EvaluateFold_PerfectPredictions_ZeroPinball()
    {
        var (obs, pred) = Perfect(10);
        var result = _sut.EvaluateFold(0, obs, pred, 12.0);
        result.PinballLoss10.Should().BeApproximately(0.0, 1e-6);
    }

    // ── Coverage ─────────────────────────────────────────────────────────────

    [Fact]
    public void EvaluateFold_NullBounds_CoverageIsNaN()
    {
        var (obs, pred) = Perfect(5);
        var result = _sut.EvaluateFold(0, obs, pred, 12.0);
        result.CoverageRate90.Should().Be(double.NaN);
    }

    [Fact]
    public void EvaluateFold_AllWithinBounds_CoverageIsOne()
    {
        var obs   = new float[] { 48f, 50f, 52f };
        var pred  = new float[] { 48f, 50f, 52f };
        var lower = new float[] { 40f, 42f, 44f };
        var upper = new float[] { 56f, 58f, 60f };
        var result = _sut.EvaluateFold(0, obs, pred, 12.0, lower, upper);
        result.CoverageRate90.Should().BeApproximately(1.0, 1e-6);
    }

    // ── Kendall τ ────────────────────────────────────────────────────────────

    [Fact]
    public void EvaluateFold_PerfectRanking_KendallIsOne()
    {
        var obs  = new float[] { 10f, 20f, 30f, 40f };
        var pred = new float[] { 11f, 21f, 31f, 41f };  // same order
        var result = _sut.EvaluateFold(0, obs, pred, 12.0);
        result.KendallTau.Should().BeApproximately(1.0, 1e-6);
    }

    [Fact]
    public void EvaluateFold_ReversedRanking_KendallIsNegativeOne()
    {
        var obs  = new float[] { 10f, 20f, 30f, 40f };
        var pred = new float[] { 40f, 30f, 20f, 10f };  // perfect reversal
        var result = _sut.EvaluateFold(0, obs, pred, 12.0);
        result.KendallTau.Should().BeApproximately(-1.0, 1e-6);
    }

    // ── Aggregation ──────────────────────────────────────────────────────────

    [Fact]
    public void AggregateFolds_IdenticalFolds_ZeroStd()
    {
        var obs  = new float[] { 10f, 20f };
        var pred = new float[] { 12f, 22f };  // mae = 2 for each fold

        var folds = Enumerable.Range(0, 3)
            .Select(i => _sut.EvaluateFold(i, obs, pred, 12.0))
            .ToList();

        var agg = _sut.AggregateFolds(folds);
        agg.MaeMean.Should().BeApproximately(2.0, 1e-6);
        agg.MaeStd.Should().BeApproximately(0.0, 1e-6);
    }

    [Fact]
    public void AggregateFolds_EmptyList_Throws()
    {
        var act = () => _sut.AggregateFolds(new List<FoldMetrics>());
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void EvaluateFold_EmptyArrays_Throws()
    {
        var act = () => _sut.EvaluateFold(0, Array.Empty<float>(), Array.Empty<float>(), 12.0);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void EvaluateFold_MismatchedLengths_Throws()
    {
        var act = () => _sut.EvaluateFold(0, new float[] { 1f }, new float[] { 1f, 2f }, 12.0);
        act.Should().Throw<ArgumentException>();
    }
}
