using FluentAssertions;
using SolarPipe.Training.Evaluation;

namespace SolarPipe.Tests.Unit.Phase7;

[Trait("Category", "Unit")]
public sealed class NnlsEnsembleOptimizerTests
{
    private readonly NnlsEnsembleOptimizer _sut = new();

    // ── Two-member ensemble ───────────────────────────────────────────────────

    [Fact]
    public void Optimize_TwoMembers_WeightsSumToOne()
    {
        var obs = new float[] { 48f, 60f, 36f, 72f, 55f };
        var m1  = new float[] { 46f, 58f, 34f, 70f, 53f };
        var m2  = new float[] { 50f, 62f, 38f, 74f, 57f };

        var weights = _sut.Optimize(
            new Dictionary<string, float[]> { ["m1"] = m1, ["m2"] = m2 },
            obs);

        float total = weights.Values.Sum();
        total.Should().BeApproximately(1.0f, 1e-4f);
    }

    [Fact]
    public void Optimize_TwoMembers_AllWeightsNonNegative()
    {
        var obs = new float[] { 48f, 60f, 36f, 72f, 55f };
        var m1  = new float[] { 46f, 58f, 34f, 70f, 53f };
        var m2  = new float[] { 50f, 62f, 38f, 74f, 57f };

        var weights = _sut.Optimize(
            new Dictionary<string, float[]> { ["m1"] = m1, ["m2"] = m2 },
            obs);

        weights.Values.Should().AllSatisfy(w => w.Should().BeGreaterThanOrEqualTo(0f));
    }

    // ── Three-member ensemble ─────────────────────────────────────────────────

    [Fact]
    public void Optimize_ThreeMembers_WeightsSumToOne()
    {
        var obs = new float[] { 50f, 40f, 60f, 55f, 45f };
        var m1  = new float[] { 48f, 38f, 58f, 53f, 43f };
        var m2  = new float[] { 52f, 42f, 62f, 57f, 47f };
        var m3  = new float[] { 50f, 40f, 60f, 55f, 45f };

        var weights = _sut.Optimize(
            new Dictionary<string, float[]> { ["m1"] = m1, ["m2"] = m2, ["m3"] = m3 },
            obs);

        float total = weights.Values.Sum();
        total.Should().BeApproximately(1.0f, 1e-4f);
    }

    // ── Single member (degenerate) ────────────────────────────────────────────

    [Fact]
    public void Optimize_SingleMember_WeightIsOne()
    {
        var obs = new float[] { 48f, 50f, 52f };
        var m1  = new float[] { 46f, 49f, 51f };

        var weights = _sut.Optimize(
            new Dictionary<string, float[]> { ["only"] = m1 },
            obs);

        weights["only"].Should().BeApproximately(1.0f, 1e-6f);
    }

    // ── Degenerate: one perfect member ───────────────────────────────────────

    [Fact]
    public void Optimize_OnePerfectMember_GetsHigherWeight()
    {
        var obs    = new float[] { 48f, 60f, 36f, 55f };
        var good   = obs.ToArray();                              // perfect
        var bad    = obs.Select(v => v + 20f).ToArray();        // +20h off

        var weights = _sut.Optimize(
            new Dictionary<string, float[]> { ["good"] = good, ["bad"] = bad },
            obs);

        weights["good"].Should().BeGreaterThan(weights["bad"]);
    }

    // ── Input validation ──────────────────────────────────────────────────────

    [Fact]
    public void Optimize_EmptyMembers_Throws()
    {
        var act = () => _sut.Optimize(
            new Dictionary<string, float[]>(),
            new float[] { 1f });
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Optimize_LengthMismatch_Throws()
    {
        var act = () => _sut.Optimize(
            new Dictionary<string, float[]> { ["m1"] = new float[] { 1f, 2f } },
            new float[] { 1f });
        act.Should().Throw<ArgumentException>();
    }

    // ── Constraint invariants via Optimize ───────────────────────────────────

    [Fact]
    public void Optimize_UniformPredictions_WeightsStillSumToOne()
    {
        // All members predict same values → optimizer should still respect simplex
        var obs = new float[] { 48f, 60f, 36f };
        var pred = new float[] { 50f, 62f, 38f };

        var weights = _sut.Optimize(
            new Dictionary<string, float[]>
            {
                ["m1"] = pred,
                ["m2"] = pred.ToArray(),
            },
            obs);

        weights.Values.Sum().Should().BeApproximately(1.0f, 1e-4f);
        weights.Values.Should().AllSatisfy(w => w.Should().BeGreaterThanOrEqualTo(0f));
    }

    [Fact]
    public void Optimize_ConvergesWithManyMembers_SumToOne()
    {
        var obs = new float[] { 50f, 40f, 60f, 55f, 45f };
        var members = Enumerable.Range(0, 5)
            .ToDictionary(
                i => $"m{i}",
                i => obs.Select(v => v + i).ToArray());

        var weights = _sut.Optimize(members, obs);

        weights.Values.Sum().Should().BeApproximately(1.0f, 1e-3f);
        weights.Values.Should().AllSatisfy(w => w.Should().BeGreaterThanOrEqualTo(0f));
    }
}
