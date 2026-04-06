using FluentAssertions;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Data.Transforms;
using SolarPipe.Tests.Unit.Fixtures;

namespace SolarPipe.Tests.Unit;

[Trait("Category", "Unit")]
public class TransformEngineTests
{
    // ─── helpers ────────────────────────────────────────────────────────────────

    private static InMemoryDataFrame MakeSingleCol(string name, float[] values)
    {
        var schema = new DataSchema([new ColumnInfo(name, ColumnType.Float, false)]);
        return new InMemoryDataFrame(schema, [values]);
    }

    private static InMemoryDataFrame MakeMultiCol(string[] names, float[][] data)
    {
        var cols = names.Select(n => new ColumnInfo(n, ColumnType.Float, false)).ToList();
        return new InMemoryDataFrame(new DataSchema(cols), data);
    }

    // ─── NormalizeTransform ──────────────────────────────────────────────────────

    [Fact]
    public void Normalize_ScalesToZeroOne()
    {
        using var frame = MakeSingleCol("speed", [0f, 500f, 1000f]);
        var t = new NormalizeTransform("speed");
        using var result = t.Apply(frame);

        var col = result.GetColumn("speed_norm");
        col[0].Should().BeApproximately(0f, 1e-5f);
        col[1].Should().BeApproximately(0.5f, 1e-5f);
        col[2].Should().BeApproximately(1f, 1e-5f);
    }

    [Fact]
    public void Normalize_NanInput_ProducesNanOutput()
    {
        using var frame = MakeSingleCol("speed", [float.NaN, 500f, 1000f]);
        var t = new NormalizeTransform("speed");
        using var result = t.Apply(frame);

        float.IsNaN(result.GetColumn("speed_norm")[0]).Should().BeTrue();
    }

    [Fact]
    public void Normalize_FixedRange_UsesProvidedRange()
    {
        // Range [0, 2000], value 500 → 0.25
        using var frame = MakeSingleCol("speed", [500f]);
        var t = new NormalizeTransform("speed", min: 0f, max: 2000f);
        using var result = t.Apply(frame);

        result.GetColumn("speed_norm")[0].Should().BeApproximately(0.25f, 1e-5f);
    }

    // ─── StandardizeTransform ────────────────────────────────────────────────────

    [Fact]
    public void Standardize_ZeroMeanUnitVariance()
    {
        // mean=2, std=sqrt(2/3) for [1,2,3]
        using var frame = MakeSingleCol("v", [1f, 2f, 3f]);
        var t = new StandardizeTransform("v");
        using var result = t.Apply(frame);

        var col = result.GetColumn("v_std");
        // sum of standardized values must be ~0 (mean=0 post-standardization)
        double sum = col.Sum(x => (double)x);
        sum.Should().BeApproximately(0.0, 1e-4);
        // sum of squares / n should be ~1 (unit variance)
        double var_ = col.Sum(x => (double)x * x) / col.Length;
        var_.Should().BeApproximately(1.0, 1e-4);
    }

    [Fact]
    public void Standardize_NanInput_ProducesNanOutput()
    {
        using var frame = MakeSingleCol("v", [float.NaN, 1f, 2f]);
        var t = new StandardizeTransform("v");
        using var result = t.Apply(frame);

        float.IsNaN(result.GetColumn("v_std")[0]).Should().BeTrue();
    }

    // ─── LogScaleTransform ───────────────────────────────────────────────────────

    [Fact]
    public void LogScale_PositiveValues_Correct()
    {
        // log(100 + 1) ≈ 4.6151
        using var frame = MakeSingleCol("density", [100f]);
        var t = new LogScaleTransform("density");
        using var result = t.Apply(frame);

        result.GetColumn("density_log")[0]
            .Should().BeApproximately(MathF.Log(101f), 1e-5f);
    }

    [Fact]
    public void LogScale_NanInput_ProducesNan()
    {
        using var frame = MakeSingleCol("density", [float.NaN]);
        var t = new LogScaleTransform("density");
        using var result = t.Apply(frame);

        float.IsNaN(result.GetColumn("density_log")[0]).Should().BeTrue();
    }

    // ─── LagTransform ────────────────────────────────────────────────────────────

    [Fact]
    public void Lag_ShiftsColumnByN()
    {
        using var frame = MakeSingleCol("bz", [1f, 2f, 3f, 4f]);
        var t = new LagTransform("bz", lagSteps: 2);
        using var result = t.Apply(frame);

        var col = result.GetColumn("bz_lag2");
        float.IsNaN(col[0]).Should().BeTrue("row 0 has no lag-2 predecessor");
        float.IsNaN(col[1]).Should().BeTrue("row 1 has no lag-2 predecessor");
        col[2].Should().BeApproximately(1f, 1e-5f);
        col[3].Should().BeApproximately(2f, 1e-5f);
    }

    [Fact]
    public void Lag_InvalidLagSteps_Throws()
    {
        var act = () => new LagTransform("bz", lagSteps: 0);
        act.Should().Throw<ArgumentOutOfRangeException>().WithMessage("*lagSteps*");
    }

    // ─── WindowStatsTransform ────────────────────────────────────────────────────

    [Fact]
    public void WindowStats_Mean_CorrectRollingMean()
    {
        using var frame = MakeSingleCol("bz", [2f, 4f, 6f, 8f]);
        var t = new WindowStatsTransform("bz", windowSize: 3, WindowStatsTransform.StatOp.Mean);
        using var result = t.Apply(frame);

        var col = result.GetColumn("bz_win3_mean");
        col[0].Should().BeApproximately(2f, 1e-4f);        // window [2]
        col[1].Should().BeApproximately(3f, 1e-4f);        // window [2,4]
        col[2].Should().BeApproximately(4f, 1e-4f);        // window [2,4,6]
        col[3].Should().BeApproximately(6f, 1e-4f);        // window [4,6,8]
    }

    [Fact]
    public void WindowStats_NanInWindow_ExcludedFromCalc()
    {
        using var frame = MakeSingleCol("bz", [2f, float.NaN, 6f]);
        var t = new WindowStatsTransform("bz", windowSize: 3, WindowStatsTransform.StatOp.Mean);
        using var result = t.Apply(frame);

        var col = result.GetColumn("bz_win3_mean");
        // row 2: window [2, NaN, 6] → NaN excluded → mean(2,6) = 4
        col[2].Should().BeApproximately(4f, 1e-4f);
    }

    // ─── TransformEngine chaining ─────────────────────────────────────────────────

    [Fact]
    public void TransformEngine_ChainMultipleTransforms_AllColumnsPresent()
    {
        using var frame = MakeSingleCol("speed", [400f, 800f, 1200f]);
        var engine = new TransformEngine()
            .Add(new NormalizeTransform("speed"))
            .Add(new StandardizeTransform("speed"))
            .Add(new LagTransform("speed", lagSteps: 1));

        using var result = engine.Apply(frame);

        result.Schema.HasColumn("speed_norm").Should().BeTrue();
        result.Schema.HasColumn("speed_std").Should().BeTrue();
        result.Schema.HasColumn("speed_lag1").Should().BeTrue();
    }

    // ─── CouplingFunctions ────────────────────────────────────────────────────────

    [Fact]
    public void VBsCoupling_SouthwardBz_PositiveCoupling()
    {
        // v=450 km/s, Bz=-10 nT → VBs = 450 * 10 * 1e-3 = 4.5 mV/m
        using var frame = MakeMultiCol(
            ["speed", "bz_gsm"],
            [[PhysicsTestFixtures.NewellCouplingValues.SolarWindSpeedKmPerSec],
             [-PhysicsTestFixtures.NewellCouplingValues.BtNt]]);

        var t = new VBsCouplingTransform("speed", "bz_gsm");
        using var result = t.Apply(frame);

        float expected = 450f * 10f * 1e-3f; // 4.5 mV/m
        result.GetColumn("vbs_coupling")[0].Should().BeApproximately(expected, 1e-3f);
    }

    [Fact]
    public void VBsCoupling_NorthwardBz_ZeroCoupling()
    {
        // Northward Bz → Bs = 0 → no coupling
        using var frame = MakeMultiCol(
            ["speed", "bz_gsm"],
            [[450f], [5f]]); // positive Bz = northward

        var t = new VBsCouplingTransform("speed", "bz_gsm");
        using var result = t.Apply(frame);

        result.GetColumn("vbs_coupling")[0].Should().BeApproximately(0f, 1e-5f);
    }

    [Fact]
    public void NewellCoupling_PurelySouthward_MaximisesCoupling()
    {
        // By=0, Bz=-10 → θ=π → sin(π/2)=1 → Φ = v^(4/3) * Bt^(2/3) * 1
        float v = PhysicsTestFixtures.NewellCouplingValues.SolarWindSpeedKmPerSec;
        float bt = PhysicsTestFixtures.NewellCouplingValues.BtNt;

        using var frame = MakeMultiCol(
            ["speed", "by_gsm", "bz_gsm"],
            [[v], [0f], [-bt]]);

        var t = new NewellCouplingTransform("speed", "by_gsm", "bz_gsm");
        using var result = t.Apply(frame);

        double expected = Math.Pow(v, 4.0 / 3) * Math.Pow(bt, 2.0 / 3) * 1.0; // sin^(8/3)(π/2)=1
        result.GetColumn("newell_coupling")[0]
            .Should().BeApproximately((float)expected, (float)(expected * 0.001));
    }

    [Fact]
    public void BorovskyCoupling_NominalConditions_Finite()
    {
        using var frame = MakeMultiCol(
            ["speed", "by_gsm", "bz_gsm", "density"],
            [[PhysicsTestFixtures.NominalSolarWind.SpeedKmPerSec],
             [0f],
             [PhysicsTestFixtures.NominalSolarWind.BzGsmNt],
             [PhysicsTestFixtures.NominalSolarWind.DensityPerCm3]]);

        var t = new BorovskyCouplingTransform("speed", "by_gsm", "bz_gsm", "density");
        using var result = t.Apply(frame);

        float val = result.GetColumn("borovsky_coupling")[0];
        float.IsNaN(val).Should().BeFalse();
        val.Should().BeGreaterThan(0f);
    }
}
