using FluentAssertions;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Tests.Unit.Fixtures;
using SolarPipe.Training.Adapters;
using SolarPipe.Training.Physics;

namespace SolarPipe.Tests.Unit;

[Trait("Category", "Unit")]
public class BurtonOdeTests
{
    private readonly PhysicsAdapter _adapter = new();

    private static StageConfig MakeBurtonConfig(IReadOnlyDictionary<string, object>? hp = null) =>
        new StageConfig("burton_stage", "Physics", "BurtonOde", "src",
            ["bz_gsm", "v_km_s"], "dst_min", hp);

    private static InMemoryDataFrame MakeSolarWindFrame(float[] bzGsm, float[] vKmS)
    {
        var schema = new DataSchema([
            new ColumnInfo("bz_gsm", ColumnType.Float, false),
            new ColumnInfo("v_km_s", ColumnType.Float, false)
        ]);
        return new InMemoryDataFrame(schema, [bzGsm, vKmS]);
    }

    [Fact]
    public void PhysicsAdapter_SupportedModels_ContainsBurtonOde()
    {
        _adapter.SupportedModels.Should().Contain("BurtonOde");
    }

    [Fact]
    public void RunOdeTimeSeries_SustainedSouthwardBz_DrivesDstMinNegative()
    {
        // Halloween storm: v=2000 km/s, Bz=-35 nT GSM → VBs = 2000 * 35 * 1e-3 = 70 mV/m
        // Sustained for 24 hours should produce Dst well below -100 nT
        double v = PhysicsTestFixtures.HalloweenStorm2003.AmbientSolarWindKmPerSec;
        double bz = PhysicsTestFixtures.HalloweenStorm2003.GsmBzNt; // -35 nT
        double vBs = v * Math.Abs(bz) * 1e-3; // mV/m

        // 24-hour time series at 1-hour steps
        double[] vBsSeries = Enumerable.Repeat(vBs, 24).ToArray();

        var (dstMin, _) = BurtonOde.RunOdeTimeSeries(
            vBsSeries,
            dst0Nt: 0.0,
            dtHours: 1.0,
            pdynNpa: 5.0);

        // Halloween storm published Dst_min ≈ -383 nT; expect significantly negative
        dstMin.Should().BeLessThan(-100.0,
            "sustained 70 mV/m VBs for 24 hours must drive Dst well below -100 nT");
        double.IsNaN(dstMin).Should().BeFalse();
    }

    [Fact]
    public void RunOdeTimeSeries_ZeroVBs_RecoveryPhase_DstApproachesZero()
    {
        // Start at Dst0 = -100 nT, no injection (VBs = 0) — should recover toward 0
        double[] vBsSeries = Enumerable.Repeat(0.0, 48).ToArray();

        var (_, dstFinal) = BurtonOde.RunOdeTimeSeries(
            vBsSeries,
            dst0Nt: -100.0,
            dtHours: 1.0,
            pdynNpa: 2.0);

        // After 48 hours of pure recovery, Dst should be less negative than -100
        dstFinal.Should().BeGreaterThan(-100.0,
            "48 hours of recovery with no injection must reduce storm intensity");
        dstFinal.Should().BeLessThan(10.0,
            "recovery should not overshoot into positive range significantly");
    }

    [Fact]
    public async Task TrainAsync_BurtonOde_ReturnsModel_AndPredictProducesFiniteResult()
    {
        var bzArr = new float[] { PhysicsTestFixtures.HalloweenStorm2003.GsmBzNt, -10f, 5f };
        var vArr = new float[] { 530f, 400f, 350f };
        using var frame = MakeSolarWindFrame(bzArr, vArr);

        var model = await _adapter.TrainAsync(MakeBurtonConfig(), frame, null, CancellationToken.None);
        model.Should().NotBeNull();
        model.StageName.Should().Be("burton_stage");

        var result = await model.PredictAsync(frame, CancellationToken.None);
        result.Values.Should().HaveCount(3);

        // Southward Bz row should produce negative Dst; northward Bz row may recover
        result.Values[0].Should().BeLessThan(0f, "southward Bz drives negative Dst");
        float.IsNaN(result.Values[0]).Should().BeFalse();
        float.IsNaN(result.Values[2]).Should().BeFalse();
    }
}
