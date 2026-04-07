using FluentAssertions;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Tests.Unit.Fixtures;
using SolarPipe.Training.Adapters;
using SolarPipe.Training.Physics;

namespace SolarPipe.Tests.Unit;

[Trait("Category", "Unit")]
public class NewellCouplingTests
{
    private readonly PhysicsAdapter _adapter = new();

    private static StageConfig MakeNewellConfig(IReadOnlyDictionary<string, object>? hp = null) =>
        new StageConfig("newell_stage", "Physics", "NewellCoupling", "src",
            ["by_gsm", "bz_gsm", "v_km_s"], "coupling", hp);

    private static InMemoryDataFrame MakeImfFrame(float[] by, float[] bz, float[] v)
    {
        var schema = new DataSchema([
            new ColumnInfo("by_gsm", ColumnType.Float, false),
            new ColumnInfo("bz_gsm", ColumnType.Float, false),
            new ColumnInfo("v_km_s", ColumnType.Float, false)
        ]);
        return new InMemoryDataFrame(schema, [by, bz, v]);
    }

    [Fact]
    public void PhysicsAdapter_SupportedModels_ContainsNewellCoupling()
    {
        _adapter.SupportedModels.Should().Contain("NewellCoupling");
    }

    [Fact]
    public void Compute_SouthwardBz_ClockAngle180_MaxCoupling()
    {
        // By=0, Bz=-10 nT → θ_c = atan2(0, -10) = 180° → sin(90°) = 1 → maximum coupling
        double couplingMax = NewellCoupling.Compute(
            vKmPerSec: PhysicsTestFixtures.NewellCouplingValues.SolarWindSpeedKmPerSec,
            byGsmNt: 0.0,
            bzGsmNt: -PhysicsTestFixtures.NewellCouplingValues.BtNt,
            normalize: true);

        // By=0, Bz=+10 nT → θ_c = atan2(0, +10) = 0° → sin(0°) = 0 → zero coupling
        double couplingNorth = NewellCoupling.Compute(
            vKmPerSec: PhysicsTestFixtures.NewellCouplingValues.SolarWindSpeedKmPerSec,
            byGsmNt: 0.0,
            bzGsmNt: PhysicsTestFixtures.NewellCouplingValues.BtNt,
            normalize: true);

        couplingMax.Should().BeGreaterThan(couplingNorth,
            "southward IMF (clock angle 180°) must produce higher coupling than northward");
        couplingMax.Should().BeGreaterThan(0.0, "southward coupling must be positive");
    }

    [Fact]
    public void Compute_NorthwardBz_ZeroCoupling()
    {
        // Purely northward IMF: By=0, Bz>0 → clock angle = 0° → sin(0/2) = 0 → coupling = 0
        double coupling = NewellCoupling.Compute(
            vKmPerSec: 400.0,
            byGsmNt: 0.0,
            bzGsmNt: 10.0,
            normalize: true);

        coupling.Should().BeApproximately(0.0, 1e-10,
            "northward IMF (clock angle 0°) produces zero reconnection");
    }

    [Fact]
    public void Compute_MonotonicInSpeed_HigherSpeedGivesHigherCoupling()
    {
        // Newell coupling scales as v^(4/3): higher speed must produce higher coupling
        // At fixed southward Bz=-10 nT, By=0
        double coupling400 = NewellCoupling.Compute(400.0, 0.0, -10.0, normalize: true);
        double coupling600 = NewellCoupling.Compute(600.0, 0.0, -10.0, normalize: true);
        double coupling800 = NewellCoupling.Compute(800.0, 0.0, -10.0, normalize: true);

        coupling600.Should().BeGreaterThan(coupling400,
            "coupling must increase with solar wind speed (v^4/3 scaling)");
        coupling800.Should().BeGreaterThan(coupling600,
            "coupling must increase with solar wind speed (v^4/3 scaling)");

        // Verify exponent: coupling(v2)/coupling(v1) ≈ (v2/v1)^(4/3)
        double ratio = coupling600 / coupling400;
        double expectedRatio = Math.Pow(600.0 / 400.0, 4.0 / 3.0);
        ratio.Should().BeApproximately(expectedRatio, 1e-6,
            "coupling ratio must follow v^(4/3) scaling law");
    }

    [Fact]
    public async Task TrainAsync_NewellCoupling_ReturnsModel_PredictProducesValues()
    {
        var by = new float[] { 0f, 5f, 0f };
        var bz = new float[] { -10f, -8f, 5f };
        var v = new float[] { 450f, 400f, 350f };
        using var frame = MakeImfFrame(by, bz, v);

        var model = await _adapter.TrainAsync(MakeNewellConfig(), frame, null, CancellationToken.None);
        model.Should().NotBeNull();
        model.StageName.Should().Be("newell_stage");

        var result = await model.PredictAsync(frame, CancellationToken.None);
        result.Values.Should().HaveCount(3);

        // Southward Bz rows → positive coupling
        result.Values[0].Should().BeGreaterThan(0f, "southward Bz row must produce positive coupling");
        result.Values[1].Should().BeGreaterThan(0f, "southward Bz row must produce positive coupling");
        // Northward Bz → coupling ≈ 0
        result.Values[2].Should().BeApproximately(0f, 1e-4f, "northward Bz row must produce near-zero coupling");
    }
}
