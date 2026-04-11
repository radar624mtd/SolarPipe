using FluentAssertions;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Training.Features;

namespace SolarPipe.Tests.Unit.Phase8;

[Trait("Category", "Unit")]
public sealed class DomainFeatureTransformTests
{
    // ── Helper ─────────────────────────────────────────────────────────────────

    private static IDataFrame MakeFrame(params (string Name, float[] Values)[] cols)
    {
        var columns = cols.Select(c => new ColumnInfo(c.Name, ColumnType.Float, true)).ToList();
        var schema  = new DataSchema(columns);
        var data    = cols.Select(c => c.Values).ToArray();
        return new InMemoryDataFrame(schema, data);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // AddOriginationFeatures
    // ═══════════════════════════════════════════════════════════════════════════

    [Fact]
    public void AddOriginationFeatures_MagneticComplexityRatio_Computed()
    {
        // totpot / (usflux + ε)  → 100 / (50 + 1e-10) ≈ 2
        var frame = MakeFrame(
            ("totpot",  new float[] { 100f }),
            ("usflux",  new float[] { 50f }),
            ("meanshr", new float[] { 0f }),
            ("totusjz", new float[] { 0f }),
            ("flare_class_numeric", new float[] { 0f }),
            ("cme_speed_kms",       new float[] { 0f }),
            ("cme_latitude",        new float[] { 0f }),
            ("cme_longitude",       new float[] { 0f }));

        var result = DomainFeatureTransforms.AddOriginationFeatures(frame);
        var ratio  = result.GetColumn("magnetic_complexity_ratio");

        ratio[0].Should().BeApproximately(100f / (50f + 1e-10f), 1e-3f);
    }

    [Fact]
    public void AddOriginationFeatures_HelicityProxy_MeanshrTimesTotusjz()
    {
        var frame = MakeFrame(
            ("totpot",  new float[] { 0f }),
            ("usflux",  new float[] { 0f }),
            ("meanshr", new float[] { 3f }),
            ("totusjz", new float[] { 7f }),
            ("flare_class_numeric", new float[] { 0f }),
            ("cme_speed_kms",       new float[] { 0f }),
            ("cme_latitude",        new float[] { 0f }),
            ("cme_longitude",       new float[] { 0f }));

        var result = DomainFeatureTransforms.AddOriginationFeatures(frame);
        result.GetColumn("helicity_proxy")[0].Should().BeApproximately(21f, 1e-4f);
    }

    [Fact]
    public void AddOriginationFeatures_FlareSpeedCoupling_Product()
    {
        var frame = MakeFrame(
            ("totpot",  new float[] { 0f }),
            ("usflux",  new float[] { 0f }),
            ("meanshr", new float[] { 0f }),
            ("totusjz", new float[] { 0f }),
            ("flare_class_numeric", new float[] { 4f }),
            ("cme_speed_kms",       new float[] { 500f }),
            ("cme_latitude",        new float[] { 0f }),
            ("cme_longitude",       new float[] { 0f }));

        var result = DomainFeatureTransforms.AddOriginationFeatures(frame);
        result.GetColumn("flare_speed_coupling")[0].Should().BeApproximately(2000f, 1e-2f);
    }

    [Fact]
    public void AddOriginationFeatures_SourceGeometry_EquatorialCentralMeridian_IsOne()
    {
        // lat=0, lon=0 → cos(0)*1 = 1
        var frame = MakeFrame(
            ("totpot",  new float[] { 0f }),
            ("usflux",  new float[] { 0f }),
            ("meanshr", new float[] { 0f }),
            ("totusjz", new float[] { 0f }),
            ("flare_class_numeric", new float[] { 0f }),
            ("cme_speed_kms",       new float[] { 0f }),
            ("cme_latitude",        new float[] { 0f }),
            ("cme_longitude",       new float[] { 0f }));

        var result = DomainFeatureTransforms.AddOriginationFeatures(frame);
        result.GetColumn("source_geometry_factor")[0].Should().BeApproximately(1f, 1e-5f);
    }

    [Fact]
    public void AddOriginationFeatures_SourceGeometry_LimbCME_IsZero()
    {
        // |lon| = 90 → lonFrac = 0 → product = 0
        var frame = MakeFrame(
            ("totpot",  new float[] { 0f }),
            ("usflux",  new float[] { 0f }),
            ("meanshr", new float[] { 0f }),
            ("totusjz", new float[] { 0f }),
            ("flare_class_numeric", new float[] { 0f }),
            ("cme_speed_kms",       new float[] { 0f }),
            ("cme_latitude",        new float[] { 0f }),
            ("cme_longitude",       new float[] { 90f }));

        var result = DomainFeatureTransforms.AddOriginationFeatures(frame);
        result.GetColumn("source_geometry_factor")[0].Should().BeApproximately(0f, 1e-5f);
    }

    [Fact]
    public void AddOriginationFeatures_NaNPropagation_WhenOperandIsNaN()
    {
        var frame = MakeFrame(
            ("totpot",  new float[] { float.NaN }),
            ("usflux",  new float[] { 50f }),
            ("meanshr", new float[] { float.NaN }),
            ("totusjz", new float[] { 7f }),
            ("flare_class_numeric", new float[] { float.NaN }),
            ("cme_speed_kms",       new float[] { 500f }),
            ("cme_latitude",        new float[] { float.NaN }),
            ("cme_longitude",       new float[] { 0f }));

        var result = DomainFeatureTransforms.AddOriginationFeatures(frame);

        float.IsNaN(result.GetColumn("magnetic_complexity_ratio")[0]).Should().BeTrue();
        float.IsNaN(result.GetColumn("helicity_proxy")[0]).Should().BeTrue();
        float.IsNaN(result.GetColumn("flare_speed_coupling")[0]).Should().BeTrue();
        float.IsNaN(result.GetColumn("source_geometry_factor")[0]).Should().BeTrue();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // AddTransitFeatures
    // ═══════════════════════════════════════════════════════════════════════════

    [Fact]
    public void AddTransitFeatures_DynamicPressure_Formula()
    {
        // 1.67e-6 * 5 * 400² = 1.67e-6 * 800000 = 1.336 nPa
        var frame = MakeFrame(
            ("sw_density_ambient", new float[] { 5f }),
            ("sw_speed_ambient",   new float[] { 400f }),
            ("sw_bz_ambient",      new float[] { -5f }));

        var result = DomainFeatureTransforms.AddTransitFeatures(frame);
        result.GetColumn("dynamic_pressure_sw")[0]
            .Should().BeApproximately(1.67e-6f * 5f * 400f * 400f, 1e-6f);
    }

    [Fact]
    public void AddTransitFeatures_NewellCoupling_Formula()
    {
        // v^(4/3) * |Bz|^(2/3): v=400, Bz=-5
        var frame = MakeFrame(
            ("sw_density_ambient", new float[] { 5f }),
            ("sw_speed_ambient",   new float[] { 400f }),
            ("sw_bz_ambient",      new float[] { -5f }));

        var result = DomainFeatureTransforms.AddTransitFeatures(frame);
        double expected = Math.Pow(400.0, 4.0 / 3.0) * Math.Pow(5.0, 2.0 / 3.0);
        result.GetColumn("newell_coupling_approx")[0]
            .Should().BeApproximately((float)expected, 1f);
    }

    [Fact]
    public void AddTransitFeatures_NewellCoupling_NearZeroBz_ReturnsZero()
    {
        var frame = MakeFrame(
            ("sw_density_ambient", new float[] { 5f }),
            ("sw_speed_ambient",   new float[] { 400f }),
            ("sw_bz_ambient",      new float[] { 0f }));

        var result = DomainFeatureTransforms.AddTransitFeatures(frame);
        result.GetColumn("newell_coupling_approx")[0].Should().BeApproximately(0f, 1e-6f);
    }

    [Fact]
    public void AddTransitFeatures_NaNPropagation()
    {
        var frame = MakeFrame(
            ("sw_density_ambient", new float[] { float.NaN }),
            ("sw_speed_ambient",   new float[] { 400f }),
            ("sw_bz_ambient",      new float[] { -5f }));

        var result = DomainFeatureTransforms.AddTransitFeatures(frame);
        float.IsNaN(result.GetColumn("dynamic_pressure_sw")[0]).Should().BeTrue();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // AddDeltaV
    // ═══════════════════════════════════════════════════════════════════════════

    [Fact]
    public void AddDeltaV_ComputesDifference()
    {
        var frame = MakeFrame(
            ("pred_arrival_speed_kms", new float[] { 600f }),
            ("sw_speed_ambient",       new float[] { 400f }));

        var result = DomainFeatureTransforms.AddDeltaV(frame);
        result.GetColumn("delta_v_kms")[0].Should().BeApproximately(200f, 1e-4f);
    }

    [Fact]
    public void AddDeltaV_MissingPredColumn_ReturnsSameFrame()
    {
        var frame  = MakeFrame(("sw_speed_ambient", new float[] { 400f }));
        var result = DomainFeatureTransforms.AddDeltaV(frame);

        result.Schema.HasColumn("delta_v_kms").Should().BeFalse();
    }

    [Fact]
    public void AddDeltaV_NaNArrivalSpeed_ProducesNaN()
    {
        var frame = MakeFrame(
            ("pred_arrival_speed_kms", new float[] { float.NaN }),
            ("sw_speed_ambient",       new float[] { 400f }));

        var result = DomainFeatureTransforms.AddDeltaV(frame);
        float.IsNaN(result.GetColumn("delta_v_kms")[0]).Should().BeTrue();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // AddImpactFeatures
    // ═══════════════════════════════════════════════════════════════════════════

    [Fact]
    public void AddImpactFeatures_VbsCoupling_SouthwardBz()
    {
        // vbs = v * max(0, -Bz) * 1e-3 = 400 * 5 * 1e-3 = 2.0 mV/m
        var frame = MakeFrame(
            ("sw_speed_ambient",   new float[] { 400f }),
            ("sw_bz_ambient",      new float[] { -5f }),
            ("sw_density_ambient", new float[] { 5f }));

        var result = DomainFeatureTransforms.AddImpactFeatures(frame);
        result.GetColumn("vbs_coupling")[0].Should().BeApproximately(2.0f, 1e-4f);
    }

    [Fact]
    public void AddImpactFeatures_VbsCoupling_NorthwardBz_IsZero()
    {
        // Bz > 0 → max(0, -Bz) = 0 → vbs = 0
        var frame = MakeFrame(
            ("sw_speed_ambient",   new float[] { 400f }),
            ("sw_bz_ambient",      new float[] { 5f }),
            ("sw_density_ambient", new float[] { 5f }));

        var result = DomainFeatureTransforms.AddImpactFeatures(frame);
        result.GetColumn("vbs_coupling")[0].Should().BeApproximately(0f, 1e-6f);
    }

    [Fact]
    public void AddImpactFeatures_RingCurrentProxy_NegativeWhenSouthwardDense()
    {
        // Bz=-5, density=10 → -50
        var frame = MakeFrame(
            ("sw_speed_ambient",   new float[] { 400f }),
            ("sw_bz_ambient",      new float[] { -5f }),
            ("sw_density_ambient", new float[] { 10f }));

        var result = DomainFeatureTransforms.AddImpactFeatures(frame);
        result.GetColumn("ring_current_proxy")[0].Should().BeApproximately(-50f, 1e-4f);
    }

    [Fact]
    public void AddImpactFeatures_NaNPropagation()
    {
        var frame = MakeFrame(
            ("sw_speed_ambient",   new float[] { float.NaN }),
            ("sw_bz_ambient",      new float[] { -5f }),
            ("sw_density_ambient", new float[] { 5f }));

        var result = DomainFeatureTransforms.AddImpactFeatures(frame);
        float.IsNaN(result.GetColumn("vbs_coupling")[0]).Should().BeTrue();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // AddStormDuration (synthetic target)
    // ═══════════════════════════════════════════════════════════════════════════

    [Fact]
    public void AddStormDuration_LargeStorm_UsesFormula()
    {
        // |Dst| = 100 → 0.060 * 100 + 16.65 = 22.65 > 6.15 → 22.65
        var frame = MakeFrame(("dst_min_nt", new float[] { -100f }));

        var result = DomainFeatureTransforms.AddStormDuration(frame);
        result.GetColumn("storm_duration_hours")[0].Should().BeApproximately(22.65f, 1e-3f);
    }

    [Fact]
    public void AddStormDuration_SmallStorm_UsesMinimum()
    {
        // |Dst| = 1 → 0.06 + 16.65 = 16.71 > 6.15 → 16.71
        var frame = MakeFrame(("dst_min_nt", new float[] { -1f }));

        var result = DomainFeatureTransforms.AddStormDuration(frame);
        // Even weak storm: 0.060 * 1 + 16.65 = 16.71
        result.GetColumn("storm_duration_hours")[0].Should().BeApproximately(16.71f, 1e-3f);
    }

    [Fact]
    public void AddStormDuration_ZeroDst_UsesMinimumSixPointFifteen()
    {
        // 0.060 * 0 + 16.65 = 16.65 > 6.15 → 16.65
        var frame  = MakeFrame(("dst_min_nt", new float[] { 0f }));
        var result = DomainFeatureTransforms.AddStormDuration(frame);
        result.GetColumn("storm_duration_hours")[0].Should().BeApproximately(16.65f, 1e-3f);
    }

    [Fact]
    public void AddStormDuration_NaNDst_ProducesNaN()
    {
        var frame  = MakeFrame(("dst_min_nt", new float[] { float.NaN }));
        var result = DomainFeatureTransforms.AddStormDuration(frame);
        float.IsNaN(result.GetColumn("storm_duration_hours")[0]).Should().BeTrue();
    }

    [Fact]
    public void AddStormDuration_Idempotent_DoesNotRecompute()
    {
        var frame   = MakeFrame(
            ("dst_min_nt",          new float[] { -100f }),
            ("storm_duration_hours", new float[] { 999f }));
        var result  = DomainFeatureTransforms.AddStormDuration(frame);

        // Already present → must return same value unchanged (999, not 22.65)
        result.GetColumn("storm_duration_hours")[0].Should().BeApproximately(999f, 1e-4f);
    }

    [Fact]
    public void AddStormDuration_MissingDstColumn_ProducesNaNColumn()
    {
        var frame  = MakeFrame(("cme_speed_kms", new float[] { 500f, 600f }));
        var result = DomainFeatureTransforms.AddStormDuration(frame);

        result.Schema.HasColumn("storm_duration_hours").Should().BeTrue();
        float.IsNaN(result.GetColumn("storm_duration_hours")[0]).Should().BeTrue();
        float.IsNaN(result.GetColumn("storm_duration_hours")[1]).Should().BeTrue();
    }
}
