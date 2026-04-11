using FluentAssertions;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Training.Features;

namespace SolarPipe.Tests.Unit.Phase8;

[Trait("Category", "Unit")]
public sealed class MissingnessEnricherTests
{
    private readonly MissingnessFeatureEnricher _sut = new();

    // ── Helper ─────────────────────────────────────────────────────────────────

    private static IDataFrame MakeFrame(params (string Name, float[] Values)[] cols)
    {
        var columns = cols.Select(c => new ColumnInfo(c.Name, ColumnType.Float, true)).ToList();
        var schema = new DataSchema(columns);
        var data   = cols.Select(c => c.Values).ToArray();
        return new InMemoryDataFrame(schema, data);
    }

    // ── Basic presence detection ───────────────────────────────────────────────

    [Fact]
    public void Enrich_ColumnFullyPresent_IndicatorIsAllOnes()
    {
        var frame  = MakeFrame(("usflux", new float[] { 1f, 2f, 3f }));
        var result = _sut.Enrich(frame);

        var indicator = result.GetColumn("has_sharp_obs");
        indicator.Should().AllSatisfy(v => v.Should().BeApproximately(1f, 1e-6f));
    }

    [Fact]
    public void Enrich_ColumnFullyNaN_IndicatorIsAllZeros()
    {
        var frame  = MakeFrame(("usflux", new float[] { float.NaN, float.NaN, float.NaN }));
        var result = _sut.Enrich(frame);

        var indicator = result.GetColumn("has_sharp_obs");
        indicator.Should().AllSatisfy(v => v.Should().BeApproximately(0f, 1e-6f));
    }

    [Fact]
    public void Enrich_MixedNaN_IndicatorMatchesPresence()
    {
        var frame = MakeFrame(("usflux", new float[] { 1f, float.NaN, 3f, float.NaN }));
        var result = _sut.Enrich(frame);

        var indicator = result.GetColumn("has_sharp_obs");
        indicator.Should().BeEquivalentTo(new float[] { 1f, 0f, 1f, 0f });
    }

    // ── Absent source column → all zeros ──────────────────────────────────────

    [Fact]
    public void Enrich_SourceColumnAbsent_IndicatorIsAllZeros()
    {
        // Frame has no usflux column at all
        var frame  = MakeFrame(("cme_speed_kms", new float[] { 500f, 600f }));
        var result = _sut.Enrich(frame);

        result.Schema.HasColumn("has_sharp_obs").Should().BeTrue();
        var indicator = result.GetColumn("has_sharp_obs");
        indicator.Should().AllSatisfy(v => v.Should().BeApproximately(0f, 1e-6f));
    }

    // ── All four indicators are added ─────────────────────────────────────────

    [Fact]
    public void Enrich_AddsAllFourIndicators()
    {
        var frame  = MakeFrame(
            ("usflux",              new float[] { 1f }),
            ("flare_class_numeric", new float[] { float.NaN }),
            ("sw_bz_ambient",       new float[] { -5f }),
            ("cme_mass_grams",      new float[] { float.NaN }));

        var result = _sut.Enrich(frame);

        result.Schema.HasColumn("has_sharp_obs").Should().BeTrue();
        result.Schema.HasColumn("has_flare_obs").Should().BeTrue();
        result.Schema.HasColumn("has_bz_obs").Should().BeTrue();
        result.Schema.HasColumn("has_mass_obs").Should().BeTrue();
    }

    [Fact]
    public void Enrich_CorrectValuesForEachIndicator()
    {
        var frame = MakeFrame(
            ("usflux",              new float[] { 1f,       float.NaN }),
            ("flare_class_numeric", new float[] { float.NaN, 2f       }),
            ("sw_bz_ambient",       new float[] { -5f,      -3f       }),
            ("cme_mass_grams",      new float[] { float.NaN, float.NaN }));

        var result = _sut.Enrich(frame);

        result.GetColumn("has_sharp_obs").Should().BeEquivalentTo(new float[] { 1f, 0f });
        result.GetColumn("has_flare_obs").Should().BeEquivalentTo(new float[] { 0f, 1f });
        result.GetColumn("has_bz_obs").Should().BeEquivalentTo(new float[]    { 1f, 1f });
        result.GetColumn("has_mass_obs").Should().BeEquivalentTo(new float[]  { 0f, 0f });
    }

    // ── Idempotency ───────────────────────────────────────────────────────────

    [Fact]
    public void Enrich_CalledTwice_DoesNotDuplicateIndicators()
    {
        var frame   = MakeFrame(("usflux", new float[] { 1f, 2f }));
        var once    = _sut.Enrich(frame);
        var twice   = _sut.Enrich(once);

        // Column count for has_sharp_obs should be exactly 1
        var colCount = twice.Schema.Columns.Count(c => c.Name == "has_sharp_obs");
        colCount.Should().Be(1);
    }

    // ── Row count preserved ───────────────────────────────────────────────────

    [Fact]
    public void Enrich_RowCountUnchanged()
    {
        var frame  = MakeFrame(("usflux", new float[] { 1f, 2f, 3f, 4f, 5f }));
        var result = _sut.Enrich(frame);

        result.RowCount.Should().Be(5);
    }
}
