using FluentAssertions;
using SolarPipe.Core.Domain;
using SolarPipe.Core.Models;
using SolarPipe.Tests.Unit.Fixtures;

namespace SolarPipe.Tests.Unit;

[Trait("Category", "Unit")]
public class CoreModelsTests
{
    [Fact]
    public void DataSchema_HasColumn_IsCaseInsensitive()
    {
        var schema = new DataSchema([new ColumnInfo("BzGsm", ColumnType.Float, true)]);
        schema.HasColumn("bzgsm").Should().BeTrue();
        schema.HasColumn("BZGSM").Should().BeTrue();
        schema.HasColumn("nonexistent").Should().BeFalse();
    }

    [Fact]
    public void DataSchema_IndexOf_ReturnsCorrectIndex()
    {
        var schema = new DataSchema([
            new ColumnInfo("speed", ColumnType.Float, false),
            new ColumnInfo("density", ColumnType.Float, false)
        ]);
        schema.IndexOf("speed").Should().Be(0);
        schema.IndexOf("density").Should().Be(1);
        schema.IndexOf("missing").Should().Be(-1);
    }

    [Fact]
    public void RadialSpeed_Create_ThrowsForOutOfRange()
    {
        var act1 = () => RadialSpeed.Create(100f);
        var act2 = () => RadialSpeed.Create(4000f);
        act1.Should().Throw<ArgumentOutOfRangeException>();
        act2.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void RadialSpeed_Create_AcceptsValidRange()
    {
        var speed = RadialSpeed.Create(PhysicsTestFixtures.ModerateEvent.RadialSpeed.KmPerSec);
        speed.KmPerSec.Should().Be(800f);
    }

    [Fact]
    public void SkyPlaneSpeed_Create_ThrowsForNegative()
    {
        var act = () => SkyPlaneSpeed.Create(-10f);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void SpaceWeatherTimeParser_ParseOmni_ParsesHalloweenStorm()
    {
        var dt = SpaceWeatherTimeParser.ParseOmni(PhysicsTestFixtures.OmniTimestamps.ValidOmni);
        dt.Should().Be(new DateTime(2003, 10, 28, 11, 30, 0, DateTimeKind.Utc));
    }

    [Fact]
    public void SpaceWeatherTimeParser_ParseCdaw_ParsesSlashFormat()
    {
        var dt = SpaceWeatherTimeParser.ParseCdaw(PhysicsTestFixtures.CdawTimestamps.ValidSlash);
        dt.Should().Be(new DateTime(2003, 10, 28, 11, 30, 0, DateTimeKind.Utc));
    }

    [Fact]
    public void SpaceWeatherTimeParser_ParseCdaw_ParsesIsoFormat()
    {
        var dt = SpaceWeatherTimeParser.ParseCdaw(PhysicsTestFixtures.CdawTimestamps.ValidIso);
        dt.Should().Be(new DateTime(2003, 10, 28, 11, 30, 0, DateTimeKind.Utc));
    }

    [Fact]
    public void SpaceWeatherTimeParser_ParseOmni_ThrowsForEmpty()
    {
        var act = () => SpaceWeatherTimeParser.ParseOmni("");
        act.Should().Throw<FormatException>();
    }

    [Fact]
    public void GseVector_And_GsmVector_AreDistinctTypes()
    {
        var gse = new GseVector(1f, 2f, -5f);
        var gsm = new GsmVector(1f, 2f, -5f);
        gse.Bz.Should().Be(gsm.Bz);
        // Compile-time check: gse = gsm would fail — distinct types enforce coordinate safety
        gse.GetType().Should().NotBe(gsm.GetType());
    }

    [Fact]
    public void PhysicalConstants_HaveExpectedValues()
    {
        PhysicalConstants.EarthRadiusKm.Should().BeApproximately(6371f, 1f);
        PhysicalConstants.SolarRadiusKm.Should().BeApproximately(695700f, 100f);
        PhysicalConstants.AuKm.Should().BeApproximately(1.496e8f, 1e5f);
    }
}
