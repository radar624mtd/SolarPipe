using FluentAssertions;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;

namespace SolarPipe.Tests.Unit;

[Trait("Category", "Unit")]
public class InMemoryDataFrameTests
{
    private static DataSchema MakeSchema(params string[] names) =>
        new(names.Select(n => new ColumnInfo(n, ColumnType.Float, true)).ToList());

    private static InMemoryDataFrame Make2Col(int rows = 5)
    {
        var schema = MakeSchema("speed", "density");
        var speed = Enumerable.Range(0, rows).Select(i => (float)i * 100f).ToArray();
        var density = Enumerable.Range(0, rows).Select(i => (float)i * 5f).ToArray();
        return new InMemoryDataFrame(schema, [speed, density]);
    }

    [Fact]
    public void Constructor_ThrowsOnColumnCountMismatch()
    {
        var schema = MakeSchema("a", "b");
        var act = () => new InMemoryDataFrame(schema, [new float[5]]);
        act.Should().Throw<ArgumentException>().WithMessage("*1 columns*schema has 2*");
    }

    [Fact]
    public void Constructor_ThrowsOnUnequalRowCounts()
    {
        var schema = MakeSchema("a", "b");
        var act = () => new InMemoryDataFrame(schema, [new float[5], new float[3]]);
        act.Should().Throw<InvalidOperationException>().WithMessage("*3 rows*expected 5*");
    }

    [Fact]
    public void GetColumn_ByName_ReturnsCorrectData()
    {
        var df = Make2Col();
        var speed = df.GetColumn("speed");
        speed.Should().BeEquivalentTo(new float[] { 0f, 100f, 200f, 300f, 400f });
    }

    [Fact]
    public void GetColumn_ByIndex_ReturnsCorrectData()
    {
        var df = Make2Col();
        var density = df.GetColumn(1);
        density.Should().BeEquivalentTo(new float[] { 0f, 5f, 10f, 15f, 20f });
    }

    [Fact]
    public void GetColumn_UnknownName_Throws()
    {
        var df = Make2Col();
        var act = () => df.GetColumn("nonexistent");
        act.Should().Throw<KeyNotFoundException>().WithMessage("*nonexistent*");
    }

    [Fact]
    public void RowCount_IsCorrect()
    {
        Make2Col(7).RowCount.Should().Be(7);
    }

    [Fact]
    public void Slice_ReturnsSubsetRows()
    {
        var df = Make2Col(10);
        var sliced = (InMemoryDataFrame)df.Slice(2, 3);
        sliced.RowCount.Should().Be(3);
        sliced.GetColumn("speed").Should().BeEquivalentTo(new float[] { 200f, 300f, 400f });
    }

    [Fact]
    public void Slice_ThrowsForOutOfRange()
    {
        var df = Make2Col(5);
        var act = () => df.Slice(3, 10);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void SelectColumns_ReturnsSubsetSchema()
    {
        var df = Make2Col();
        var selected = df.SelectColumns("density");
        selected.Schema.Columns.Should().HaveCount(1);
        selected.Schema.Columns[0].Name.Should().Be("density");
        selected.GetColumn("density").Should().BeEquivalentTo(new float[] { 0f, 5f, 10f, 15f, 20f });
    }

    [Fact]
    public void SelectColumns_ThrowsForUnknownColumn()
    {
        var df = Make2Col();
        var act = () => df.SelectColumns("missing");
        act.Should().Throw<KeyNotFoundException>().WithMessage("*missing*");
    }

    [Fact]
    public void AddColumn_AppendsColumn()
    {
        var df = Make2Col(3);
        var extra = new float[] { 1f, 2f, 3f };
        var extended = df.AddColumn("pressure", extra);
        extended.Schema.Columns.Should().HaveCount(3);
        extended.GetColumn("pressure").Should().BeEquivalentTo(extra);
    }

    [Fact]
    public void AddColumn_ThrowsOnDuplicateName()
    {
        var df = Make2Col(3);
        var act = () => df.AddColumn("speed", new float[3]);
        act.Should().Throw<InvalidOperationException>().WithMessage("*speed*already exists*");
    }

    [Fact]
    public void AddColumn_ThrowsOnLengthMismatch()
    {
        var df = Make2Col(3);
        var act = () => df.AddColumn("extra", new float[5]);
        act.Should().Throw<ArgumentException>().WithMessage("*5*RowCount=3*");
    }

    [Fact]
    public void NaN_PropagatesCorrectly()
    {
        var schema = MakeSchema("bz");
        var bz = new float[] { -5f, float.NaN, -15f };
        var df = new InMemoryDataFrame(schema, [bz]);
        float.IsNaN(df.GetColumn("bz")[1]).Should().BeTrue();
    }

    [Fact]
    public void ToArray_ReturnsCopy()
    {
        var df = Make2Col(3);
        var arr = df.ToArray();
        arr.Should().HaveCount(2);
        arr[0].Should().HaveCount(3);
        // Verify independence — mutating copy doesn't affect original
        arr[0][0] = 9999f;
        df.GetColumn(0)[0].Should().Be(0f);
    }

    [Fact]
    public void Empty_HasZeroRows()
    {
        var schema = MakeSchema("a", "b");
        var df = InMemoryDataFrame.Empty(schema);
        df.RowCount.Should().Be(0);
        df.Schema.Columns.Should().HaveCount(2);
    }

    [Fact]
    public void ResampleAndAlign_ThrowsNotSupportedInPhase1()
    {
        var df = Make2Col();
        var act = () => df.ResampleAndAlign(TimeSpan.FromHours(1));
        act.Should().Throw<NotSupportedException>().WithMessage("*Phase 1*");
    }
}
