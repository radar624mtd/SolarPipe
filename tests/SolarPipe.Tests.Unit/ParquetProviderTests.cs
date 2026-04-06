using FluentAssertions;
using ParquetSharp;
using SolarPipe.Core.Models;
using SolarPipe.Data.Providers;

namespace SolarPipe.Tests.Unit;

[Trait("Category", "Unit")]
public sealed class ParquetProviderTests : IDisposable
{
    private readonly List<string> _tempFiles = new();
    private readonly ParquetProvider _sut = new();

    public void Dispose()
    {
        foreach (var f in _tempFiles)
            if (File.Exists(f)) File.Delete(f);
    }

    // 1. Schema discovery — correct column count and names
    [Fact]
    public async Task DiscoverSchemaAsync_SimpleFile_ReturnsCorrectSchema()
    {
        var path = WriteParquet(
            ("speed", new float[] { 400f, 500f, 600f }),
            ("bz_gsm", new float[] { -5f, -10f, 0f }));

        var config = MakeConfig(path);
        var schema = await _sut.DiscoverSchemaAsync(config, CancellationToken.None);

        schema.Columns.Should().HaveCount(2);
        schema.HasColumn("speed").Should().BeTrue();
        schema.HasColumn("bz_gsm").Should().BeTrue();
        schema.Columns[0].Type.Should().Be(ColumnType.Float);
    }

    // 2. Load — correct row count and values
    [Fact]
    public async Task LoadAsync_SimpleFile_LoadsAllRows()
    {
        var path = WriteParquet(
            ("speed", new float[] { 400f, 500f, 600f }),
            ("bz_gsm", new float[] { -5f, -10f, 0f }));

        var config = MakeConfig(path);
        var frame = await _sut.LoadAsync(config, new DataQuery(), CancellationToken.None);

        frame.RowCount.Should().Be(3);
        frame.GetColumn("speed")[0].Should().BeApproximately(400f, 0.001f);
        frame.GetColumn("bz_gsm")[2].Should().BeApproximately(0f, 0.001f);
    }

    // 3. Sentinel value conversion — 9999.9 → NaN (RULE-120)
    [Fact]
    public async Task LoadAsync_SentinelValues_ConvertedToNaN()
    {
        var path = WriteParquet(
            ("dst", new float[] { -50f, 9999.9f, -30f }));

        var config = MakeConfig(path);
        var frame = await _sut.LoadAsync(config, new DataQuery(), CancellationToken.None);

        var col = frame.GetColumn("dst");
        col[0].Should().BeApproximately(-50f, 0.001f);
        float.IsNaN(col[1]).Should().BeTrue("9999.9 is a sentinel value");
        col[2].Should().BeApproximately(-30f, 0.001f);
    }

    // 4. Nullable columns — nulls become NaN
    [Fact]
    public async Task LoadAsync_NullableColumn_NullsAreNaN()
    {
        var path = WriteNullableParquet(
            ("speed", new float?[] { 400f, null, 600f }));

        var config = MakeConfig(path);
        var frame = await _sut.LoadAsync(config, new DataQuery(), CancellationToken.None);

        var col = frame.GetColumn("speed");
        col[0].Should().BeApproximately(400f, 0.001f);
        float.IsNaN(col[1]).Should().BeTrue("null maps to NaN");
        col[2].Should().BeApproximately(600f, 0.001f);
    }

    // 5. Query limit — only first N rows returned
    [Fact]
    public async Task LoadAsync_WithLimit_ReturnsOnlyNRows()
    {
        var path = WriteParquet(
            ("speed", new float[] { 100f, 200f, 300f, 400f, 500f }));

        var config = MakeConfig(path);
        var frame = await _sut.LoadAsync(config, new DataQuery(Limit: 3), CancellationToken.None);

        frame.RowCount.Should().Be(3);
        frame.GetColumn("speed")[2].Should().BeApproximately(300f, 0.001f);
    }

    // 6. FileNotFound throws cleanly
    [Fact]
    public async Task LoadAsync_MissingFile_ThrowsFileNotFoundException()
    {
        var config = MakeConfig("/nonexistent/path.parquet");

        var act = async () => await _sut.LoadAsync(config, new DataQuery(), CancellationToken.None);

        await act.Should().ThrowAsync<FileNotFoundException>()
            .WithMessage("*ParquetProvider*");
    }

    // --- helpers ---

    private string WriteParquet(params (string name, float[] values)[] columns)
    {
        var path = Path.GetTempFileName() + ".parquet";
        _tempFiles.Add(path);

        var cols = columns.Select(c => (Column)new Column<float>(c.name)).ToArray();
        using var writer = new ParquetFileWriter(path, cols);
        using var rg = writer.AppendRowGroup();

        foreach (var (name, values) in columns)
        {
            using var colWriter = rg.NextColumn().LogicalWriter<float>();
            colWriter.WriteBatch(values);
        }

        writer.Close();
        return path;
    }

    private string WriteNullableParquet(params (string name, float?[] values)[] columns)
    {
        var path = Path.GetTempFileName() + ".parquet";
        _tempFiles.Add(path);

        var cols = columns.Select(c => (Column)new Column<float?>(c.name)).ToArray();
        using var writer = new ParquetFileWriter(path, cols);
        using var rg = writer.AppendRowGroup();

        foreach (var (name, values) in columns)
        {
            using var colWriter = rg.NextColumn().LogicalWriter<float?>();
            colWriter.WriteBatch(values);
        }

        writer.Close();
        return path;
    }

    private static DataSourceConfig MakeConfig(string path) =>
        new("test_parquet", "parquet", path);
}
