using FluentAssertions;
using SolarPipe.Core.Models;
using SolarPipe.Data.Providers;
using SolarPipe.Tests.Unit.Fixtures;

namespace SolarPipe.Tests.Unit;

[Trait("Category", "Unit")]
public sealed class CsvProviderTests : IDisposable
{
    private readonly List<string> _tempFiles = new();
    private readonly CsvProvider _sut = new();

    public void Dispose()
    {
        foreach (var f in _tempFiles)
            if (File.Exists(f)) File.Delete(f);
    }

    // 1. Comma-delimited file — header detection and correct row count
    [Fact]
    public async Task LoadAsync_CommaCsv_LoadsAllRows()
    {
        var path = WriteTempCsv("timestamp,speed,bz_gsm\n" +
            "2003-01-01,450.0,-5.0\n" +
            "2003-01-02,460.0,-6.0\n" +
            "2003-01-03,470.0,-7.0\n");

        var config = MakeConfig(path);
        var frame = await _sut.LoadAsync(config, new DataQuery(), CancellationToken.None);

        frame.RowCount.Should().Be(3);
        frame.Schema.Columns.Should().HaveCount(3);
        frame.Schema.HasColumn("speed").Should().BeTrue();
    }

    // 2. Tab-delimited file
    [Fact]
    public async Task LoadAsync_TabDelimiter_ParsesCorrectly()
    {
        var path = WriteTempCsv("speed\tdensity\tbz\n450.0\t7.0\t-5.0\n460.0\t7.5\t-6.0\n");
        var config = MakeConfig(path, delimiter: "tab");

        var frame = await _sut.LoadAsync(config, new DataQuery(), CancellationToken.None);

        frame.RowCount.Should().Be(2);
        frame.GetColumn("speed")[0].Should().BeApproximately(450.0f, 1e-4f);
    }

    // 3. Sentinel value 9999.9 → NaN (RULE-120)
    [Fact]
    public async Task LoadAsync_SentinelValue_ConvertedToNaN()
    {
        var path = WriteTempCsv("speed,density,bz_gsm\n" +
            "450.0,7.0,-5.0\n" +
            "9999.9,8.0,-6.0\n" +   // speed is sentinel
            "460.0,9999.9,-7.0\n"); // density is sentinel

        var config = MakeConfig(path);
        var frame = await _sut.LoadAsync(config, new DataQuery(), CancellationToken.None);

        float.IsNaN(frame.GetColumn("speed")[1]).Should().BeTrue("9999.9 speed must be NaN");
        float.IsNaN(frame.GetColumn("density")[2]).Should().BeTrue("9999.9 density must be NaN");
    }

    // 4. Missing/empty field → NaN
    [Fact]
    public async Task LoadAsync_EmptyField_ConvertedToNaN()
    {
        var path = WriteTempCsv("speed,density\n450.0,\n460.0,7.5\n");

        var config = MakeConfig(path);
        var frame = await _sut.LoadAsync(config, new DataQuery(), CancellationToken.None);

        float.IsNaN(frame.GetColumn("density")[0]).Should().BeTrue("empty field must be NaN");
        frame.GetColumn("density")[1].Should().BeApproximately(7.5f, 1e-4f);
    }

    // 5. Limit parameter respected
    [Fact]
    public async Task LoadAsync_WithLimit_RespectsLimit()
    {
        var path = WriteTempCsv("speed\n100\n200\n300\n400\n500\n");
        var config = MakeConfig(path);

        var frame = await _sut.LoadAsync(config, new DataQuery(Limit: 2), CancellationToken.None);

        frame.RowCount.Should().Be(2);
    }

    // 6. DiscoverSchemaAsync infers float type from sample rows
    [Fact]
    public async Task DiscoverSchemaAsync_FloatColumns_InferredCorrectly()
    {
        var path = WriteTempCsv("speed,density,bz_gsm\n450.0,7.0,-5.0\n460.0,7.5,-6.0\n");
        var config = MakeConfig(path);

        var schema = await _sut.DiscoverSchemaAsync(config, CancellationToken.None);

        schema.Columns.Should().HaveCount(3);
        schema.Columns.Should().OnlyContain(c => c.Type == ColumnType.Float);
    }

    // 7. Semicolon delimiter
    [Fact]
    public async Task LoadAsync_SemicolonDelimiter_ParsesCorrectly()
    {
        var path = WriteTempCsv("speed;density\n450.0;7.0\n460.0;7.5\n");
        var config = MakeConfig(path, delimiter: ";");

        var frame = await _sut.LoadAsync(config, new DataQuery(), CancellationToken.None);

        frame.RowCount.Should().Be(2);
        frame.GetColumn("density")[1].Should().BeApproximately(7.5f, 1e-4f);
    }

    // 8. File from fixtures contains known sentinel row (test.csv)
    [Fact]
    public async Task LoadAsync_FixtureFile_SentinelRowIsNaN()
    {
        // Walk up from test bin directory to solution root
        var dir = AppContext.BaseDirectory;
        while (!File.Exists(Path.Combine(dir, "SolarPipe.sln")) && Directory.GetParent(dir) is { } p)
            dir = p.FullName;
        var path = Path.Combine(dir, "tests", "fixtures", "test.csv");
        var config = MakeConfig(path);

        var frame = await _sut.LoadAsync(config, new DataQuery(), CancellationToken.None);

        // Row 8 (index 8) in test.csv has feature_a = 9999.9
        float.IsNaN(frame.GetColumn("feature_a")[8]).Should().BeTrue();
    }

    // --- helpers ---

    private DataSourceConfig MakeConfig(string path, string? delimiter = null)
    {
        var opts = delimiter is null
            ? null
            : (IReadOnlyDictionary<string, string>)new Dictionary<string, string> { ["delimiter"] = delimiter };
        return new DataSourceConfig("test", "csv", path, opts);
    }

    private string WriteTempCsv(string content)
    {
        var path = Path.GetTempFileName() + ".csv";
        File.WriteAllText(path, content);
        _tempFiles.Add(path);
        return path;
    }
}
