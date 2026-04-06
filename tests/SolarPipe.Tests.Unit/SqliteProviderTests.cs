using FluentAssertions;
using Microsoft.Data.Sqlite;
using SolarPipe.Core.Models;
using SolarPipe.Data.Providers;
using SolarPipe.Tests.Unit.Fixtures;

namespace SolarPipe.Tests.Unit;

[Trait("Category", "Unit")]
public sealed class SqliteProviderTests : IDisposable
{
    private readonly SqliteConnection _keepAlive;
    private readonly string _connectionString;
    private readonly SqliteProvider _sut;

    public SqliteProviderTests()
    {
        // Use shared-cache in-memory DB so multiple connections share the same data
        _connectionString = "Data Source=SqliteProviderTests;Mode=Memory;Cache=Shared";
        _keepAlive = new SqliteConnection(_connectionString);
        _keepAlive.Open();
        Seed(_keepAlive);
        _sut = new SqliteProvider();
    }

    public void Dispose() => _keepAlive.Dispose();

    // 1. Schema discovery — correct column count and names
    [Fact]
    public async Task DiscoverSchemaAsync_ValidTable_ReturnsExpectedColumns()
    {
        var config = MakeConfig("solar_wind");
        var schema = await _sut.DiscoverSchemaAsync(config, CancellationToken.None);

        schema.Columns.Should().HaveCount(6);  // id, timestamp, speed, density, bz_gsm, dst
        schema.HasColumn("id").Should().BeTrue();
        schema.HasColumn("timestamp").Should().BeTrue();
        schema.HasColumn("speed").Should().BeTrue();
        schema.HasColumn("density").Should().BeTrue();
        schema.HasColumn("bz_gsm").Should().BeTrue();
        schema.HasColumn("dst").Should().BeTrue();
    }

    // 2. LoadAsync returns all rows
    [Fact]
    public async Task LoadAsync_NoLimit_ReturnsAllRows()
    {
        var config = MakeConfig("solar_wind");
        var frame = await _sut.LoadAsync(config, new DataQuery(), CancellationToken.None);

        frame.RowCount.Should().Be(10);
    }

    // 3. Query with explicit SQL (parameterized) — RULE-091
    [Fact]
    public async Task LoadAsync_WithSqlQuery_ReturnsFilteredRows()
    {
        var config = MakeConfig("solar_wind");
        var query = new DataQuery(Sql: "SELECT * FROM \"solar_wind\" WHERE id <= @maxId",
            Parameters: new Dictionary<string, object> { ["maxId"] = 5L });

        var frame = await _sut.LoadAsync(config, query, CancellationToken.None);

        frame.RowCount.Should().Be(5);
    }

    // 4. Limit respected
    [Fact]
    public async Task LoadAsync_WithLimit_RespectsLimit()
    {
        var config = MakeConfig("solar_wind");
        var query = new DataQuery(Limit: 3);

        var frame = await _sut.LoadAsync(config, query, CancellationToken.None);

        frame.RowCount.Should().Be(3);
    }

    // 5. Sentinel value (9999.9) converted to NaN — RULE-120
    [Fact]
    public async Task LoadAsync_SentinelValues_ConvertedToNaN()
    {
        var config = MakeConfig("solar_wind");
        var frame = await _sut.LoadAsync(config, new DataQuery(), CancellationToken.None);

        // Row index 4 has bz_gsm = 9999.9 (seeded as sentinel)
        var bz = frame.GetColumn("bz_gsm");
        float.IsNaN(bz[4]).Should().BeTrue("sentinel 9999.9 must become NaN");
    }

    // 6. NULL column values converted to NaN
    [Fact]
    public async Task LoadAsync_NullColumn_ConvertedToNaN()
    {
        var config = MakeConfig("solar_wind");
        var frame = await _sut.LoadAsync(config, new DataQuery(), CancellationToken.None);

        // Row index 7 has density = NULL (seeded as nullable)
        var density = frame.GetColumn("density");
        float.IsNaN(density[7]).Should().BeTrue("NULL must become NaN");
    }

    // 7. Missing table raises clear error
    [Fact]
    public async Task DiscoverSchemaAsync_MissingTable_ThrowsWithContext()
    {
        var config = MakeConfig("nonexistent_table");
        var act = () => _sut.DiscoverSchemaAsync(config, CancellationToken.None);

        await act.Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("*nonexistent_table*");
    }

    // 8. CanHandle returns false for non-sqlite provider
    [Fact]
    public void CanHandle_WrongProvider_ReturnsFalse()
    {
        var config = new DataSourceConfig("x", "csv", "file.csv");
        _sut.CanHandle(config).Should().BeFalse();
    }

    // --- helpers ---

    private DataSourceConfig MakeConfig(string table) =>
        new("test", "sqlite", _connectionString,
            new Dictionary<string, string> { ["table"] = table });

    private static void Seed(SqliteConnection conn)
    {
        using var cmd = conn.CreateCommand();
        cmd.CommandText = """
            CREATE TABLE IF NOT EXISTS solar_wind (
                id       INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                speed    REAL NOT NULL,
                density  REAL,
                bz_gsm   REAL NOT NULL,
                dst      REAL NOT NULL
            );
            """;
        cmd.ExecuteNonQuery();

        // 10 rows; row 4 (id=5) has bz_gsm sentinel; row 7 (id=8) has density NULL
        var rows = new (int id, string ts, double speed, double? density, double bz, double dst)[]
        {
            (1,  "2003-01-01", 450.0, 7.0,  -5.0,  -20.0),
            (2,  "2003-01-02", 460.0, 7.5,  -6.0,  -25.0),
            (3,  "2003-01-03", 470.0, 8.0,  -7.0,  -30.0),
            (4,  "2003-01-04", 480.0, 8.5,  -8.0,  -35.0),
            (5,  "2003-01-05", 490.0, 9.0,  9999.9, -40.0),  // bz sentinel
            (6,  "2003-01-06", 500.0, 9.5,  -10.0, -45.0),
            (7,  "2003-01-07", 510.0, 10.0, -11.0, -50.0),
            (8,  "2003-01-08", 520.0, null, -12.0, -55.0),   // density NULL
            (9,  "2003-01-09", 530.0, 11.0, -13.0, -60.0),
            (10, "2003-01-10", 540.0, 11.5, -14.0, -65.0),
        };

        foreach (var (id, ts, speed, density, bz, dst) in rows)
        {
            using var insert = conn.CreateCommand();
            insert.CommandText =
                "INSERT INTO solar_wind VALUES (@id, @ts, @speed, @density, @bz, @dst)";
            insert.Parameters.AddWithValue("@id", id);
            insert.Parameters.AddWithValue("@ts", ts);
            insert.Parameters.AddWithValue("@speed", speed);
            insert.Parameters.AddWithValue("@density", density.HasValue ? density.Value : DBNull.Value);
            insert.Parameters.AddWithValue("@bz", bz);
            insert.Parameters.AddWithValue("@dst", dst);
            insert.ExecuteNonQuery();
        }
    }
}
