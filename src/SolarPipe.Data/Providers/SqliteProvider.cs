using Microsoft.Data.Sqlite;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Data.Providers;

public sealed class SqliteProvider : IDataSourceProvider
{
    public string ProviderName => "sqlite";

    public bool CanHandle(DataSourceConfig config) =>
        config.Provider.Equals("sqlite", StringComparison.OrdinalIgnoreCase)
        && !string.IsNullOrWhiteSpace(config.ConnectionString);

    public async Task<DataSchema> DiscoverSchemaAsync(DataSourceConfig config, CancellationToken ct)
    {
        var table = RequireTable(config);
        using var conn = new SqliteConnection(config.ConnectionString);
        await conn.OpenAsync(ct);

        var cols = new List<ColumnInfo>();
        using var cmd = conn.CreateCommand();
        // PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
        cmd.CommandText = $"PRAGMA table_info({QuoteIdentifier(table)})";
        using var reader = await cmd.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
        {
            var name = reader.GetString(1);
            var rawType = reader.GetString(2).ToUpperInvariant();
            var notNull = reader.GetInt32(3) != 0;
            var colType = InferColumnType(rawType);
            cols.Add(new ColumnInfo(name, colType, IsNullable: !notNull));
        }

        if (cols.Count == 0)
            throw new InvalidOperationException(
                $"SqliteProvider.DiscoverSchemaAsync: table '{table}' not found or has no columns. " +
                $"Connection: {config.ConnectionString}");

        return new DataSchema(cols);
    }

    public async Task<IDataFrame> LoadAsync(DataSourceConfig config, DataQuery query, CancellationToken ct)
    {
        using var conn = new SqliteConnection(config.ConnectionString);
        await conn.OpenAsync(ct);

        var schema = await DiscoverSchemaAsync(config, ct);
        var sql = BuildQuery(config, query, schema);

        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        AddParameters(cmd, query);

        using var reader = await cmd.ExecuteReaderAsync(ct);

        // Buffer rows; space weather datasets are << 1M rows in Phase 1
        var columnBuffers = new List<float>[schema.Columns.Count];
        for (int i = 0; i < columnBuffers.Length; i++)
            columnBuffers[i] = new List<float>();

        while (await reader.ReadAsync(ct))
        {
            for (int col = 0; col < schema.Columns.Count; col++)
            {
                float value = reader.IsDBNull(col)
                    ? float.NaN
                    : ToFloat(reader.GetValue(col), schema.Columns[col]);
                columnBuffers[col].Add(value);
            }
        }

        var data = columnBuffers.Select(b => b.ToArray()).ToArray();
        return new DataFrame.InMemoryDataFrame(schema, data);
    }

    // --- helpers ---

    private static string RequireTable(DataSourceConfig config)
    {
        if (config.Options is not null && config.Options.TryGetValue("table", out var t) && !string.IsNullOrWhiteSpace(t))
            return t;
        throw new ArgumentException(
            $"SqliteProvider requires Options[\"table\"] to identify the target table. " +
            $"Config name: '{config.Name}'.");
    }

    private static string BuildQuery(DataSourceConfig config, DataQuery query, DataSchema schema)
    {
        if (!string.IsNullOrWhiteSpace(query.Sql))
            return query.Sql!;

        var table = query.Table ?? RequireTable(config);
        var sql = $"SELECT * FROM {QuoteIdentifier(table)}";

        if (query.Limit.HasValue)
            sql += $" LIMIT {query.Limit.Value}";

        return sql;
    }

    private static void AddParameters(SqliteCommand cmd, DataQuery query)
    {
        if (query.Parameters is null) return;
        foreach (var (key, value) in query.Parameters)
            cmd.Parameters.AddWithValue("@" + key, value ?? DBNull.Value);
    }

    // Infer ColumnType from SQLite affinity/declared type string
    private static ColumnType InferColumnType(string rawType) =>
        rawType switch
        {
            var t when t.Contains("INT") => ColumnType.Float,      // store as float (RULE-120)
            var t when t.Contains("REAL") || t.Contains("FLOA") || t.Contains("DOUB") || t.Contains("NUMER") => ColumnType.Float,
            var t when t.Contains("TEXT") || t.Contains("CHAR") || t.Contains("CLOB") => ColumnType.String,
            var t when t.Contains("DATE") || t.Contains("TIME") => ColumnType.DateTime,
            _ => ColumnType.Float  // SQLite default affinity is NUMERIC → float
        };

    private static float ToFloat(object value, ColumnInfo col)
    {
        var f = value switch
        {
            long l => (float)l,
            double d => (float)d,
            string s when float.TryParse(s, out var parsed) => parsed,
            string => float.NaN,
            _ => float.NaN
        };

        // RULE-120: convert sentinel values to NaN at load time
        return IsSentinel(f) ? float.NaN : f;
    }

    private static bool IsSentinel(float v) =>
        v is 9999.9f or 999.9f or 999f or -1e31f;

    // Safe identifier quoting — prevents SQL injection via table names (RULE-091)
    internal static string QuoteIdentifier(string identifier)
    {
        if (identifier.Any(c => c == '"' || c == '\0'))
            throw new ArgumentException(
                $"Table/column identifier contains illegal characters: '{identifier}'");
        return $"\"{identifier}\"";
    }
}
