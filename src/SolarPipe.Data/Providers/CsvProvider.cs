using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Data.Providers;

public sealed class CsvProvider : IDataSourceProvider
{
    public string ProviderName => "csv";

    public bool CanHandle(DataSourceConfig config) =>
        config.Provider.Equals("csv", StringComparison.OrdinalIgnoreCase)
        && !string.IsNullOrWhiteSpace(config.ConnectionString);

    public async Task<DataSchema> DiscoverSchemaAsync(DataSourceConfig config, CancellationToken ct)
    {
        var delimiter = GetDelimiter(config);
        using var reader = OpenReader(config.ConnectionString);
        using var csv = new CsvReader(reader, MakeCsvConfig(delimiter));

        await csv.ReadAsync();
        csv.ReadHeader();

        var headers = csv.HeaderRecord
            ?? throw new InvalidOperationException(
                $"CsvProvider.DiscoverSchemaAsync: no header row in '{config.ConnectionString}'.");

        // Sample up to 20 rows to infer types
        var samples = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase);
        foreach (var h in headers) samples[h] = new List<string>();

        int sampled = 0;
        while (sampled < 20 && await csv.ReadAsync())
        {
            ct.ThrowIfCancellationRequested();
            foreach (var h in headers)
                samples[h].Add(csv.GetField(h) ?? "");
            sampled++;
        }

        var cols = headers.Select(h => new ColumnInfo(h, InferType(samples[h]), IsNullable: true)).ToList();
        return new DataSchema(cols);
    }

    public async Task<IDataFrame> LoadAsync(DataSourceConfig config, DataQuery query, CancellationToken ct)
    {
        var schema = await DiscoverSchemaAsync(config, ct);
        var delimiter = GetDelimiter(config);

        using var reader = OpenReader(config.ConnectionString);
        using var csv = new CsvReader(reader, MakeCsvConfig(delimiter));

        await csv.ReadAsync();
        csv.ReadHeader();

        var columnBuffers = new List<float>[schema.Columns.Count];
        for (int i = 0; i < columnBuffers.Length; i++)
            columnBuffers[i] = new List<float>();

        int rowsRead = 0;
        while (await csv.ReadAsync())
        {
            ct.ThrowIfCancellationRequested();

            if (query.Limit.HasValue && rowsRead >= query.Limit.Value)
                break;

            for (int col = 0; col < schema.Columns.Count; col++)
            {
                var raw = csv.GetField(schema.Columns[col].Name) ?? "";
                columnBuffers[col].Add(ParseFloat(raw));
            }
            rowsRead++;
        }

        var data = columnBuffers.Select(b => b.ToArray()).ToArray();
        return new DataFrame.InMemoryDataFrame(schema, data);
    }

    // --- helpers ---

    private static TextReader OpenReader(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"CsvProvider: file not found: '{path}'");
        return new StreamReader(path);
    }

    private static CsvConfiguration MakeCsvConfig(char delimiter) =>
        new(CultureInfo.InvariantCulture)
        {
            Delimiter = delimiter.ToString(),
            HasHeaderRecord = true,
            MissingFieldFound = null,
            BadDataFound = null,
        };

    private static char GetDelimiter(DataSourceConfig config)
    {
        if (config.Options is not null && config.Options.TryGetValue("delimiter", out var d))
        {
            return d.ToLowerInvariant() switch
            {
                "tab" or "\t" => '\t',
                "semicolon" or ";" => ';',
                "pipe" or "|" => '|',
                _ when d.Length == 1 => d[0],
                _ => ','
            };
        }
        return ',';
    }

    private static ColumnType InferType(List<string> samples)
    {
        bool allFloat = true, allDateTime = true;

        foreach (var s in samples.Where(v => !string.IsNullOrWhiteSpace(v)))
        {
            if (!float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out _))
                allFloat = false;
            if (!DateTime.TryParse(s, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind, out _))
                allDateTime = false;
        }

        if (allDateTime && allFloat) return ColumnType.DateTime; // prefer DateTime for ISO strings
        if (allFloat) return ColumnType.Float;
        if (allDateTime) return ColumnType.DateTime;
        return ColumnType.String;
    }

    private static float ParseFloat(string raw)
    {
        if (string.IsNullOrWhiteSpace(raw))
            return float.NaN;

        if (!float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out float v))
            return float.NaN;

        // RULE-120: convert sentinel values to NaN at load time
        return IsSentinel(v) ? float.NaN : v;
    }

    private static bool IsSentinel(float v) =>
        v is 9999.9f or 999.9f or 999f or -1e31f;
}
