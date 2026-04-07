using System.Text.Json;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;

namespace SolarPipe.Data.Providers;

// RestApiProvider: fetches real-time solar wind and CME catalog data from NOAA SWPC and NASA DONKI.
//
// ConnectionString: full URL (e.g. https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json)
// Options:
//   endpoint_type: "swpc_solar_wind" | "donki_cme" | "generic_array"  (default: auto-detect)
//   limit: max rows to load (default: all)
//
// SWPC solar wind JSON: array of objects with fields time_tag, bx_gsm, by_gsm, bz_gsm,
//   speed (or proton_speed), density (or proton_density), temperature, propagated_time_tag
//   Missing/null values → float.NaN (RULE-120: sentinel conversion at load time)
//
// DONKI CME JSON: array of CME events, flattened to startTime, speed, halfAngle, latitude, longitude
//   Missing analyses → float.NaN
//
// generic_array: flattens first-level numeric fields into IDataFrame columns
public sealed class RestApiProvider : IDataSourceProvider
{
    private readonly HttpClient _http;

    public string ProviderName => "rest_api";

    public RestApiProvider(HttpClient? httpClient = null)
    {
        _http = httpClient ?? new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
    }

    public bool CanHandle(DataSourceConfig config) =>
        config.Provider.Equals("rest_api", StringComparison.OrdinalIgnoreCase)
        && !string.IsNullOrWhiteSpace(config.ConnectionString);

    public async Task<DataSchema> DiscoverSchemaAsync(DataSourceConfig config, CancellationToken ct)
    {
        var endpointType = GetEndpointType(config);
        return endpointType switch
        {
            "swpc_solar_wind" => SwpcSolarWindSchema(),
            "donki_cme" => DonkiCmeSchema(),
            _ => await DiscoverGenericSchemaAsync(config.ConnectionString, ct)
        };
    }

    public async Task<IDataFrame> LoadAsync(DataSourceConfig config, DataQuery query, CancellationToken ct)
    {
        var endpointType = GetEndpointType(config);
        int? limit = GetLimit(config, query);

        return endpointType switch
        {
            "swpc_solar_wind" => await LoadSwpcSolarWindAsync(config.ConnectionString, limit, ct),
            "donki_cme" => await LoadDonkiCmeAsync(config.ConnectionString, limit, ct),
            _ => await LoadGenericArrayAsync(config.ConnectionString, limit, ct)
        };
    }

    // ─── SWPC solar wind ──────────────────────────────────────────────────────

    private static DataSchema SwpcSolarWindSchema() => new([
        new ColumnInfo("time_tag", ColumnType.DateTime, IsNullable: false),
        new ColumnInfo("bx_gsm", ColumnType.Float, IsNullable: true),
        new ColumnInfo("by_gsm", ColumnType.Float, IsNullable: true),
        new ColumnInfo("bz_gsm", ColumnType.Float, IsNullable: true),
        new ColumnInfo("speed", ColumnType.Float, IsNullable: true),
        new ColumnInfo("density", ColumnType.Float, IsNullable: true),
        new ColumnInfo("temperature", ColumnType.Float, IsNullable: true),
    ]);

    private async Task<IDataFrame> LoadSwpcSolarWindAsync(string url, int? limit, CancellationToken ct)
    {
        using var response = await _http.GetAsync(url, ct);
        response.EnsureSuccessStatusCode();
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);

        var root = doc.RootElement;
        if (root.ValueKind != JsonValueKind.Array)
            throw new InvalidOperationException(
                $"RestApiProvider: expected JSON array from SWPC endpoint '{url}', got {root.ValueKind}.");

        var schema = SwpcSolarWindSchema();
        var cols = schema.Columns;

        // Skip the timestamp column for float buffers (col index 1+)
        var floatBufs = new List<float>[cols.Count - 1];
        for (int i = 0; i < floatBufs.Length; i++)
            floatBufs[i] = new List<float>();

        int count = 0;
        foreach (var elem in root.EnumerateArray())
        {
            if (limit.HasValue && count >= limit.Value) break;
            ct.ThrowIfCancellationRequested();

            floatBufs[0].Add(ReadFloatOrNaN(elem, "bx_gsm", "Bx_GSM"));
            floatBufs[1].Add(ReadFloatOrNaN(elem, "by_gsm", "By_GSM"));
            floatBufs[2].Add(ReadFloatOrNaN(elem, "bz_gsm", "Bz_GSM"));
            floatBufs[3].Add(ReadFloatOrNaN(elem, "speed", "proton_speed", "v"));
            floatBufs[4].Add(ReadFloatOrNaN(elem, "density", "proton_density", "n"));
            floatBufs[5].Add(ReadFloatOrNaN(elem, "temperature", "proton_temperature", "t"));
            count++;
        }

        // Build schema without timestamp (InMemoryDataFrame is float-only)
        var floatSchema = new DataSchema([
            new ColumnInfo("bx_gsm", ColumnType.Float, IsNullable: true),
            new ColumnInfo("by_gsm", ColumnType.Float, IsNullable: true),
            new ColumnInfo("bz_gsm", ColumnType.Float, IsNullable: true),
            new ColumnInfo("speed", ColumnType.Float, IsNullable: true),
            new ColumnInfo("density", ColumnType.Float, IsNullable: true),
            new ColumnInfo("temperature", ColumnType.Float, IsNullable: true),
        ]);

        float[][] arrays = floatBufs.Select(b => b.ToArray()).ToArray();
        return new InMemoryDataFrame(floatSchema, arrays);
    }

    // ─── DONKI CME catalog ────────────────────────────────────────────────────

    private static DataSchema DonkiCmeSchema() => new([
        new ColumnInfo("speed", ColumnType.Float, IsNullable: true),
        new ColumnInfo("half_angle", ColumnType.Float, IsNullable: true),
        new ColumnInfo("latitude", ColumnType.Float, IsNullable: true),
        new ColumnInfo("longitude", ColumnType.Float, IsNullable: true),
    ]);

    private async Task<IDataFrame> LoadDonkiCmeAsync(string url, int? limit, CancellationToken ct)
    {
        using var response = await _http.GetAsync(url, ct);
        response.EnsureSuccessStatusCode();
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);

        var root = doc.RootElement;
        if (root.ValueKind != JsonValueKind.Array)
            throw new InvalidOperationException(
                $"RestApiProvider: expected JSON array from DONKI endpoint '{url}', got {root.ValueKind}.");

        var speeds = new List<float>();
        var halfAngles = new List<float>();
        var lats = new List<float>();
        var lons = new List<float>();

        int count = 0;
        foreach (var cme in root.EnumerateArray())
        {
            if (limit.HasValue && count >= limit.Value) break;
            ct.ThrowIfCancellationRequested();

            // DONKI: each CME event may have a cmeAnalyses array; take first entry
            if (cme.TryGetProperty("cmeAnalyses", out var analyses)
                && analyses.ValueKind == JsonValueKind.Array
                && analyses.GetArrayLength() > 0)
            {
                var first = analyses[0];
                speeds.Add(ReadFloatOrNaN(first, "speed"));
                halfAngles.Add(ReadFloatOrNaN(first, "halfAngle"));
                lats.Add(ReadFloatOrNaN(first, "latitude"));
                lons.Add(ReadFloatOrNaN(first, "longitude"));
            }
            else
            {
                // Event with no analysis — emit NaN row
                speeds.Add(float.NaN);
                halfAngles.Add(float.NaN);
                lats.Add(float.NaN);
                lons.Add(float.NaN);
            }
            count++;
        }

        var schema = DonkiCmeSchema();
        float[][] arrays = [speeds.ToArray(), halfAngles.ToArray(), lats.ToArray(), lons.ToArray()];
        return new InMemoryDataFrame(schema, arrays);
    }

    // ─── Generic JSON array ───────────────────────────────────────────────────

    private async Task<DataSchema> DiscoverGenericSchemaAsync(string url, CancellationToken ct)
    {
        using var response = await _http.GetAsync(url, ct);
        response.EnsureSuccessStatusCode();
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);

        var root = doc.RootElement;
        if (root.ValueKind != JsonValueKind.Array)
            throw new InvalidOperationException(
                $"RestApiProvider: expected JSON array from '{url}', got {root.ValueKind}.");

        var cols = new List<ColumnInfo>();
        if (root.GetArrayLength() > 0)
        {
            foreach (var prop in root[0].EnumerateObject())
            {
                var colType = prop.Value.ValueKind switch
                {
                    JsonValueKind.Number => ColumnType.Float,
                    JsonValueKind.String => ColumnType.String,
                    _ => ColumnType.Float
                };
                cols.Add(new ColumnInfo(prop.Name, colType, IsNullable: true));
            }
        }
        return new DataSchema(cols);
    }

    private async Task<IDataFrame> LoadGenericArrayAsync(string url, int? limit, CancellationToken ct)
    {
        using var response = await _http.GetAsync(url, ct);
        response.EnsureSuccessStatusCode();
        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);

        var root = doc.RootElement;
        if (root.ValueKind != JsonValueKind.Array)
            throw new InvalidOperationException(
                $"RestApiProvider: expected JSON array from '{url}', got {root.ValueKind}.");

        if (root.GetArrayLength() == 0)
        {
            var emptySchema = new DataSchema([]);
            return new InMemoryDataFrame(emptySchema, []);
        }

        // Collect numeric columns from first element
        var columnNames = new List<string>();
        foreach (var prop in root[0].EnumerateObject())
        {
            if (prop.Value.ValueKind == JsonValueKind.Number
                || prop.Value.ValueKind == JsonValueKind.Null
                || prop.Value.ValueKind == JsonValueKind.True
                || prop.Value.ValueKind == JsonValueKind.False)
                columnNames.Add(prop.Name);
        }

        var bufs = columnNames.Select(_ => new List<float>()).ToArray();
        int count = 0;
        foreach (var elem in root.EnumerateArray())
        {
            if (limit.HasValue && count >= limit.Value) break;
            ct.ThrowIfCancellationRequested();

            for (int i = 0; i < columnNames.Count; i++)
            {
                if (elem.TryGetProperty(columnNames[i], out var val))
                    bufs[i].Add(JsonElementToFloat(val));
                else
                    bufs[i].Add(float.NaN);
            }
            count++;
        }

        var schema = new DataSchema(columnNames
            .Select(n => new ColumnInfo(n, ColumnType.Float, IsNullable: true))
            .ToList());

        float[][] arrays = bufs.Select(b => b.ToArray()).ToArray();
        return new InMemoryDataFrame(schema, arrays);
    }

    // ─── helpers ─────────────────────────────────────────────────────────────

    private static string GetEndpointType(DataSourceConfig config)
    {
        if (config.Options != null
            && config.Options.TryGetValue("endpoint_type", out var et))
            return et;

        // Auto-detect from URL
        var url = config.ConnectionString;
        if (url.Contains("swpc.noaa.gov") || url.Contains("rtsw"))
            return "swpc_solar_wind";
        if (url.Contains("donki") || url.Contains("CME"))
            return "donki_cme";
        return "generic_array";
    }

    private static int? GetLimit(DataSourceConfig config, DataQuery query)
    {
        if (query.Limit.HasValue) return query.Limit;
        if (config.Options != null
            && config.Options.TryGetValue("limit", out var lStr)
            && int.TryParse(lStr, out var lInt))
            return lInt;
        return null;
    }

    // Read a float from one of several candidate property names.
    // Null JSON values → float.NaN (RULE-120: sentinel conversion at load time).
    private static float ReadFloatOrNaN(JsonElement elem, params string[] candidates)
    {
        foreach (var name in candidates)
        {
            if (!elem.TryGetProperty(name, out var val)) continue;
            return JsonElementToFloat(val);
        }
        return float.NaN;
    }

    // Convert JsonElement to float; null/missing → NaN (RULE-120).
    private static float JsonElementToFloat(JsonElement val) => val.ValueKind switch
    {
        JsonValueKind.Number => val.TryGetSingle(out var f) ? f : (float)val.GetDouble(),
        JsonValueKind.Null => float.NaN,
        JsonValueKind.String when val.GetString() is { } s && float.TryParse(s, out var pf) => pf,
        JsonValueKind.String => float.NaN, // non-numeric string → NaN
        _ => float.NaN
    };
}
