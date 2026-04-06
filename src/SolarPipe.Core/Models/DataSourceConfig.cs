namespace SolarPipe.Core.Models;

public record DataSourceConfig(
    string Name,
    string Provider,
    string ConnectionString,
    IReadOnlyDictionary<string, string>? Options = null);

public record DataQuery(
    string? Table = null,
    string? Sql = null,
    IReadOnlyDictionary<string, object>? Parameters = null,
    int? Limit = null);
