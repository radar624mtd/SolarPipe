using SolarPipe.Core.Models;

namespace SolarPipe.Core.Interfaces;

public interface IDataSourceProvider
{
    string ProviderName { get; }
    bool CanHandle(DataSourceConfig config);
    Task<DataSchema> DiscoverSchemaAsync(DataSourceConfig config, CancellationToken ct);
    Task<IDataFrame> LoadAsync(DataSourceConfig config, DataQuery query, CancellationToken ct);
}
