using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Data;

public sealed class DataSourceRegistry
{
    private readonly Dictionary<string, IDataSourceProvider> _providers = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, DataSourceConfig> _sources = new(StringComparer.OrdinalIgnoreCase);

    public void RegisterProvider(IDataSourceProvider provider)
    {
        if (provider is null) throw new ArgumentNullException(nameof(provider));
        _providers[provider.ProviderName] = provider;
    }

    public void RegisterSource(DataSourceConfig config)
    {
        if (config is null) throw new ArgumentNullException(nameof(config));
        _sources[config.Name] = config;
    }

    public async Task<IDataFrame> LoadAsync(string sourceName, DataQuery query, CancellationToken ct)
    {
        if (!_sources.TryGetValue(sourceName, out var config))
            throw new KeyNotFoundException(
                $"DataSourceRegistry: no data source named '{sourceName}'. " +
                $"Registered sources: [{string.Join(", ", _sources.Keys)}].");

        var provider = _providers.Values.FirstOrDefault(p => p.CanHandle(config))
            ?? throw new InvalidOperationException(
                $"DataSourceRegistry: no provider can handle source '{sourceName}' " +
                $"(provider='{config.Provider}'). " +
                $"Registered providers: [{string.Join(", ", _providers.Keys)}].");

        return await provider.LoadAsync(config, query, ct);
    }

    public async Task<DataSchema> DiscoverSchemaAsync(string sourceName, CancellationToken ct)
    {
        if (!_sources.TryGetValue(sourceName, out var config))
            throw new KeyNotFoundException(
                $"DataSourceRegistry: no data source named '{sourceName}'.");

        var provider = _providers.Values.FirstOrDefault(p => p.CanHandle(config))
            ?? throw new InvalidOperationException(
                $"DataSourceRegistry: no provider can handle source '{sourceName}' " +
                $"(provider='{config.Provider}').");

        return await provider.DiscoverSchemaAsync(config, ct);
    }
}
