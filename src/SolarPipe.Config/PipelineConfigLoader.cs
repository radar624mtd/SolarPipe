using System.Reflection;
using SolarPipe.Config.Models;
using YamlDotNet.Core;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace SolarPipe.Config;

public class PipelineConfigLoader
{
    private readonly IDeserializer _deserializer = new DeserializerBuilder()
        .WithNamingConvention(UnderscoredNamingConvention.Instance)
        .WithTypeConverter(new Yaml12BooleanConverter())  // RULE-020: YAML 1.2 booleans
        .Build();

    public async Task<PipelineConfig> LoadAsync(string path, CancellationToken ct = default)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Pipeline config not found: {path}");

        var yaml = await File.ReadAllTextAsync(path, ct);
        return LoadFromString(yaml);
    }

    public PipelineConfig LoadFromString(string yaml)
    {
        PipelineConfig config;
        try
        {
            config = _deserializer.Deserialize<PipelineConfig>(yaml)
                ?? throw new InvalidOperationException("YAML deserialization returned null. The file may be empty.");
        }
        catch (YamlException ex)
        {
            throw new InvalidOperationException(
                $"YAML parse error at {ex.Start}: {ex.Message}", ex);
        }

        // RULE-021: Reject unexpected nulls on non-nullable properties
        ValidateNoUnexpectedNulls(config, nameof(PipelineConfig));
        foreach (var (key, ds) in config.DataSources)
            ValidateNoUnexpectedNulls(ds, $"DataSources[{key}]");
        foreach (var (key, stage) in config.Stages)
            ValidateNoUnexpectedNulls(stage, $"Stages[{key}]");

        ValidateReferences(config);

        return config;
    }

    // RULE-021: Walks non-nullable reference-type properties and throws if null after
    // deserialization. YamlDotNet silently assigns null to non-nullable properties
    // when the YAML key is missing or has an unquoted null literal (GitHub #763).
    private static void ValidateNoUnexpectedNulls(object obj, string path)
    {
        var ctx = new NullabilityInfoContext();
        foreach (var prop in obj.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance))
        {
            if (!prop.CanRead) continue;
            var value = prop.GetValue(obj);
            if (value is not null) continue;

            var info = ctx.Create(prop);
            if (info.WriteState == NullabilityState.NotNull)
                throw new InvalidOperationException(
                    $"Required property '{path}.{prop.Name}' is null after YAML deserialization. " +
                    $"Ensure the key is present and not an unquoted null literal.");
        }
    }

    private static void ValidateReferences(PipelineConfig config)
    {
        foreach (var (stageName, stage) in config.Stages)
        {
            if (!config.DataSources.ContainsKey(stage.DataSource))
                throw new ArgumentException(
                    $"Stage '{stageName}' references data source '{stage.DataSource}' " +
                    $"which is not defined in data_sources. " +
                    $"Available: {string.Join(", ", config.DataSources.Keys)}");

            if (stage.Features.Count == 0)
                throw new ArgumentException(
                    $"Stage '{stageName}' has no features defined.");
        }
    }
}
