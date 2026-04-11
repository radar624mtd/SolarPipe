using SolarPipe.Config.Models;
using YamlDotNet.Core;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace SolarPipe.Config;

public class SweepConfigLoader
{
    private readonly IDeserializer _deserializer = new DeserializerBuilder()
        .WithNamingConvention(UnderscoredNamingConvention.Instance)
        .WithTypeConverter(new Yaml12BooleanConverter())
        .Build();

    public async Task<SweepConfig> LoadAsync(string path, CancellationToken ct = default)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Sweep config not found: {path}");

        var yaml = await File.ReadAllTextAsync(path, ct);
        return LoadFromString(yaml);
    }

    public SweepConfig LoadFromString(string yaml)
    {
        SweepConfig config;
        try
        {
            config = _deserializer.Deserialize<SweepConfig>(yaml)
                ?? throw new InvalidOperationException("YAML deserialization returned null.");
        }
        catch (YamlException ex)
        {
            throw new InvalidOperationException(
                $"Sweep YAML parse error at {ex.Start}: {ex.Message}", ex);
        }

        Validate(config);
        return config;
    }

    private static void Validate(SweepConfig config)
    {
        if (string.IsNullOrWhiteSpace(config.Sweep.Name))
            throw new InvalidOperationException("Sweep config: sweep.name is required.");

        if (config.Sweep.Hypotheses.Count == 0)
            throw new InvalidOperationException("Sweep config: at least one hypothesis is required.");

        if (config.Stages.Count == 0)
            throw new InvalidOperationException("Sweep config: at least one stage is required.");

        if (config.Sweep.Cv.Folds < 2)
            throw new InvalidOperationException(
                $"Sweep config: cv.folds must be >= 2, got {config.Sweep.Cv.Folds}.");

        var knownStages = config.Stages.Keys.ToHashSet(StringComparer.OrdinalIgnoreCase);

        foreach (var h in config.Sweep.Hypotheses)
        {
            if (string.IsNullOrWhiteSpace(h.Id))
                throw new InvalidOperationException("Sweep config: each hypothesis must have an id.");

            if (string.IsNullOrWhiteSpace(h.Compose))
                throw new InvalidOperationException(
                    $"Sweep config: hypothesis '{h.Id}' is missing a compose expression.");

            if (h.Stages.Count == 0)
                throw new InvalidOperationException(
                    $"Sweep config: hypothesis '{h.Id}' has no stages.");

            foreach (var s in h.Stages)
            {
                if (!knownStages.Contains(s))
                    throw new InvalidOperationException(
                        $"Sweep config: hypothesis '{h.Id}' references stage '{s}' " +
                        $"which is not defined in stages. Known: [{string.Join(", ", knownStages)}].");
            }
        }

        foreach (var (stageName, stage) in config.Stages)
        {
            if (string.IsNullOrWhiteSpace(stage.Framework))
                throw new InvalidOperationException(
                    $"Sweep stage '{stageName}': framework is required.");

            if (stage.Features.Count == 0)
                throw new InvalidOperationException(
                    $"Sweep stage '{stageName}': features list is empty.");

            if (string.IsNullOrWhiteSpace(stage.Target))
                throw new InvalidOperationException(
                    $"Sweep stage '{stageName}': target is required.");
        }
    }
}
