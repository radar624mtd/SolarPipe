using SolarPipe.Config.Models;
using SolarPipe.Core.Models;
using YamlDotNet.Core;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace SolarPipe.Config;

public sealed class DomainSweepConfigLoader
{
    private readonly IDeserializer _deserializer = new DeserializerBuilder()
        .WithNamingConvention(UnderscoredNamingConvention.Instance)
        .WithTypeConverter(new Yaml12BooleanConverter())
        .Build();

    public async Task<DomainSweepConfig> LoadAsync(string path, CancellationToken ct = default)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Domain sweep config not found: {path}");

        var yaml = await File.ReadAllTextAsync(path, ct);
        return LoadFromString(yaml);
    }

    public DomainSweepConfig LoadFromString(string yaml)
    {
        DomainSweepConfig config;
        try
        {
            config = _deserializer.Deserialize<DomainSweepConfig>(yaml)
                ?? throw new InvalidOperationException("YAML deserialization returned null.");
        }
        catch (YamlException ex)
        {
            throw new InvalidOperationException(
                $"Domain sweep YAML parse error at {ex.Start}: {ex.Message}", ex);
        }

        Validate(config);
        return config;
    }

    // Converts YAML model → core StageConfig for a named stage.
    public StageConfig ToStageConfig(DomainSweepConfig config, string stageName)
    {
        if (!config.Stages.TryGetValue(stageName, out var s))
            throw new InvalidOperationException(
                $"Stage '{stageName}' not found in domain sweep config.");

        return new StageConfig(stageName, s.Framework, s.ModelType, string.Empty,
            s.Features, s.Target, s.Hyperparameters);
    }

    private static void Validate(DomainSweepConfig config)
    {
        var meta = config.DomainPipeline;

        if (string.IsNullOrWhiteSpace(meta.Name))
            throw new InvalidOperationException("Domain sweep: domain_pipeline.name is required.");

        if (string.IsNullOrWhiteSpace(meta.DataSource.ConnectionString))
            throw new InvalidOperationException("Domain sweep: domain_pipeline.data_source.connection_string is required.");

        if (string.IsNullOrWhiteSpace(meta.DataSource.Table))
            throw new InvalidOperationException("Domain sweep: domain_pipeline.data_source.table is required.");

        if (meta.Cv.Folds < 2)
            throw new InvalidOperationException(
                $"Domain sweep: cv.folds must be >= 2, got {meta.Cv.Folds}.");

        var knownStages = config.Stages.Keys.ToHashSet(StringComparer.OrdinalIgnoreCase);

        ValidateDomainGroup("origination", meta.Origination, knownStages, singleStage: true);
        ValidateDomainGroup("transit",     meta.Transit,     knownStages, singleStage: false);
        ValidateDomainGroup("impact",      meta.Impact,      knownStages, singleStage: false);

        if (meta.MetaLearner.Stages.Count == 0)
            throw new InvalidOperationException("Domain sweep: meta_learner.stages must have at least one output.");

        foreach (var (output, stageName) in meta.MetaLearner.Stages)
        {
            if (!knownStages.Contains(stageName))
                throw new InvalidOperationException(
                    $"Domain sweep: meta_learner output '{output}' references unknown stage '{stageName}'.");
        }

        foreach (var (stageName, stage) in config.Stages)
        {
            if (string.IsNullOrWhiteSpace(stage.Framework))
                throw new InvalidOperationException($"Stage '{stageName}': framework is required.");
            if (stage.Features.Count == 0)
                throw new InvalidOperationException($"Stage '{stageName}': features list is empty.");
            if (string.IsNullOrWhiteSpace(stage.Target))
                throw new InvalidOperationException($"Stage '{stageName}': target is required.");
        }
    }

    private static void ValidateDomainGroup(
        string domainName,
        DomainGroupConfig group,
        HashSet<string> knownStages,
        bool singleStage)
    {
        if (string.IsNullOrWhiteSpace(group.Target))
            throw new InvalidOperationException(
                $"Domain sweep: {domainName}.target is required.");

        if (singleStage)
        {
            if (string.IsNullOrWhiteSpace(group.Stage))
                throw new InvalidOperationException(
                    $"Domain sweep: {domainName}.stage is required.");
            if (!knownStages.Contains(group.Stage!))
                throw new InvalidOperationException(
                    $"Domain sweep: {domainName}.stage '{group.Stage}' is not defined in stages.");
        }
        else
        {
            if (string.IsNullOrWhiteSpace(group.PhysicsStage))
                throw new InvalidOperationException(
                    $"Domain sweep: {domainName}.physics_stage is required.");
            if (string.IsNullOrWhiteSpace(group.ResidualStage))
                throw new InvalidOperationException(
                    $"Domain sweep: {domainName}.residual_stage is required.");
            if (!knownStages.Contains(group.PhysicsStage!))
                throw new InvalidOperationException(
                    $"Domain sweep: {domainName}.physics_stage '{group.PhysicsStage}' is not defined.");
            if (!knownStages.Contains(group.ResidualStage!))
                throw new InvalidOperationException(
                    $"Domain sweep: {domainName}.residual_stage '{group.ResidualStage}' is not defined.");
        }
    }
}
