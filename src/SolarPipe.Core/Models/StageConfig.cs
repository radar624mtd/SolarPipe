namespace SolarPipe.Core.Models;

public record StageConfig(
    string Name,
    string Framework,
    string ModelType,
    string DataSource,
    IReadOnlyList<string> Features,
    string Target,
    IReadOnlyDictionary<string, object>? Hyperparameters = null,
    string? MockDataStrategy = null);
