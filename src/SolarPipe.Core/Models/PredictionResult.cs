namespace SolarPipe.Core.Models;

public record PredictionResult(
    float[] Values,
    float[]? LowerBound,
    float[]? UpperBound,
    string ModelId,
    DateTime GeneratedAt,
    IReadOnlyDictionary<string, object>? Metadata = null);
