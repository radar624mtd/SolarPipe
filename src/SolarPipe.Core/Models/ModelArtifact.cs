namespace SolarPipe.Core.Models;

public record ModelArtifact
{
    public required string ModelId { get; init; }
    public required string Version { get; init; }
    public required string StageName { get; init; }
    public required StageConfig Config { get; init; }
    public required ModelMetrics Metrics { get; init; }
    public required string DataFingerprint { get; init; }
    public required DateTime TrainedAt { get; init; }
    public required string ArtifactPath { get; init; }
}
