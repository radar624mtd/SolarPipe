using SolarPipe.Core.Models;

namespace SolarPipe.Core.Interfaces;

public interface IModelRegistry
{
    Task RegisterAsync(ModelArtifact artifact, ITrainedModel model, CancellationToken ct);
    Task<ITrainedModel> LoadAsync(string modelId, string version, CancellationToken ct);
    Task<IReadOnlyList<ModelArtifact>> ListAsync(string? stageName = null, CancellationToken ct = default);
}
