using SolarPipe.Core.Models;

namespace SolarPipe.Core.Interfaces;

public interface ITrainedModel
{
    string ModelId { get; }
    string StageName { get; }
    ModelMetrics Metrics { get; }

    Task<PredictionResult> PredictAsync(IDataFrame input, CancellationToken ct);
    Task SaveAsync(string path, CancellationToken ct);
    Task LoadAsync(string path, CancellationToken ct);
}
