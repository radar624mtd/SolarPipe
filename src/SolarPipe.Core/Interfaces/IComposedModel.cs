using SolarPipe.Core.Models;

namespace SolarPipe.Core.Interfaces;

public interface IComposedModel
{
    string Name { get; }
    Task<PredictionResult> PredictAsync(IDataFrame input, CancellationToken ct);
}
