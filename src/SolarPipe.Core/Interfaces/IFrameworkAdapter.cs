using SolarPipe.Core.Models;

namespace SolarPipe.Core.Interfaces;

public interface IFrameworkAdapter
{
    FrameworkType FrameworkType { get; }
    IReadOnlyList<string> SupportedModels { get; }
    Task<ITrainedModel> TrainAsync(
        StageConfig config,
        IDataFrame trainingData,
        IDataFrame? validationData,
        CancellationToken ct);
}
