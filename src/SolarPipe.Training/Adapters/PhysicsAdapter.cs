using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Training.Physics;

namespace SolarPipe.Training.Adapters;

// PhysicsAdapter dispatches to registered IPhysicsEquation implementations.
// "Training" for a physics model extracts hyperparameters from StageConfig — no ML fitting.
// RULE-150: No CPU-bound work here; physics "training" is instant parameter extraction.
public sealed class PhysicsAdapter : IFrameworkAdapter
{
    private readonly IReadOnlyDictionary<string, IPhysicsEquation> _equations;

    public FrameworkType FrameworkType => FrameworkType.Physics;

    public IReadOnlyList<string> SupportedModels { get; }

    public PhysicsAdapter()
    {
        var dragModel = new DragEquation();
        var registry = new Dictionary<string, IPhysicsEquation>(StringComparer.OrdinalIgnoreCase)
        {
            ["DragBased"] = dragModel,
        };
        _equations = registry;
        SupportedModels = ["DragBased", "DragBasedV2", "BurtonOde", "NewellCoupling"];
    }

    public Task<ITrainedModel> TrainAsync(
        StageConfig config,
        IDataFrame trainingData,
        IDataFrame? validationData,
        CancellationToken ct)
    {
        // Physics "training" = extract and validate hyperparameters from config
        ITrainedModel model = config.ModelType switch
        {
            var t when t.Equals("DragBased", StringComparison.OrdinalIgnoreCase)
                => new DragBasedModel(config),
            var t when t.Equals("DragBasedV2", StringComparison.OrdinalIgnoreCase)
                => new DragBasedModelWithArrivalSpeed(config),
            var t when t.Equals("BurtonOde", StringComparison.OrdinalIgnoreCase)
                => new BurtonOde(config),
            var t when t.Equals("NewellCoupling", StringComparison.OrdinalIgnoreCase)
                => new NewellCoupling(config),
            _ => throw new NotSupportedException(
                $"PhysicsAdapter does not support model type '{config.ModelType}'. " +
                $"Supported: [{string.Join(", ", SupportedModels)}] (stage={config.Name}).")
        };

        return Task.FromResult(model);
    }
}
