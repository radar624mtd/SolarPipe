using SolarPipe.Core.Models;

namespace SolarPipe.Training.Physics;

// IPhysicsEquation implementation wrapping DragBasedModel ODE for direct use.
// Separates the equation logic from ITrainedModel interface concerns.
public sealed class DragEquation : IPhysicsEquation
{
    public string Name => "DragBased";

    public IReadOnlyList<string> SupportedModels => ["DragBased"];

    public Task<PhysicsResult> ComputeAsync(PhysicsInput input, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var (_, arrivalHours) = DragBasedModel.RunOde(
            input.InitialSpeedKmPerSec,
            input.AmbientWindSpeedKmPerSec,
            input.DragGammaKmInv,
            input.StartDistanceSolarRadii,
            input.TargetDistanceSolarRadii);

        // No ML metrics for a pure physics model
        var metrics = new ModelMetrics(0.0, 0.0, 0.0);
        return Task.FromResult(new PhysicsResult(arrivalHours, 0.0, metrics));
    }
}
