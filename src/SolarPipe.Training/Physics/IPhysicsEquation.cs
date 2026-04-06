using SolarPipe.Core.Models;

namespace SolarPipe.Training.Physics;

public interface IPhysicsEquation
{
    string Name { get; }
    IReadOnlyList<string> SupportedModels { get; }
    Task<PhysicsResult> ComputeAsync(PhysicsInput input, CancellationToken ct);
}

public record PhysicsInput(
    double InitialSpeedKmPerSec,
    double AmbientWindSpeedKmPerSec,
    double DragGammaKmInv,
    double StartDistanceSolarRadii,
    double TargetDistanceSolarRadii);

public record PhysicsResult(
    double ArrivalTimeHours,
    double ArrivalSpeedKmPerSec,
    ModelMetrics Metrics);
