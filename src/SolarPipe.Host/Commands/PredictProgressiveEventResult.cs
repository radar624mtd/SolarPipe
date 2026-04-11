using SolarPipe.Training.Physics;

namespace SolarPipe.Host.Commands;

// Phase 9 M3: per-event result record for predict-progressive.
// Lifted out of PredictProgressiveCommand so PredictProgressiveOutput can
// reference it without the command exposing a public nested type.
internal sealed record PredictProgressiveEventResult(
    string ActivityId,
    DateTime LaunchTime,
    double InitialSpeedKmPerSec,
    double? ObservedTransitHours,
    double? ArrivalTimeHours,
    double? ErrorHours,
    string TerminationReason,
    bool ShockArrived,
    int NMissingHours,
    double DensityCoverage,
    double GammaEffMean,
    double GammaEffFinal,
    ProgressiveTrajectory Trajectory);
