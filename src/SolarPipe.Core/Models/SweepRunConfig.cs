namespace SolarPipe.Core.Models;

// Sweep configuration passed to ModelSweep — lives in Core so Training can reference it.
// Populated from SweepConfig (in SolarPipe.Config) at the Host layer.
public sealed record SweepRunConfig(
    string Name,
    bool Parallel,
    int Folds,
    int GapBufferDays,
    int MinTestEvents,
    IReadOnlyList<SweepHypothesis> Hypotheses,
    IReadOnlyDictionary<string, StageConfig> Stages);

public sealed record SweepHypothesis(
    string Id,
    string ComposeExpression,
    IReadOnlyList<string> Stages);
