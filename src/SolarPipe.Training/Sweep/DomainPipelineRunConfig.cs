using SolarPipe.Core.Models;

namespace SolarPipe.Training.Sweep;

// Config records passed to DomainPipelineSweep — lives in Training (not Core) because
// this pipeline is Phase 8-specific and doesn't need Core-layer visibility.
public sealed record DomainPipelineRunConfig(
    string Name,
    int Folds,
    int GapBufferDays,
    int MinTestEvents,
    DomainGroupRunConfig Origination,
    DomainGroupRunConfig Transit,
    DomainGroupRunConfig Impact,
    IReadOnlyDictionary<string, string> MetaLearnerStages,  // output_name → stage_name
    IReadOnlyDictionary<string, StageConfig> Stages,
    // When set, bypasses CV and does a single train/test split at this UTC date (ISO-8601).
    // All rows before this date are training; rows on/after are test.
    string? HeldOutAfter = null);

// Per-domain config — either single-stage (origination) or physics+residual (transit/impact).
public sealed record DomainGroupRunConfig(
    string Target,
    string? Stage,           // single-stage domain (origination)
    string? PhysicsStage,    // physics baseline stage
    string? ResidualStage);  // ML residual stage

// Per-event prediction row for detailed evaluation.
public sealed record EventPrediction(
    float LaunchTimeUnix,   // Unix seconds (from launch_time column)
    float ObsTransit,
    float PredTransit,
    float ObsDst,
    float PredDst,
    float PhysicsTransit);  // DragBased-only prediction

// Per-fold results for one domain output.
public sealed record DomainFoldMetrics(
    int FoldIndex,
    double MaeTransit,
    double MaeDst,
    double MaeDuration,
    double MaePhysicsBaseline,
    IReadOnlyList<EventPrediction>? EventPredictions = null);

// Aggregated result across all folds.
public sealed record DomainPipelineResult(
    string PipelineName,
    IReadOnlyList<DomainFoldMetrics> FoldMetrics,
    double MeanMaeTransit,
    double MeanMaeDst,
    double MeanMaeDuration,
    double MeanMaePhysicsBaseline);
