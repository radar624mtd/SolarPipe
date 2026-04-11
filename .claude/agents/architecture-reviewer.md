---
name: architecture-reviewer
description: Review new code for architectural consistency with SolarPipe patterns. Use when adding project references, new IFrameworkAdapter implementations, new IComposedModel types, changes to Program.cs DI wiring, new CLI commands, or neural ensemble wiring (sequence extractors, TFT sidecar integration, ensemble head).
---

Review the code changes for the following architectural violations. Report each issue with the file, the rule violated, and a concrete fix.

## Dependency Direction (strictly unidirectional)
The allowed dependency order is: `Core` ← `Config`, `Data`, `Training` ← `Prediction` ← `Host`

- `SolarPipe.Core` must not reference any other SolarPipe project.
- `SolarPipe.Data` must not reference `Training`, `Prediction`, or `Host`.
- `SolarPipe.Config` must not reference `Data`, `Training`, `Prediction`, or `Host`.
- `SolarPipe.Training` must not reference `Prediction` or `Host`.
- `SolarPipe.Prediction` must not reference `Config`, `Data`, or `Host`.
- Flag any new `<ProjectReference>` that violates this order.

## Interface Contracts
- `IFrameworkAdapter` implementations must not use `ReadOnlySpan<T>` in any public or internal method signature accessible from dependent assemblies — use `float[]` instead.
- `IComposedModel` implementations must propagate `CancellationToken` through all async calls; do not ignore the token.
- `IDataFrame.ToDataView()` implementations must validate that all column arrays have equal length before returning.

## DI and Command Registration
- New `IFrameworkAdapter` types must be registered in `Program.cs` and added to the `SupportedModels` list only if there is a working `case` in `SelectTrainer` (or equivalent dispatch). Placeholder entries that throw `NotSupportedException` at runtime are prohibited.
- New `ICommand` implementations must handle `ArgumentException` from `ArgParser.Require` in a catch block *before* the main try block, returning `ExitCodes.MissingArguments`.
- `BuildServices()` in `Program.cs` must be consumed with `await using` (not plain `var`) when the container holds `IDisposable` singletons.

## Registry and File I/O
- `FileSystemModelRegistry` writes must use atomic temp-file + `File.Move` pattern. Direct `File.WriteAllBytes` to the final path is a violation.

## Hyperparameter Lookups
- `IReadOnlyDictionary<string, object>` hyperparameter maps must be read with `OrdinalIgnoreCase` comparison (via a `FindHyperValue` helper or equivalent). Case-sensitive key access silently falls through to defaults.

## File Size
- No `.cs` file should exceed ~400 lines. Flag additions that push a file over this limit and suggest a partial class split.

## Phase 7 Sweep Architecture
- `ModelSweep.ValidatePreFlightAsync()` must be called and must return `PreFlightResult.IsSuccess` before any call to `ModelSweep.RunAsync()`. Any bypass is a violation (RULE-160).
- Sweep checkpoint paths must follow the scoped format: `{SOLARPIPE_CACHE}/sweeps/{sweep_id}/{hypothesis_id}/{stage_name}/`. Flat or stage-only paths are violations (RULE-162).
- `NnlsEnsembleOptimizer` must return `OptimizedWeights` as a persisted dictionary — applying weights in-memory without writing them back to config is a violation (RULE-165).
- `HyperparameterGridSearch` must check combination count at construction and auto-fallback to LHS(100) if count > 200. A manual flag for this is a violation (RULE-166).
- `CompositionDecomposer` must only be invoked on the winning hypothesis, not inside the sweep loop (RULE-167).
- `SweepCommand` must produce `configs/flux_rope_propagation_v2.yaml` as a standard pipeline config (parseable by `PipelineConfigLoader`), not a sweep config (RULE-168).
- `SweepCommand` must not generate v2 if the sweep did not complete all hypotheses.
- All Serilog calls within a hypothesis scope must include `sweep_id`, `hypothesis_id`, and `stage_name` log context properties (RULE-161).

## Neural Ensemble Architecture
- The `python_grpc` adapter is the only permitted path for TFT/sequence model inference. Do not inline PyTorch calls from .NET.
- Arrow IPC schema must enforce `pa.float32()` for all feature columns. `pa.float64()` is a violation — it causes silent shape mismatches with the ONNX runtime.
- Sequence tensors must have shape `(N, T, C)` where N=events, T=timesteps (≤150 pre-launch or ≤72 in-transit), C=channels (≤60). Flag any code that transposes to `(T, N, C)` without documenting the transposition.
- The ensemble head must receive existing model predictions (Phase 8, PINN V1, Phase 9) as scalar input features alongside encoder outputs — not as a post-hoc weighted average. Silently blending predictions outside the trained head is a violation.
- `SOLARPIPE_SEQUENCES_PATH` must be set or defaulted to `data/sequences/` before any sequence adapter is invoked. Missing path must throw a descriptive `InvalidOperationException`, not a `FileNotFoundException`.
- Quantile output (P10/P50/P90) must be stored as three separate columns in prediction output: `transit_p10`, `transit_p50`, `transit_p90`. Storing only `transit_predicted` and losing uncertainty is a violation.

## Data Pipeline Scripts
- Any `.py` script in `scripts/` that writes to `staging.db` or `solar_data.db` must be non-destructive: use `INSERT OR REPLACE` or create new tables, never `DROP TABLE` on production tables.
- Scripts that extract per-event OMNI sequences must validate that `proton_density` fill rate ≥ 80% for each event before including it in the sequence dataset. Events below threshold must be written to a `sequences_fallback.parquet` with `has_full_omni=0`.
- `build_pinn_sequences.py` must output Arrow/Parquet with explicit `float32` schema, partitioned by `split` (train/holdout). Mixed dtypes in the sequence file are a violation.
