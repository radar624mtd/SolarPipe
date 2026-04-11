# Phase 7 Development Rules (RULE-160 – RULE-168)

These rules extend the base rule set for Phase 7 implementation. All prior rules remain in effect.

---

## RULE-160: Pre-Flight Gate Is Mandatory and Atomic

**What:** The sweep pre-flight validation gate must run to completion before any hypothesis
training begins. Every check must pass. The gate is all-or-nothing — no partial sweep starts.

**Why:** Parallel hypotheses share framework adapters and data sources. A mid-sweep
discovery that the sidecar is down (H7) or a data file is corrupt means wasted compute
on all other hypotheses and a partially-written checkpoint tree that must be manually cleaned.

**How to apply:**
- `SweepCommand` calls `ModelSweep.ValidatePreFlightAsync()` before `RunAsync()`
- `ValidatePreFlightAsync()` returns `PreFlightResult` with a list of failures
- If `PreFlightResult.HasFailures`, log each failure as `[preflight:FAIL component=X]` and return `ExitCodes.PreFlightFailed`
- No hypothesis training starts unless `PreFlightResult.IsSuccess`

---

## RULE-161: Each Hypothesis Gets an Isolated Log Tag

**What:** Every log entry produced during a sweep run must include `sweep_id`, `hypothesis_id`,
and `stage_name` in its structured JSON payload. Plain log lines without these fields are prohibited
during sweep execution.

**Why:** Parallel hypothesis runs interleave in `logs/dotnet_latest.json`. Without tags, failures
are unattributable and debugging requires reconstructing execution order from timestamps.

**How to apply:**
- Inject a `SweepLogContext` (record: `SweepId`, `HypothesisId`, `StageName`) into each hypothesis scope
- All Serilog calls within a hypothesis scope use `Log.ForContext(ctx)` or push to `LogContext.PushProperty`
- Format: `[sweep:{HypothesisId} stage:{StageName}]` as the log message prefix
- Python sidecar log entries must include `sweep_id` and `hypothesis_id` fields (passed as gRPC metadata)

---

## RULE-162: Sweep Checkpoints Are Hypothesis-Scoped

**What:** Checkpoint paths for sweep runs must include the sweep ID and hypothesis ID:
`{SOLARPIPE_CACHE}/sweeps/{sweep_id}/{hypothesis_id}/{stage_name}/`

**Why:** Multiple sweep runs (e.g., grid search iterations) and multiple hypotheses must not
share checkpoint namespaces. A resumed sweep must restore the correct hypothesis checkpoint,
not a checkpoint from a different hypothesis that happens to share a stage name.

**How to apply:**
- `ModelSweep` generates a deterministic `sweep_id` from the sweep config hash at start
- `CheckpointManager` receives the full scoped path; it does not construct paths itself during sweeps
- `--resume` flag on `SweepCommand` reuses the existing `sweep_id`; `--fresh` clears and regenerates

---

## RULE-163: Metrics Must Be Computed Per Fold Before Aggregation

**What:** `ComprehensiveMetricsEvaluator` must compute all 10 metrics independently on each
CV fold's predictions. Aggregation (mean ± std) happens only after all folds complete.
Pooling predictions across folds before metric computation is prohibited.

**Why:** Pooling inflates effective sample size and masks fold-to-fold variance, which is
critical information for the deferred significance testing decision. Per-fold metrics preserve
the variance structure needed later.

**How to apply:**
- `ComprehensiveMetricsEvaluator.EvaluateFoldAsync(predictions, observed)` returns `FoldMetrics`
- `AggregateFolds(IReadOnlyList<FoldMetrics>)` computes mean and std per metric
- Skill score vs DBM is computed per fold using that fold's DBM baseline MAE from scoreboard

---

## RULE-164: Calibration Fold Is Strictly Held Out

**What:** The last CV fold is reserved exclusively for ensemble weight optimization (NNLS).
It must not be used in training any stage, in feature importance computation, or in
hyperparameter grid search evaluation.

**Why:** Using the calibration fold in any upstream computation leaks information into
the weight optimization step, producing overfit weights that won't generalize.

**How to apply:**
- `ExpandingWindowCV` splits into `TrainingFolds` (1–4) and `CalibrationFold` (5)
- `HyperparameterGridSearch` evaluates on `TrainingFolds` only
- `FeatureImportanceAnalyzer` runs on `TrainingFolds` only
- `NnlsEnsembleOptimizer` receives only `CalibrationFold` predictions

---

## RULE-165: NNLS Optimizer Writes Weights Back to Config

**What:** After `NnlsEnsembleOptimizer` solves for optimal weights, the result must be
written back to the sweep config output as explicit `weights:` entries. The optimizer
must not silently apply weights in memory without persisting them.

**Why:** Reproducibility. If the sweep is re-run or the winning hypothesis is promoted
to v2 config, the weights must be identical. Implicit in-memory weights are unauditable.

**How to apply:**
- `NnlsEnsembleOptimizer` returns `OptimizedWeights: IReadOnlyDictionary<string, float>`
- `SweepCommand` writes these into the hypothesis's output config section
- The v2 config generated at phase end includes explicit `weights:` under ensemble stage definitions
- Weights are also logged as `[sweep:HN stage:ensemble] weights={...}`

---

## RULE-166: Grid Search Respects Wall-Clock Budget via Latin Hypercube Fallback

**What:** If the full hyperparameter grid exceeds 200 combinations for a single stage,
`HyperparameterGridSearch` automatically falls back to a Latin hypercube sample of 100
combinations. Full grid and LHS paths must produce identical output schemas.

**Why:** A 2,000-combination full grid across all parameters is impractical on a 661-row
dataset with 5-fold CV. LHS preserves parameter space coverage while keeping the sweep
tractable. The fallback must be automatic, not a manual flag.

**How to apply:**
- `HyperparameterGridSearch` counts total combinations at construction
- If count > 200, switch to `LatinHypercubeSampler(n=100, seed=42)`
- Log the fallback: `[grid:LHS combinations=100 reason=grid_too_large]`
- Output schema is identical: `GridSearchResult` with `(Hyperparameters, FoldMetrics)[]`

---

## RULE-167: Composition Decomposition Runs Only on Winning Hypothesis

**What:** `CompositionDecomposer` runs post-selection, on the winning hypothesis only,
against the full training set and the external validation set. It does not run on all
hypotheses during the sweep.

**Why:** Decomposition requires re-running each stage independently (bypassing composition),
which doubles training cost per hypothesis. Applying it to all 7 hypotheses is wasteful;
the actionable output is only needed for the model being promoted to production.

**How to apply:**
- `SweepCommand` selects winner by minimum mean MAE across training folds
- Only then invokes `CompositionDecomposer.DecomposeAsync(winnerHypothesis, fullData, validationData)`
- Decomposition results written to `output/composition_decomposition_{hypothesis_id}.json`

---

## RULE-168: v2 Config Is the Only Sweep Output Promoted to Production

**What:** The output of Phase 7 is exactly one file: `configs/flux_rope_propagation_v2.yaml`.
It contains the winning hypothesis's compose expression, all stage definitions, tuned
hyperparameters from grid search, and (if applicable) NNLS-optimized ensemble weights.
All other sweep outputs (leaderboard JSON, decomposition reports) are artifacts, not config.

**Why:** Downstream consumers (TrainCommand, PredictCommand, ValidateEventsCommand) read
a single pipeline config. A sweep should not produce multiple candidate configs that require
manual selection. The sweep's job is to select; v2 is the result of that selection.

**How to apply:**
- `SweepCommand` generates `configs/flux_rope_propagation_v2.yaml` as its final action
- v2 config is a standard pipeline config (not a sweep config) — it uses the existing
  `PipelineConfigLoader` schema, not the sweep schema extension
- v2 is not generated if the sweep did not complete all hypotheses (pre-flight failure,
  sidecar crash, etc.)
