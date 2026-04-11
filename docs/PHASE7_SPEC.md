# Phase 7: Hypothesis-Driven Model Validation & Refinement

## Purpose

Systematically evaluate candidate model sequences to identify the optimal composition
of physics baselines, ML corrections, and ensemble patterns for CME transit time
prediction. Configuration-driven: all hypotheses are declared in YAML, evaluated
under identical temporal CV conditions, and ranked by a comprehensive metric suite.

No new physics models, framework adapters, or composition operators are introduced.
This phase uses only what already exists and produces a winning `flux_rope_propagation_v2.yaml`.

---

## Decisions (Locked)

| Decision | Resolution |
|----------|------------|
| H7 (TFT via gRPC sidecar) | Included |
| Ensemble weight calibration | Last CV fold (held-out, not used in training) |
| Sweep parallelism | Enabled; each hypothesis tagged `[sweep:HN]` in structured logs |
| Statistical significance threshold | Deferred — evaluate fold reality first, expand before committing |
| Sidecar unavailability | Hard abort — pre-flight must pass for all hypotheses before any training starts |

---

## Pre-Flight Validation Gate

Runs before any training. All checks must pass or the sweep aborts with the failing
component identified. No partial runs.

| Check | What It Verifies |
|-------|-----------------|
| Data sources | SQLite `cme_catalog.db` readable, Parquet `enlil_ensemble_v1.parquet` readable, all validation CSVs present |
| Framework adapters | MLContext initializes, ONNX runtime loads, PhysicsAdapter constructs without error |
| Sidecar health | gRPC channel to `SOLARPIPE_SIDECAR_ADDRESS` ready within 10s timeout |
| Config validity | All 7 hypothesis configs parse, all stage framework types are known, no missing required hyperparameter keys |
| Registry & cache | Write permissions confirmed on `SOLARPIPE_REGISTRY` and `SOLARPIPE_CACHE` paths |

---

## Hypothesis Definitions

All hypotheses share the same training data, feature set, and CV protocol.
Differences are composition structure and stage selection only.

| ID | Compose Expression | Physics Baseline | ML Layer | Notes |
|----|-------------------|-----------------|----------|-------|
| H1 | `drag_baseline ^ rf_correction` | DragBased | FastForest residual | Current production |
| H2 | `burton_ode ^ rf_correction` | BurtonOde | FastForest residual | Dst-coupled baseline |
| H3 | `drag_baseline + burton_ode` | DragBased + BurtonOde | None | Equal-weight physics ensemble |
| H4 | `(drag_baseline + burton_ode) ^ rf_correction` | Ensemble physics | FastForest residual | Physics ensemble + ML correction |
| H5 | `drag_baseline ? (rf_correction, burton_ode)` | DragBased (gate) | Gated by CME speed class | Fast/slow routing |
| H6 | `drag_baseline → rf_correction` | DragBased | Chained (correction sees baseline output) | Chained composition |
| H7 | `drag_baseline ^ tft_correction` | DragBased | TFT via Python gRPC sidecar | Neural correction |

### Sweep Config Format (new YAML schema extension)

```yaml
sweep:
  name: phase7_hypothesis_sweep
  parallel: true
  log_tag_prefix: "sweep"          # produces [sweep:H1], [sweep:H2], etc.
  cv:
    strategy: expanding_window
    folds: 5
    gap_buffer_days: 5
    min_test_events: 50
    calibration_fold: last         # last fold held out for ensemble weight optimization

  hypotheses:
    - id: H1
      compose: "drag_baseline ^ rf_correction"
      stages: [drag_baseline, rf_correction]
    - id: H2
      compose: "burton_ode ^ rf_correction"
      stages: [burton_ode, rf_correction]
    # ... etc.

  stages:
    drag_baseline:
      framework: physics
      model_type: DragBased
      hyperparameters:
        drag_parameter: 0.2e-7
        background_speed_km_s: 400
        r_start_rs: 21.5
        r_stop_rs: 215.0
        solver: dormand_prince

    burton_ode:
      framework: physics
      model_type: BurtonOde
      hyperparameters:
        solver: dormand_prince

    rf_correction:
      framework: mlnet
      model_type: FastForest
      features: [cme_speed_kms, bz_gsm_proxy_nt, sw_density_n_cc,
                 sw_speed_ambient_kms, delta_v_kms, sw_bt_nt,
                 speed_ratio, speed_x_bz, speed_x_density]
      target: transit_hours_observed
      hyperparameters:
        number_of_trees: 100
        number_of_leaves: 20
        feature_fraction: 0.7

    tft_correction:
      framework: python_grpc
      model_type: TFT
      features: [cme_speed_kms, bz_gsm_proxy_nt, sw_density_n_cc,
                 sw_speed_ambient_kms, delta_v_kms, sw_bt_nt,
                 speed_ratio, speed_x_bz, speed_x_density]
      target: transit_hours_observed
      hyperparameters:
        hidden_size: 64
        attention_heads: 4
        dropout: 0.1
        max_epochs: 100
```

---

## Training Protocol

### Cross-Validation
- Strategy: `ExpandingWindowCV` (RULE-051 — temporal, no random k-fold)
- Folds: 5
- Gap buffer: 5 days
- Minimum test events per fold: 50
- Calibration fold: last fold held out exclusively for ensemble weight optimization

### Parallel Execution
- Each hypothesis trains on an independent DI scope
- Log entries prefixed `[sweep:HN stage:stage_name]` in `logs/dotnet_latest.json`
- Shared Trace ID per sweep run (different from per-hypothesis Trace IDs)
- Checkpoint per hypothesis-stage combination: `{cache}/{sweep_id}/{hypothesis_id}/{stage_name}/`

### Sidecar (H7 only)
- H7 stages dispatch to gRPC sidecar via `GrpcSidecarAdapter`
- Arrow IPC file exchange (RULE-125)
- Training progress via server-streaming `StreamTrain` RPC
- Log entries in `logs/python_latest.json` with matching `[sweep:H7]` tag

---

## Metrics Suite (R3)

All metrics computed per fold, then aggregated (mean ± std across folds).

| Metric | Formula / Method | Decision Role |
|--------|-----------------|---------------|
| MAE | mean(|ŷ - y|) | Primary ranking |
| RMSE | √mean((ŷ-y)²) | Sensitivity to large errors |
| R² | 1 - SS_res/SS_tot | Variance explained |
| Bias | mean(ŷ - y) | Systematic error direction |
| Skill vs DBM | 1 - MAE_model/MAE_DBM | Operational benchmark comparison |
| Hit rate ±6h | count(|ŷ-y|≤6)/n | Event-level decision accuracy |
| Hit rate ±12h | count(|ŷ-y|≤12)/n | Broader event-level accuracy |
| Pinball loss (α=0.10) | mean(max(α(y-ŷ), (α-1)(y-ŷ))) | Interval calibration quality |
| Coverage rate 90% | count(y ∈ [lb,ub])/n | Does 90% interval cover 90%? |
| Kendall τ | rank correlation of ŷ vs y | Event ordering correctness |

DBM baseline MAE sourced from `data/validation/ccmc_scoreboard_2026.csv`.

---

## Feature Importance & Selection (R4)

Runs after CV for each hypothesis's ML stage(s) using `FeatureImportanceAnalyzer`
(permutation importance, 5 repeats per feature).

Output per hypothesis:
- Ranked feature list (mean importance ± std)
- Features with importance < 0.01 flagged as removal candidates
- Features with high variance across folds flagged as unstable
- Interaction candidates: any two features with individual importance > 0.1
  and joint rank correlation < 0.3 (potential unexploited interaction)

This output informs the v2 config feature list but does not automatically modify it.

---

## Ensemble Weight Optimization (R5)

Applies to H3 and H4 only (ensemble composition).

Protocol:
1. Train all ensemble member stages on folds 1–4
2. Collect per-event predictions from each member on fold 5 (calibration fold)
3. Solve NNLS: `min ‖Aw - y‖²` subject to `1ᵀw = 1, w ≥ 0`
   where A = [predictions_member1, predictions_member2, ...], y = observed
4. Write optimal weights as `weights: [w1, w2]` in the sweep config output
5. Re-evaluate on external validation set (`observed_transits_2026.csv`) with optimized weights

---

## Hyperparameter Sensitivity Analysis (R6)

Grid search on highest-impact parameters. Evaluated under full expanding-window CV.
Applied to H1 (current best) first; winner carries into other hypotheses if applicable.

| Parameter | Stage | Grid Values |
|-----------|-------|------------|
| `drag_parameter` | DragBased | [0.1e-7, 0.15e-7, 0.2e-7, 0.3e-7, 0.5e-7] |
| `background_speed_km_s` | DragBased | [350, 375, 400, 425, 450] |
| `number_of_trees` | FastForest | [50, 100, 150, 200] |
| `number_of_leaves` | FastForest | [10, 20, 30, 40] |
| `feature_fraction` | FastForest | [0.5, 0.6, 0.7, 0.8, 0.9] |

Total grid: 5×5×4×4×5 = 2,000 combinations. Reduce to Latin hypercube sample of
100 if full grid exceeds wall-clock budget. Winning configuration replaces YAML defaults.

---

## Composition Decomposition (R8)

For the winning hypothesis, post-prediction analysis:

1. **Per-stage scatter**: Each stage's raw prediction vs observed (independent of composition)
2. **Residual reduction**: Error at each composition step
   - After physics baseline: MAE_baseline
   - After ML correction: MAE_final
   - Reduction: (MAE_baseline - MAE_final) / MAE_baseline × 100%
3. **Stage correlation matrix**: Pearson correlation between all stage predictions
   - High correlation = stages not adding independent information
   - Low correlation = stages are complementary (good for ensemble)
4. **Error attribution**: Fraction of final variance explained by each stage

Output written to `output/composition_decomposition_{hypothesis_id}.json`.

---

## Deliverables

| Artifact | Path | Description |
|----------|------|-------------|
| Sweep config | `configs/phase7_sweep.yaml` | All 7 hypotheses, shared stages, CV settings |
| `ModelSweep` executor | `src/SolarPipe.Training/Sweep/ModelSweep.cs` | Parallel hypothesis runner with pre-flight |
| `SweepCommand` CLI | `src/SolarPipe.Host/Commands/SweepCommand.cs` | `dotnet run -- sweep --config configs/phase7_sweep.yaml` |
| `ComprehensiveMetricsEvaluator` | `src/SolarPipe.Training/Evaluation/ComprehensiveMetricsEvaluator.cs` | Full metric suite |
| `NnlsEnsembleOptimizer` | `src/SolarPipe.Training/Evaluation/NnlsEnsembleOptimizer.cs` | Weight optimization |
| `HyperparameterGridSearch` | `src/SolarPipe.Training/Sweep/HyperparameterGridSearch.cs` | Grid search runner |
| `CompositionDecomposer` | `src/SolarPipe.Prediction/CompositionDecomposer.cs` | Error attribution |
| Sweep results | `output/phase7_sweep_results.json` | Full leaderboard, per-fold metrics |
| Decomposition report | `output/composition_decomposition_{id}.json` | Per-winning-hypothesis |
| v2 config | `configs/flux_rope_propagation_v2.yaml` | Winning hypothesis, tuned hyperparameters, optimized weights |

---

## New Rules (Phase 7)

See DEVELOPMENT_RULES.md RULE-160 through RULE-168.

---

## File Size Budget

| New File | Estimated Lines | Split Strategy |
|----------|----------------|----------------|
| `ModelSweep.cs` | ~350 | Single file |
| `SweepCommand.cs` | ~200 | Single file |
| `ComprehensiveMetricsEvaluator.cs` | ~300 | Single file |
| `NnlsEnsembleOptimizer.cs` | ~150 | Single file |
| `HyperparameterGridSearch.cs` | ~250 | Single file |
| `CompositionDecomposer.cs` | ~200 | Single file |

All within 400-line limit. No partial classes needed at this estimate.

---

## Test Requirements

Minimum unit tests per new component:

| Component | Min Tests |
|-----------|-----------|
| `ModelSweep` pre-flight (all pass, each failure type) | 8 |
| `ComprehensiveMetricsEvaluator` (each metric, edge cases) | 12 |
| `NnlsEnsembleOptimizer` (2-member, 3-member, degenerate) | 6 |
| `HyperparameterGridSearch` (grid construction, latin hypercube) | 5 |
| `CompositionDecomposer` (2-stage, 3-stage, NaN propagation) | 6 |
| Sweep YAML config parsing (valid, invalid hypothesis refs) | 5 |

Total: ≥42 new unit tests.
