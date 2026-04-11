---
name: neural-pipeline-validator
description: Validate sequence shape, Arrow dtype, TFT config, ensemble head wiring, and quantile output for the SolarPipe neural ensemble pipeline. Use before training or modifying the TFT sidecar, sequence extractor, or ensemble head config.
---

Validate the neural ensemble pipeline against the following rules. Report each violation with the file/config key, the rule violated, and a concrete fix.

## Sequence Shape Rules
- Pre-launch sequence shape must be `(N, 150, C)` where C ≤ 60 (OMNI channels). Fewer timesteps indicate a truncated window — flag with the actual T value.
- In-transit sequence shape must be `(N, T, C)` where T ≤ 72. T varies per event (expanding window) — confirm the collation function uses right-padding with a mask token (not zero-padding without masking).
- Static covariates must be `(N, S)` where S = number of scalar features from `pinn_expanded_flat`.
- Ensemble head input must be `(N, encoder_dim + n_existing_models)` where `n_existing_models` ≥ 3 (Phase 8, PINN V1, Phase 9 or physics ODE).

## Arrow / dtype Rules
- All feature tensors must be `pa.float32()` (not float64). Check the schema of every Arrow IPC file written by `build_pinn_sequences.py` and the sidecar's `_read_arrow_ipc()`.
- Categorical features (cluster_id_k5, cluster_id_k8, cluster_id_k12) must be cast to `float32` before inclusion — not object or int64.
- The `activity_id` column must be present as a string index column but excluded from the feature tensor. Verify it is not accidentally included in the channel dimension.

## TFT Configuration Rules
- `model_type` in the YAML config must be `TFT` (not `LSTM`, `TCN`, or `NeuralOde`). The stub `_SimpleTftModel` in `solarpipe_server.py` is a single-layer LSTM — it must be replaced with pytorch-forecasting's `TemporalFusionTransformer` for the goal architecture.
- TFT must distinguish:
  - `time_varying_known_reals` — OMNI channels (observed up to launch)
  - `time_varying_unknown_reals` — in-transit OMNI (unknown beyond current query time)
  - `static_reals` — CME geometry, SHARP, CDAW, cluster labels
  - `known_future_reals` — F10.7 forecast, day-of-year
- Quantile loss must specify three quantiles: `[0.1, 0.5, 0.9]`. Single-point MSE loss is a violation for the goal architecture (loses uncertainty output).
- `hidden_size` ≥ 32 and `attention_head_size` ≥ 4. Smaller models underfit on 1,884 training sequences.

## Ensemble Head Rules
- Existing model predictions (Phase 8, PINN V1, Phase 9) must be input features to the ensemble head — not post-hoc weighted averages applied after training.
- Out-of-fold predictions must be used for train events. If out-of-fold predictions are not available for a model, that model's prediction must be excluded from the train set features (not zero-filled).
- The ensemble head output must include three columns: `transit_p10`, `transit_p50`, `transit_p90`. Missing any quantile is a violation.

## Temporal CV Rules (inherited from existing rules)
- CV strategy must be `expanding_window`. Random k-fold is prohibited.
- `gap_buffer_days` ≥ 5 between train and validation folds.
- Last fold is calibration-only (RULE-164) — not used for training, feature importance, or hyperparameter search.

## OMNI Coverage Gate
- Before training, verify `proton_density` fill rate per event from `omni_hourly`. Events with < 80% coverage in the pre-launch window must use their scalar-only fallback features (from `pinn_expanded_flat`) and receive `has_full_omni=0`.
- The sequence dataset file `data/sequences/train_sequences.parquet` must include a `has_full_omni` boolean column. Models must respect this flag when deciding whether to use the sequence encoder output.

## OMNI Gap Check
- The current OMNI gap starts at 2026-03-29 01:00. Any prediction for events after this date must check `omni_hourly` availability and fall back to Phase 9 static or Phase 8 baseline. Never return a sequence-model prediction when the in-transit OMNI window is entirely NULL.

## Output
List all violations found. If none, confirm the neural pipeline is ready for training.
