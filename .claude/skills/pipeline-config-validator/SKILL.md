---
name: pipeline-config-validator
description: Validate a SolarPipe YAML pipeline config for semantic correctness beyond schema — compose expressions, framework types, feature lists, data source paths, and physics constraints.
---

Given a YAML pipeline config path, validate it against the following rules. Report each violation with the YAML key path, the rule violated, and a suggested fix.

## Compose Expression Rules
- All stage names referenced in compose expressions (`^`, `+`, `|`) must correspond to a defined stage in the config.
- The `^` operator (residual) requires exactly two operands: a baseline and a correction stage.
- Stage names must be unique within the config.

## Framework Rules
- `framework` must be one of: `mlnet`, `onnx`, `physics`, `python_grpc`.
- Physics stages (`framework: physics`) must specify `solver: dormand_prince` in their parameters.
- `python_grpc` stages must have a `sidecar_address` parameter or rely on the `SOLARPIPE_SIDECAR_ADDRESS` environment variable.

## Feature Rules
- `features` list must not be empty for ML stages (`mlnet`, `onnx`, `python_grpc`).
- Feature names must not duplicate fields that have an explicit `column_role` (e.g., the label column should not also appear in features).
- Hyperparameter keys must use `snake_case`. Flag any `PascalCase` keys (they silently fall through to defaults at runtime).

## Data Source Rules
- Every `data_source` path referenced must exist relative to the working directory. Check both SQLite `.db` paths and Parquet `.parquet` paths.
- SQLite sources must specify a `table` or `query`. Parquet sources must specify a `file` path.

## Evaluation Rules
- Cross-validation strategy must be `expanding_window` or `sliding_window`. Random k-fold (`kfold`) is prohibited for time-series data.
- `gap_buffer_days` must be >= 1 when a CV strategy is specified.

## Output
List all violations found. If none, confirm the config is semantically valid.
