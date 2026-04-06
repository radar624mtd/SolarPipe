---
name: pipeline-config-validator
description: Validate SolarPipe YAML pipeline configurations against schema and semantic rules
tags: ["validation", "dsl", "configuration", "yaml"]
---

# Pipeline Config Validator Skill

Automated validation of SolarPipe YAML pipeline configurations. Checks schema compliance, semantic correctness, and data bindings.

## Validation Checks

### 1. Schema Validation
- Pipeline name is defined and non-empty
- data_sources, stages, and compose blocks exist (if needed)
- All required fields per stage are present

### 2. Data Source Validation
- All referenced data sources in stages exist in data_sources
- Provider type is supported: `sqlite`, `csv`, `parquet`, `api`
- Required provider fields present (connection, query, path, endpoint)
- Column definitions include name and role (feature/target/key/timestamp)

### 3. Stage Validation
- Stage name is valid C# identifier
- Source references existing data source
- Framework type is valid: `ml_net`, `onnx`, `python`, `physics`
- Model type is supported by framework
- Task type is valid: `regression`, `classification`, `ranking`, `timeseries`
- Features list contains column names that exist in source data
- Target column exists in source (if not physics)
- Hyperparameters match selected model's expected types

### 4. Framework-Specific Validation

#### ML.NET
- Supported models: FastForest, FastTree, LightGbm, Gam, Sdca, OrdinaryLeastSquares
- Hyperparameters: number_of_trees (int), number_of_leaves (int), feature_fraction (float 0-1), etc.

#### ONNX
- Model file referenced in hyperparameters exists
- Input/output tensor names match data schema

#### Physics
- Equation name is valid: drag_based_ensemble_v3, burton_ode, newell_coupling
- Required inputs for equation are available in features
- Parameters are physically reasonable (no negative speeds, masses, etc.)

#### Python
- Sidecar target is specified (localhost:port or remote service)
- Model type is valid: tft, neural_ode, lstm, custom
- Training hyperparameters are JSON-serializable

### 5. Composition Validation
- Composition expressions parse correctly
- Operators are valid: `->` (chain), `+` (ensemble), `^` (residual), `?` (gate)
- All referenced stages in composition expressions exist
- Composition types are compatible:
  - Chain: output columns of left match input requirements of right
  - Ensemble: output shapes are compatible
  - Residual: right model can accept left's output as feature
  - Gate: classifier outputs probabilities, models have matching output shapes

### 6. Validation Strategy Validation
- Method is valid: `leave_one_out`, `expanding_window`, `k_fold`
- Gap buffer format is valid (e.g., "6h", "5d")
- Metrics list contains supported names: rmse, mae, r2, rmsle, feature_importance

### 7. Mock Data Validation
- Source exists in data_sources
- Strategy is valid: `pretrain_then_finetune`, `mixed_training`, `residual_calibration`
- Sample weights sum to 1.0 (if specified)

## Usage

### Run via CLI

```bash
solarpipe validate --config configs/cme_propagation.yaml
```

### Run via Skill

```
/pipeline-config-validator --config configs/cme_propagation.yaml
/pipeline-config-validator --config configs/cme_propagation.yaml --verbose
```

## Output

### Success Case
```
✓ Configuration valid
  Pipeline: flux_rope_propagation_v1
  Stages: 2 (drag_baseline, rotation_predictor)
  Composition: drag_baseline ^ rotation_predictor
  Data sources: 2 (cme_catalog, enlil_ensemble)
```

### Validation Error
```
✗ Configuration invalid

Error [Stage: rotation_predictor]:
  Feature column 'initial_axis_angle' not found in data source 'cme_catalog'
  Available columns: chirality, cme_speed, cme_mass, ...

Error [Composition: propagation_prediction]:
  Stage 'drag_baseline' outputs arrival_time (scalar)
  Stage 'rotation_predictor' expects rotation features (list)
  Cannot compose with ^ operator - shapes incompatible

Warnings:
  - Feature 'flare_class_numeric' has no scale specified; recommend normalize(min=0, max=5)
```

## Examples

### Valid Simple Pipeline
```yaml
pipeline: test_regression
data_sources:
  simple_csv:
    provider: csv
    path: "data/test.csv"
    columns:
      - name: feature_1
        role: feature
      - name: feature_2
        role: feature
      - name: target
        role: target

stages:
  basic_rf:
    source: simple_csv
    framework: ml_net
    model: FastForest
    task: regression
    features: [feature_1, feature_2]
    target: target
    hyperparameters:
      number_of_trees: 100
      feature_fraction: 0.8
```

### Complex Physics + ML Pipeline
```yaml
pipeline: cme_propagation
data_sources:
  cme_catalog:
    provider: sqlite
    connection: "Data Source=./data/cme_catalog.db"
    query: "SELECT * FROM cme_events WHERE quality_flag >= 3"
    columns:
      - name: cme_speed
        role: feature
        transform: normalize(min=200, max=3500)
      - name: observed_rotation_angle
        role: target

stages:
  drag_baseline:
    source: cme_catalog
    framework: physics
    equation: drag_based_ensemble_v3
    params: [cme_speed, cme_mass, sw_density_ambient, sw_speed_ambient]
    output: predicted_arrival_time

  rotation_predictor:
    source: cme_catalog
    framework: ml_net
    model: FastForest
    task: regression
    features: [chirality, cme_speed, hcs_tilt_angle]
    target: observed_rotation_angle
    hyperparameters:
      number_of_trees: 300
      feature_fraction: 0.7

compose:
  final_prediction:
    expr: "drag_baseline ^ rotation_predictor"
    description: "Physics baseline + ML correction"
```

## Implementation Notes

The validator is implemented in `SolarPipe.Config/PipelineConfigValidator.cs` and called by:
- CLI `validate` command (SolarPipe.Host)
- ConfigurationLoader before building execution graph
- Integration tests before running pipelines

For programmatic access:

```csharp
var loader = new PipelineConfigLoader();
var result = await loader.ValidateAsync("configs/cme_propagation.yaml", registry);
if (!result.IsValid)
{
    foreach (var error in result.Errors)
        Console.WriteLine($"Error: {error}");
}
```
