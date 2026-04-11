# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SolarPipe is a .NET 8 declarative ML orchestration framework for CME (coronal mass ejection) propagation modeling and geomagnetic storm prediction. Pipelines are defined in YAML — topology, model selection, and composition rules are configuration, not code.

**Core concept:** YAML compose expressions wire stages together (e.g., `drag_baseline ^ rf_correction` means residual correction on top of a physics baseline). The runtime resolves this into `IComposedModel` implementations.

## Commands

```bash
# Build (run after every .cs edit — TreatWarningsAsErrors is on)
dotnet build --no-restore

# Unit tests only
dotnet test --filter Category=Unit --no-build

# All tests
dotnet test

# Run a single test class
dotnet test --filter "FullyQualifiedName~CheckpointManagerTests" --no-build

# CLI commands
dotnet run --project src/SolarPipe.Host -- validate --config configs/flux_rope_propagation_v1.yaml
dotnet run --project src/SolarPipe.Host -- train --config configs/flux_rope_propagation_v1.yaml
dotnet run --project src/SolarPipe.Host -- predict --config configs/flux_rope_propagation_v1.yaml --input data/...
dotnet run --project src/SolarPipe.Host -- validate-events --config configs/validate_2026.yaml
dotnet run --project src/SolarPipe.Host -- inspect --config configs/flux_rope_propagation_v1.yaml

# Add a NuGet package (NEVER use dotnet add package)
# Edit Directory.Packages.props, then:
dotnet restore

# Python sidecar — NEVER use venv; use system python directly
python solarpipe_server.py --port 50051

# Compile proto stubs (required after any proto change; NOT committed to git)
python -m grpc_tools.protoc \
    -I python/ --python_out=python/ --grpc_python_out=python/ python/solarpipe.proto

# Python sidecar tests
cd python && python -m pytest tests/ -v              # all
cd python && python -m pytest tests/ -v -m "not live" # unit only
cd python && python -m pytest tests/ -v -m live       # live gRPC server tests

# Docker
docker compose up sidecar        # Start gRPC sidecar only
docker compose run --rm host train --config configs/...
```

## Python

**NEVER use venv.** Always invoke `python` directly — no `.venv`, no virtual environments of any kind.

## Phase 9 Data Reality (read before writing any script)

The real ground-truth dataset is **535 IPS→CME transit events (2010–2026)** from `donki_ips` JOIN `donki_cme` in `solar_data.db` — not 71. The 71-event set is 2026-only and is the holdout.

**280 of 535 events are multi-CME interactions** — this is the majority case, not an edge case. Multi-CME IPS events have identifiable OMNI signatures: density spikes 20–49 cm⁻³, speed gradients >100 km/s in the 72h pre-arrival window.

Key tables for the full feature matrix:
- `donki_ips` — 644 shock arrivals with CME linkage (`linked_event_ids`)
- `donki_cme` — 8K CMEs: speed, half-angle, lat/lon, analysis_type
- `donki_flare` — 3K flares linked to CMEs: class, source location, active region
- `donki_gst` — 192 geomagnetic storms linked to CMEs and IPS
- `omni_hourly` — 561K rows, 60 columns, ~95% fill 2010+: Bz_GSM, flow_pressure, plasma_beta, Alfvén Mach, AE, Dst, electric_field
- `cdaw_cme` — 42K CMEs with linear+2nd-order speed, acceleration, angular width, mass, kinetic energy
- `symh_hourly` — 1-min Sym-H 1981–2026 (higher resolution than Dst)
- `gfz_kp_ap` — 3-hourly Kp/Ap back to 1932 + daily F10.7
- `goes_xrs_flares` — 70K flares with location (Earth-facing source region check)
- `mag_results` — existing ML runs: FastTree/LGB predict Dst at 1h (R²=0.75), storm occurrence ROC-AUC=0.996
- `ml_clusters` — 54,973 events in 5–12 ambient regime clusters (use for medium state classification)

**The objective is a 3-stage PINN pipeline:**
1. Regime classifier — ambient medium state from OMNI + cluster membership
2. CME-CME interaction detector — preceding CME density in 48h window, angular separation
3. Transit time PINN — ODE drag prior + ML residual; asymmetric loss (late penalty 1.5× early)

**Do not** tune drag parameters on 71 events. **Do not** re-derive the event catalog without querying `donki_ips`. **Do not** write any data script without reading the relevant table schema and row counts first.

## Solution Structure

```
src/
  SolarPipe.Core/       # Interfaces, domain models, physics constants, coordinate types
  SolarPipe.Config/     # YAML pipeline loader, compose expression parser
  SolarPipe.Data/       # DataFrame abstraction, CSV/SQLite/Parquet/REST providers
  SolarPipe.Training/   # Framework adapters, physics ODE solver, checkpointing
  SolarPipe.Prediction/ # Composed model types (Residual, Chained, Ensemble, Gated)
  SolarPipe.Host/       # CLI entry point, DI wiring, 5 commands
tests/
  SolarPipe.Tests.Unit/
  SolarPipe.Tests.Integration/
  SolarPipe.Tests.Pipeline/
python/                 # gRPC sidecar (TFT + NeuralODE via PyTorch)
configs/                # YAML pipeline configurations
data/                   # Parquet files, validation CSVs, staging.db (data/data/staging/)
solar_data.db           # 11 GB OMNI L1 archive at repo root — primary source for omni_hourly
                        # (Phase 9 predict-progressive default; see docs/DATA_SCHEMA_REFERENCE.md §1)
```

**Dependency order (strictly unidirectional):**
`Core` ← `Config`, `Data`, `Training` ← `Prediction` ← `Host`

## Architecture

### Pipeline execution flow
1. `PipelineConfigLoader` parses YAML → `PipelineConfig` with `StageConfig[]`
2. `ComposeExpressionParser` resolves compose expressions into a model graph
3. Each stage maps to a `IFrameworkAdapter` (MlNet, Onnx, Physics, PythonGrpc)
4. Adapters produce `ITrainedModel` instances
5. `IComposedModel` wires them: `ResidualModel` (`^`), `ChainedModel`, `EnsembleModel`, `GatedModel`
6. `PredictAsync(IDataFrame)` runs the composed graph

### Key interfaces (SolarPipe.Core)
- `IDataFrame` — unified data abstraction over CSV/SQLite/Parquet/Arrow (7 methods: `GetColumn`, `Slice`, `SelectColumns`, `AddColumn`, `ToDataView`, `ToArray`, `ResampleAndAlign`)
- `IFrameworkAdapter` — trains a single stage, returns `ITrainedModel`
- `IComposedModel` — `PredictAsync(IDataFrame, CancellationToken) → PredictionResult`
- `IModelRegistry` — stores/loads `ModelArtifact` (implemented by `FileSystemModelRegistry` with atomic writes)

### Physics baseline
`DragBasedModel` solves CME transit time using Dormand-Prince RK4(5) ODE (`DormandPrinceSolver`). All physics equations use GSM-frame Bz (not GSE). Coordinate conversion is in `CoordinateTransform`. Never use bare floats for spatial vectors — use `GseVector`/`GsmVector`.

### Python gRPC sidecar
Data transfers use Arrow IPC files (not inline proto bytes). The proto defines `Train`, `StreamTrain`, `Predict`, `ExportOnnx`. The sidecar logs structured JSON to `logs/python_latest.json`. The .NET side logs to `logs/dotnet_latest.json`. Both share a Trace ID.

### Checkpointing
`CheckpointManager` in `SolarPipe.Training` caches completed stage artifacts so training can resume. `TrainCommand` calls it per-stage before invoking the adapter.

## Critical Rules

| Rule | What to do |
|------|------------|
| No `ReadOnlySpan<T>` in interfaces | Use `float[]` |
| Validate column lengths in `ToDataView()` | Check before ML.NET ingestion |
| Set `FeatureFraction=0.7` explicitly | ML.NET FastForest default differs |
| Register `Yaml12BooleanConverter` | YAML 1.1 Norway problem (`no` → false) |
| Use Dormand-Prince solver | Not MathNet ODE solvers |
| Physics equations use GSM-frame Bz | Not GSE |
| Atomic file write for registry | temp file + `File.Move` |
| Temporal CV with gap buffers | Never random k-fold |
| `OrdinalIgnoreCase` for hyperparameter keys | YAML authors mix `snake_case`/`PascalCase` |
| `await using` for ServiceProvider | Has `IDisposable` singletons |
| No XML doc comments | No IntelliSense in this workflow |

## Python Sidecar Gotchas

| Gotcha | Detail |
|--------|--------|
| **NEVER use venv** | Always invoke `python` directly — no `.venv`, no `python/.venv/Scripts/python`, no virtual environments of any kind |
| `onnx` package required | `torch.onnx.export` needs `onnx` installed; included in `requirements.txt` |
| Max ONNX opset is 20 | PyTorch 2.5.1 supports opsets up to 20, not 21; server clamps via `min(opset, 20)` |
| Proto stubs not committed | `solarpipe_pb2.py` / `solarpipe_pb2_grpc.py` are generated — recompile after any proto change |
| `live` pytest mark | Tests that spin up an in-process gRPC server are `@pytest.mark.live`; defined in `python/pytest.ini` |
| Predict input columns must match training | `_SimpleTftModel` input size is fixed at train time; predict Arrow file must have same feature columns only |

## New Components (Phase 7)

| File | Project | Purpose |
|------|---------|---------|
| `Sweep/ModelSweep.cs` | Training | Pre-flight + parallel hypothesis runner |
| `Sweep/HyperparameterGridSearch.cs` | Training | Grid search with LHS fallback |
| `Evaluation/ComprehensiveMetricsEvaluator.cs` | Training | 10-metric suite per fold |
| `Evaluation/NnlsEnsembleOptimizer.cs` | Training | NNLS weight optimization on calibration fold |
| `CompositionDecomposer.cs` | Prediction | Error attribution for winning hypothesis |
| `Commands/SweepCommand.cs` | Host | CLI entry point for sweep |

All files stay under 400 lines. Sweep checkpoint path: `{SOLARPIPE_CACHE}/sweeps/{sweep_id}/{hypothesis_id}/{stage_name}/`

## Testing Conventions

- All tests use `[Trait("Category", "Unit")]` / `"Integration"` / `"Pipeline"`
- Physics test data always comes from `PhysicsTestFixtures.cs` — never `Random.NextDouble()` for domain values
- Unit tests use NSubstitute for `ITrainedModel` and `IDataFrame`
- Integration tests write temp YAML configs and invoke CLI commands directly
- Sentinel values (`9999.9`, `-1e31`) must be converted to `NaN` at data load time

## Active Phase

**Phase 7: Hypothesis-Driven Model Validation & Refinement**
Spec: `docs/PHASE7_SPEC.md` | Rules: `docs/PHASE7_RULES.md`

Systematic evaluation of 7 candidate model sequences (H1–H7) to select the optimal
composition of physics baselines, ML corrections, and ensemble patterns.

**New CLI command:**
```bash
# Run full hypothesis sweep (pre-flight → parallel CV → leaderboard → v2 config)
dotnet run --project src/SolarPipe.Host -- sweep --config configs/phase7_sweep.yaml

# Resume an interrupted sweep
dotnet run --project src/SolarPipe.Host -- sweep --config configs/phase7_sweep.yaml --resume

# Fresh sweep (clears checkpoints)
dotnet run --project src/SolarPipe.Host -- sweep --config configs/phase7_sweep.yaml --fresh
```

**Key Phase 7 rules (RULE-160–168):**

| Rule | Summary |
|------|---------|
| RULE-160 | Pre-flight gate is mandatory and atomic — all checks pass or sweep aborts |
| RULE-161 | Every log entry in sweep scope must include `sweep_id`, `hypothesis_id`, `stage_name` |
| RULE-162 | Checkpoint paths are hypothesis-scoped: `{cache}/sweeps/{sweep_id}/{hypothesis_id}/{stage_name}/` |
| RULE-163 | Metrics computed per fold independently — no pooling across folds before aggregation |
| RULE-164 | Last CV fold is calibration-only — not used in training, feature importance, or grid search |
| RULE-165 | NNLS optimizer writes weights back to config — no silent in-memory application |
| RULE-166 | Grid search auto-falls back to LHS(100) if combinations > 200 |
| RULE-167 | `CompositionDecomposer` runs only on the winning hypothesis, not in sweep loop |
| RULE-168 | v2 config is the only sweep output promoted to production; not generated on incomplete sweep |

**Hypothesis summary:**

| ID | Compose | Notes |
|----|---------|-------|
| H1 | `drag_baseline ^ rf_correction` | Current production |
| H2 | `burton_ode ^ rf_correction` | Dst-coupled baseline |
| H3 | `drag_baseline + burton_ode` | Physics ensemble |
| H4 | `(drag_baseline + burton_ode) ^ rf_correction` | Ensemble + ML residual |
| H5 | `drag_baseline ? (rf_correction, burton_ode)` | Gated by speed class |
| H6 | `drag_baseline → rf_correction` | Chained composition |
| H7 | `drag_baseline ^ tft_correction` | Neural correction via sidecar |

## Automations

**Hook (automatic):** Every `.cs` edit triggers `dotnet build --no-restore` automatically. Edits to `Directory.Packages.props` print a `dotnet restore` reminder.

**Skills (invoke by name):**
- `/dotnet-test-runner [Unit|Integration|Pipeline|<ClassName>]` — filtered test runs
- `/pipeline-config-validator` — semantic validation of standard pipeline YAML configs
- `/sweep-validator` — semantic validation of Phase 7 sweep configs (RULE-160–168 compliance)
- `/db-schema-lookup` — look up table schemas, column names, and feature aliases before writing SQL or YAML feature lists

**Data schema reference:** `docs/DATA_SCHEMA_REFERENCE.md` — authoritative mapping of all DB files, table schemas, row counts, and the YAML-alias → actual-column-name translation table. Consult before any data work.

**Subagents (invoke via Agent tool):**
- `physics-validator` — use when editing anything in `SolarPipe.Training/Physics/`, `DragBasedModel`, `DormandPrinceSolver`, coordinate transforms; also covers Phase 7 physics-in-sweep rules
- `architecture-reviewer` — use when adding project references, new adapter/command types, `Program.cs` DI wiring, or any Phase 7 sweep components (`ModelSweep`, `SweepCommand`, `NnlsEnsembleOptimizer`, `HyperparameterGridSearch`, `CompositionDecomposer`)

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `SOLARPIPE_REGISTRY` | Model artifact storage path |
| `SOLARPIPE_CACHE` | Checkpoint cache path |
| `SOLARPIPE_SIDECAR_ADDRESS` | gRPC sidecar address (default: `http://localhost:50051`) |
