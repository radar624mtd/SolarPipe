# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SolarPipe is a .NET 8 declarative ML orchestration framework for CME (coronal mass ejection) propagation modeling and geomagnetic storm prediction. Pipelines are defined in YAML — topology, model selection, and composition rules are configuration, not code.

**Core concept:** YAML compose expressions wire stages together (e.g., `drag_baseline ^ rf_correction` means residual correction on top of a physics baseline). The runtime resolves this into `IComposedModel` implementations.

## Primary Goal: MAE ≤ 3h ± 2h on CME Transit Time

The target is a **two-headed neural ensemble** that reaches ≤ 3h MAE on the 90-event holdout:

1. **Pre-launch encoder** — full 150h × 60-channel OMNI time series + expanded scalar features (CDAW accel/curvature, 14 SHARP magnetic parameters, multi-cluster labels). Replaces the 45-column hand-engineered pinn_training_flat.
2. **In-transit encoder** — expanding L1 observation window (0–72h × 60 channels), causally masked. Learned version of Phase 9 density-modulated drag.
3. **Ensemble head** — concatenates both encoder outputs with existing model predictions (Phase 8, PINN V1, Phase 9 progressive, physics ODE baseline) as input features → quantile output (P10/P50/P90).

**Current performance baseline:**
| Model | MAE | Events |
|---|---|---|
| Phase 7 H1 reference | 20.26h | sweep |
| Physics ODE only | 17.80h | 71 |
| Phase 8 domain ML | 12.33h | 71 |
| PINN V1 (LGB residual) | 8.69h | 90 holdout |
| Phase 9 density-modulated | 4.42h | 15-event OMNI subset |

**Key insight:** `pinn_training_flat` collapses 150h × 60-channel OMNI to 2 scalars, drops 4 CDAW kinematic fields (accel, 2nd-order speeds), and drops 14 SHARP magnetic parameters. The neural network's first job is to see what the hand-engineered pipeline was forced to compress.

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
dotnet run --project src/SolarPipe.Host -- predict-progressive --config configs/flux_rope_propagation_v1.yaml

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

# Feature matrix & sequence extraction
python scripts/build_pinn_feature_matrix.py          # rebuild pinn_training_flat (45 cols)
python scripts/build_expanded_feature_matrix.py      # expanded matrix: CDAW accel + SHARP + multi-cluster
python scripts/build_pinn_sequences.py               # extract 150h OMNI sequences per event → Parquet

# OMNI backfill
python scripts/backfill_omni_gap_ace1h.py            # fill post-2026-03-29 gap from ACE 1h feed

# Docker
docker compose up sidecar        # Start gRPC sidecar only
docker compose run --rm host train --config configs/...
```

## Python

**NEVER use venv.** Always invoke `python` directly — no `.venv`, no virtual environments of any kind.

## Data Reality (read before writing any script)

The real ground-truth dataset is **1,974 IPS→CME transit events (2010–2026)** in `staging.db` (`pinn_training_flat`): 1,884 train (pre-2026) + 90 holdout (2026).

**39% of events are multi-CME interactions** (`is_multi_cme`). These are NOT edge cases — they are the majority of large-error events and the primary reason for Phase 9.

### Feature engineering is artificially narrow — the untapped data
Current `pinn_training_flat` (45 cols) deliberately drops:

| Dropped source | Fields omitted | Where to find |
|---|---|---|
| CDAW CME catalog | `second_order_speed_init/final/20Rs`, `accel_kms2`, `mpa_deg` | `solar_data.db:cdaw_cme` |
| SHARP magnetic keywords | 14 parameters: `meangam`, `meangbt/bz/bh`, `meanjzd`, `totusjz`, `meanjzh`, `totusjh`, `absnjzh`, `meanalp`, `savncpp`, `meanpot`, `totpot`, `meanshr`, `shrgt45`, `r_value` | `staging.db:sharp_keywords` |
| OMNI time series | Full 150h × 60-channel pre-launch sequence | `solar_data.db:omni_hourly` (needs per-event join) |
| Alternative clusters | k=8, k=12, dbscan(96) cluster IDs | `solar_data.db:ml_clusters` |
| Existing model outputs | Phase 8 pred, PINN V1 pred, Phase 9 progressive pred | `output/` JSON files |

### Key tables
- `donki_ips` (solar_data.db) — 644 shock arrivals with CME linkage
- `donki_cme` (solar_data.db) — 8K CMEs: speed, half-angle, lat/lon, analysis_type
- `omni_hourly` (solar_data.db) — 561K rows, 60 columns, 1963–2026; **OMNI gap: all columns NULL from 2026-03-29 onward** (needs ACE backfill)
- `cdaw_cme` (solar_data.db) — 42K CMEs with linear+2nd-order speed, acceleration, mass, KE
- `pinn_training_flat` (staging.db) — 1,974 events, 45 engineered columns, train/holdout split
- `sharp_keywords` (staging.db) — 102K rows, 1,930 active regions, 24 magnetic parameters
- `ml_clusters` (solar_data.db) — 219K events, 4 clustering methods (k=5/8/12, dbscan=96)
- `mag_predictions` (solar_data.db) — 2.7M rows: Phase 8 Dst predictions by 5 models
- `symh_hourly` (solar_data.db) — 1-min Sym-H 1981–2026
- `gfz_kp_ap` (solar_data.db) — 3-hourly Kp/Ap + daily F10.7
- `goes_xrs_flares` (solar_data.db) — 70K flares with location

**Do not** tune drag parameters on 71 events. **Do not** re-derive the event catalog without querying `donki_ips`. **Do not** write any data script without reading the relevant table schema and row counts first.

## Solution Structure

```
src/
  SolarPipe.Core/       # Interfaces, domain models, physics constants, coordinate types
  SolarPipe.Config/     # YAML pipeline loader, compose expression parser
  SolarPipe.Data/       # DataFrame abstraction, CSV/SQLite/Parquet/REST providers
  SolarPipe.Training/   # Framework adapters, physics ODE solver, checkpointing
  SolarPipe.Prediction/ # Composed model types (Residual, Chained, Ensemble, Gated)
  SolarPipe.Host/       # CLI entry point, DI wiring, commands
tests/
  SolarPipe.Tests.Unit/
  SolarPipe.Tests.Integration/
  SolarPipe.Tests.Pipeline/
python/                 # gRPC sidecar (TFT + NeuralODE via PyTorch)
scripts/                # Data pipeline scripts (Python)
configs/                # YAML pipeline configurations
data/                   # Parquet files, validation CSVs, staging.db (data/data/staging/)
output/                 # Model results JSON (phase8, phase9_m*, pinn_v1)
solar_data.db           # 11 GB OMNI L1 archive at repo root — primary source for omni_hourly
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

The `_SimpleTftModel` in `python/solarpipe_server.py` is a **stub LSTM** — it must be replaced with a full pytorch-forecasting TFT for the neural ensemble goal. The `python_grpc` adapter in .NET is the correct integration path.

### Phase 9 progressive propagation
`ProgressiveDragPropagator` in `SolarPipe.Training/Physics/` implements density-modulated drag: `γ_eff(t) = γ₀·(n_obs(t)/n_ref)` with `n_ref=5 cm⁻³`. CLI entry: `predict-progressive`. Achieves 4.42h MAE on the 15-event OMNI-covered subset. Blocks on OMNI gap post-2026-03-29.

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
| Read schema before writing SQL | Column names differ between tables; use /db-schema-lookup |
| Sequence shape: (N, T, C) | N=events, T=timesteps, C=channels; float32 throughout |
| Arrow schema enforcement | `pa.float32()` exactly at both ends of gRPC channel |

## Python Sidecar Gotchas

| Gotcha | Detail |
|--------|--------|
| **NEVER use venv** | Always invoke `python` directly — no `.venv`, no `python/.venv/Scripts/python`, no virtual environments of any kind |
| `onnx` package required | `torch.onnx.export` needs `onnx` installed; included in `requirements.txt` |
| Max ONNX opset is 20 | PyTorch 2.5.1 supports opsets up to 20, not 21; server clamps via `min(opset, 20)` |
| Proto stubs not committed | `solarpipe_pb2.py` / `solarpipe_pb2_grpc.py` are generated — recompile after any proto change |
| `live` pytest mark | Tests that spin up an in-process gRPC server are `@pytest.mark.live`; defined in `python/pytest.ini` |
| Predict input columns must match training | TFT/LSTM input size is fixed at train time; predict Arrow file must have same feature columns |
| `_SimpleTftModel` is a stub | Single-layer LSTM only; full TFT requires pytorch-forecasting |
| Sequence extraction not pre-computed | Must run `build_pinn_sequences.py` before training sequence models |

## Active Phase

**Goal: MAE ≤ 3h ± 2h via two-headed neural ensemble**

Implementation phases (sequential):

| Phase | Task | Expected MAE | Status |
|---|---|---|---|
| P1 | OMNI backfill (ACE 1h feed) | unblocks Mar-30/Apr-01 | blocked on data |
| P2 | Expand feature matrix (CDAW accel + SHARP + multi-cluster) | ~7h | ready |
| P3 | Sequence extraction: 150h OMNI per event → Parquet | prereq for P4 | ready |
| P4 | Full TFT in sidecar (replace stub LSTM) | ~5–6h | prereq: P3 |
| P5 | In-transit encoder + Phase 9 integration | ~3–4h | prereq: P4 |
| P6 | Ensemble head (TFT + Phase 8 + PINN V1 + Phase 9 as inputs) | ~2.5–3.5h | prereq: P4+P5 |
| P7 | Quantile calibration (isotonic, P10/P50/P90) | ±2h bound | prereq: P6 |

**Key Phase 7 rules still in effect (sweep infrastructure, RULE-160–168):**

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

## New Components (Phase 7 sweep infrastructure)

| File | Project | Purpose |
|------|---------|---------|
| `Sweep/ModelSweep.cs` | Training | Pre-flight + parallel hypothesis runner |
| `Sweep/HyperparameterGridSearch.cs` | Training | Grid search with LHS fallback |
| `Evaluation/ComprehensiveMetricsEvaluator.cs` | Training | 10-metric suite per fold |
| `Evaluation/NnlsEnsembleOptimizer.cs` | Training | NNLS weight optimization on calibration fold |
| `CompositionDecomposer.cs` | Prediction | Error attribution for winning hypothesis |
| `Commands/SweepCommand.cs` | Host | CLI entry point for sweep |
| `Physics/ProgressiveDragPropagator.cs` | Training | Density-modulated drag assimilation (Phase 9) |

All files stay under 400 lines. Sweep checkpoint path: `{SOLARPIPE_CACHE}/sweeps/{sweep_id}/{hypothesis_id}/{stage_name}/`

## Testing Conventions

- All tests use `[Trait("Category", "Unit")]` / `"Integration"` / `"Pipeline"`
- Physics test data always comes from `PhysicsTestFixtures.cs` — never `Random.NextDouble()` for domain values
- Unit tests use NSubstitute for `ITrainedModel` and `IDataFrame`
- Integration tests write temp YAML configs and invoke CLI commands directly
- Sentinel values (`9999.9`, `-1e31`) must be converted to `NaN` at data load time
- Sequence model tests must validate output shape `(N, 1)` and confirm float32 dtype

## Automations

**Hooks (automatic):**
- Every `.cs` edit triggers `dotnet build --no-restore`
- Every `.py` edit in `scripts/` triggers a syntax check (`python -m py_compile <file>`)
- Edits to `Directory.Packages.props` print a `dotnet restore` reminder

**Skills (invoke by name):**
- `/dotnet-test-runner [Unit|Integration|Pipeline|<ClassName>]` — filtered test runs
- `/pipeline-config-validator` — semantic validation of standard pipeline YAML configs
- `/sweep-validator` — semantic validation of Phase 7 sweep configs (RULE-160–168 compliance)
- `/db-schema-lookup` — look up table schemas, column names, and feature aliases before writing SQL or YAML feature lists
- `/feature-matrix-builder` — expand pinn_training_flat with CDAW accel, SHARP magnetic, multi-cluster labels
- `/neural-pipeline-validator` — validate sequence shape, Arrow dtype, TFT config, and ensemble head wiring

**Data schema reference:** `docs/DATA_SCHEMA_REFERENCE.md` — authoritative mapping of all DB files, table schemas, row counts, and the YAML-alias → actual-column-name translation table. Consult before any data work.

**Subagents (invoke via Agent tool):**
- `physics-validator` — use when editing anything in `SolarPipe.Training/Physics/`, `DragBasedModel`, `DormandPrinceSolver`, coordinate transforms, or `ProgressiveDragPropagator`
- `architecture-reviewer` — use when adding project references, new adapter/command types, `Program.cs` DI wiring, sweep components, or neural ensemble wiring

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `SOLARPIPE_REGISTRY` | Model artifact storage path |
| `SOLARPIPE_CACHE` | Checkpoint cache path |
| `SOLARPIPE_SIDECAR_ADDRESS` | gRPC sidecar address (default: `http://localhost:50051`) |
| `SOLARPIPE_SEQUENCES_PATH` | Path to Parquet sequence dataset (default: `data/sequences/`) |
