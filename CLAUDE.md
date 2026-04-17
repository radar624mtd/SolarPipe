# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SolarPipe is a .NET 8 declarative ML orchestration framework for CME (coronal mass ejection) propagation modeling and geomagnetic storm prediction. Pipelines are defined in YAML — topology, model selection, and composition rules are configuration, not code.

**Core concept:** YAML compose expressions wire stages together (e.g., `drag_baseline ^ rf_correction` means residual correction on top of a physics baseline). The runtime resolves this into `IComposedModel` implementations.

## Target Language: C# (Python is exploration only)

**C# is the production target for all data access, physics computation, and ML orchestration.**

Python files in `data/src/` and `scripts/` serve two roles only:
1. **Bootstrapping ingest** — populate SQLite DBs that C# reads from
2. **Endpoint exploration** — prototype API calls, characterize sentinels/rate-limits/pagination before C# implementation

Once a data-source endpoint is fully characterized in Python, it must be mapped to a C# implementation in `src/SolarPipe.Data/`. The promotion lifecycle and all endpoint knowledge are maintained in:

**[`docs/SOURCE_TO_SCHEMA_REFERENCE.md`](docs/SOURCE_TO_SCHEMA_REFERENCE.md)** — canonical per-channel reference: for every model input channel, all upstream sources ranked by reliability, wire format, sentinel values, ingest method, and exact DB schema mapping. **Consult before writing any ingest script, SQL query, or feature join.**

**[`docs/ENDPOINT_CSHARP_MAPPING.md`](docs/ENDPOINT_CSHARP_MAPPING.md)** — per-endpoint promotion lifecycle (Exploring → Characterized → Promoted), C# implementation notes, and the promotion checklist.

**[`docs/ENDPOINT_AUDIT_2026_04_15.md`](docs/ENDPOINT_AUDIT_2026_04_15.md)** — live endpoint audit (2026-04-15): verified response structures, confirmed sentinels, 16 bugs found, optimal ingest strategy per channel. **Read before touching any ingest client.** Contains a bug table (§9) — all 🔴 items must be fixed before the next ingest run.

**Known live bugs in Python ingest clients (from audit — ALL B1–B10 🔴 bugs fixed 2026-04-15):**
| Bug | File | Status |
|---|---|---|
| `mc_flag`/`bde_flag` column indices swapped | `ingest_rc_icme.py` | ✅ Fixed |
| `bz_min_nt` maps to nonexistent column | `ingest_rc_icme.py` | ✅ Fixed (removed from row dict) |
| MC flag values are `0/1/2/2H`, not Y/N/H | `ingest_rc_icme.py` + schema | ✅ Fixed (`_parse_mc_flag` added) |
| `api.nasa.gov` DONKI fallback → 503 | `donki.py` | ✅ Fixed (removed) |
| CDAW Type II freq unit is kHz not MHz | `ingest_radio_bursts.py` | ✅ Fixed (÷1000 applied) |
| CDAW Type II end-date missing year | `ingest_radio_bursts.py` | ✅ Fixed (`_infer_end_date` with rollover) |
| `Halo` CPA not → NULL | `ingest_radio_bursts.py` | ✅ Already handled in `_clean()` |
| HEK `event_peaktime` stored as `"masked"` | `hek.py` | ✅ Fixed (`_is_masked()` guard) |
| CDAW sentinel `"----"` (4 dashes) missing | RULE-204 | ✅ Fixed (added to sentinel sets) |
| GOES archive endpoint → 404 | `noaa_indices.py` | ✅ Fixed (replaced with Hesperia archive) |
| GFZ Kp/F10.7 API → HTTP 500 | any Kp ingest | Switch to OMNI2 dat primary + GFZ single-index gap-fill (B13, 🟡) |
| ACE MAG (`AC_H0_MFI`) ends 2026-03-22 | `pyspedas_l1.py` | Use Wind/MFI (`WI_H0_MFI`) as fallback — confirmed live through 2026-04-07 (see §11c) |

**Accepted data gaps (web-searched 2026-04-15 — see audit §11 for full detail):**
| Source | Gap | Alternative |
|---|---|---|
| WSO HCS tilt | CR 2302+ (2025-10 onward) | GONG PFSS → `hcs_tilt_radial` only; classic not replicable |
| CDAW Type II radio | Sep 2024+ | None for DH range; CDAWeb raw CDF requires detection pipeline |
| Wind/WAVES Type III | All dates | No pre-built catalog; raw CDFs ~54 GB — skipped, explicit approval 2026-04-16 |
| R&C ICME catalog | Sep 2025+ | HELIO4CAST ICMECAT v2.3 CSV (through Jul 2025, no MC flag) |
| HEK | All 2026 events | None |
| ACE MAG | 2026-03-23+ | **Wind/MFI `WI_H0_MFI`** — native GSM, hourly, confirmed live ✅ |

**Rules:**
- Before writing any new ingest script or SQL join: check `docs/SOURCE_TO_SCHEMA_REFERENCE.md` for the authoritative column names, sentinel rules, and ingest method
- Before running any ingest client: check the bug table in `docs/ENDPOINT_AUDIT_2026_04_15.md` §9 and verify the relevant 🔴 bugs are fixed
- Never write a new C# data client without first checking `docs/ENDPOINT_CSHARP_MAPPING.md`
- Never merge a C# client that hasn't completed the Promotion Checklist in §3 of that document
- After any Python endpoint exploration session, update all three docs before ending the session
- Python clients in `data/src/solarpipe_data/clients/` that have been fully promoted to C# are kept as reference implementations only — they must not be extended unless the C# client is updated in the same PR

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

# Python sidecar — NEVER use venv; always use Python 3.12 directly
# Python 3.12 path: C:/Users/radar/AppData/Local/Programs/Python/Python312/python.exe
# In bash/hooks use: /c/Users/radar/AppData/Local/Programs/Python/Python312/python.exe
python3.12 solarpipe_server.py --port 50051

# Compile proto stubs (required after any proto change; NOT committed to git)
python3.12 -m grpc_tools.protoc \
    -I python/ --python_out=python/ --grpc_python_out=python/ python/solarpipe.proto

# Python sidecar tests
cd python && python3.12 -m pytest tests/ -v              # all
cd python && python3.12 -m pytest tests/ -v -m "not live" # unit only
cd python && python3.12 -m pytest tests/ -v -m live       # live gRPC server tests

# Feature matrix & sequence extraction
python3.12 scripts/build_pinn_feature_matrix.py          # rebuild pinn_training_flat (45 cols)
python3.12 scripts/build_expanded_feature_matrix.py      # expanded matrix: CDAW accel + SHARP + multi-cluster
python3.12 scripts/build_pinn_sequences.py               # extract 150h OMNI sequences per event → Parquet

# OMNI backfill
python3.12 scripts/backfill_omni_gap_ace1h.py            # fill post-2026-03-29 gap from ACE 1h feed

# Docker
docker compose up sidecar        # Start gRPC sidecar only
docker compose run --rm host train --config configs/...
```

## Python

**NEVER use venv.** Always invoke `python3.12` directly — no `.venv`, no virtual environments of any kind. Full path: `C:/Users/radar/AppData/Local/Programs/Python/Python312/python.exe`. In bash scripts and hooks use: `/c/Users/radar/AppData/Local/Programs/Python/Python312/python.exe`.

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

**`solar_data.db`** — 11 GB L1 archive at repo root (`solar_data.db`):
- `donki_ips` — 644 shock arrivals with CME linkage
- `donki_cme` — 8K CMEs: speed, half-angle, lat/lon, analysis_type
- `omni_hourly` — 561K rows, 60 columns, 1963–2026; OMNI gap post-2026-03-29 **closed** via ACE/DSCOVR backfill (all 9 affected holdout events have 100% bz_gsm coverage)
- `cdaw_cme` — 42K CMEs with linear+2nd-order speed, acceleration, mass, KE
- `ml_clusters` — 219K events, 4 clustering methods (k=5/8/12, dbscan=96)
- `mag_predictions` — 2.7M rows: Phase 8 Dst predictions by 5 models
- `symh_hourly` — 1-min Sym-H 1981–2026
- `gfz_kp_ap` — 3-hourly Kp/Ap + daily F10.7
- `goes_xrs_flares` — 70K flares with location

**`data/data/staging/staging.db`** — feature engineering and model tables:
- `pinn_training_flat` — 1,974 events, 45 engineered columns, train/holdout split
- `pinn_expanded_flat` — 1,974 events, 108 columns (Tier 1+2 features populated — see implementation state below)
- `training_features` — SQLite VIEW: LEFT JOIN of `feature_vectors` (9,418 rows) with `pinn_expanded_flat` (1,974 rows) → 133 columns. **Primary C# access point for neural ensemble training.** Use `filter: split IS NOT NULL` to select training/holdout events.
- `sharp_keywords` — 102K rows, 1,930 active regions, 24 magnetic parameters
- `hek_events` — HEK coronal dimmings/EIT waves/filaments/sigmoids (staging.db, NOT solar_data.db)
- `sep_events` — DONKI SEP events linked to CMEs
- `rc_icme` — Richardson & Cane ICME catalog
- `pfss_topology` — PFSS open/closed field topology per CME
- `wind_waves_type2` — CDAW Type II radio bursts
- `hcs_tilt` — WSO/GONG heliospheric current sheet tilt
- `flares` — GOES XRS flares (Hesperia archive + DONKI)

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
| stdout progress on every action | Every loop, stage, and task must print `step X of Y — <action> [status]` to stdout at each increment; program-level start/end banners required; no silent execution |

## Python Sidecar Gotchas

| Gotcha | Detail |
|--------|--------|
| **NEVER use venv** | Always invoke `python3.12` (`/c/Users/radar/AppData/Local/Programs/Python/Python312/python.exe`) — no `.venv`, no `python/.venv/Scripts/python`, no virtual environments of any kind |
| `onnx` package required | `torch.onnx.export` needs `onnx` installed; included in `requirements.txt` |
| Max ONNX opset is 20 | PyTorch 2.5.1 supports opsets up to 20, not 21; server clamps via `min(opset, 20)` |
| Proto stubs not committed | `solarpipe_pb2.py` / `solarpipe_pb2_grpc.py` are generated — recompile after any proto change |
| `live` pytest mark | Tests that spin up an in-process gRPC server are `@pytest.mark.live`; defined in `python/pytest.ini` |
| Predict input columns must match training | TFT/LSTM input size is fixed at train time; predict Arrow file must have same feature columns |
| `_SimpleTftModel` is a stub | Single-layer LSTM only; full TFT requires pytorch-forecasting |
| Sequence extraction not pre-computed | Must run `build_pinn_sequences.py` before training sequence models |

## Active Phase

**Goal: MAE ≤ 3h ± 2h via two-headed neural ensemble**

**Data pipeline status (2026-04-17):**
- ✅ Tier 1+2 ingest complete: `pinn_expanded_flat` = 1,974 × 108 cols
- ✅ `training_features` VIEW = 9,418 × 133 cols (C# access point)
- ✅ `data/sequences/` Parquet: 1,884 train + 90 holdout × 222 timesteps × 20 OMNI channels
- ✅ G1 schema contract (`python/feature_schema.py`, 25 tests) — commit fbc1138
- ✅ G2 masked dataset loader (`python/datasets/training_features_loader.py`, 18 tests) — commit f7c6074
- ✅ `configs/neural_ensemble_v1.yaml` — 105-feature list, training + eval config, G7 quality gate
- ✅ G3 TFT+PINN model (`python/tft_pinn_model.py`, 15 tests) — commit e55a7ce; hand-rolled `nn.TransformerEncoder` (pytorch-forecasting rejected; see plan §G3)
- ✅ G4 physics loss (`python/physics_loss.py`, 30 tests) — commit 2748bbe; γ₀ in km⁻¹ matching `DragBasedModel.cs`, clamped to [1e-9, 1e-6]
- ⏳ G5: ONNX export + parity test + `use_tft_pinn` server flag (deferred from G3)
- ⏳ G6: C# `OnnxAdapter` wiring round-trip
- ⏳ G7: Holdout MAE ≤ 6h quality gate
- ⏳ SuperMAG (SME/SMR): blocked on account activation — sequence expansion to 24 channels pending (RULE-213 atomic)

Full implementation tracking: `docs/NEURAL_ENSEMBLE_PLAN.md`

Implementation phases (sequential):

| Phase | Task | Expected MAE | Status |
|---|---|---|---|
| P1 | OMNI backfill (ACE 1h feed) | unblocks Mar-30/Apr-01 | ✅ complete (2026-04-10) |
| P2 | Expand feature matrix (CDAW accel + SHARP + multi-cluster + Tier 1+2) | ~7h | ✅ complete (2026-04-17) |
| P3 | Sequence extraction: 222-step OMNI per event → Parquet | prereq for P4 | ✅ complete (existing Parquet) |
| P4 | Full TFT+PINN in sidecar (replace stub LSTM) | ~5–6h | ⏳ G3 in progress |
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
- `/feature-matrix-builder` — expand pinn_training_flat with CDAW accel, SHARP magnetic, multi-cluster labels, and Tier 1/2 ingest columns
- `/neural-pipeline-validator` — validate sequence shape, Arrow dtype, TFT config, and ensemble head wiring
- `/data-ingest-runner` — run or review Tier 1/2 data acquisition batch runs (HEK, PFSS, ACE/DSCOVR, SEP, STEREO, GOES MAG, SuperMAG); enforces RULE-200–216
- `/omni-gap-fixer` — diagnose and fix the OMNI data gap post-2026-03-29; unblocks Phase 9 for the 4 affected holdout events

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
| `STANDFORD_JSOC_EMAIL` | JSOC/DRMS email for solar data access (note: typo in key name is correct) |
| `NASA_API_KEY` | NASA DONKI / HelioViewer / CCMC API key |
| `NASA_API_EMAIL` | NASA API registration email |

## Solar Data Ingest (solar_toolkit)

`solar_ingest_processing_guide.json` is in this workspace root — the authoritative
API reference for all SunPy ecosystem packages. Consult it before writing any
solar data fetch or calibration code.

**Package:** `solar_toolkit 0.1.0` (editable install at `C:/Users/radar/solar_toolkit/`)

```python
import solar_toolkit as st   # lazy-loads 15 sub-packages
# Preferred canonical imports:
from sunpy.map import Map
from sunpy.net import Fido, attrs as a
from aiapy.calibrate import register, update_pointing, correct_degradation
from aiapy.calibrate.utils import get_correction_table, get_pointing_table
import drms, os

# Credentials — always from env vars:
JSOC_EMAIL = os.environ["STANDFORD_JSOC_EMAIL"]   # typo in key name is the actual key
NASA_KEY   = os.environ["NASA_API_KEY"]
```

**Do NOT use** `st.aia`, `st.search`, `st.image` — these SimpleNamespace shortcuts
do not exist in the installed version.

### Key ingest patterns for SolarPipe

| Data need | Source | Pattern |
|-----------|--------|---------|
| AIA EUV images | SDO/JSOC | `Fido.search(a.Time(...), a.Instrument.aia, a.Wavelength(...))` |
| HMI magnetograms | SDO/JSOC | `Fido.search(..., a.Instrument.hmi, a.Physobs.los_magnetic_field)` |
| OMNI in-situ (L1) | CDAweb | `pyspedas.omni.data(trange=...)` |
| GOES XRS flares | NOAA | `pyspedas.goes.xrs(trange=...)` or `sunkit_instruments.goes_xrs` |
| PSP/WIND/ACE | CDA/SPDF | `pyspedas.<mission>.<instrument>(trange=...)` |
| Solar Orbiter | SOAR | `import sunpy_soar; Fido.search(..., a.Source("Solar Orbiter"))` |
| CDAW CME catalog | already in solar_data.db | query `cdaw_cme` table directly |
| DONKI events | NASA API | `requests.get(f"https://api.nasa.gov/DONKI/FLR?api_key={NASA_KEY}")` |

### Quality gates for Python ingest scripts
```bash
python -m py_compile scripts/<file.py>
ruff check --select E,F scripts/<file.py>
```

---

## ABSOLUTE IMPERATIVE: Data Acquisition Roadmap (RULE-200–RULE-215)

**THIS SECTION IS NON-NEGOTIABLE. ALL RULES BELOW ARE MANDATORY. DEVIATION FROM THE IMPLEMENTATION ORDER OR ARTIFACT SPECIFICATIONS REQUIRES EXPLICIT USER APPROVAL BEFORE ANY CODE IS WRITTEN.**

This roadmap was approved by the project owner on 2026-04-12. Its purpose is to drive SolarPipe from the current 4.42h MAE (Phase 9, 15-event subset) to the target MAE ≤ 3h ± 2h on the full 90-event holdout via a two-headed neural ensemble. Every data source listed below has been evaluated for physics relevance, implementation cost, and match rate against the 1,974-event training corpus. The ordering is final.

| Rule | Mandate |
|------|---------|
| RULE-200 | Implement data sources in Tier order (Tier 1 → Tier 2 → Tier 3). Never start a Tier 2 source before all Tier 1 sources have passed verification. All ingest batch runs use `multiprocessing.Pool` — NOT `threading.Thread`. Python's GIL means threads cannot achieve true parallelism; only separate OS processes can. Implementation mandates: (1) `multiprocessing.get_context("spawn").Pool(processes=12)` — 12 independent OS processes, each with its own GIL and CPU core; (2) worker function must be a top-level module function (not a lambda or nested def) so it is picklable by the spawn context; (3) within each worker process, all sub-queries for a single item (e.g. 4 HEK event types) fire concurrently via `ThreadPoolExecutor(max_workers=4)` + `as_completed` — GIL releases during blocking network I/O, giving 4× throughput per process; (4) two-phase execution: Phase 1 = bulk download via `pool.imap_unordered(fn, work, chunksize=1)` — all worker processes fetch into memory, parent collects results as they stream in; Phase 2 = bulk write in parent process only — single-threaded, no DB access in worker processes; (5) DB writes use bulk `INSERT OR IGNORE` (`on_conflict_do_nothing()`) in batches of 500 — never per-row SELECT+INSERT; (6) parent prints `step X of Y` for every result received from `imap_unordered` (RULE-216); (7) pool closed and joined (`pool.close()` + `pool.join()`) before function returns — no fire-and-forget. |
| RULE-201 | Never skip a Tier 1 source. All 5 must be implemented: R&C ICME, HEK, DONKI SEP, ACE/DSCOVR L1, PFSS. |
| RULE-202 | Every new client must follow BaseClient pattern (async httpx, RateLimiter, SHA256 file cache, retry on 429/5xx) UNLESS the data source requires a non-HTTP protocol (HEK uses sunpy.net.hek, PFSS uses local compute, pyspedas uses CDAWeb). Non-HTTP exceptions must be documented in the client file docstring. |
| RULE-203 | Every new ingest module must use `queries.upsert()` with `sqlalchemy.dialects.sqlite.insert` (RULE-030). Never use raw INSERT or executemany. |
| RULE-204 | Sentinel conversion is mandatory at ingest boundary (RULE-003). Values matching `{"---", "...", "****", "N/A", "", "9999", "-9999"}` or abs(v) > 9990 must become NULL before any row dict is assembled. |
| RULE-205 | The `pinn_expanded_flat` table must have exactly 1,974 rows after every feature matrix rebuild. Row count parity with `pinn_training_flat` is a hard gate — abort and diagnose if violated. |
| RULE-206 | Train/holdout split (1,884 train / 90 holdout, pre-2026 / 2026) must be preserved exactly. Never add holdout events to the training set. |
| RULE-207 | When adding new flat features, extend `_NEW_COLS` in `build_expanded_feature_matrix.py` and add corresponding NULL-rate checks to the audit report section. |
| RULE-208 | Sequence shape is (N, T, C) — float32 throughout. When adding new sequence channels (Tier 2), update OMNI_CHANNELS in `build_pinn_sequences.py`, update `n_seq_channels` in `python/tft_model.py`, and update `SEQ_CHANNELS` in `python/solarpipe_server.py` atomically. |
| RULE-209 | PFSS computation uses `sunkit_magex.pfss` with HMI synoptic maps (`hmi.synoptic_mr_polfil_720s`). GONG synoptic maps are the fallback only. Never substitute a simpler magnetic model. |
| RULE-210 | ACE/DSCOVR gap-fill uses COALESCE upsert — never overwrite existing non-null OMNI values. Spacecraft tag must be 'ACE_DIRECT' or 'DSCOVR_DIRECT'. |
| RULE-211 | HEK queries use sunpy.net.hek.HEKClient with ThreadPoolExecutor timeout. Query window is ±2h around CME launch time. Event types: CD (coronal dimming), CW (EIT wave), FI (filament), SG (sigmoid). |
| RULE-212 | R&C ICME matching uses ±6h window on disturbance_datetime vs arrival_time. MC flag values are numeric in source: 0=no cloud, 1=cloud, 2=probable cloud, 2H=hybrid. `_parse_mc_flag()` converts to int. |
| RULE-213 | All Tier 2 sequence channels (GOES_Hp, GOES_Bt, SME_nT, SMR_nT) must be added in a single atomic update to sequences + model + server — no partial states where sequences have 22 channels but model expects 20. |
| RULE-214 | Tier 3 sources are contingent on Tier 1+2 model holdout evaluation showing residual error addressable by those sources. Do not implement Tier 3 without running the full Tier 1+2 ensemble first. |
| RULE-215 | All new Python files in `data/src/` and `scripts/` must pass `python -m py_compile` and `ruff check --select E,F` before being considered complete. |
| RULE-216 | Every Python script and ingest function must emit structured stdout progress at each action boundary. Required format: `print(f"step {i} of {n} — {action} [{status}]", flush=True)`. Rules: (1) loops over events/rows/files print at every increment; (2) every top-level function prints a start banner (`=== <function_name> start ===`) and end banner (`=== <function_name> done — N processed, M skipped, K errors ===`); (3) `flush=True` is mandatory on all progress prints so output is not buffered; (4) errors print inline as `step X of Y — <action> [ERROR: <msg>]` and do not suppress the counter. No script may run silently. |

### Current Implementation State (as of 2026-04-16)

**Tier 1 — COMPLETE (code and data)**

| Source | Client | Ingest | Schema | Feature cols | Data populated |
|--------|--------|--------|--------|-------------|----------------|
| 1.1 R&C ICME | `clients/rc_icme.py` ✓ | `ingest_rc_icme.py` ✓ | `RichardsonCaneICME` ✓ | rc_icme_type, rc_bde_flag, rc_v_icme, rc_b_max, rc_bz_min, rc_matched | 576/1974 matched (29%) ✓ |
| 1.2 HEK | `clients/hek.py` ✓ | `ingest_hek_events.py` ✓ | `HekEvent` ✓ | has_coronal_dimming, has_eit_wave, has_filament_eruption, has_sigmoid | 1671/1974 (84.7%) ✓ — table in staging.db |
| 1.3 DONKI SEP | `clients/donki.py` ✓ | `ingest_donki_sep.py` ✓ | `SepEvent` ✓ | has_sep, sep_onset_delay_hours | 97/1974 matched (5%) ✓ |
| 1.4 ACE/DSCOVR L1 | `clients/pyspedas_l1.py` ✓ | `ingest_ace_dscovr_direct.py` ✓ | uses SolarWindHourly | (no flat features — fills OMNI gaps) | OMNI gap closed ✓ — all 9 holdout events 100% covered |
| 1.5 PFSS | `clients/pfss_compute.py` ✓ | `ingest_pfss_topology.py` ✓ | `PfssTopology` ✓ | pfss_field_type, pfss_open_fraction_10deg, pfss_ch_distance_deg, pfss_polarity | 1749/1974 (88.6%) ✓ |

**`pinn_expanded_flat` current state:** 1,974 rows × 108 columns. Tier 1+2 features populated. `training_features` view (133 cols) is the C# access point. Note: `eit_wave_speed_kms` and `dimming_area` are NULL by design — HEK CD/CW FRMs do not populate quantitative fields (existence-only flags per `hek.py` docstring). SuperMAG sequence channels (SME_nT, SMR_nT) pending account activation.

**Tier 1 verification gates (RULE-200) — ALL PASSED:**
1. ✓ Schema gate: all 4 Tier 1 tables created without error
2. ✓ Ingest gate: each source ingested, row count > 0, idempotent (skip logic verified in code)
3. ✓ Feature matrix gate: 1,974 rows, NULL rates within expected bounds
4. ✓ Sequence gate: shape unchanged at (N, 222, 20), float32

### Tier 1 — Full Specification

#### 1.1 Richardson & Cane ICME Catalog
Gold-standard manually curated ICME list. ICME type (magnetic cloud vs ejecta) splits the transit-time distribution bimodally.

- Client: `data/src/solarpipe_data/clients/rc_icme.py` — BaseClient, source URL: `izw1.caltech.edu/ACE/ASC/DATA/level3/icmetable2.htm`
- Ingest: `data/src/solarpipe_data/ingestion/ingest_rc_icme.py` — BeautifulSoup HTML parse, 18-cell row format
- Schema: `RichardsonCaneICME` — PK: `disturbance_datetime`, cols: mc_flag, bde_flag, v_icme_kms, v_max_kms, b_max_nt, bz_min_nt, dst_min_nt
- Features: `rc_icme_type` (0=N, 1=Y, 2=H), `rc_bde_flag`, `rc_v_icme`, `rc_b_max`, `rc_bz_min`, `rc_matched`

#### 1.2 HEK Coronal Dimmings & EIT Waves
Coronal dimmings = CME mass evacuation (r=0.6 correlation with CME mass). EIT waves = CME lateral extent. ~35% coverage (SDO era 2010+).

- Client: `data/src/solarpipe_data/clients/hek.py` — sunpy.net.hek.HEKClient, ThreadPoolExecutor timeout, NOT BaseClient
- Ingest: `data/src/solarpipe_data/ingestion/ingest_hek_events.py` — event types CD/CW/FI/SG, ±2h window per CME
- Schema: `HekEvent` — PK: autoincrement id, cols: event_type, start/end/peak time, hpc_x/y, area_atdiskcenter, wave_speed_kms, activity_id
- Features: `dimming_area`, `dimming_asymmetry`, `has_eit_wave`, `eit_wave_speed_kms`, `has_filament_eruption`, `has_sigmoid`

#### 1.3 DONKI SEP Events
SEP = proxy for CME-driven shock strength. ~12% of events have associated SEP.

- Client: `fetch_sep()` method added to `data/src/solarpipe_data/clients/donki.py`
- Ingest: `data/src/solarpipe_data/ingestion/ingest_donki_sep.py`
- Schema: `SepEvent` — PK: sep_id, cols: event_time, instruments, linked_event_ids
- Features: `has_sep`, `sep_onset_delay_hours`

#### 1.4 ACE/DSCOVR Direct L1 via pyspedas
Fills OMNI gap post-2026-03-29. Provides redundancy for full 2010–2026 record. Unblocks Phase 9 for full 90-event holdout.

- Client: `data/src/solarpipe_data/clients/pyspedas_l1.py` — pyspedas.ace.mfi/swe, pyspedas.dscovr.mag/fc, NOT BaseClient
- Ingest: `data/src/solarpipe_data/ingestion/ingest_ace_dscovr_direct.py` — COALESCE upsert, fills NULL rows only
- Schema: uses existing `SolarWindHourly` table; spacecraft = 'ACE_DIRECT' / 'DSCOVR_DIRECT'
- Features: none new — enables existing OMNI sequences to cover gap period

#### 1.5 PFSS Open/Closed Field Topology
Whether CME propagates through open vs closed field determines ambient drag coefficient. ~65% coverage (events with known source location + HMI synoptic maps 2010+).

- Client: `data/src/solarpipe_data/clients/pfss_compute.py` — sunkit_magex.pfss local compute, JSOC HMI synoptic maps, NOT BaseClient
- Ingest: `data/src/solarpipe_data/ingestion/ingest_pfss_topology.py` — per-CME, Carrington rotation → synoptic map → field line trace
- Schema: `PfssTopology` — PK: activity_id, cols: carrington_rotation, source_lat/lon_cr, field_type, open_flux_fraction_10deg, nearest_ch_distance_deg, polarity, rss
- Features: `pfss_field_type`, `pfss_open_fraction_10deg`, `pfss_ch_distance_deg`, `pfss_polarity`

### Tier 2 — Full Specification (implement after Tier 1 verification)

#### 2.1 STEREO-A Beacon MAG/PLASTIC
STEREO-A orbits ahead of Earth; solar wind measurements preview structures the CME will encounter. ~40% coverage (angular separation 30–120 deg).

- Client: `data/src/solarpipe_data/clients/stereo.py` — pyspedas.stereo.mag(), pyspedas.stereo.plastic()
- Ingest: `data/src/solarpipe_data/ingestion/ingest_stereo_wind.py`
- Schema: `StereoWindHourly` — PK: datetime+spacecraft, cols: B_total, Bz, density, speed, temp, stereo_carr_lon
- Features: `stereo_lead_hours`, `stereo_speed_ambient`, `stereo_density_ambient`, `stereo_available`

#### 2.2 GOES Geosynchronous Magnetic Field
Near-Earth B-field compression = precursor signal for ICME arrival. Feeds in-transit encoder.

- Client: `data/src/solarpipe_data/clients/goes_mag.py` — pyspedas.goes.fgm()
- Ingest: `data/src/solarpipe_data/ingestion/ingest_goes_mag.py`
- Schema: `GoesMagHourly` — PK: datetime, cols: Hp, He, Hn, Bt, satellite_id
- Sequence channels: `GOES_Hp`, `GOES_Bt` (2 new channels, in-transit window)

#### 2.3 Wind/WAVES Type III Radio Bursts — **SKIPPED (dataset size)**
No pre-built catalog exists (audited 2026-04-16). Detection requires downloading raw Wind/WAVES L2 RAD1+RAD2 CDF spectrograms (~54 GB for 2010–2026 at 20s/82s cadence). Dataset size is prohibitive. e-CALLISTO is listed as Tier 3.4 but is ground-based with RFI/weather gaps. Skip condition met — explicit user approval granted 2026-04-16.

#### 2.4 DONKI Magnetopause Crossings (MPC)
Extreme geomagnetic compression events. Binary flag for the most impactful ICMEs.

- Client: `fetch_mpc()` method added to `data/src/solarpipe_data/clients/donki.py`
- Ingest: `data/src/solarpipe_data/ingestion/ingest_donki_mpc.py`
- Schema: `MagnetopauseCrossing` — PK: mpc_id, cols: event_time, instruments, linked_event_ids
- Features: `has_mpc`, `mpc_delay_hours`

#### 2.5 SuperMAG Indices
1-min resolution geomagnetic indices from 100+ stations. SME/SMR resolve sub-storm structure better than Kp/AE. Feeds in-transit encoder.

- Client: `data/src/solarpipe_data/clients/supermag.py` — BaseClient, API at supermag.jhuapl.edu, requires `SUPERMAG_LOGON` env var
- Ingest: `data/src/solarpipe_data/ingestion/ingest_supermag.py` — hourly averages
- Schema: `SupermagHourly` — PK: datetime, cols: SML, SMU, SME, SMR, n_stations
- Sequence channels: `SME_nT`, `SMR_nT` (2 new channels, in-transit window)

#### 2.6 CDAW Mass Quality Filtering
CME mass (~40% NULL) has quality flags in `remarks` not currently used.

- No new client or ingest module — enhancement to `build_expanded_feature_matrix.py` only
- Feature: `cdaw_mass_quality` (0=poor event, 1=fair, 2=good) parsed from cdaw_cme.remarks

### Tier 3 — Mandatory (implement after Tier 2 verification gate)

Tier 3 is **not conditional**. Every source must be fully sourced, audited, validated, and mapped. Only if a source proves technically impossible (endpoint dead, coverage < 5 events, no viable substitute) may it be skipped — and only with explicit user approval before any code is written. C# promotion does not begin until all viable Tier 3 sources are complete.

| Source | Rationale | Skip condition (requires explicit approval) |
|--------|-----------|----------------------------------------------|
| 3.1 SDO/AIA Direct Dimming | Quantitative dimming area/depth from AIA 193Å difference images; replaces HEK proxies | Only if JSOC AIA 193Å coverage < 5 matched events or pipeline compute prohibitive |
| 3.2 Solar Orbiter SWA/MAG | In-situ 0.3–1.0 AU; sparse coverage (launched 2020) | Only if matched event count < 5 after full catalog scan |
| 3.3 PSP Inner Heliosphere | < 0.25 AU during perihelia; < 20 coincident events | Only if matched event count < 5 after full catalog scan |
| 3.4 eCALLISTO Ground-Based Radio | 25–600 MHz Type II/III network; incremental over Wind/WAVES | Only if network API is unavailable or < 5 matched events |
| 3.5 STEREO-A 3D Geometry Quality | Angular separation → GCS model quality flag; depends on 2.1 | Only if GCS reconstruction data is unavailable for > 95% of events |

### Verification Gates (RULE-200 enforcement)

Each tier must pass ALL gates before the next tier begins:

**Tier 1 Verification:**
1. Schema gate: `init_db('test.db')` creates all 4 new tables without error
2. Ingest gate: Each source ingests a 30-day window (2024-01-01 to 2024-01-31), row count > 0, idempotent re-run produces identical row count
3. Feature matrix gate: `pinn_expanded_flat` has exactly 1,974 rows, no NaN/Inf in new columns, NULL rates within expected bounds (R&C ~70%, HEK ~65%, SEP ~95%, PFSS ~35%)
4. Sequence gate: Shape (N, 222, 20), all float32 — no change until Tier 2

**Tier 2 Verification — COMPLETE (2026-04-17):**
1. ✓ All Tier 1 gates still passing
2. ✓ Ingest complete: STEREO-A (133,969 rows), GOES MAG (43,104 rows), MPC (95 rows); Type III SKIPPED (54 GB raw CDFs, no catalog); SuperMAG BLOCKED (account not activating)
3. ✓ Feature matrix gate: `pinn_expanded_flat` 1,974 rows × 108 cols; `training_features` view 9,418 rows × 133 cols in staging.db
4. ✓ Value audit passed: physical range checks, sentinel checks, cross-source validation all clean
5. ⏳ GOES + SuperMAG sequence integration (Shape → N, 222, 24): pending SuperMAG account activation
6. ⏳ Ensemble holdout MAE with Tier 1+2 features: pending neural ensemble training (P4-P6)

**Tier 3 Verification:**
1. All Tier 3 sources fully sourced, audited, validated, and mapped (or explicitly skipped with user approval)
2. Tier 1+2 ensemble MAE documented as baseline before Tier 3 integration
3. Final ensemble MAE target: ≤ 3h ± 2h on 90-event holdout
4. C# promotion begins only after all viable Tier 3 sources pass this gate
