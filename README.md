# SolarPipe

A declarative ML orchestration framework for coronal mass ejection (CME) transit time prediction and geomagnetic storm forecasting.

**Target:** MAE ≤ 3h ± 2h on a 90-event holdout spanning 1818–2026.  
**Pre-tuning baseline:** 30.02h MAE (G7 gate active, target ≤ 6h).  
**Stack:** C# .NET 8 (orchestration, physics, data) · Python 3.12 (neural training via gRPC sidecar) · ONNX Runtime (inference).

---

## What it does

SolarPipe predicts how long a CME takes to travel from the Sun to Earth (L1). The pipeline is defined entirely in YAML — topology, model composition, and stage wiring are configuration, not code. The C# runtime resolves compose expressions like `drag_baseline ^ rf_correction` into `IComposedModel` implementations.

The two-headed neural ensemble architecture separates:

1. **Pre-launch encoder** — 150-hour OMNI time series (222 timesteps × 20 channels) + 105 scalar features from CDAW, SHARP magnetic parameters, flare data, PFSS topology, multi-CME context, and pre-launch solar wind state.
2. **In-transit encoder** — expanding L1 observation window (causally masked), learned version of density-modulated drag.
3. **Ensemble head** — concatenates both encoder outputs with physics baseline predictions → quantile output (P10/P50/P90).

---

## Architecture

```
configs/neural_ensemble_v1.yaml
        │
        ▼
SolarPipe.Config  ←── YAML pipeline/sweep loader, compose expression parser
        │
        ▼
SolarPipe.Host    ←── CLI: train, predict, sweep, validate, inspect commands
        │
        ├─► SolarPipe.Data       ←── SQLite, Parquet, REST, FITS providers
        │       ├─ InMemoryDataFrame  (typed, resample, transform)
        │       ├─ L1ObservationStream
        │       ├─ HmiFitsReader / JsocSynopticMapClient
        │       └─ Pfss/          (GPU-accelerated PFSS field line tracer)
        │
        ├─► SolarPipe.Training   ←── Model adapters, physics, sweep, CV, checkpoint
        │       ├─ Adapters/      (gRPC sidecar, ONNX, ML.NET, physics)
        │       ├─ Physics/       (DragBased, DensityModulatedDrag, BurtonODE,
        │       │                  NewellCoupling, DormandPrince solver)
        │       ├─ Sweep/         (HyperparameterGridSearch, DomainPipelineSweep)
        │       ├─ Validation/    (PurgedCV, ExpandingWindowCV, KFoldCV,
        │       │                  ConformalPredictor, EnbPiPredictor)
        │       └─ Checkpoint/    (atomic tmp+move, dependency ordering)
        │
        ├─► SolarPipe.Prediction ←── Composition: Ensemble, Residual, Chained, Gated
        │
        └─► python/              ←── PyTorch sidecar (gRPC)
                ├─ tft_pinn_model.py       (hand-rolled TFT transformer + PINN)
                ├─ physics_loss.py         (drag ODE residual, bound, quantile order)
                ├─ feature_schema.py       (133-column contract, enforced at every run)
                ├─ solarpipe_server.py     (gRPC server: train, predict, export ONNX)
                └─ datasets/
                    └─ training_features_loader.py  (masked dataset, split-leak guard)
```

---

## Data sources

The database spans 1818–2026 across eleven tables:

| Source | Content |
|--------|---------|
| OMNI (NASA/GSFC) | Hourly solar wind: speed, density, IMF Bz/Bt, pressure, AE/Dst/Kp indices |
| CDAW | CME catalog: speed, width, mass, acceleration, 2nd-order kinematics |
| DONKI | CME, SEP, MPC events; instrument-linked associations |
| SHARP (JSOC/HMI) | 14 photospheric magnetic parameters per active region |
| STEREO-A | In-situ ambient solar wind from a second vantage point |
| Richardson & Cane ICME | ICME type, Bz extremes, speed, bidirectional electron flags |
| GOES/NOAA | Flare class, peak flux, location |
| HEK | Coronal dimmings, EIT waves, filament eruptions, sigmoids |
| PFSS (JSOC synoptic) | Field topology: open/closed fraction, CH proximity, polarity |
| Solar indices | F10.7, international sunspot number |
| Ephemeris | ICRF positions + velocities, 16-worker parallel ingest, 1818–2026 |

A 16-bug audit identified and fixed all critical ingestion errors before the current training run. See [`docs/ENDPOINT_AUDIT_2026_04_15.md`](docs/ENDPOINT_AUDIT_2026_04_15.md).

---

## Neural model

The neural component is a hand-rolled **Temporal Fusion Transformer + Physics-Informed Neural Network** trained via a Python gRPC sidecar and deployed via ONNX Runtime.

**Architecture (TFT+PINN):**
- Flat encoder: 105 scalar features → MLP with residual blocks → 256-dim embedding
- Null handling: dense columns imputed; sparse columns get learnable null embeddings + binary mask
- Sequence encoder: 222 timesteps × 20 OMNI channels → multi-head attention (4 heads, 2 layers, d=128)
- Quantile head: P10/P50/P90 transit time predictions

**Physics losses (active):**
- `lambda_bound=0.1` — transit time bounds [12h, 120h]
- `lambda_qorder=0.5` — quantile ordering consistency (P10 ≤ P50 ≤ P90)
- `lambda_ode / lambda_mono = 0.0` — drag ODE residual and monotonic decay warm-started off; enabled post-convergence

**Hardware:** Quadro M4000 (Maxwell SM 5.2, 8 GB VRAM) · CUDA 12.1 · ONNX opset ≤ 20 with MatMul surgery for SM 5.2 compatibility.

---

## Gate board

| Gate | Name | Status | Exit commit |
|------|------|--------|-------------|
| G1 | Schema contract | ✅ | fbc1138 |
| G2 | Masked dataset loader | ✅ | f7c6074 |
| G3 | TFT+PINN model | ✅ | e55a7ce |
| G4 | Physics loss | ✅ | 2748bbe |
| G5 | ONNX export | ✅ | pending-commit |
| G6 | C# wiring + YAML | ✅ | 184ec83 / 6740308 |
| G6.5 | HPC validation (10 red-team items) | ✅ | pending-commit |
| **G7** | **Holdout quality gate** | **☐ active** | — |

G7 target: MAE ≤ 6h on 90-event holdout. Current baseline: 30.02h (underfitting; hyperparameter tuning in progress).

---

## Project layout

```
SolarPipe/
├── configs/                  # YAML pipeline/sweep definitions
│   ├── neural_ensemble_v1.yaml
│   ├── flux_rope_propagation_v1.yaml
│   ├── flux_rope_propagation_v2.yaml
│   └── phase8_*/  validate_2026.yaml  ...
├── src/
│   ├── SolarPipe.Config/     # YAML loader, compose expression parser
│   ├── SolarPipe.Core/       # Interfaces, domain types, physical constants
│   ├── SolarPipe.Data/       # Data providers, PFSS engine, DataFrame
│   ├── SolarPipe.Host/       # CLI entry point, all commands
│   ├── SolarPipe.Prediction/ # Composition implementations
│   ├── SolarPipe.Training/   # Training adapters, physics, sweep, CV
│   └── SolarPipe.Monitor/    # Run monitor
├── python/                   # gRPC sidecar (PyTorch training + ONNX export)
│   ├── tft_pinn_model.py
│   ├── physics_loss.py
│   ├── feature_schema.py     # authoritative 133-column contract
│   ├── solarpipe_server.py
│   └── tests/
├── docs/                     # Architecture, endpoint audit, source-schema reference
├── models/                   # ONNX + PyTorch checkpoints (gitignored — large binaries)
├── data/                     # Staging DB and sequences (gitignored — up to 11 GB)
├── SolarPipe.sln
├── global.json               # .NET 8 SDK pin
└── Dockerfile
```

---

## Build and run

**Prerequisites:** .NET 8 SDK · Python 3.12 · `pip install torch onnx grpcio` · CUDA 12.1 (optional)

```bash
# Build
dotnet build --no-restore

# Unit tests
dotnet test --filter Category=Unit --no-build

# All tests
dotnet test

# Train (runs gRPC sidecar automatically)
dotnet run --project src/SolarPipe.Host -- train --config configs/neural_ensemble_v1.yaml

# Predict against holdout
dotnet run --project src/SolarPipe.Host -- predict --config configs/neural_ensemble_v1.yaml

# Hyperparameter sweep
dotnet run --project src/SolarPipe.Host -- sweep --config configs/phase8_domain_sweep.yaml

# Validate event catalog
dotnet run --project src/SolarPipe.Host -- validate-events

# Inspect a trained model
dotnet run --project src/SolarPipe.Host -- inspect --model models/baselines/g6_tft_pinn_943e0a87/
```

**Python sidecar (standalone):**
```bash
cd python
python solarpipe_server.py   # starts gRPC server on :50051
```

---

## Key design decisions

**Compose expressions** — `drag_baseline ^ rf_correction` means residual correction on a physics baseline. The parser resolves this into a `ResidualModel` wrapping a `DragBasedModel` and an ML.NET correction layer.

**Physics-first baseline** — All training runs start from a physics drag model (DBM). Neural components learn residuals, not raw transit times.

**Causal masking** — The in-transit encoder sees only L1 observations available at prediction time. No future data leaks.

**Temporal CV** — `PurgedCV` with a 5-day gap buffer (RULE-051) prevents train/val leakage across the time dimension.

**Conformal prediction** — `SplitConformalPredictor` and `EnbPiPredictor` provide distribution-free prediction intervals calibrated on the validation fold.

**Atomic checkpoints** — All model artifacts written via tmp-file + `File.Move`; dependency ordering enforced (no stage enqueued until all `DependsOn` IDs complete).

---

## Credentials

The pipeline reads credentials from environment variables:

```
STANDFORD_JSOC_EMAIL   # JSOC/DRMS access (typo in key name is intentional)
NASA_API_KEY           # DONKI, HelioViewer, CCMC
NASA_API_EMAIL         # same address as JSOC
```

Never commit `.env` or `appsettings.local.json`.

---

## Documentation

| Document | Purpose |
|----------|---------|
| [`docs/NEURAL_ENSEMBLE_PLAN.md`](docs/NEURAL_ENSEMBLE_PLAN.md) | Gate board, per-model MAE history, active tuning state |
| [`docs/SOURCE_TO_SCHEMA_REFERENCE.md`](docs/SOURCE_TO_SCHEMA_REFERENCE.md) | Per-channel source ranking, wire format, sentinel values, DB schema |
| [`docs/ENDPOINT_AUDIT_2026_04_15.md`](docs/ENDPOINT_AUDIT_2026_04_15.md) | Live endpoint audit: 16 bugs found, all 🔴 items fixed |
| [`docs/ENDPOINT_CSHARP_MAPPING.md`](docs/ENDPOINT_CSHARP_MAPPING.md) | Promotion lifecycle: Exploring → Characterized → Promoted |
| [`docs/HPC_OPTIMIZATION_PLAN.md`](docs/HPC_OPTIMIZATION_PLAN.md) | Hardware-constrained optimization (Quadro M4000 SM 5.2) |
| [`docs/HPC_REDTEAM_REVIEW_2026_04_18.md`](docs/HPC_REDTEAM_REVIEW_2026_04_18.md) | Red-team review: all 10 items ✅ |

---

## License

MIT
