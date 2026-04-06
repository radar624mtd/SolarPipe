# SolarPipe Project Status Tracker

**Project**: ML Orchestration Framework for Space Weather Forecasting
**Status**: 🟢 In Progress — Phase 1
**Last Updated**: 2026-04-06 (Task 4.3 complete — Phase 1 done)
**Target Completion**: Q3 2026 (19 weeks from start — extended from 16 with 25% per-phase buffer)

---

## 📊 Executive Summary

| Metric | Status | Notes |
|--------|--------|-------|
| **Architecture** | ✅ Complete | Documented in SolarPipe_Architecture_Plan.docx |
| **CLAUDE.md** | ✅ Complete | Development guide created |
| **Automation Setup** | ✅ Complete | 2 skills, 2 agents, 3 hooks configured |
| **Implementation** | 🟢 In Progress | Phase 2 — Tasks 5.1, 5.2, 5.3, 7.2 done |
| **Overall Progress** | 37% | 14 of ~40 implementation tasks done; 120 tests passing (120 unit, 0 integration, 0 pipeline) |

---

## 🎯 Project Phases & Timeline

### Phase 1: Foundation (Weeks 1–5) — ✅ Complete
**Goal**: Core framework skeleton with basic ML.NET model training + CLI-agent infrastructure

**Deliverables**:
- [x] .NET 8 solution with all 6 projects + `Directory.Packages.props` (Central Package Management)
- [x] `TreatWarningsAsErrors` enabled in all `.csproj` files (clean terminal output)
- [x] Core interfaces: IDataFrame (with `AddColumn`, `ResampleAndAlign`), IDataSourceProvider, IFrameworkAdapter, ITrainedModel, IComposedModel, IModelRegistry
- [x] Domain-driven value types: `GseVector`, `GsmVector`, `SkyPlaneSpeed`, `RadialSpeed`, `PhysicalConstants`, `SpaceWeatherTimeParser`
- [x] InMemoryDataFrame (partial classes, `float[][]` backing) with NaN-as-missing policy
- [x] SqliteProvider and CsvProvider with sentinel value conversion (9999.9 → NaN)
- [x] YAML configuration loader with `Yaml12BooleanConverter` + null validation
- [x] `validate` CLI command with DAG validation and AI-readable error messages
- [x] MlNetAdapter with FastForest (FeatureFraction=0.7, dual seed pinning, LongRunning tasks)
- [x] FileSystemModelRegistry with atomic writes (temp + File.Move)
- [x] CLI host with `train`, `predict`, `validate` commands + structured JSON logging
- [x] `PhysicsTestFixtures.cs` with validated parameter sets
- [x] Unit tests with FluentAssertions, NSubstitute, `[Trait("Category", "Unit")]` segmentation

**Success Criteria**: `solarpipe validate --config test.yaml` passes in <1s → `solarpipe train --config test.yaml` trains FastForest → atomic registry save → `solarpipe predict` generates output. Zero build warnings.

**Estimated Effort**: 96–120 hours (5 weeks × 20–24 hrs/week, includes 25% buffer)

**Dependencies**: None

**Risks**:
- ⚠️ IDataFrame abstraction complexity; may need iteration on interface design
- ⚠️ YAML deserialization error handling
- ⚠️ ArrayPool lifecycle management in Dispose() pattern

---

### Phase 2: Physics & Composition (Weeks 6–9) — 🟡 Pending
**Goal**: Physics models, composition algebra, and early gRPC validation

**Deliverables**:
- [ ] PhysicsAdapter with DragBasedModel implementation (accepts `RadialSpeed`, not bare float)
- [ ] DragBasedModel: **Dormand-Prince RK4(5)** adaptive ODE solver (~200 lines, NOT MathNet RK4)
- [ ] NaN propagation guard after every ODE step (RULE-121)
- [ ] ComposeExpressionParser completion: all operators (chain, ensemble, residual, gate)
- [ ] ChainedModel, ResidualModel, EnsembleModel, GatedModel implementations
- [ ] EnsembleModel uses `TaskCreationOptions.LongRunning` for parallel training (ADR-014)
- [ ] **gRPC sidecar stub** (Week 6): proto schema, deterministic mock service, Arrow IPC validation (ADR-011)
- [ ] Composition algebra tested against both ML.NET (in-process) and gRPC stub (out-of-process)
- [ ] Data transforms: normalize, standardize, log_scale, lag, window_stats
- [ ] `IDataFrame.ResampleAndAlign()` temporal alignment primitive (RULE-122)
- [ ] Coupling functions: Newell, VBs, Borovsky
- [ ] Coordinate transform utility: Hapgood 1992 GSE↔GSM with `GseVector`/`GsmVector` types
- [ ] Integration tests for composition algebra

**Success Criteria**: Two-stage pipeline (`physics_baseline ^ rf_correction`) trains and predicts correctly. gRPC stub validates serialization round-trip. `GseVector` → `GsmVector` transform matches published test vectors.

**Estimated Effort**: 80–100 hours (includes 25% buffer)

**Dependencies**: Phase 1 complete

**Risks**:
- ⚠️ Dormand-Prince implementation correctness (numerical stability for extreme events)
- ⚠️ Composition type checking complexity
- ⚠️ gRPC serialization overhead may require IPC redesign (caught early via stub)

---

### Phase 3: Mock Data & Validation (Weeks 10–14) — 🟡 Pending
**Goal**: Synthetic data integration, rigorous validation, and pipeline resilience

**Deliverables**:
- [ ] ParquetProvider using **ParquetSharp** (NOT Parquet.Net) with safe disposal patterns (ADR-005)
- [ ] MockDataStrategy framework: pretrain_then_finetune, mixed_training, residual_calibration
- [ ] ResidualCalibrator: trains correction model on (obs - synthetic) residuals
- [ ] **ENLIL temporal isolation** enforcement (RULE-053)
- [ ] Expanding-window temporal cross-validation with gap buffers (3-7 days) (ADR-008)
- [ ] Purged cross-validation (de Prado 2018) for overlapping events
- [ ] Solar-cycle-aware splits (SC24 + SC25 in each fold)
- [ ] **EnbPI** adaptive conformal prediction for time-series UQ (~150 lines) (ADR-012)
- [ ] Standard split conformal prediction as fallback for non-temporal use cases
- [ ] **Pipeline state checkpointing** with `--resume-from-stage` (ADR-015)
- [ ] Data invariant tests: sentinel value rejection, NaN propagation, out-of-bounds detection
- [ ] Feature importance extraction and analysis
- [ ] Complete flux_rope_propagation_v1 pipeline configuration
- [ ] Integration tests with synthetic + real data

**Success Criteria**: Full flux_rope_propagation_v1 pipeline runs with temporal CV. Pipeline crash at Stage 4 resumes from checkpoint without re-running Stages 1-3. EnbPI intervals adapt across solar cycle phases.

**Estimated Effort**: 100–125 hours (includes 25% buffer)

**Dependencies**: Phase 2 complete

**Risks**:
- ⚠️ Mock data strategy effectiveness (need real ENLIL data)
- ⚠️ Temporal cross-validation edge cases (CME transit times)
- ⚠️ EnbPI sliding window parameter sensitivity to solar cycle phase transitions
- ⚠️ ParquetSharp P/Invoke: use-after-dispose produces access violations, not managed exceptions

---

### Phase 4: Python Sidecar & Advanced Models (Weeks 15–19) — 🟡 Pending
**Goal**: Deep learning via Python sidecar, ONNX export, full framework operational

**Deliverables**:
- [ ] Full Python sidecar server (replaces Phase 2 stub) with `IHostedService` lifecycle (RULE-060)
- [ ] **Sidecar uses workspace-relative Python path** (RULE-062, `${SOLARPIPE_ROOT}/python/.venv/bin/python`)
- [ ] Parent-process heartbeat — sidecar auto-kills if .NET PID dies (orphan prevention)
- [ ] Server-streaming RPCs for training with cooperative CancellationToken (RULE-061)
- [ ] **Large arrays via file-based Arrow IPC** (RULE-125), not inline Protobuf
- [ ] TFT (Temporal Fusion Transformer) trainer (tft_trainer.py)
- [ ] Neural ODE trainer — dynamics network export only to ONNX (ADR-006, RULE-070)
- [ ] PythonSidecarAdapter in .NET (gRPC client, Arrow IPC schema enforcement)
- [ ] OnnxAdapter with C# ODE solver wrapper for Neural ODE inference (~200 ORT calls/prediction)
- [ ] RestApiProvider for real-time L1 solar wind data (DSCOVR/ACE, DONKI API)
- [ ] BurtonOde physics implementation (Dormand-Prince solver, GSM-frame Bz only)
- [ ] NewellCoupling physics implementation
- [ ] Python sidecar Dockerfile + `logs/python_latest.json` structured logging (ADR-010)
- [ ] **CLI exit code translation** (RULE-141): `137 → OOM`, `139 → SIGSEGV`, etc.
- [ ] End-to-end tests with all framework adapters

**Success Criteria**: Framework fully operational with all 4 adapters. Sidecar crash produces readable error in `logs/`. Sidecar auto-terminates when host exits. Neural ODE dynamics network runs through ONNX + C# solver.

**Estimated Effort**: 120–150 hours (includes 25% buffer)

**Dependencies**: Phase 3 complete; Phase 2 gRPC stub validates proto schema

**Risks**:
- ⚠️ Python sidecar orphan processes — heartbeat mechanism must be bulletproof
- ⚠️ Neural ODE dynamics-only export may lose accuracy vs. full adaptive solver
- ⚠️ PyTorch model serialization and ONNX export compatibility
- ⚠️ Python environment management — agent must never use bare `python` command

---

## 📋 Detailed Task Breakdown

### Phase 1 Detailed Tasks

#### Week 1: Project Setup, Core Interfaces & CLI-Agent Infrastructure
- [x] Task 1.1: Create .NET 8 solution with 6 projects + build infrastructure
  - Create SolarPipe.sln with all projects (Core, Config, Data, Training, Prediction, Host)
  - Configure project references (unidirectional DAG)
  - Create `Directory.Packages.props` for Central Package Management (RULE-112)
  - Add `<TreatWarningsAsErrors>true</TreatWarningsAsErrors>` to all `.csproj` (RULE-113)
  - Create `logs/`, `cache/` directories with `.gitignore`
  - Configure Serilog structured JSON logging to `logs/dotnet_latest.json` (ADR-010)
  - Estimated: 5 hours

- [x] Task 1.2: Implement Core interfaces + domain-driven types
  - IDataFrame (GetColumn, Slice, SelectColumns, AddColumn, ResampleAndAlign, ToDataView, ToArray)
  - IDataSourceProvider (DiscoverSchemaAsync, LoadAsync)
  - IFrameworkAdapter (TrainAsync)
  - ITrainedModel (PredictAsync, SaveAsync, LoadAsync)
  - IComposedModel (PredictAsync)
  - IModelRegistry (RegisterAsync, LoadAsync, ListAsync)
  - Models: DataSchema (with ColumnInfo.MissingReason), PredictionResult, ModelMetrics, ModelArtifact
  - Domain types: GseVector, GsmVector, SkyPlaneSpeed, RadialSpeed (ADR-013)
  - PhysicalConstants.cs (RULE-132)
  - SpaceWeatherTimeParser.cs (RULE-133)
  - Estimated: 10 hours

- [x] Task 1.3: Test infrastructure
  - PhysicsTestFixtures.cs with validated parameter sets (NEVER use random floats)
  - Configure FluentAssertions + NSubstitute
  - `[Trait("Category", "Unit")]` segmentation
  - test fixtures directory with sample_config.yaml, test.csv, test.db
  - Estimated: 3 hours

- [x] Task 1.4: Implement InMemoryDataFrame
  - float[][] backing storage
  - Column access (by name and index)
  - Slicing and column selection
  - ToDataView conversion (MLContext compatibility)
  - Tests: 10 test cases
  - Estimated: 6 hours

#### Week 2: Data & Configuration
- [x] Task 2.1: SqliteProvider implementation
  - Connection management (Microsoft.Data.Sqlite)
  - Schema discovery (PRAGMA table_info)
  - Parameterized query execution
  - Type inference (int, float, string)
  - Tests: 8 test cases (multiple tables, nullable columns, query filters)
  - Estimated: 8 hours

- [x] Task 2.2: CsvProvider implementation
  - CsvHelper integration
  - Header detection and column mapping
  - Delimiter support (comma, tab, semicolon)
  - Type inference for columns
  - Tests: 6 test cases (multiple delimiters, missing values, type casting) + DataSourceRegistry
  - Estimated: 6 hours

- [x] Task 2.3: YAML configuration loader
  - YamlDotNet integration
  - PipelineConfig deserialization
  - Reference validation (data sources, stages, features)
  - Estimated: 8 hours

#### Week 3: ML.NET Adapter & Registry
- [x] Task 3.1: MlNetAdapter for FastForest
  - Framework dispatch pattern
  - Feature pipeline construction
  - Hyperparameter mapping (number_of_trees → NumberOfTrees, etc.)
  - ML.NET Regression metrics collection
  - Tests: 6 test cases (train/predict, metrics, hyperparameters)
  - Estimated: 8 hours

- [x] Task 3.2: FileSystemModelRegistry
  - Model artifact storage (versioned directories)
  - JSON metadata serialization
  - Semantic versioning (1.0.0)
  - Data fingerprinting
  - Tests: 5 test cases (register, load, list, versioning)
  - Estimated: 6 hours

#### Week 4: CLI & Integration
- [x] Task 4.1: CLI Host implementation
  - Program.cs entry point
  - DependencyInjection setup (IServiceCollection)
  - Command pattern for train, predict, validate, inspect
  - Estimated: 8 hours

- [x] Task 4.2: TrainCommand & PredictCommand
  - Config loading, stage selection, training orchestration
  - Model registry interaction
  - Output formatting (JSON, human-readable)
  - Tests: 4 integration tests (ValidateCommand valid/invalid, TrainCommand registers model, PredictCommand writes JSON)
  - Estimated: 6 hours

- [x] Task 4.3: Phase 1 integration test
  - Full end-to-end workflow: CSV → train → predict
  - Single stage, single model type (FastForest)
  - Estimated: 4 hours

---

### Phase 2 Detailed Tasks

#### Week 5-6: Physics & Composition Parsing
- [x] Task 5.1: ComposeExpressionParser
  - Tokenizer (→, +, ^, ?, parentheses, identifiers)
  - Recursive descent parser (operator precedence)
  - AST representation
  - Tests: 18 test cases (all operators, nested expressions, precedence, error cases)
  - Estimated: 10 hours

- [x] Task 5.2: DragBasedModel physics implementation
  - ODE solver (Dormand-Prince RK4(5) adaptive, FSAL)
  - Drag coefficient computation (γ·(v-w)·|v-w|)
  - Arrival time calculation (segment integration with crossing detection)
  - Input/output validation (RULE-032: v₀∈[200,3500], γ∈[0.2e-7,2.1e-7], dist≥20 R☉)
  - Tests: 12 test cases (slow/fast CMEs, Carrington-class stability, NaN propagation, save/load)
  - Estimated: 8 hours

- [x] Task 5.3: PhysicsAdapter framework
  - Equation registry pattern (IPhysicsEquation interface)
  - PhysicsAdapter implements IFrameworkAdapter (FrameworkType.Physics)
  - DragEquation + DragBasedModel registration
  - Tests: 4 test cases (framework type, supported models, train+predict end-to-end)
  - Estimated: 4 hours

#### Week 6-7: Composition Models
- [ ] Task 6.1: ChainedModel implementation
  - Sequential model chaining
  - Column name mapping (output columns → input columns)
  - Error handling for mismatched shapes
  - Tests: 4 test cases
  - Estimated: 4 hours

- [ ] Task 6.2: ResidualModel implementation
  - Baseline prediction
  - Feature augmentation (add baseline prediction as feature)
  - Residual prediction (observed - baseline)
  - Final output composition
  - Uncertainty propagation
  - Tests: 6 test cases
  - Estimated: 6 hours

- [ ] Task 6.3: EnsembleModel implementation
  - Weighted averaging
  - Weight normalization
  - Output shape compatibility
  - Tests: 4 test cases
  - Estimated: 4 hours

- [ ] Task 6.4: GatedModel implementation
  - Routing based on classifier output
  - Soft gating (weighted combination)
  - Uncertainty from routing entropy
  - Tests: 4 test cases
  - Estimated: 4 hours

#### Week 7-8: Data Transforms & Integration
- [ ] Task 7.1: TransformEngine & transforms
  - Normalize, standardize, log_scale, lag, window_stats
  - Coupling functions (Newell, VBs, Borovsky)
  - Transform chaining
  - Tests: 10 test cases
  - Estimated: 8 hours

- [x] Task 7.2: Dormand-Prince RK4(5) ODE solver implementation (~200 lines C#)
  - Reusable for DragBasedModel and BurtonOde (ADR-003, RULE-030)
  - FSAL tableau, embedded error estimation, adaptive step control (atol=1e-8, rtol=1e-6)
  - NaN propagation guard after every ODE step (RULE-121)
  - Scalar + vector overloads; both tested
  - Estimated: 6 hours

- [ ] Task 7.3: Phase 2 integration tests
  - Physics baseline + ML correction pipeline
  - Residual composition correctness verification
  - Estimated: 6 hours

---

### Phase 3 Detailed Tasks

#### Week 9-10: Data Providers & Mock Data
- [ ] Task 8.1: ParquetProvider implementation
  - ParquetSharp integration (NOT Parquet.Net — see ADR-005, RULE-050)
  - Schema discovery
  - Predicate pushdown
  - Partition handling
  - Tests: 6 test cases
  - Estimated: 6 hours

- [ ] Task 8.2: MockDataStrategy framework
  - Strategy pattern: pretrain_then_finetune, mixed_training, residual_calibration
  - Sample weighting
  - ResidualCalibrator: trains correction model on (obs - synthetic)
  - Tests: 6 test cases
  - Estimated: 8 hours

#### Week 10-11: Validation & Cross-Validation
- [ ] Task 9.1: Cross-validation implementations
  - LOOCV (leave-one-out)
  - Expanding-window temporal CV (gap buffer handling)
  - K-fold CV
  - Tests: 8 test cases
  - Estimated: 8 hours

- [ ] Task 9.2: Uncertainty quantification & feature importance
  - Tree variance extraction (RF predictions per tree)
  - Uncertainty propagation through compositions
  - Feature importance from tree-based models
  - Feature stability analysis
  - Tests: 6 test cases
  - Estimated: 8 hours

#### Week 11-12: Configuration & Integration
- [ ] Task 10.1: flux_rope_propagation_v1 configuration
  - Complete YAML configuration file
  - Data sources (SQLite CME catalog, Parquet ENLIL ensemble)
  - Two-stage pipeline (drag baseline, RF correction)
  - Mock data integration (residual calibration)
  - Estimated: 4 hours

- [ ] Task 10.2: Phase 3 integration tests
  - Full pipeline with mock + real data
  - LOOCV validation
  - Metrics collection and reporting
  - Estimated: 6 hours

---

### Phase 4 Detailed Tasks

#### Week 13: gRPC Sidecar & Python
- [ ] Task 11.1: gRPC proto definition
  - PythonTrainer service (Train, Predict, ExportOnnx)
  - Message types (TrainRequest, TrainResponse, etc.)
  - Arrow IPC serialization
  - Estimated: 4 hours

- [ ] Task 11.2: Python sidecar server
  - Flask/FastAPI with gRPC
  - Health checks and restart logic
  - Error handling and logging
  - Process management from .NET host
  - Tests: 4 test cases
  - Estimated: 8 hours

- [ ] Task 11.3: TFT trainer implementation
  - PyTorch Time Series Transformer setup
  - Training loop (epochs, validation)
  - Model serialization
  - Tests: 3 test cases
  - Estimated: 8 hours

#### Week 14: ONNX & Advanced Physics
- [ ] Task 12.1: OnnxAdapter implementation
  - ONNX Runtime integration
  - Model loading and inference
  - Input/output tensor mapping
  - Tests: 4 test cases
  - Estimated: 6 hours

- [ ] Task 12.2: Physics implementations (Phase 2 stub completion)
  - BurtonOde: dDst*/dt = Q(t) - Dst*/τ(VBs)
  - NewellCoupling: v^4/3 × B_T^2/3 × sin^8/3(θ_c/2)
  - ODE integration with adaptive time-stepping
  - Tests: 6 test cases (reference data validation)
  - Estimated: 8 hours

#### Week 15: REST API & Neural ODE
- [ ] Task 13.1: RestApiProvider
  - HTTP client setup (System.Net.Http)
  - JSON deserialization
  - NOAA DSCOVR/ACE real-time data support
  - DONKI API integration (CME catalog)
  - Tests: 4 test cases
  - Estimated: 6 hours

- [ ] Task 13.2: Neural ODE trainer
  - torchdiffeq integration
  - Training loop
  - ONNX export
  - Tests: 2 test cases
  - Estimated: 6 hours

#### Week 19: Integration & Polish
- [ ] Task 14.1: PythonSidecarAdapter
  - gRPC client implementation
  - Data serialization (Arrow IPC → IDataFrame)
  - Health monitoring and restart
  - Tests: 4 test cases
  - Estimated: 6 hours

- [ ] Task 14.2: Docker & deployment
  - Python sidecar Dockerfile
  - .NET host Docker setup
  - docker-compose for local development
  - Estimated: 4 hours

- [ ] Task 14.3: Phase 4 end-to-end tests
  - All framework adapters (ML.NET, ONNX, Physics, Python)
  - All composition operators
  - Full pipeline with all features
  - Estimated: 6 hours

---

## 🏗️ Module Dependency Map

```
SolarPipe.Core (interfaces & base models)
    ↓
    ├──→ SolarPipe.Config (YAML parsing, graph builder)
    ├──→ SolarPipe.Data (data providers, transforms)
    │     └──→ SolarPipe.Data.Providers (SQLite, CSV, Parquet, REST)
    ├──→ SolarPipe.Training (framework adapters)
    │     ├──→ SolarPipe.Training.MlNet
    │     ├──→ SolarPipe.Training.Onnx
    │     ├──→ SolarPipe.Training.Physics
    │     ├──→ SolarPipe.Training.Python (gRPC)
    │     └──→ SolarPipe.Training.Registry (model storage)
    ├──→ SolarPipe.Prediction (composition engine)
    │     └──→ SolarPipe.Prediction.Compose
    └──→ SolarPipe.Host (CLI entry point)

No circular dependencies; unidirectional DAG enables independent testing.
```

---

## 🎯 Success Criteria by Phase

### Phase 1
- ✅ `solarpipe train --config test.yaml` with single FastForest stage runs end-to-end
- ✅ All Core interfaces implemented
- ✅ InMemoryDataFrame, SqliteProvider, CsvProvider work correctly
- ✅ Model registry stores and retrieves models
- ✅ >80% code coverage for Core, Config, Data modules

### Phase 2
- ✅ Two-stage pipeline (physics ^ ML) trains and predicts
- ✅ All composition operators (→, +, ^, ?) work correctly
- ✅ Physics drag model produces reasonable arrival times
- ✅ Data transforms (normalize, lag, coupling) work
- ✅ >80% code coverage for Training, Prediction modules

### Phase 3
- ✅ flux_rope_propagation_v1 pipeline runs end-to-end
- ✅ LOOCV validation produces consistent results
- ✅ Mock data integration (residual calibration) improves model
- ✅ Feature importance and stability analysis functional
- ✅ >80% code coverage for validation strategies

### Phase 4
- ✅ All 4 framework adapters (ML.NET, ONNX, Physics, Python) operational
- ✅ gRPC sidecar trains and exports models correctly
- ✅ REST API provider fetches real-time solar wind data
- ✅ Burton ODE and Newell coupling produce correct physics
- ✅ Full end-to-end pipeline with all features
- ✅ >80% overall code coverage

---

## 📦 NuGet Dependencies (By Phase)

### Phase 1
- Microsoft.ML (ML.NET)
- YamlDotNet (YAML parsing)
- Microsoft.Data.Sqlite (SQLite)
- CsvHelper (CSV parsing)
- Microsoft.Extensions.DependencyInjection (DI)
- Microsoft.Extensions.Logging (logging)

### Phase 2
- MathNet.Numerics (matrix operations ONLY — NOT for ODE solving, see ADR-003)

### Phase 3
- ParquetSharp (Parquet support — NOT Parquet.Net, see ADR-005)

### Phase 4
- Microsoft.ML.OnnxRuntime (ONNX inference)
- Grpc.Net.Client (gRPC)
- Google.Protobuf (protobuf)
- Apache.Arrow (Arrow serialization)

Optional:
- Serilog (structured logging)
- OpenTelemetry (observability)

---

## 🚨 Comprehensive Risk Register

*Updated 2026-04-06 from pre-implementation audit (37 risks, 8 critical). See ARCHITECTURAL_DECISIONS.md for resolution details.*

### Critical Risks (Must Resolve Before/During Coding)

| # | Risk | Class | Phase | Failure Mode | Mitigation |
|---|------|-------|-------|-------------|------------|
| 1a | IDataView row-count mismatch in custom IDataFrame wrapper | Technical | 1 | Silent wrong predictions if columns have unequal length | Implement `GetRowCount()` returning verified `long?`; unit-test with `mlContext.Data.CreateEnumerable<T>(reuseRowObject: false)` round-trips |
| 2a | YAML 1.1 implicit typing ("Norway problem") | Technical | 1 | Config values silently wrong — `NO` → `false`, `1.0` → float when string expected | Write custom `IYamlTypeConverter` enforcing YAML 1.2 boolean rules (only true/false); enforce quoted strings in YAML configs |
| 3a | No adaptive/stiff ODE solver in MathNet.Numerics | Technical | 2 | Divergence in extreme geomagnetic storms (τ < 0.36 h → RK4 unstable) | Hand-code Dormand-Prince RK4(5) with adaptive stepping (~200 lines C#); implement BDF2 fallback for Carrington-class events |
| 5a | Coordinate frame confusion (GSE vs GSM) | Domain | 2 | Completely wrong Bz values — GSM requires dipole tilt angle correction | Use validated coordinate library (SunPy/SpacePy in sidecar); implement Hapgood 1992 transforms with unit tests against published test vectors |
| 7 | Neural ODE ONNX export is fundamentally impossible | Technical | 4 | Cannot deploy Neural ODE to .NET — adaptive solvers use dynamic control flow | Export only dynamics network f(y,t,θ) to ONNX; implement ODE solver in C# calling ORT per step; or keep inference in Python sidecar permanently |
| 2b | Null scalar silent corruption in YamlDotNet | Technical | 1 | Missing parameters in physics equations — `null` assigned to non-nullable properties | Enable `WithEnforceNullability()`; add post-deserialization validator walking all properties |
| 9 | Linux advisory-only file locking in model registry | Operational | 1 | Silent registry corruption — concurrent writes on Linux ignore `FileShare.None` | Use atomic write pattern: write to `.tmp_{Guid}`, then `File.Move(src, dest, overwrite: true)` (POSIX `rename()` is atomic) |
| 12 | FastForest hides individual tree predictions | Technical | 3 | Cannot compute proper uncertainty quantification — tree variance inaccessible | Implement split conformal prediction (~50 lines C#): sorted residuals on calibration set, (1−α) quantile as interval half-width |

### High-Severity Risks

| # | Risk | Class | Phase | Failure Mode | Mitigation |
|---|------|-------|-------|-------------|------------|
| 1b | FeatureFraction=1.0 disables RF feature bagging | Technical | 1 | Overfitting, no Random Forest diversity mechanism | Explicitly set `FeatureFraction = 0.7` and `FeatureFractionPerSplit = 0.7` in all FastForest configs |
| 1c | ReadOnlySpan\<float\> banned from async methods | Technical | 1 | Compile errors across entire pipeline — ref struct cannot be heap-allocated | Define all cross-layer interfaces using `float[]` or `Memory<float>`; confine Span to synchronous leaf methods |
| 4a | Sidecar orphan processes on host crash | Integration | 4 | GPU memory leak, port conflicts — Python survives .NET crash | Use Windows Job Objects (P/Invoke) / Linux `PR_SET_PDEATHSIG`; implement `IHostedService` for lifecycle |
| 5b | CDAW sky-plane speed projection bias | Domain | 2 | Underestimated arrival times — halo CME speeds dramatically underestimated | Document bias in pipeline output; consider de-projection correction factor |
| 6a | No temporal CV in ML.NET | Technical | 3 | Inflated performance metrics — random k-fold ignores temporal ordering | Implement expanding-window CV from scratch in C#; use ≥50 events per test fold; purge overlapping events |
| 6b | ENLIL data dependency leaks into test events | Data | 3 | Leakage through shared CME input parameters | Use ENLIL augmentation only from events strictly before test period; or perturb inputs ±20% |
| 8 | Parquet.Net loads entire row groups into memory (BANNED — use ParquetSharp) | Technical | 3 | OOM on GB-scale ENLIL ensemble files | Use ParquetSharp (4–10× faster, row-group random access); chunk files into ≤64 MB row groups |
| 4b | gRPC idle connection failure after ~9 hours | Integration | 4 | Training job failures on long runs | Use server-streaming RPCs with periodic progress; set `KeepAliveTime` and `KeepAlivePingTimeout` |

### Medium-Severity Risks

| # | Risk | Class | Phase | Failure Mode | Mitigation |
|---|------|-------|-------|-------------|------------|
| 10 | WSL2 DNS resolution failures | Operational | 4 | Developer environment instability under VPN | Disable auto-generated resolv.conf; use Docker Compose service names instead of `host.docker.internal` |
| 11a | Moq incompatible with Span/ref struct types | Test | 1 | Shapes all interface design — cannot mock `ReadOnlySpan<T>` | Use `float[]`/`Memory<float>` in interfaces; write hand-rolled fakes for span-consuming APIs |
| 3b | YAML merge keys partially broken | Technical | 1 | Config override rules violated | Avoid `<<` merge keys; use explicit YAML anchors with `*alias` instead |
| 14 | MLContext PredictionEngine not thread-safe | Technical | 1 | `IndexOutOfRangeException` under concurrent inference | Use `Microsoft.Extensions.ML.PredictionEnginePool<TIn, TOut>` from `Microsoft.Extensions.ML` package |
| 15 | ML.NET seed + thread interaction | Test | 1 | Non-deterministic test results | Pin `NumberOfThreads = 1`; set both `MLContext(seed: 42)` and `FeatureSelectionSeed` per trainer |
| 16 | Physics adapter versioning — no binary artifact | Technical | 2 | `LoadAsync` tries to open nonexistent model file | Metadata-only registry entries for physics; `LoadAsync` reconstructs from config, not binary |
| 17 | Data leakage via pre-split transforms | Data | 3 | Transforms fitted on full dataset before CV splits | Fit transforms only within each CV fold's training partition |
| 18 | Coupling function edge cases (B_T=0, θ_c=0) | Domain | 2 | Floating-point underflow, NaN propagation | Guard with `Math.Max(B_T, 1e-10)` before exponentiation; return 0.0 for Newell when B_T < threshold |

### Domain-Specific Risks

| # | Risk | Class | Phase | Impact | Verification Step |
|---|------|-------|-------|--------|-------------------|
| D1 | ENLIL ensemble availability — are 5,000 runs pre-generated? | Data | 3 | Blocks Phase 3 mock data integration | Confirm dataset exists, format, and access path before Phase 3 |
| D2 | solar_data.db catalog completeness — quality_flag ≥ 3 events with in-situ fits | Data | 1 | Insufficient training data if curation needed | Query database for event count, column presence, quality distribution |
| D3 | Observed_rotation_angle poorly defined for ambiguous events | Domain | 3 | Noisy target variable degrades model | Define inclusion criteria; filter events without clear magnetic cloud fits |
| D4 | Solar cycle phase labels missing from database | Data | 3 | Cannot perform cycle-aware CV splits | Add SC23/SC24/SC25 phase column; verify against SIDC sunspot number data |
| D5 | Coordinate frame inconsistency (HCI vs HAE vs GSM vs RTN) | Domain | 2 | Target variable corruption — silent | Document frame for every column in schema; implement Hapgood 1992 transforms |
| D6 | NOAA SWPC API downtime propagation | Operational | 4 | Real-time pipeline stalls on API outage | Implement caching with stale-while-revalidate; fallback to OMNI archive |
| D7 | Publication bias — well-studied events overweighted | Data | 3 | Model memorizes famous storms | Deduplicate catalog; verify no event appears more than once |

---

## 📈 Burndown & Milestones

```
Week    Phase 1     Phase 2     Phase 3     Phase 4     Total Tasks
────────────────────────────────────────────────────────────────
1       ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  10%
2       ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  20%
3       ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  30%
4       ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  40%
        ─────────────────────────────────────────────────────
5                   ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  42%
6                   ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  50%
7                   ████████████░░░░░░░░░░░░░░░░░░░░░░░░  60%
8                   ████████████████░░░░░░░░░░░░░░░░░░░░  65%
        ─────────────────────────────────────────────────────
9                                ████░░░░░░░░░░░░░░░░░░░░  68%
10                               ████████░░░░░░░░░░░░░░░░  72%
11                               ████████████░░░░░░░░░░░░  78%
12                               ████████████████░░░░░░░░  85%
        ─────────────────────────────────────────────────────
13                                           ████░░░░░░░  87%
14                                           ████████░░░░  92%
15                                           ████████████░  95%
16                                           ████████████████ 100%
```

---

## 🔍 Tracking & Monitoring

### Weekly Status Updates
- [ ] Tasks completed this week
- [ ] Blockers encountered
- [ ] Actual vs. planned hours
- [ ] Risk assessment
- [ ] Next week preview

### Phase Completion Checklist
- [ ] All tasks marked complete
- [ ] Code review passed (architecture-reviewer agent)
- [ ] >80% code coverage
- [ ] Integration tests passing
- [ ] Success criteria met
- [ ] Documentation updated

### Metrics to Track
- Lines of code per phase
- Test coverage by module
- Bug escape rate (bugs found in QA vs. development)
- Task estimation accuracy (actual hours vs. planned)
- Code review feedback cycles

---

## 📝 Notes & Constraints

1. **Greenfield Implementation**: No existing codebase to integrate with; clean start
2. **Single Developer**: Can be parallelized with future team; currently sequential planning
3. **Research Domain**: Space weather physics is novel area; may encounter unforeseen complexities
4. **Estimation Buffer**: 19-week timeline includes 25% per-phase buffer (extended from original 16-week estimate)
5. **Iteration Expected**: Architecture is solid, but implementation details may change
6. **Test-Driven**: Every feature has unit + integration test requirements
7. **Reference Pipeline**: flux_rope_propagation_v1 is primary validation target

---

## ✅ Sign-Off

- **Created**: 2026-04-06
- **Status**: Ready for Phase 1 Kickoff
- **Next Action**: Begin Week 1 tasks (project setup & Core interfaces)
- **Review Frequency**: Weekly status updates; phase-end reviews
