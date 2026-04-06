# SolarPipe Architectural Decision Records (ADRs)

**Created**: 2026-04-06
**Purpose**: Document key architectural decisions, especially those forced by the pre-implementation risk audit, with context, alternatives considered, and consequences.

---

## ADR-001: Use float[] instead of ReadOnlySpan<float> in IDataFrame interface

**Date**: 2026-04-06
**Status**: Accepted
**Audit Risk**: 1c (Critical), 11a (Medium)

**Context**: The original architecture plan specified `ReadOnlySpan<float>` for `IDataFrame.GetColumn()` to enable zero-copy column access. However:
- `ReadOnlySpan<T>` is a ref struct in C# 12/.NET 8 and **cannot exist as a local in any async method**
- The async state machine stores locals as heap fields, which ref structs cannot be
- C# 13 relaxes this slightly but still prohibits spanning an `await` boundary
- Moq/NSubstitute cannot mock `ReadOnlySpan<T>` parameters (compile error)
- All pipeline operations (`TrainAsync`, `LoadAsync`, `PredictAsync`) are async

**Decision**: All cross-layer interfaces use `float[]`. `ReadOnlySpan<float>` is confined to synchronous leaf methods inside `InMemoryDataFrame` internals only (via `internal GetColumnSpan()`).

**Alternatives considered**:
1. `Memory<float>` — Heap-safe, but adds complexity and `.Span` indirection at every use site. Would work but adds ceremony for small datasets.
2. `ReadOnlyMemory<float>` — Same as above; slightly better semantics but unfamiliar to ML.NET ecosystem.
3. Keep `ReadOnlySpan<float>` and make all methods synchronous — Would eliminate async throughout the pipeline, defeating the purpose of non-blocking I/O.

**Consequences**:
- Small allocation overhead for returning `float[]` (array reference) vs zero-copy span
- Negligible for SolarPipe's dataset sizes (~500 events × 20 features)
- Enables full mockability of IDataFrame in unit tests
- All interfaces are async-safe

---

## ADR-002: Custom YAML 1.2 Boolean Type Converter

**Date**: 2026-04-06
**Status**: Accepted
**Audit Risk**: 2a (Critical)

**Context**: YamlDotNet 16.x implements YAML 1.1, which defines 22 boolean literals: `yes`, `no`, `on`, `off`, `y`, `n` and their case variants. In a scientific configuration system for space weather, values like `coordinate_frame: NO` or `use_gse: on` would silently deserialize as booleans instead of strings. This is known as the "Norway problem."

**Decision**: Implement a custom `IYamlTypeConverter` (`Yaml12BooleanConverter`) that enforces YAML 1.2 boolean rules: only `true`/`false`/`True`/`False`/`TRUE`/`FALSE` are valid booleans. All other values throw a descriptive error. Register this converter in every `DeserializerBuilder`.

**Alternatives considered**:
1. Switch to a YAML 1.2 parser — No mature .NET YAML 1.2 parser exists. YamlDotNet is the de facto standard.
2. Require all string values to be quoted — Too error-prone; YAML users don't naturally quote strings.
3. Use TOML or JSON for configuration — Would lose YAML's readability advantages; TOML lacks nested structures needed for pipeline config.
4. Post-deserialization type checking — Too late; by the time we check, the value is already `false` not `"NO"`.

**Consequences**:
- Every `DeserializerBuilder` must include the converter (enforced by RULE-020)
- Pipeline YAML authors who try to use YAML 1.1 booleans get clear error messages
- Marginal deserialization overhead (negligible)

---

## ADR-003: Hand-coded Dormand-Prince RK4(5) ODE Solver

**Date**: 2026-04-06
**Status**: Accepted
**Audit Risk**: 3a (Critical)

**Context**: MathNet.Numerics 5.0.0 provides only fixed-step RK2 and RK4 plus Adams-Bashforth in its `OdeSolvers` namespace. There is no adaptive step-size, no error estimation, no implicit solvers, no BDF methods. Fixed-step RK4's stability region on the negative real axis extends to |hλ| < 2.785. For Burton ODE with τ < 0.36h (Carrington-class events), |hλ| exceeds this and the solution diverges.

**Decision**: Hand-code Dormand-Prince RK4(5) with adaptive step-size control (~200 lines C#). The FSAL (First Same As Last) tableau provides embedded error estimation between 4th and 5th order solutions. Step-size control: `h_new = h * min(5, max(0.2, 0.9 * (tol/err)^0.2))`. Use `atol=1e-8, rtol=1e-6` as defaults.

**Alternatives considered**:
1. CenterSpace NMath `RungeKutta45OdeSolver` — Commercial license; adds vendor dependency.
2. SUNDIALS CVODE via P/Invoke — Production-grade stiff solver but heavy C interop complexity, deployment issues.
3. Fixed-step RK4 with very small step size (h=0.01h) — Wastes ~100x compute for the 99% of events where τ > 6h; still unstable for extreme events.
4. Port `scipy.integrate.solve_ivp` logic — Essentially the same as hand-coding Dormand-Prince, which is what scipy uses internally.

**Consequences**:
- ~200 lines of well-documented C# to implement and maintain
- Must be thoroughly tested: convergence order, stability boundary, analytical solution comparison
- Reusable for both drag-based CME model and Burton ODE
- For truly stiff scenarios (τ < 0.36h), may need BDF2 or implicit Euler in Phase 4 (deferred)

---

## ADR-004: Atomic File Write Pattern for Model Registry

**Date**: 2026-04-06
**Status**: Accepted
**Audit Risk**: 9 (High)

**Context**: The model registry uses file-system-based storage. On Linux, .NET's `FileStream` with `FileShare.None` uses `flock()`, which is **advisory-only** — non-cooperating processes can read/write the file without acquiring the lock. This makes concurrent model writes in Docker/Linux deployments silently dangerous.

**Decision**: Use atomic write pattern: write to a temp file (`$".tmp_{Guid.NewGuid():N}"`), compute SHA-256 fingerprint, then `File.Move(tempPath, finalPath, overwrite: true)`. POSIX `rename()` is atomic on the same filesystem. For cross-process coordination, use named `Mutex` (`Global\\SolarPipeRegistry`).

**Alternatives considered**:
1. `FileStream` with `FileShare.None` — Advisory-only on Linux; unsafe.
2. SQLite-backed registry — Adds dependency; SQLite has its own locking semantics that work cross-platform.
3. Redis/distributed lock — Overkill for file-system registry; appropriate for production MLflow migration.
4. `.lock` file convention — Same advisory problem; race conditions between check and create.

**Consequences**:
- All writes go through temp-then-move pattern
- Named Mutex provides cross-process safety (implemented on Linux via shared memory)
- Temp files must be cleaned up on crash recovery (scan for `.tmp_*` on startup)
- SHA-256 fingerprint computed on every write (fast; ~100MB/s on modern hardware)

---

## ADR-005: ParquetSharp Instead of Parquet.Net

**Date**: 2026-04-06
**Status**: Accepted
**Audit Risk**: 8 (High)

**Context**: Parquet.Net (pure managed C#) loads entire row groups into memory. Issue #59 documented 1 GB CSV→Parquet consuming 7+ GB RAM because all data was in a single row group. Issue #81 documented a float data integrity bug producing incorrect values starting at index 38 with Snappy compression.

**Decision**: Use ParquetSharp (G-Research, wrapping Apache Parquet C++ via P/Invoke) for all Parquet operations. It is 4-10x faster and supports random-access row group reading via `file.RowGroup(i)`.

**Alternatives considered**:
1. Parquet.Net with multiple small row groups — Mitigates memory but not the float bug; still slower.
2. Apache Arrow .NET with Parquet support — Arrow C# can read Parquet but has its own 2 GiB buffer limit; less mature Parquet integration.
3. Convert all Parquet to CSV — Loses columnar storage benefits (predicate pushdown, compression).

**Consequences**:
- P/Invoke dependency — use-after-dispose produces access violations, not managed exceptions
- All native handles must be wrapped in `using` statements
- ParquetSharp NuGet includes native binaries for Windows/Linux/macOS
- Must chunk files into ≤64 MB row groups during write

---

## ADR-006: Neural ODE Inference Strategy

**Date**: 2026-04-06
**Status**: Accepted
**Audit Risk**: 7 (Critical)

**Context**: torchdiffeq's adaptive ODE solvers use data-dependent while loops, step rejection logic, and error estimation — all dynamic control flow that ONNX's static graph model cannot represent. The adjoint method uses custom `autograd.Function` that `torch.jit.trace` and `torch.jit.script` cannot capture. PyTorch maintainers confirmed this limitation (#121280). This blocks ONNX export of the full Neural ODE model.

**Decision**: Three-tier strategy:
1. **Primary (production)**: Export only the dynamics network `f(y, t, θ)` to ONNX. Implement the ODE solver (Dormand-Prince RK4(5)) in C#, calling ONNX Runtime once per solver step. With RK4 and 50 steps → ~200 ORT inference calls per prediction, feasible for SolarPipe batch sizes.
2. **Interim**: Keep Neural ODE inference entirely in Python sidecar via gRPC.
3. **Alternative**: Fixed-step Euler/RK4 unrolling into static ONNX graph (accuracy trade-off).

**Alternatives considered**:
1. Switch to torchode (Lienen & Günnemann 2022) — JIT-compatible for forward-mode only; adjoint backward pass still incompatible.
2. Wait for PyTorch ONNX exporter improvements — No roadmap for dynamic control flow support.
3. Abandon Neural ODE entirely — Loses a promising architecture for space weather dynamics.
4. Use TensorRT or other inference frameworks — Same static graph limitations as ONNX.

**Consequences**:
- Neural ODE will always require either the Python sidecar or a C# ODE solver wrapper
- The dynamics network export path requires careful validation that f(y,t,θ) alone produces correct outputs
- ~200 ORT calls per prediction adds latency vs. single inference call; acceptable for offline batch prediction
- C# solver is already implemented (ADR-003), so reuse is straightforward

---

## ADR-007: Split Conformal Prediction for Uncertainty Quantification

**Date**: 2026-04-06
**Status**: Accepted
**Audit Risk**: 12 (High)

**Context**: ML.NET's `FastForestRegressionModelParameters` does not expose individual tree predictions through the public API. The naive tree variance approach (computing σ² across B tree predictions) is systematically overconfident — it captures estimation uncertainty but ignores irreducible noise, producing intervals that are too narrow (Wager et al. 2014, JMLR).

**Decision**: Use split conformal prediction (~50 lines of C#). Train the forest, compute sorted absolute residuals on a held-out calibration set, use the (1−α) quantile as the prediction interval half-width. This provides **finite-sample coverage guarantees** regardless of the underlying model, without needing access to individual trees.

**Alternatives considered**:
1. Tree variance from individual trees — ML.NET hides individual trees; even if accessible, intervals are overconfident.
2. Ensemble of forests (different seeds) — Crude but implementable; high compute cost, no coverage guarantees.
3. ONNX extraction of individual trees — Complex parsing of ONNX graph; fragile across ML.NET versions.
4. Conformalized Quantile Regression (CQR) — Better (heteroscedastic intervals) but requires quantile regression models not easily available in ML.NET.
5. QOOB method (Gupta et al. 2022) — State-of-the-art but requires custom implementation; candidate for Phase 4+.

**Consequences**:
- Simple implementation (~50 lines), easy to test and validate
- Coverage guarantees hold for any model, not just RF
- Requires a calibration set (split from training data) — reduces effective training size by ~20%
- Intervals are exchangeable (same width for all predictions) — does not capture heteroscedastic uncertainty
- For storm forecasting, conservative intervals are preferable to overconfident ones

---

## ADR-008: Custom Temporal Cross-Validation Engine

**Date**: 2026-04-06
**Status**: Accepted
**Audit Risk**: 6a (High), 6b (High)

**Context**: ML.NET's `CrossValidate()` performs random k-fold only. There is no temporal cross-validation, no gap buffer support, and no purged CV implementation. For space weather events, temporal leakage through overlapping storms, rolling statistics, and ENLIL shared parameters produces inflated performance metrics.

**Decision**: Implement custom `ExpandingWindowCV` in C# with:
- 5 folds, ≥50 events per test set
- Configurable gap buffers (3-7 days)
- Purged CV (de Prado 2018) — remove training samples whose event windows overlap with test events
- Solar-cycle-aware stratification
- ENLIL temporal isolation enforcement

**Alternatives considered**:
1. Delegate to Python sidecar using scikit-learn's `TimeSeriesSplit` — Adds dependency on sidecar for validation; gap parameter only available since v0.24.
2. Use `timeseriescv` Python package `PurgedKFoldCV` — Same sidecar dependency; more feature-complete.
3. Random k-fold with caveats documented — Unacceptable for publication-quality results; reviewers would reject.
4. LOOCV — High variance for N=300-500; not recommended for hyperparameter tuning.

**Consequences**:
- Custom C# implementation (~300-500 lines) to build and maintain
- Must handle edge cases: events spanning fold boundaries, variable-length storms, solar cycle transitions
- Validation results will be more pessimistic (but more honest) than random k-fold
- ENLIL augmentation constrained to pre-test-period events reduces synthetic data utility

---

## ADR-009: ArrayPool<float> for Large Object Heap Mitigation

**Date**: 2026-04-06
**Status**: Accepted
**Analysis Risk**: float[] LOH allocation under high-cadence multi-year datasets

**Context**: ADR-001 chose `float[]` over `ReadOnlySpan<float>` in interfaces. However, any `float[]` exceeding 85KB (≈21,250 elements) lands on the Large Object Heap, which is only compacted during expensive Gen 2 collections. Multi-year 1-minute cadence datasets (525,600 rows/year × 20 columns) will rapidly fragment the LOH, causing GC pauses and eventually `OutOfMemoryException` during Phase 3 training runs — with no profiler available in a CLI-only environment.

**Decision**: Use `ArrayPool<float>.Shared.Rent()/Return()` inside `InMemoryDataFrame` internals for backing storage. The public interface still returns `float[]` (satisfying ADR-001), but the implementation leases from the pool and copies only when data escapes the dataframe's lifetime via `ToArray()`. Add a `--profile-memory` CLI flag that logs `GC.GetTotalMemory()` and LOH size after every pipeline stage to `logs/memory_profile.json`.

**Alternatives considered**:
1. `ReadOnlyMemory<float>` in interfaces — Would fix LOH but rejected in ADR-001 for async/mock complexity.
2. Accept LOH fragmentation — Acceptable for Phase 1 (500 events), breaks in Phase 3 (5000 synthetic + multi-year OMNI).
3. Manual memory-mapped files — Too complex; appropriate if datasets exceed available RAM.

**Consequences**:
- `InMemoryDataFrame.Dispose()` must return all rented arrays — failure leaks pool memory
- Pool arrays may be larger than requested; always track actual length separately
- `GetColumn()` returns a correctly-sized `float[]` (copy from pool array) — slight overhead but satisfies interface contract
- Memory profiling flag enables diagnosis without external tools

---

## ADR-010: Structured JSON Logging with Separate Process Streams

**Date**: 2026-04-06
**Status**: Accepted
**Analysis Risk**: Terminal output splicing between C# and Python processes

**Context**: In Phase 4, the C# host and Python sidecar run concurrently. Their stdout/stderr streams interleave chaotically in a single terminal. When a PyTorch tensor shape mismatch causes a crash, the agent reads garbled output mixing .NET async state machine logs with Python tracebacks, making root cause diagnosis extremely expensive in token context.

**Decision**: Force structured JSON logging to independent files:
- C#: `logs/dotnet_latest.json` via Serilog `WriteTo.File(new CompactJsonFormatter(), ...)`
- Python: `logs/python_latest.json` via Python `logging` with `JSONFormatter`
- Both include a shared `TraceId` (passed via gRPC metadata) for cross-process correlation
- Console output shows only: stage progress, completion messages, and critical errors
- CLI provides `solarpipe parse-logs --trace-id <id>` to join both log files by trace

**Alternatives considered**:
1. Single interleaved console log — Unreadable for both humans and agents.
2. Serilog Seq sink — Requires running a Seq server; overkill for single-developer CLI workflow.
3. OpenTelemetry collector — Appropriate for production; too heavy for development iteration.

**Consequences**:
- Agent reads clean log files instead of parsing garbled terminal
- Log rotation needed (keep last 10 runs, ~50MB cap)
- Slight latency from file I/O; acceptable for debugging use

---

## ADR-011: Early gRPC Sidecar Stub in Phase 2

**Date**: 2026-04-06
**Status**: Accepted
**Analysis Risk**: Composition algebra validated only against ML.NET creates false confidence

**Context**: The composition algebra (Chain, Ensemble, Residual, Gate) is built in Phase 2, but the Python sidecar is Phase 4. If composition is only tested against simple ML.NET models, latency, serialization overhead, and multi-output model handling from out-of-process inference are invisible until Phase 4 — when discovery of problems could force rewriting the core composition engine.

**Decision**: Create a lightweight gRPC stub service in Phase 2 (Week 6) that:
- Implements the same proto interface as the real sidecar
- Returns deterministic mock predictions (no actual PyTorch)
- Introduces real gRPC latency and serialization overhead
- Tests composition algebra with both in-process (ML.NET) and out-of-process (stub) models
- Validates Arrow IPC schema enforcement end-to-end

**Alternatives considered**:
1. Defer all sidecar work to Phase 4 — Risk of discovering fundamental composition issues too late.
2. Build full sidecar in Phase 2 — Too much scope; PyTorch integration isn't needed for stub testing.
3. Mock gRPC with in-process fake — Misses real serialization and latency characteristics.

**Consequences**:
- ~2 days of Phase 2 time for stub + proto definitions + integration tests
- Composition algebra is validated against both synchronous and asynchronous model backends early
- Proto schema is locked in Phase 2, not Phase 4

---

## ADR-012: Adaptive Conformal Prediction (EnbPI) for Time-Series UQ

**Date**: 2026-04-06
**Status**: Accepted (amends ADR-007)
**Analysis Risk**: Standard split conformal prediction assumes exchangeability; space weather is non-stationary

**Context**: ADR-007 chose split conformal prediction for UQ. However, standard split conformal fundamentally assumes data is **exchangeable** (i.i.d.). Space weather data is highly non-stationary due to the 11-year solar cycle — solar maximum events are qualitatively different from solar minimum events. Using standard conformal prediction produces miscalibrated intervals: too wide during solar minimum (wasting forecast precision), dangerously narrow during solar maximum (underestimating extreme event risk).

**Decision**: Implement Ensemble Predictors with Prediction Intervals (EnbPI, Xu & Xie 2021) as the primary UQ method. EnbPI:
- Does not assume exchangeability
- Uses a sliding window of recent residuals instead of a fixed calibration set
- Adapts interval width to changing data distributions (solar cycle phases)
- Still provides asymptotic coverage guarantees for time series
- Implementation is ~150 lines of C# (vs. ~50 for standard conformal)

Retain standard split conformal as a fallback for non-temporal use cases.

**Alternatives considered**:
1. Standard split conformal (ADR-007 original) — Exchangeability assumption violated; miscalibrated.
2. Conformalized Quantile Regression (CQR) — Better for heteroscedastic intervals but requires quantile loss; deferred to Phase 4+.
3. Bayesian credible intervals — ML.NET FastForest hides individual trees; not implementable.
4. QOOB (Gupta et al. 2022) — State-of-the-art but complex; candidate for future phases.

**Consequences**:
- ~150 lines of C# instead of ~50 — still manageable for single developer
- Requires a sliding window parameter (default: last 100 events) that should be tuned per solar cycle phase
- Must be tested against synthetic data with known distribution shifts
- Standard conformal retained for non-temporal predictions (e.g., feature importance)

---

## ADR-013: Domain-Driven Type Safety for Physics

**Date**: 2026-04-06
**Status**: Accepted
**Analysis Risk**: AI agents cannot reliably enforce domain constraints through code review alone

**Context**: Space weather physics involves coordinate systems (GSE vs GSM), uncorrected vs corrected speeds (sky-plane vs radial), and physical constants. AI agents generating code will inevitably mix coordinate frames, use uncorrected CDAW speeds directly, or scatter magic numbers in formulas. Code review (human or AI) catches these intermittently; the type system catches them always.

**Decision**: Implement domain-driven value types as `readonly record struct`:
- `GseVector` / `GsmVector` — No implicit conversion; explicit transform required
- `SkyPlaneSpeed` / `RadialSpeed` — DragBasedModel accepts only `RadialSpeed`
- `PhysicalConstants` static class — All constants centralized; no inline float literals for physical quantities
- `SpaceWeatherTimeParser` — Handles OMNI (Year+DOY+Hour+Min), CDAW, and ACE catalog formats; replaces `DateTime.Parse()`

**Alternatives considered**:
1. Runtime validation only (RULE-032) — Catches at runtime, not compile time; agent may not run tests.
2. Comment-based documentation — Agent ignores comments; types are enforced by the compiler.
3. Full units-of-measure library (UnitsNet) — Overkill; we need ~5 domain types, not a full unit system.

**Consequences**:
- Small overhead per value type creation (~10 lines each)
- Physics equations become self-documenting: `DragBasedModel.Solve(RadialSpeed v0, ...)` clearly shows corrected speed is required
- Compile-time safety eliminates entire classes of silent domain bugs
- Agent-generated code that passes `SkyPlaneSpeed` to a `RadialSpeed` parameter fails at compile time

---

## ADR-014: LongRunning Tasks for CPU-Bound ML Training

**Date**: 2026-04-06
**Status**: Accepted
**Analysis Risk**: Thread pool starvation from CPU-bound ML.NET training wrapped in Task.Run()

**Context**: ML.NET FastForest/FastTree training is intensely CPU-bound and synchronous internally. The `IFrameworkAdapter` interface mandates `Task<ITrainedModel> TrainAsync(...)` for async compatibility. Wrapping CPU-bound work in `Task.Run()` borrows a thread pool thread. If `EnsembleModel` trains multiple adapters concurrently, all pool threads are consumed by CPU work, leaving none for I/O callbacks — the classic thread pool starvation deadlock.

**Decision**: Use `Task.Factory.StartNew(..., TaskCreationOptions.LongRunning, TaskScheduler.Default)` for all ML training dispatches. This creates a dedicated thread outside the pool. For `EnsembleModel` concurrent training, use `SemaphoreSlim` to limit parallelism to `Environment.ProcessorCount - 1`.

**Alternatives considered**:
1. `Task.Run()` — Default; causes thread pool starvation under concurrent training.
2. Custom `TaskScheduler` with dedicated thread pool — More robust but complex; candidate for Phase 4.
3. Sequential training only — Simple but wastes wall-clock time for ensembles.
4. `ConfigureAwait(false)` everywhere — Does not solve the root cause (CPU work consuming pool threads).

**Consequences**:
- Each `TrainAsync` call creates a new OS thread (~1MB stack) — acceptable for SolarPipe's 2-5 concurrent stages
- Ensemble parallelism capped to prevent memory exhaustion
- Must pair with `CancellationToken` timeout (RULE-151) to prevent infinite hangs

---

## ADR-015: Pipeline State Checkpointing

**Date**: 2026-04-06
**Status**: Accepted
**Analysis Risk**: Multi-stage pipeline failures force complete re-execution

**Context**: SolarPipe pipelines chain 3-6 stages. Training runs for multi-year datasets can take hours. If the Python sidecar crashes in Stage 4 (e.g., CUDA OOM), the user must re-run Stages 1-3 from scratch. In a CLI-agent workflow, this wastes significant time per iteration cycle.

**Decision**: Implement per-stage checkpointing:
- After each stage completes, serialize its output (`IDataFrame` + `ITrainedModel` reference) to `cache/{pipeline_name}/{stage_name}.checkpoint`
- CLI supports `--resume-from-stage <name>` to skip completed stages
- CLI supports `--no-cache` to force full re-execution
- Checkpoints include a SHA-256 fingerprint of the stage config + input data hash; stale checkpoints are automatically invalidated when config changes
- `IDataFrame` checkpoint format: Arrow IPC (reusable by Python sidecar)

**Alternatives considered**:
1. No checkpointing — Acceptable for Phase 1 (single stage, fast); unacceptable for Phase 3+ (multi-stage, hours-long).
2. Full MLflow experiment tracking — Too heavy for MVP; planned for production migration (ADR-011 future).
3. Manual file caching — Error-prone; fingerprint-based invalidation prevents stale data bugs.

**Consequences**:
- ~200 lines for checkpoint manager + CLI integration
- Disk usage: ~10-50MB per stage checkpoint (Arrow IPC is compact)
- Config change detection prevents using stale cached results
- Significantly faster iteration cycles during development

---

## ADR-016: Dry-Run DAG Validation Before Execution

**Date**: 2026-04-06
**Status**: Accepted
**Analysis Risk**: YAML typos cause failures hours into ML training runs

**Context**: SolarPipe's "configuration over code" philosophy means YAML errors are the most common failure mode. Without an IDE validating YAML as you type, invalid configurations are only caught at runtime. If a schema mismatch between Stage A's output and Stage B's input is discovered after 2 hours of training, the entire run is wasted. In an agent workflow, the resulting stack trace consumes significant context.

**Decision**: The `validate` command performs a complete dry-run compilation:
1. Parse YAML and enforce YAML 1.2 semantics (Yaml12BooleanConverter)
2. Build the full Directed Acyclic Graph (DAG) of pipeline stages
3. Verify topological sort (detect cycles)
4. Check that output schema of each stage matches input requirements of downstream stages
5. Verify all data sources are reachable (file exists, SQLite table exists, REST endpoint responds)
6. Validate all hyperparameters against framework-specific constraints
7. Complete in <1 second
8. Output structured, AI-readable error messages: `"Error at stage 'rf_model': feature 'Bz_gsm' not found in data source 'omni_hourly'. Available: [Bx, By, Bz_gse, Vp, Np]"`

**Alternatives considered**:
1. Validate-on-first-error during execution — Catches issues too late; wastes training time.
2. JSON Schema validation only — Catches structural errors but not semantic errors (missing columns, framework constraints).
3. IDE-based validation plugins — No IDE in this workflow.

**Consequences**:
- `solarpipe validate --config X` becomes the mandatory first step before any training
- Error messages designed for agent parsing (structured, include context, suggest fixes)
- Schema compatibility checking requires all providers to implement `DiscoverSchemaAsync` early
- Minimal implementation cost (~150 lines); high value for iteration speed

---

## Future ADRs (To be decided during implementation)

- **ADR-017**: Choice of CLI parsing library (System.CommandLine vs. Spectre.Console.Cli vs. manual)
- **ADR-018**: Model registry migration path (file system → MLflow or database)
- **ADR-019**: Sidecar Python environment management (venv vs. conda vs. Docker)

---

**Document Version History**:
- v2.0 (2026-04-06): Added ADRs 009-016 from CLI-agent workflow analysis
- v1.0 (2026-04-06): Initial creation with 8 ADRs from pre-implementation risk audit
