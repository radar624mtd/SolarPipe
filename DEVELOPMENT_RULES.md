# SolarPipe Development Rules

**Version**: 2.0
**Created**: 2026-04-06
**Updated**: 2026-04-06
**Source**: Pre-implementation risk audit (37 risks, 8 critical) + CLI-agent workflow analysis (25 additional risks)
**Purpose**: Concrete, enforceable rules that prevent known failure modes. Every rule has a rationale and violation pattern — if you're about to break one, read the rationale first.

---

## Interface & Type System Rules

### RULE-001: No ReadOnlySpan<T> in public interfaces
**Rationale**: `ReadOnlySpan<T>` is a ref struct banned from async methods in C# 12/.NET 8. All pipeline operations are async. Moq/NSubstitute also cannot mock ref struct parameters.
**Applies to**: All interfaces in `SolarPipe.Core/Interfaces/`
**Violation pattern**: `ReadOnlySpan<float> GetColumn(...)` in an interface
**Correct pattern**: `float[] GetColumn(...)` in interfaces; `ReadOnlySpan<float>` only in synchronous internal methods (e.g., `InMemoryDataFrame.GetColumnSpan()`)
**Enforcement**: Compile error in async callers; code review

### RULE-002: Validate column lengths match before IDataView conversion
**Rationale**: If columns have unequal lengths in the backing `float[][]`, `IDataView` cursor-based access silently uses wrong feature values. Predictions are incorrect with no error.
**Applies to**: `InMemoryDataFrame.ToDataView()`
**Violation pattern**: Creating IDataView without verifying all `_data[i].Length == RowCount`
**Correct pattern**: `Debug.Assert(_data.All(col => col.Length == RowCount))` before conversion; unit test with `CreateEnumerable<T>(reuseRowObject: false)` round-trip
**Enforcement**: Unit test, assertion

### RULE-003: All async interfaces return Task<T>, never void
**Rationale**: Async void methods cannot be awaited, swallow exceptions, and crash the process on unhandled exceptions.
**Applies to**: All interfaces and public methods
**Violation pattern**: `public async void TrainAsync(...)`
**Correct pattern**: `public async Task<ITrainedModel> TrainAsync(...)`
**Enforcement**: Code review, analyzer rule (CA2007)

---

## ML.NET Rules

### RULE-010: FeatureFraction must be explicitly set ≤ 0.8
**Rationale**: ML.NET FastForest/FastTree default `FeatureFraction` and `FeatureFractionPerSplit` to **1.0**, disabling the feature bagging mechanism that makes Random Forest work. With 10-20 features, this produces correlated trees with no diversity.
**Applies to**: All `FastForestRegressionTrainer.Options` and `FastTreeRegressionTrainer.Options`
**Violation pattern**: Omitting `FeatureFraction` or setting it to `1.0`
**Correct pattern**: `FeatureFraction = 0.7f, FeatureFractionPerSplit = 0.7f` (≥0.5 with <10 features)
**Enforcement**: Code review, config validation

### RULE-011: Pin both MLContext seed AND trainer-specific seed
**Rationale**: `MLContext(seed: N)` controls global RNG but FastTree has a separate `FeatureSelectionSeed`. Both must be pinned for reproducibility. `NumberOfThreads` must be 1 — multi-threading introduces non-determinism.
**Applies to**: All ML.NET training code
**Violation pattern**: Setting only `MLContext(seed: 42)` without `FeatureSelectionSeed` and `NumberOfThreads = 1`
**Correct pattern**: Set all three: `MLContext(seed: 42)`, `FeatureSelectionSeed = 42`, `NumberOfThreads = 1`
**Enforcement**: Code review

### RULE-012: Never use PredictionEngine directly in concurrent code
**Rationale**: `PredictionEngine<TIn, TOut>` throws `IndexOutOfRangeException` under concurrent access (ML.NET issues #1718, #5582).
**Applies to**: All inference code that may be called concurrently
**Violation pattern**: `mlContext.Model.CreatePredictionEngine<>()` in a service or API endpoint
**Correct pattern**: Register `PredictionEnginePool<TIn, TOut>` via `services.AddPredictionEnginePool<>()` from `Microsoft.Extensions.ML`
**Enforcement**: Code review; single-threaded CLI is exempt

### RULE-013: Fresh MLContext per test method
**Rationale**: MLContext's internal RNG is shared state. Parallel test classes sharing a static MLContext consume random numbers non-deterministically, producing flaky tests.
**Applies to**: All xUnit test classes that use ML.NET
**Violation pattern**: `private static readonly MLContext _mlContext = new(42);` as a shared field
**Correct pattern**: `var mlContext = new MLContext(seed: 42);` inside each test method
**Enforcement**: Code review; use `[Collection("ML")]` for sequential execution of tests sharing expensive resources

---

## YAML Configuration Rules

### RULE-020: Register Yaml12BooleanConverter in every DeserializerBuilder
**Rationale**: YamlDotNet 16.x implements YAML 1.1, which treats `yes`, `no`, `on`, `off`, `y`, `n` (and case variants) as boolean literals. A config value like `coordinate_frame: NO` silently becomes `false`. This is called the "Norway problem."
**Applies to**: Every `DeserializerBuilder` instantiation
**Violation pattern**: `new DeserializerBuilder().Build()` without `.WithTypeConverter(new Yaml12BooleanConverter())`
**Correct pattern**: Always include `.WithTypeConverter(new Yaml12BooleanConverter())`
**Enforcement**: Unit test that deserializes `{key: NO}` and verifies it throws

### RULE-021: Post-deserialization null validation on all config objects
**Rationale**: YamlDotNet assigns `null` to non-nullable C# properties without throwing (GitHub issues #763, #591). The string "NULL" writes unquoted, reads back as actual null.
**Applies to**: All deserialized configuration objects
**Violation pattern**: Using deserialized config properties without null checking non-nullable types
**Correct pattern**: Call `ValidateNoUnexpectedNulls()` immediately after deserialization, walking all properties via reflection
**Enforcement**: Unit test with null/NULL/~ scalar values

### RULE-022: Never use YAML merge keys (<<)
**Rationale**: YamlDotNet's `MergingParser` violates the spec's override rules (issues #388, #487). Duplicate key errors appear where overrides should succeed.
**Applies to**: All pipeline YAML configuration files
**Violation pattern**: `<<: *alias` in YAML configs
**Correct pattern**: Explicit YAML anchors with `*alias` references; or duplicate the keys
**Enforcement**: Config validation, documentation

---

## ODE Solver & Physics Rules

### RULE-030: Use Dormand-Prince RK4(5), not MathNet.Numerics ODE solvers
**Rationale**: MathNet.Numerics 5.0.0 provides only fixed-step RK2/RK4 with no adaptive step-size, no error estimation, no implicit solvers. Fixed-step RK4 diverges for Burton ODE when τ < 0.36h (Carrington-class storms).
**Applies to**: All ODE integration in physics models
**Violation pattern**: `MathNet.Numerics.OdeSolvers.RungeKutta.FourthOrder()`
**Correct pattern**: Custom Dormand-Prince RK4(5) with FSAL tableau, embedded error estimation, adaptive step control (atol=1e-8, rtol=1e-6)
**Enforcement**: Unit test: verify solver stability for τ = 0.4h; convergence test: halving step halves error by ~2⁵

### RULE-031: All physics equations use GSM coordinate frame for Bz
**Rationale**: Burton ODE, Newell coupling function, and all geoeffectiveness calculations require GSM-frame Bz. Solar wind data arrives in GSE coordinates. Using GSE Bz produces completely wrong results with no error signal.
**Applies to**: All code using Bz for geomagnetic calculations
**Violation pattern**: Using `Bz_gse` directly in Burton/Newell equations
**Correct pattern**: Transform GSE→GSM using Hapgood 1992 (requires dipole tilt angle from date/UT); unit test against published test vectors
**Enforcement**: Assert that input data has `Bz_gsm` column; fail fast if only `Bz_gse` is available without transform

### RULE-032: Validate physics parameter ranges at entry
**Rationale**: Out-of-range inputs (negative speeds, impossible drag coefficients) produce nonsensical results that propagate silently through the composition algebra.
**Applies to**: DragBasedModel, BurtonOde, NewellCoupling
**Violation pattern**: Passing unchecked user input to physics equations
**Correct pattern**: Validate and throw `ArgumentOutOfRangeException` at method entry: v₀ ∈ [200, 3500] km/s, γ ∈ [0.2e-7, 2e-7] km⁻¹, starting distance ≥ 20 R☉

---

## File & Registry Rules

### RULE-040: Atomic write for all persistent file operations
**Rationale**: On Linux, `FileStream` with `FileShare.None` uses advisory-only `flock()` — non-cooperating processes can ignore the lock. Concurrent writes to the model registry silently corrupt data.
**Applies to**: `FileSystemModelRegistry`, any file write that must be durable
**Violation pattern**: `File.WriteAllTextAsync(finalPath, data)` or `FileStream` with `FileShare.None`
**Correct pattern**: Write to `tempPath = $".tmp_{Guid.NewGuid():N}"`, then `File.Move(tempPath, finalPath, overwrite: true)` — POSIX `rename()` is atomic on the same filesystem
**Enforcement**: Code review; unit test verifying concurrent writes don't corrupt

### RULE-041: SHA-256 for all data fingerprints
**Rationale**: Birthday-paradox threshold requires ~2¹²⁸ hashes for 50% collision probability. SHA-256 is computationally fast and collision-free in practice.
**Applies to**: Model registry fingerprints, data integrity checks
**Violation pattern**: Using MD5, CRC32, or string-based hashing for data integrity
**Correct pattern**: `Convert.ToHexString(SHA256.HashData(fileBytes))` or streaming `SHA256.HashDataAsync(stream)`

---

## Data & Validation Rules

### RULE-050: Use ParquetSharp, not Parquet.Net
**Rationale**: Parquet.Net loads entire row groups into memory (1 GB CSV → 7+ GB RAM). A float integrity bug (issue #81) produced incorrect values at index 38 with Snappy compression. ParquetSharp wraps Apache Parquet C++ and is 4-10x faster.
**Applies to**: `ParquetProvider` implementation
**Violation pattern**: NuGet reference to `Parquet.Net`
**Correct pattern**: NuGet reference to `ParquetSharp`; wrap native handles in `using` statements (P/Invoke: use-after-dispose = access violation, not managed exception)

### RULE-051: Temporal cross-validation with gap buffers, never random k-fold for time-series
**Rationale**: ML.NET `CrossValidate()` does random k-fold which leaks future information. For N=300-500 CME events, expanding-window CV with 5 folds and 3-7 day gap buffers is required. Each fold's test set must contain ≥50 events for RMSE stability.
**Applies to**: All model evaluation and hyperparameter tuning
**Violation pattern**: `mlContext.Regression.CrossValidate(data, numberOfFolds: 5)`
**Correct pattern**: Custom `ExpandingWindowCV` with gap buffer (3-7 days), ≥50 events per test fold, ENLIL data temporally isolated
**Enforcement**: Code review; test that verifies no test-period events appear in training fold

### RULE-052: Never compute rolling statistics on full dataset before train/test split
**Rationale**: `df['rolling_mean'] = df['Dst'].rolling(24).mean()` computed on the full dataset leaks test-period information into training features.
**Applies to**: All feature engineering with time-windowed aggregations
**Violation pattern**: Rolling/window operations before temporal split
**Correct pattern**: Split first, then compute rolling statistics within each fold independently

### RULE-053: ENLIL synthetic data must be temporally isolated from test events
**Rationale**: ENLIL uses observed CME parameters as input. If ENLIL simulations for event X are in training while event X's observations are in testing, information leaks through the shared input parameters.
**Applies to**: All mock data integration strategies
**Violation pattern**: Including ENLIL data for event X in training when event X is in the test fold
**Correct pattern**: Only use ENLIL data from events strictly before the test period

---

## Python Sidecar Rules

### RULE-060: Manage sidecar lifecycle via IHostedService
**Rationale**: `Process.Start` creates an independent OS process. If .NET crashes, Python keeps running as an orphan holding GPU memory and file locks.
**Applies to**: All sidecar process management
**Violation pattern**: `Process.Start("python", "server.py")` without lifecycle management
**Correct pattern**: Implement `IHostedService`; register `IHostApplicationLifetime.ApplicationStopping`; on Windows use Job Objects (P/Invoke), on Linux set `PR_SET_PDEATHSIG`; register `Process.Exited` for crash detection

### RULE-061: Server-streaming RPCs for long-running training, not unary with deadlines
**Rationale**: gRPC deadlines abort the HTTP/2 stream on the client side but the server-side method keeps running until it checks `CancellationToken`. Multi-hour training jobs become zombies. Known: connections fail after ~9 hours idle (grpc-dotnet #2628).
**Applies to**: All training RPCs
**Violation pattern**: Unary RPC with `CallOptions.Deadline` for training
**Correct pattern**: Server-streaming RPC with periodic progress messages; cooperative `CancellationToken` propagation; client-side heartbeat monitoring

### RULE-062: Python gRPC server must bind to 0.0.0.0, not localhost
**Rationale**: Binding to `localhost` inside a Docker container makes the server unreachable from other containers (grpc/grpc #16682).
**Applies to**: Python sidecar `server.py`
**Violation pattern**: `server.add_insecure_port('localhost:50051')`
**Correct pattern**: `server.add_insecure_port('0.0.0.0:50051')`

### RULE-063: Enforce explicit Arrow schema at both ends of gRPC channel
**Rationale**: Pandas defaults to float64; `pa.Table.from_pandas(df)` creates float64 unless schema specifies `pa.float32()`. The .NET side has distinct `FloatArray`/`DoubleArray` types. Mismatched precision causes silent accuracy loss.
**Applies to**: All Arrow IPC serialization/deserialization
**Violation pattern**: `pa.Table.from_pandas(df)` without explicit schema
**Correct pattern**: `pa.Table.from_pandas(df, schema=pa.schema([('col', pa.float32()), ...]))` and validate column types on .NET receipt

---

## ONNX Rules

### RULE-070: Neural ODE models export dynamics network only, not full solver
**Rationale**: torchdiffeq adaptive ODE solvers use dynamic control flow (data-dependent while loops, step rejection) incompatible with ONNX's static graph. The adjoint method's custom autograd.Function is not JIT-capturable.
**Applies to**: Neural ODE model export
**Violation pattern**: `torch.onnx.export(neural_ode_model, ...)` with the full ODE solver
**Correct pattern**: Export only `f(y, t, θ)` dynamics network to ONNX; implement ODE solver (Dormand-Prince) in C# calling ONNX Runtime per step; OR keep inference in Python sidecar permanently

---

## Testing Rules

### RULE-080: Tolerance-based assertions for all floating-point comparisons
**Rationale**: Floating-point arithmetic is not exact. Exact equality assertions produce flaky tests across platforms, compiler optimizations, and library versions.
**Applies to**: All test assertions involving float/double values
**Violation pattern**: `Assert.Equal(0.7f, result)`
**Correct pattern**: `Assert.InRange(result, 0.699f, 0.701f)` or custom `AssertApproxEqual(expected, actual, tolerance)`

### RULE-081: Pin NuGet package versions exactly
**Rationale**: ML.NET model outputs change between releases. An unpinned version can make previously passing tests fail after `dotnet restore` pulls a new version.
**Applies to**: All `.csproj` files
**Violation pattern**: `<PackageReference Include="Microsoft.ML" Version="4.*" />`
**Correct pattern**: `<PackageReference Include="Microsoft.ML" Version="4.0.1" />`

### RULE-082: Physics validation against published reference values
**Rationale**: Space weather equations have specific published coefficients and test cases. Implementing from memory or approximation produces subtle errors.
**Applies to**: All physics equation implementations
**Violation pattern**: Implementing Burton ODE without O'Brien & McPherron (2000) Table 1 coefficients
**Correct pattern**: Use golden reference values: τ floor = 6.15 hr, α ≈ -4.5 nT/hr per mV/m, pressure correction b = 7.26 nT/√nPa. Validate against NASA OMNI data and SuperMAG pre-computed values

---

## Security Rules

### RULE-090: Validate all file paths against path traversal
**Rationale**: `Path.Combine(@"C:\safe", @"C:\evil\file")` returns the absolute path `C:\evil\file`, bypassing any intended sandbox. This applies to YAML config file references and model registry paths.
**Applies to**: All file path handling from configuration or user input
**Violation pattern**: `Path.Combine(basePath, userInput)` without checking result starts with basePath
**Correct pattern**: `var full = Path.GetFullPath(Path.Combine(basePath, userInput)); if (!full.StartsWith(basePath)) throw ...;`

### RULE-091: Parameterized queries only for SQLite
**Rationale**: SQL injection through YAML config file fields (table names, column filters) is possible if string concatenation is used.
**Applies to**: `SqliteProvider`
**Violation pattern**: `$"SELECT * FROM {tableName} WHERE {filter}"`
**Correct pattern**: Use parameterized queries; validate table/column names against discovered schema

---

## Domain Rules

### RULE-100: Document irreducible uncertainty bounds, don't hide them
**Rationale**: Drag-based CME arrival time predictions have >12-hour MAE floor. CDAW catalog has 79% completeness. These are fundamental limits, not engineering problems.
**Applies to**: All prediction output, documentation, metrics reporting
**Violation pattern**: Reporting model RMSE without noting the physics floor or catalog incompleteness
**Correct pattern**: Include system-level uncertainty bounds in prediction output; document that performance below physics floor likely indicates overfitting

### RULE-101: CDAW sky-plane speed requires correction for Earth-directed CMEs
**Rationale**: CDAW catalog reports projected plane-of-sky speeds which dramatically underestimate true radial speed for halo CMEs directed at Earth. Using uncorrected speeds produces underestimated arrival times.
**Applies to**: All feature engineering using CME speed from CDAW catalog
**Violation pattern**: Using `V_CDAW` directly as radial speed
**Correct pattern**: Apply cone-model correction or use 3D reconstructions from STEREO when available; document correction method and residual uncertainty

---

## Agent Workflow & Build Discipline Rules

### RULE-110: Build after every file edit
**Rationale**: Without an IDE, cross-file compilation breaks (e.g., interface changes breaking 5 implementations) are invisible until you build. Batching edits across files then building causes cascading errors that waste agent context diagnosing multiple breaks simultaneously.
**Applies to**: All `.cs` file modifications
**Violation pattern**: Editing 3 files then running `dotnet build`
**Correct pattern**: Edit one file → `dotnet build --no-restore` → fix breaks → next file
**Enforcement**: Agent instruction in CLAUDE.md

### RULE-111: No XML doc comments unless explicitly requested
**Rationale**: In a CLI/agent workflow with no IntelliSense, XML `<summary>` tags consume thousands of tokens per session while providing zero value. Self-documenting code with clear naming is sufficient.
**Applies to**: All generated C# code
**Violation pattern**: `/// <summary>Loads data from a data source</summary>` on every method
**Correct pattern**: Clear method names (`LoadAsync`, `ValidateColumnLengths`) without XML docs
**Enforcement**: Agent instruction in CLAUDE.md

### RULE-112: Central Package Management only — no dotnet add package
**Rationale**: `dotnet add package` can pick versions incompatible with ML.NET or .NET 8. All versions must be locked centrally in `Directory.Packages.props` to prevent dependency hell.
**Applies to**: All NuGet package changes
**Violation pattern**: `dotnet add package Microsoft.ML --version 4.0.2`
**Correct pattern**: Edit `Directory.Packages.props`, then `dotnet restore`
**Enforcement**: Agent instruction in CLAUDE.md

### RULE-113: TreatWarningsAsErrors in all projects
**Rationale**: Build warnings (nullable types, unused variables) clog terminal output. The agent wastes context reading and potentially "fixing" irrelevant warnings instead of the actual error.
**Applies to**: All `.csproj` files
**Violation pattern**: `<TreatWarningsAsErrors>false</TreatWarningsAsErrors>` or omitting the property
**Correct pattern**: `<TreatWarningsAsErrors>true</TreatWarningsAsErrors>` with `<NoWarn>CS1591</NoWarn>` (suppresses XML doc warnings)

### RULE-114: Maximum 400 lines per .cs file — use partial classes
**Rationale**: Large files exhaust agent token context. When the agent needs to modify one method in a 1000-line file, it must load the entire file, risking token exhaustion and accidental deletion.
**Applies to**: All implementation files (interfaces exempt)
**Violation pattern**: `InMemoryDataFrame.cs` at 800+ lines
**Correct pattern**: `InMemoryDataFrame.Core.cs`, `InMemoryDataFrame.Transforms.cs`, `InMemoryDataFrame.IO.cs`
**Enforcement**: Code review

### RULE-115: Git status before every implementation step
**Rationale**: If uncommitted changes exist and the agent makes new changes, merge conflicts will arise. CLI agents struggle immensely with `<<<<<<< HEAD` markers, often hallucinating broken merges.
**Applies to**: All agent workflow
**Violation pattern**: Starting implementation with dirty working tree
**Correct pattern**: `git status` → commit/stash dirty changes → begin work

---

## Data Handling & Pipeline Resilience Rules

### RULE-120: Convert sentinel values to NaN at load time
**Rationale**: Space weather catalogs use sentinel values like `9999.9`, `-1e31`, or `999.99` for missing data. If these flow into training as real values, models learn nonsensical relationships with no error signal.
**Applies to**: All `IDataSourceProvider.LoadAsync()` implementations
**Violation pattern**: Passing `9999.9` through to `InMemoryDataFrame` as a valid float
**Correct pattern**: Check provider-specific sentinel values (OMNI: `9999.9`, CDAW: `-1.0`), convert to `float.NaN`, track provenance in `ColumnInfo.MissingReason`

### RULE-121: NaN propagation guard after every ODE step
**Rationale**: If plasma density drops to 0 or a parameter goes out of range during ODE solving, the equation produces NaN. C# propagates NaN silently through the entire array. The agent will not know which time-step caused it.
**Applies to**: All ODE solver implementations (Dormand-Prince, physics equations)
**Violation pattern**: Running ODE solver without per-step NaN checking
**Correct pattern**: After each RK step, check `float.IsNaN(y)`. If found, throw with exact step index and variable states: `throw new PhysicsNaNException(step: 47, variable: "v_cme", state: {...})`

### RULE-122: Temporal alignment is a framework primitive
**Rationale**: Space weather datasets have mismatched cadences (1-min ACE, 1-hour OMNI, 5-min DSCOVR). Without a built-in alignment method, ad-hoc merge scripts proliferate, consuming token context and creating inconsistent resampling logic.
**Applies to**: `IDataFrame` interface
**Violation pattern**: Writing custom LINQ joins for temporal alignment each time
**Correct pattern**: `IDataFrame.ResampleAndAlign(cadence: TimeSpan.FromMinutes(5))` as a core method

### RULE-123: Pipeline state checkpointing for multi-stage runs
**Rationale**: If a pipeline fails at Stage 4, Stages 1-3 should not re-execute. Without checkpointing, every failure forces full re-training which wastes hours.
**Applies to**: `PipelineExecutor` and CLI commands
**Violation pattern**: `solarpipe train` always runs all stages from scratch
**Correct pattern**: Each stage serializes its output to `cache/{pipeline}/{stage}.checkpoint`. `--resume-from-stage` flag skips completed stages. `--no-cache` forces full re-run.

### RULE-124: All YAML configs must pass dry-run validation before execution
**Rationale**: YAML typos or schema mismatches cause failures hours into an ML run. The `validate` command must build the full DAG, verify topological sort, check output→input schema compatibility between chained stages, and fail in milliseconds.
**Applies to**: `PipelineConfigLoader` and `PipelineGraphBuilder`
**Violation pattern**: Attempting `TrainAsync` without prior DAG validation
**Correct pattern**: `solarpipe validate --config X` runs full schema checking, stage compatibility, and data source reachability in <1 second. Error messages are structured and AI-readable: `"Error at stage 'rf_model': feature 'Bz_gsm' not found in data source 'omni_hourly'. Available columns: [Bx, By, Bz_gse, Vp, Np]"`

### RULE-125: Large arrays never sent inline over gRPC
**Rationale**: gRPC has a 4MB default message size limit. A 500,000-row float[] will fail or cause severe serialization latency. Protobuf is not designed for bulk array transfer.
**Applies to**: All C#↔Python data transfer
**Violation pattern**: Embedding a `repeated float` with 500K values in a Protobuf message
**Correct pattern**: C# writes Arrow IPC to a temp file, sends the file path over gRPC. Python reads the file directly. Both sides validate the Arrow schema.

---

## Domain-Driven Type Safety Rules

### RULE-130: Coordinate vectors use typed structs — no bare floats
**Rationale**: Space weather uses GSE and GSM coordinate systems. If vectors are just `float x, y, z`, GSE and GSM Bz will inevitably be mixed — a silent catastrophe. The type system must make this impossible.
**Applies to**: All coordinate-related code in physics equations
**Violation pattern**: `float bz` passed between coordinate systems
**Correct pattern**: `GseVector` and `GsmVector` record structs with no implicit conversion. Transform via `CoordinateTransform.GseToGsm(gse, tiltAngle)` only.

### RULE-131: CDAW sky-plane speed uses SkyPlaneSpeed type, not float
**Rationale**: CDAW reports projected speeds. Using them as radial speeds underestimates arrival time. A typed struct forces cone-model correction before use in physics models.
**Applies to**: All CME speed handling
**Violation pattern**: `float cmeSpeed = cdawCatalog.Speed;` → pass to DragBasedModel
**Correct pattern**: `SkyPlaneSpeed raw = cdawCatalog.Speed;` → `RadialSpeed corrected = ConeCorrection.Apply(raw, halfAngle);` → pass to DragBasedModel

### RULE-132: No inline physics constants — use PhysicalConstants.cs
**Rationale**: If the agent scatters constants like `6371.0f` (Earth radius) or `1.6726e-27f` (proton mass) inside formulas, a hallucinated decimal place creates an unfindable bug.
**Applies to**: All physics equation implementations
**Violation pattern**: `float r = distance / 6371.0f;`
**Correct pattern**: `float r = distance / PhysicalConstants.EarthRadiusKm;`

### RULE-133: Use SpaceWeatherTimeParser for catalog timestamps
**Rationale**: Space weather catalogs use non-standard time formats (Year + Day-of-Year + Hour + Min). The agent will default to `DateTime.Parse()` which fails on these formats.
**Applies to**: All CSV/catalog data ingestion
**Violation pattern**: `DateTime.Parse(row["timestamp"])`
**Correct pattern**: `SpaceWeatherTimeParser.Parse(year, doy, hour, min)` — handles OMNI, CDAW, and ACE formats

---

## Logging & Terminal Output Rules

### RULE-140: Structured JSON logging to file, minimal console output
**Rationale**: When C# and Python processes interleave stdout, the agent cannot parse garbled terminal output. Structured logs to disk with shared Trace IDs enable clean cross-process debugging.
**Applies to**: All logging configuration
**Violation pattern**: Verbose Serilog console logging during training
**Correct pattern**: `logs/dotnet_latest.json` for C#, `logs/python_latest.json` for Python. Console shows only: progress bar, stage completion, and critical errors.

### RULE-141: CLI exit codes must be human/AI-readable
**Rationale**: If a child process exits with code 137, the agent sees "Process failed" and enters a debugging loop. Translated exit codes save context.
**Applies to**: SolarPipe.Host process management
**Violation pattern**: `throw new Exception($"Process exited with code {exitCode}")`
**Correct pattern**: Map known codes: `137 → "OOM: reduce batch size or row group count"`, `139 → "SIGSEGV: check ParquetSharp native handle disposal"`, `1 → "General error: check logs/python_latest.json"`

### RULE-142: Test output must show full diffs on failure
**Rationale**: Standard xUnit truncates array assertion output (`Expected [1.1, 2.2, ...] but got [1.1, 9.9, ...]`). The agent cannot see actual values to fix the math.
**Applies to**: All array/collection assertions in tests
**Violation pattern**: `Assert.Equal(expected, actual)` for arrays
**Correct pattern**: FluentAssertions `actual.Should().BeEquivalentTo(expected, options => options.WithStrictOrdering().Using<float>(ctx => ctx.Subject.Should().BeApproximately(ctx.Expectation, 0.001f)).WhenTypeIs<float>())`

---

## Async & Threading Rules

### RULE-150: LongRunning tasks for CPU-bound ML training
**Rationale**: ML.NET training is intensely CPU-bound and synchronous internally. Wrapping in `Task.Run()` uses thread pool threads. If EnsembleModel trains multiple models concurrently, thread pool starvation deadlocks the async state machine.
**Applies to**: All `IFrameworkAdapter.TrainAsync()` implementations
**Violation pattern**: `await Task.Run(() => pipeline.Fit(trainView), ct);`
**Correct pattern**: `await Task.Factory.StartNew(() => pipeline.Fit(trainView), ct, TaskCreationOptions.LongRunning, TaskScheduler.Default);`

### RULE-151: Pipeline stage timeout with descriptive exception
**Rationale**: If training deadlocks or hangs (e.g., `.Result` on an incomplete task), the CLI hangs forever with no output. Without an IDE debugger, infinite hangs are undiagnosable.
**Applies to**: All pipeline stage execution
**Violation pattern**: `await stage.TrainAsync(config, data, null, ct);` with no timeout
**Correct pattern**: Wrap with `CancellationTokenSource.CreateLinkedTokenSource(ct)` + timeout (default 30 min for training, 60s for prediction). On timeout: `throw new TimeoutException($"Stage '{stageName}' exceeded {timeout}. Check logs/dotnet_latest.json for last activity.")`

---

**Total Rules**: 48
**Critical Rules** (cause silent failures if violated): RULE-001, RULE-002, RULE-010, RULE-020, RULE-030, RULE-031, RULE-040, RULE-051, RULE-070, RULE-120, RULE-130

---

**Document Version History**:
- v2.0 (2026-04-06): Added 20 rules from CLI-agent workflow analysis (RULE-110 through RULE-151)
- v1.0 (2026-04-06): Initial creation from pre-implementation risk audit
