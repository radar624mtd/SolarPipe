# SolarPipe Detailed Implementation Plan

**Document Type**: Technical Implementation Guide
**Version**: 1.3 (Aligned to 19-week timeline with 25% per-phase buffer)
**Created**: 2026-04-06
**Updated**: 2026-04-06
**Target Audience**: Development team implementing SolarPipe
**Scope**: Phase 1–4, 19 weeks (April–August 2026)

---

## Table of Contents

1. [Overview & Philosophy](#overview--philosophy)
2. [Architecture Principles](#architecture-principles)
3. [Phase-by-Phase Implementation](#phase-by-phase-implementation)
4. [Code Organization & Naming Conventions](#code-organization--naming-conventions)
5. [Testing Strategy](#testing-strategy)
6. [Development Workflow](#development-workflow)
7. [Debugging & Troubleshooting](#debugging--troubleshooting)
8. [Performance & Optimization](#performance--optimization)
9. [Documentation Requirements](#documentation-requirements)

---

## Overview & Philosophy

### Design Principles

1. **Configuration over Code** — Pipeline behavior is defined in YAML, not hardcoded
2. **Framework Plurality** — No vendor lock-in; support ML.NET, ONNX, Physics, Python
3. **Composable Models** — Models are first-class objects combined via algebra (→, +, ^, ?)
4. **Unidirectional Dependencies** — Core ← Data, Config, Training, Prediction; no circular refs
5. **Test-Driven Development** — Every feature has unit + integration tests
6. **Async/Await throughout** — All I/O operations are non-blocking

### Decision-Making Framework

When faced with design choices, ask:

- **Configuration or Code?** Favor YAML configuration
- **Interface or Concrete Type?** Always use interface (IDataFrame, IFrameworkAdapter)
- **Sync or Async?** Always async (Task<T>)
- **Exception or Result?** Throw exceptions for programming errors (null checks), return results for user input errors
- **Centralized or Distributed?** Favor distributed patterns (e.g., each adapter handles its own serialization)

---

## Architecture Principles

### The Core Abstraction: IDataFrame

`IDataFrame` is the central abstraction that bridges all frameworks. Everything flows through it:

```csharp
public interface IDataFrame : IDisposable
{
    DataSchema Schema { get; }
    int RowCount { get; }
    float[] GetColumn(string name);
    float[] GetColumn(int index);
    IDataFrame Slice(int startRow, int count);
    IDataFrame SelectColumns(params string[] columns);
    IDataView ToDataView(MLContext mlContext);   // ML.NET
    float[][] ToArray();                         // Python sidecar
}
```

**Why this design?**
- Single abstraction for heterogeneous data sources (SQLite, CSV, Parquet, REST API, in-memory)
- Framework-agnostic; each framework converts to its native format
- Supports slicing and column selection without data duplication

> **AUDIT DECISION (Risk 1c — ReadOnlySpan in async methods):** The interface returns `float[]` instead of `ReadOnlySpan<float>`. `ReadOnlySpan<T>` is a `ref struct` in C# 12/.NET 8 and **cannot exist as a local in any async method** — the compiler rejects it because async state machines store locals as heap fields. Since all pipeline operations are async (`TrainAsync`, `PredictAsync`, `LoadAsync`), using `ReadOnlySpan<float>` in cross-layer interfaces would force every caller into synchronous code or awkward workarounds. Confine `ReadOnlySpan<float>` to **synchronous leaf methods** inside `InMemoryDataFrame` internals only. Extract span work into non-async local methods called between awaits.
>
> **AUDIT DECISION (Risk 11a — Moq incompatible with Span types):** Using `float[]` in interfaces also enables mocking with Moq/NSubstitute. `ReadOnlySpan<T>` cannot be a generic type argument (pre-C# 13), making `It.IsAny<ReadOnlySpan<float>>()` a compile error. Hand-written fakes/stubs are needed only for span-consuming internal APIs.

**Implementation Notes:**
- Store data as float[][] internally (dense, column-major preferred)
- Implement AddColumn() for feature augmentation (residual baseline feature)
- Implement all operations without copying data where possible (views, indices)
- Handle missing data explicitly (NaN for float, or filter rows)
- Use `ReadOnlySpan<float>` only in synchronous internal methods (e.g., inside `InMemoryDataFrame.GetColumnSpan()` for hot-path transforms); never expose in interfaces or async boundaries

### Framework Dispatch Pattern

Each framework adapter implements `IFrameworkAdapter`:

```csharp
public interface IFrameworkAdapter
{
    FrameworkType FrameworkType { get; }
    IReadOnlyList<string> SupportedModels { get; }
    Task<ITrainedModel> TrainAsync(
        StageConfig config,
        IDataFrame training,
        IDataFrame? validation,
        CancellationToken ct);
}
```

**Implementation for each framework:**

1. **ML.NET** — Direct use of MLContext and trainers
2. **Physics** — Equation selection + parameter validation, no training (output deterministic)
3. **ONNX** — Load model file, wrap in adapter, only implements Predict (no Train)
4. **Python** — gRPC client, marshal data to sidecar process

**Key Pattern**: Each adapter is responsible for:
- Hyperparameter validation (throw early if invalid)
- Data format conversion (IDataFrame → native format)
- Metrics collection (RMSE, MAE, R², feature importance)
- Error handling (throw ArgumentException for bad input)

### Composition Algebra

Models combine via four operators:

| Operator | Symbol | Semantics | Example |
|----------|--------|-----------|---------|
| Chain | `->` | Sequential: A's output feeds B's input | `classifier -> regressor` |
| Ensemble | `+` | Parallel: average or vote | `rf_model + gbm_model` |
| Residual | `^` | Correction: B learns error of A | `physics_model ^ ml_correction` |
| Gate | `?` | Conditional: C routes to A or B | `classifier ? (modelA, modelB)` |

**Implementation Notes:**

- Each composition operator is an `IComposedModel` implementation
- ResidualModel is critical for CME use case: `drag_physics ^ rf_correction`
  - Physics baseline produces prediction (deterministic)
  - RF learns (observed - baseline) residuals
  - Final = baseline + residual
- All operators must propagate uncertainty correctly
- All operators validate input/output shape compatibility

---

## Phase-by-Phase Implementation

### Phase 1: Foundation (Weeks 1–5)

#### 1.1 Week 1: Project Setup & Core Interfaces

**Objective**: Create solution skeleton and define all interfaces

**Tasks**:

1. **Create .NET 8 Solution Structure**

   **Shell note**: Claude Code on Windows runs in Git Bash. Run each `dotnet` command individually; verify exit 0 before continuing — a failed `dotnet new` will cascade into broken `dotnet sln add` and `dotnet add reference` calls.

   See `CLAUDE.md` Session 0 for the complete bootstrap command sequence with project references. `global.json` and `Directory.Packages.props` already exist in the repo root — do not regenerate them.

   Project reference rules (strict unidirectional):
   - Core: no project dependencies
   - Config, Data: depend on Core
   - Training: depends on Core + Data
   - Prediction: depends on Core + Training
   - Host: depends on all above
   - Test projects: each references its target module + xUnit packages

   **Day 1 infrastructure (ADR-016, RULE-112, RULE-113)**:
   - `Directory.Packages.props` already exists with all NuGet versions — do not recreate; add packages via `<PackageReference Include="..." />` in `.csproj` (no version — version is in props file)
   - Add `<TreatWarningsAsErrors>true</TreatWarningsAsErrors>` and `<NoWarn>CS1591</NoWarn>` to all `.csproj` files
   - Create `logs/` directory and add to `.gitignore`
   - Create `cache/` directory and add to `.gitignore`

   **Day 1 domain types (ADR-013, RULE-130, RULE-131, RULE-132)**:
   - `SolarPipe.Core/Physics/GseVector.cs` — `readonly record struct GseVector(float Bx, float By, float Bz)`
   - `SolarPipe.Core/Physics/GsmVector.cs` — `readonly record struct GsmVector(float Bx, float By, float Bz)`
   - `SolarPipe.Core/Physics/SkyPlaneSpeed.cs` — `readonly record struct SkyPlaneSpeed(float KmPerSec)`
   - `SolarPipe.Core/Physics/RadialSpeed.cs` — `readonly record struct RadialSpeed(float KmPerSec)`
   - `SolarPipe.Core/Physics/PhysicalConstants.cs` — All physical constants (EarthRadiusKm, SolarRadiusKm, AuKm, ProtonMassKg)
   - `SolarPipe.Core/Parsing/SpaceWeatherTimeParser.cs` — Handles OMNI, CDAW, ACE time formats

   **Day 1 test infrastructure (RULE-142)**:
   - `tests/SolarPipe.Tests.Unit/Fixtures/PhysicsTestFixtures.cs` — Validated space weather parameter sets (CME speeds, drag coefficients, solar wind values). NEVER generate random floats for physics tests.
   - `tests/fixtures/` directory with `sample_config.yaml`, `test.csv`, `test.db`

2. **Core Interfaces (SolarPipe.Core/Interfaces)**
   - `IDataFrame.cs` — Data abstraction (250 lines)
   - `IDataSourceProvider.cs` — Provider pattern (100 lines)
   - `IFrameworkAdapter.cs` — Framework dispatch (100 lines)
   - `ITrainedModel.cs` — Trained model interface (80 lines)
   - `IComposedModel.cs` — Composition interface (60 lines)
   - `IModelRegistry.cs` — Model storage interface (80 lines)

3. **Core Models (SolarPipe.Core/Models)**
   - `DataSchema.cs` — Schema representation (50 lines)
   - `ColumnInfo.cs` — Column metadata (40 lines)
   - `PredictionResult.cs` — Prediction output (60 lines)
   - `ModelMetrics.cs` — Metrics container (100 lines)
   - `ModelArtifact.cs` — Model versioning (80 lines)
   - Enums: `FrameworkType`, `TaskType`, `ColumnRole`

4. **Tests for Core**
   - DataSchema serialization/deserialization
   - ColumnInfo role validation
   - PredictionResult structure
   - Enum correctness

**Success Criteria**: All interfaces and models compile; all tests pass

**Estimated Hours**: 20 hours

---

#### 1.2 Week 2: Data Abstraction & Providers

**Objective**: Implement IDataFrame and first two providers (SQLite, CSV)

**Tasks**:

1. **InMemoryDataFrame Implementation (SolarPipe.Data/InMemoryDataFrame.cs)**
   ```csharp
   public class InMemoryDataFrame : IDataFrame
   {
       private readonly float[][] _data;  // columns-major
       private readonly DataSchema _schema;

       // Interface method: returns float[] for async/mock compatibility
       public float[] GetColumn(string name)
       {
           var index = _schema.ColumnIndex(name);
           return _data[index];
       }

       // Internal hot-path method: returns span for synchronous leaf methods only
       internal ReadOnlySpan<float> GetColumnSpan(string name)
       {
           var index = _schema.ColumnIndex(name);
           return _data[index].AsSpan();
       }

       // Slicing: create new dataframe with row range [start, start+count)
       public IDataFrame Slice(int startRow, int count)
       {
           // Validate bounds, create new float[][] with sliced data
       }

       // Column selection: create new dataframe with selected columns
       public IDataFrame SelectColumns(params string[] columns)
       {
           // Reorder columns, create new dataframe
       }

       // ML.NET interop
       // AUDIT NOTE (Risk 1a): Verify row count matches across all columns
       // before creating IDataView. Mismatched column lengths cause silent
       // wrong predictions. Unit-test round-trip with
       // mlContext.Data.CreateEnumerable<T>(reuseRowObject: false).
       public IDataView ToDataView(MLContext mlContext)
       {
           // Validate all columns have RowCount length
           Debug.Assert(_data.All(col => col.Length == RowCount),
               "Column length mismatch — IDataView will produce wrong results");
           // Create MLContext.Data.LoadFromEnumerable with schema
       }

       // Python interop
       public float[][] ToArray() => _data;

       // Feature augmentation (for residual models)
       public IDataFrame AddColumn(string name, float[] values)
       {
           // Append new column to data
       }
   }
   ```

   **Tests** (12 test cases):
   - GetColumn by name and index
   - Slice correctness (boundaries, empty ranges)
   - SelectColumns order preservation
   - AddColumn appends correctly
   - ToDataView compatibility
   - **ToDataView round-trip** — create IDataView, convert back to enumerable, verify values match (Risk 1a)
   - **Column length mismatch detection** — verify assertion fires if columns have unequal lengths
   - Edge case: empty dataframe, single column/row

2. **SqliteProvider Implementation (SolarPipe.Data/Providers/SqliteProvider.cs)**
   ```csharp
   public class SqliteProvider : IDataSourceProvider
   {
       public string ProviderName => "sqlite";

       public async Task<DataSchema> DiscoverSchemaAsync(
           DataSourceConfig config, CancellationToken ct)
       {
           // Open connection, execute PRAGMA table_info(table_name)
           // Build ColumnInfo for each column
           // Return DataSchema with row count estimate
       }

       public async Task<IDataFrame> LoadAsync(
           DataSourceConfig config, DataQuery query, CancellationToken ct)
       {
           // Execute parameterized query
           // Read rows into float[][]
           // Create InMemoryDataFrame with schema
       }

       public bool CanHandle(DataSourceConfig config)
           => config.Provider == "sqlite" && config.Connection != null;
   }
   ```

   **Tests** (8 test cases):
   - Connect to test SQLite database
   - Schema discovery (multiple tables)
   - Query execution with filters
   - Type inference (int, float, string → float)
   - Nullable column handling
   - Parameterized query safety
   - Row count accuracy
   - Large dataset handling (streaming not required for Phase 1)

3. **CsvProvider Implementation (SolarPipe.Data/Providers/CsvProvider.cs)**
   ```csharp
   public class CsvProvider : IDataSourceProvider
   {
       public string ProviderName => "csv";

       // Similar interface to SqliteProvider
       // Use CsvHelper for parsing
       // Infer types from sample rows
   }
   ```

   **Tests** (6 test cases):
   - Multiple delimiters (comma, tab, semicolon)
   - Header detection
   - Type inference
   - Missing value handling
   - Quoted fields
   - Large files

4. **DataSourceRegistry (SolarPipe.Data/DataSourceRegistry.cs)**
   ```csharp
   public class DataSourceRegistry
   {
       private readonly Dictionary<string, IDataSourceProvider> _providers;
       private readonly Dictionary<string, DataSourceConfig> _sources;

       public void Register(IDataSourceProvider provider)
       {
           _providers[provider.ProviderName] = provider;
       }

       public async Task<IDataFrame> LoadAsync(
           string sourceName, DataQuery query, CancellationToken ct)
       {
           var config = _sources[sourceName];
           var provider = _providers.Values.First(p => p.CanHandle(config));
           return await provider.LoadAsync(config, query, ct);
       }
   }
   ```

**Success Criteria**:
- InMemoryDataFrame passes all tests
- SqliteProvider loads from test database
- CsvProvider loads from test CSV file
- Registry correctly dispatches to providers

**Estimated Hours**: 24 hours

---

#### 1.3 Week 3: Configuration & ML.NET Adapter

**Objective**: YAML config loading and first framework adapter

**Tasks**:

1. **Configuration Models (SolarPipe.Config/Models)**
   ```csharp
   public record PipelineConfig
   {
       public string Name { get; init; }
       public Dictionary<string, DataSourceConfig> DataSources { get; init; }
       public Dictionary<string, StageConfig> Stages { get; init; }
       public Dictionary<string, ComposeConfig> Compose { get; init; }
       public EvaluationConfig? Evaluation { get; init; }
       public OutputConfig? Output { get; init; }
   }

   public record StageConfig
   {
       public string Source { get; init; }
       public FrameworkType Framework { get; init; }
       public string Model { get; init; }
       public TaskType Task { get; init; }
       public List<string> Features { get; init; }
       public string? Target { get; init; }
       public Dictionary<string, object> Hyperparameters { get; init; }
       public ValidationConfig? Validation { get; init; }
   }

   // Helper methods for type-safe hyperparameter access
   public static int GetHyperInt(this StageConfig config, string key, int @default)
       => config.Hyperparameters.TryGetValue(key, out var val) && val is int i ? i : @default;
   ```

2. **YAML Configuration Loader (SolarPipe.Config/PipelineConfigLoader.cs)**

   > **AUDIT DECISION (Risk 2a — YAML 1.1 implicit typing):** YamlDotNet 16.x implements YAML 1.1 only, which defines 22 boolean literals (`yes`, `no`, `on`, `off`, `y`, `n` and variants). A value like `coordinate_frame: NO` silently becomes `false`. A custom `IYamlTypeConverter` that enforces YAML 1.2 boolean rules (only `true`/`false`/`True`/`False`/`TRUE`/`FALSE`) is **mandatory** for scientific configuration.
   >
   > **AUDIT DECISION (Risk 2b — Null scalar corruption):** YamlDotNet assigns `null` to non-nullable properties without throwing (GitHub #763). Post-deserialization validation must walk all properties and reject unexpected nulls.

   ```csharp
   /// <summary>
   /// Custom type converter that enforces YAML 1.2 boolean semantics.
   /// Prevents YAML 1.1 "Norway problem" where NO/YES/ON/OFF become booleans.
   /// </summary>
   public class Yaml12BooleanConverter : IYamlTypeConverter
   {
       private static readonly HashSet<string> TrueValues =
           new(StringComparer.Ordinal) { "true", "True", "TRUE" };
       private static readonly HashSet<string> FalseValues =
           new(StringComparer.Ordinal) { "false", "False", "FALSE" };

       public bool Accepts(Type type) => type == typeof(bool) || type == typeof(bool?);

       public object? ReadYaml(IParser parser, Type type, ObjectDeserializer rootDeserializer)
       {
           var scalar = parser.Consume<Scalar>();
           if (TrueValues.Contains(scalar.Value)) return true;
           if (FalseValues.Contains(scalar.Value)) return false;
           throw new YamlException(scalar.Start, scalar.End,
               $"Invalid boolean '{scalar.Value}'. YAML 1.2 allows only true/false/True/False/TRUE/FALSE. " +
               $"Wrap in quotes if this is a string value.");
       }

       public void WriteYaml(IEmitter emitter, object? value, Type type, ObjectSerializer serializer)
       {
           emitter.Emit(new Scalar(value is true ? "true" : "false"));
       }
   }

   public class PipelineConfigLoader
   {
       private readonly IDeserializer _deserializer = new DeserializerBuilder()
           .WithNamingConvention(CamelCaseNamingConvention.Instance)
           .WithTypeConverter(new Yaml12BooleanConverter())  // AUDIT: YAML 1.2 booleans
           .Build();

       public async Task<PipelineConfig> LoadAsync(string path, CancellationToken ct)
       {
           var yaml = await File.ReadAllTextAsync(path, ct);
           var config = _deserializer.Deserialize<PipelineConfig>(yaml);

           // AUDIT: Reject null values in non-nullable properties (Risk 2b)
           ValidateNoUnexpectedNulls(config);

           // Validate all references exist
           ValidateReferences(config);

           return config;
       }

       /// <summary>
       /// Walks all public properties via reflection and throws if any
       /// non-nullable reference type property is null after deserialization.
       /// Prevents YamlDotNet's silent null assignment (GitHub #763).
       /// </summary>
       private void ValidateNoUnexpectedNulls(object config, string path = "")
       {
           foreach (var prop in config.GetType().GetProperties())
           {
               var value = prop.GetValue(config);
               var nullability = new NullabilityInfoContext().Create(prop);
               if (value is null && nullability.WriteState == NullabilityState.NotNull)
                   throw new InvalidOperationException(
                       $"Required property '{path}.{prop.Name}' is null after YAML deserialization. " +
                       $"Check that the YAML key exists and is not an unquoted null literal.");
           }
       }

       private void ValidateReferences(PipelineConfig config)
       {
           // Check stages reference existing data sources
           // Check features exist in data source columns
           // Check target exists in data source
           // Throw ArgumentException if invalid
       }
   }
   ```

   **Tests** (10 test cases):
   - Load valid YAML configuration
   - Validate missing data source error
   - Validate missing feature error
   - Validate invalid framework type error
   - Deserialize complex nested structures
   - Handle missing optional fields
   - **YAML 1.2 boolean enforcement** — verify `NO`, `yes`, `on`, `off` throw instead of silently converting (Risk 2a)
   - **Null rejection** — verify non-nullable property set to null throws (Risk 2b)
   - **Quoted string passthrough** — verify `"NO"` and `"yes"` survive as strings
   - **Merge key avoidance** — verify explicit anchors/aliases work correctly (Risk 2c)

3. **MlNetAdapter (SolarPipe.Training/Adapters/MlNetAdapter.cs)**

   > **AUDIT DECISION (Risk 1b — FeatureFraction=1.0):** ML.NET FastForest/FastTree default `FeatureFraction` and `FeatureFractionPerSplit` to **1.0**, meaning no feature bagging occurs. This eliminates the diversity mechanism that makes Random Forest work. Always set explicitly to 0.7 (or ≥0.5 with <10 features).
   >
   > **AUDIT DECISION (Risk 1 — Seed management):** `MLContext(seed: 0)` controls global RNG, but FastTree trainers have a separate `FeatureSelectionSeed`. Both must be pinned, and `NumberOfThreads = 1` for reproducibility.
   >
   > **AUDIT DECISION (Risk 1 — PredictionEngine thread safety):** `PredictionEngine<TIn, TOut>` is NOT thread-safe. Use `Microsoft.Extensions.ML.PredictionEnginePool<TIn, TOut>` for concurrent inference. Add NuGet: `Microsoft.Extensions.ML`.

   ```csharp
   public class MlNetAdapter : IFrameworkAdapter
   {
       public FrameworkType FrameworkType => FrameworkType.MlNet;

       public IReadOnlyList<string> SupportedModels => new[]
       {
           "FastForest", "FastTree", "LightGbm", "Gam", "Sdca", "OrdinaryLeastSquares"
       };

       public async Task<ITrainedModel> TrainAsync(
           StageConfig config, IDataFrame training,
           IDataFrame? validation, CancellationToken ct)
       {
           // AUDIT: Pin both global and trainer-level seeds for reproducibility
           var mlContext = new MLContext(seed: 42);
           var trainView = training.ToDataView(mlContext);

           // Build feature pipeline
           var featurePipeline = mlContext.Transforms.Concatenate(
               "Features", config.Features.ToArray());

           // Select trainer based on config
           var trainer = SelectTrainer(mlContext, config);

           // Fit model — AUDIT: NumberOfThreads=1 for deterministic training
           var model = await Task.Run(() => featurePipeline.Append(trainer).Fit(trainView), ct);

           // Evaluate
           var metrics = validation != null
               ? mlContext.Regression.Evaluate(model.Transform(validation.ToDataView(mlContext)))
               : null;

           return new MlNetTrainedModel(config, model, mlContext, metrics);
       }

       private IEstimator<ITransformer> SelectTrainer(MLContext mlContext, StageConfig config)
       {
           return config.Model switch
           {
               "FastForest" => mlContext.Regression.Trainers.FastForest(new FastForestRegressionTrainer.Options
               {
                   NumberOfTrees = config.GetHyperInt("number_of_trees", 100),
                   NumberOfLeaves = config.GetHyperInt("number_of_leaves", 31),
                   // AUDIT FIX (Risk 1b): NEVER default to 1.0 — disables RF diversity
                   FeatureFraction = config.GetHyperFloat("feature_fraction", 0.7f),
                   FeatureFractionPerSplit = config.GetHyperFloat("feature_fraction_per_split", 0.7f),
                   // AUDIT: Pin trainer-specific seed for reproducibility
                   FeatureSelectionSeed = config.GetHyperInt("feature_selection_seed", 42),
                   // AUDIT: Single-threaded for deterministic results
                   NumberOfThreads = config.GetHyperInt("number_of_threads", 1),
               }),
               // ... other models
               _ => throw new NotSupportedException($"Model {config.Model} not supported")
           };
       }
   }

   public class MlNetTrainedModel : ITrainedModel
   {
       private readonly ITransformer _model;
       private readonly MLContext _mlContext;
       private readonly StageConfig _config;
       private readonly RegressionMetrics? _metrics;

       public string ModelId => $"{_config.SourceConfig.SourceConfig}_{_config.Model}";
       public ModelMetrics Metrics { get; }  // Converted from RegressionMetrics

       public async Task<PredictionResult> PredictAsync(IDataFrame input, CancellationToken ct)
       {
           var dataView = input.ToDataView(_mlContext);
           var predictions = _model.Transform(dataView);

           // AUDIT FIX: Use PredictionEnginePool for thread-safe inference
           // Do NOT use _mlContext.Model.CreatePredictionEngine directly — it is
           // NOT thread-safe and throws IndexOutOfRangeException under concurrency.
           // Register PredictionEnginePool<TIn, TOut> in DI via:
           //   services.AddPredictionEnginePool<InputSchema, OutputSchema>()
           //       .FromUri(modelUri);
           // For single-threaded CLI use, CreatePredictionEngine is acceptable.
           var engine = _mlContext.Model.CreatePredictionEngine</* types */>(predictions);
           // ... build PredictionResult
       }

       public async Task SaveAsync(string path, CancellationToken ct)
       {
           // Use MLContext.Model.Save()
           // AUDIT NOTE: ML.NET uses ZIP-format binary serialization. Cross-version
           // round-trip fidelity is NOT guaranteed between major ML.NET releases.
           // ONNX export is the recommended portability path.
       }

       public async Task<ITrainedModel> LoadAsync(string path, CancellationToken ct)
       {
           // Use MLContext.Model.Load()
       }
   }
   ```

   **Tests** (6 test cases):
   - Train FastForest with valid config
   - Validate hyperparameter mapping
   - Metrics collection
   - Prediction output shape
   - Invalid model type error
   - Save/load roundtrip

**Success Criteria**:
- Load valid YAML configuration
- Train ML.NET model on synthetic data
- Metrics calculated correctly
- Model saved and loaded

**Estimated Hours**: 22 hours

---

#### 1.4 Week 4: Model Registry & CLI

**Objective**: Persistent model storage and command-line interface

**Tasks**:

1. **FileSystemModelRegistry (SolarPipe.Training/Registry/FileSystemModelRegistry.cs)**

   > **AUDIT DECISION (Risk 9 — Linux advisory-only file locking):** `FileStream` with `FileShare.None` uses `flock()` on Linux, which is **advisory-only** — non-cooperating processes can ignore it. Use atomic write pattern: write to temp file, compute SHA-256 fingerprint, then `File.Move(temp, final, overwrite: true)` which delegates to POSIX `rename()` (atomic on same filesystem). For cross-process coordination, use named `Mutex`.

   ```csharp
   public class FileSystemModelRegistry : IModelRegistry
   {
       private readonly string _basePath;  // models/
       private readonly Mutex _registryMutex = new(false, "Global\\SolarPipeRegistry");

       public async Task<ModelArtifact> RegisterAsync(ITrainedModel model, CancellationToken ct)
       {
           var artifact = new ModelArtifact
           {
               ModelId = model.ModelId,
               Version = "1.0.0",  // auto-increment for duplicates
               StageName = model.SourceConfig.StageName,
               Config = model.SourceConfig,
               Metrics = model.Metrics,
               DataFingerprint = ComputeFingerprint(/* training data */),
               TrainedAt = DateTime.UtcNow,
               ArtifactPath = Path.Combine(_basePath, model.ModelId, "1.0.0", "model.bin")
           };

           // Create directory
           Directory.CreateDirectory(Path.GetDirectoryName(artifact.ArtifactPath));

           // AUDIT FIX (Risk 9): Atomic write pattern — safe on Linux advisory locking
           // Step 1: Write to temp file
           var tempModelPath = $"{artifact.ArtifactPath}.tmp_{Guid.NewGuid():N}";
           await model.SaveAsync(tempModelPath, ct);

           // Step 2: Compute SHA-256 fingerprint of temp file
           var fileBytes = await File.ReadAllBytesAsync(tempModelPath, ct);
           artifact = artifact with {
               DataFingerprint = Convert.ToHexString(SHA256.HashData(fileBytes))
           };

           // Step 3: Atomic move (POSIX rename() is atomic on same filesystem)
           _registryMutex.WaitOne();
           try
           {
               File.Move(tempModelPath, artifact.ArtifactPath, overwrite: true);

               // Save metadata via same atomic pattern
               var tempMetaPath = $".tmp_{Guid.NewGuid():N}_metadata.json";
               var metaFinalPath = Path.Combine(_basePath, model.ModelId, "1.0.0", "metadata.json");
               await File.WriteAllTextAsync(tempMetaPath,
                   JsonConvert.SerializeObject(artifact), ct);
               File.Move(tempMetaPath, metaFinalPath, overwrite: true);
           }
           finally
           {
               _registryMutex.ReleaseMutex();
           }

           return artifact;
       }

       public async Task<ITrainedModel> LoadAsync(string modelId, string? version, CancellationToken ct)
       {
           version ??= GetLatestVersion(modelId);
           var metadataPath = Path.Combine(_basePath, modelId, version, "metadata.json");
           var metadata = JsonConvert.DeserializeObject<ModelArtifact>(
               await File.ReadAllTextAsync(metadataPath, ct));

           // Deserialize model based on framework type
           return await DeserializeModel(metadata.ArtifactPath, metadata.Config.Framework, ct);
       }

       public async Task<IReadOnlyList<ModelArtifact>> ListAsync(string? stageFilter, CancellationToken ct)
       {
           var artifacts = new List<ModelArtifact>();

           foreach (var modelDir in Directory.EnumerateDirectories(_basePath))
           {
               foreach (var versionDir in Directory.EnumerateDirectories(modelDir))
               {
                   var metadataPath = Path.Combine(versionDir, "metadata.json");
                   var artifact = JsonConvert.DeserializeObject<ModelArtifact>(
                       await File.ReadAllTextAsync(metadataPath, ct));

                   if (stageFilter == null || artifact.StageName == stageFilter)
                       artifacts.Add(artifact);
               }
           }

           return artifacts.OrderByDescending(a => a.TrainedAt).ToList();
       }

       private string ComputeFingerprint(IDataFrame data)
       {
           // Hash of row count + column hashes for data integrity
           using var hasher = System.Security.Cryptography.SHA256.Create();
           // ...
       }
   }
   ```

   **Tests** (5 test cases):
   - Register model and save metadata
   - Load model by ID
   - List all models
   - Version management
   - Corrupted metadata error handling

2. **CLI Host & Commands (SolarPipe.Host/)**
   ```csharp
   class Program
   {
       static async Task Main(string[] args)
       {
           var services = new ServiceCollection()
               .AddSingleton<IDataSourceProvider>(new SqliteProvider())
               .AddSingleton<IDataSourceProvider>(new CsvProvider())
               .AddSingleton<IFrameworkAdapter>(new MlNetAdapter())
               .AddSingleton<DataSourceRegistry>()
               .AddSingleton<IModelRegistry>(new FileSystemModelRegistry("./models"))
               .AddLogging(builder => builder.AddConsole())
               .BuildServiceProvider();

           var command = args[0];
           var handler = command switch
           {
               "validate" => services.GetRequiredService<ValidateCommand>(),
               "train" => services.GetRequiredService<TrainCommand>(),
               "predict" => services.GetRequiredService<PredictCommand>(),
               "inspect" => services.GetRequiredService<InspectCommand>(),
               _ => throw new ArgumentException($"Unknown command: {command}")
           };

           await handler.ExecuteAsync(args.Skip(1).ToArray());
       }
   }

   // TrainCommand: solarpipe train --config configs/test.yaml [--stage stagename]
   public class TrainCommand
   {
       public async Task ExecuteAsync(string[] args)
       {
           var configPath = ParseArg(args, "--config");
           var stageName = ParseArg(args, "--stage");

           var config = await _loader.LoadAsync(configPath, CancellationToken.None);

           foreach (var stage in config.Stages.Where(s => stageName == null || s.Key == stageName))
           {
               // Load data
               var data = await _registry.LoadAsync(stage.Value.Source, new DataQuery(), CancellationToken.None);

               // Split train/validation
               var split = SplitTrainValidation(data, 0.8);

               // Train
               var adapter = _adapters[stage.Value.Framework];
               var model = await adapter.TrainAsync(stage.Value, split.Train, split.Validation, CancellationToken.None);

               // Register
               await _modelRegistry.RegisterAsync(model, CancellationToken.None);

               Console.WriteLine($"✓ Trained {stage.Key}: {model.ModelId}");
           }
       }
   }

   // PredictCommand: solarpipe predict --config configs/test.yaml --input data.csv --output predictions.json
   public class PredictCommand
   {
       public async Task ExecuteAsync(string[] args)
       {
           var configPath = ParseArg(args, "--config");
           var inputPath = ParseArg(args, "--input");
           var outputPath = ParseArg(args, "--output");

           var config = await _loader.LoadAsync(configPath, CancellationToken.None);
           var input = await _csvProvider.LoadAsync(inputPath, CancellationToken.None);

           // Build and execute prediction (placeholder for Phase 2 composition)
           var result = await ExecutePipeline(config, input, CancellationToken.None);

           var json = JsonConvert.SerializeObject(result);
           await File.WriteAllTextAsync(outputPath, json, CancellationToken.None);

           Console.WriteLine($"✓ Predictions written to {outputPath}");
       }
   }
   ```

   **Tests** (4 integration tests):
   - Validate valid configuration
   - Train single stage
   - Predict with trained model
   - Output JSON correctness

**Success Criteria**:
- `solarpipe train --config test.yaml` trains and saves model
- `solarpipe predict --config test.yaml --input data.csv --output out.json` generates predictions
- Model files saved and loaded correctly
- CLI error messages are helpful

**Estimated Hours**: 18 hours

---

### Summary: Phase 1 (Weeks 1–5)

**Total Lines of Code**: ~3,200
**Total Tests**: 56
**Total Hours**: ~96 hours (includes 25% buffer)

**End-of-Phase Deliverable**:
`solarpipe validate --config test.yaml` → validates DAG and schemas in <1s → `solarpipe train --config test.yaml` → trains FastForest model on CSV data → saves to registry with atomic writes → `solarpipe predict` uses saved model

**New in Phase 1 (from CLI-agent analysis)**:
- Domain-driven value types (`GseVector`, `GsmVector`, `RadialSpeed`, `SkyPlaneSpeed`, `PhysicalConstants`)
- `PhysicsTestFixtures.cs` with validated parameter sets
- `Directory.Packages.props` for Central Package Management
- `TreatWarningsAsErrors` in all `.csproj` files
- Structured JSON logging (`logs/dotnet_latest.json`)
- `validate` CLI command with structured AI-readable error messages

**Go/No-Go Criteria**:
- All Core interfaces defined (with domain-driven value types)
- InMemoryDataFrame (partial classes, ArrayPool backing) + SqliteProvider + CsvProvider working
- YAML loader functional with Yaml12BooleanConverter and null validation
- MlNetAdapter trains and evaluates with FeatureFraction=0.7, dual seed pinning
- CLI commands functional including `validate` with AI-readable errors
- Model registry uses atomic write pattern
- `dotnet build` produces zero warnings (TreatWarningsAsErrors enabled)
- >80% code coverage for Core, Config, Data modules
- All tests passing with FluentAssertions and PhysicsTestFixtures

---

### Phase 2: Physics & Composition (Weeks 6–9)

*[Similar level of detail as Phase 1, focusing on physics models and composition algebra]*

#### Week 5-6: Composition Parser & Physics Baseline

1. **ComposeExpressionParser (SolarPipe.Config/ComposeExpressionParser.cs)**
   - Tokenizer (→, +, ^, ?, identifiers, parentheses)
   - Recursive descent parser with operator precedence
   - AST representation
   - Tests: 12 test cases (all operators, nesting, errors)

2. **DragBasedModel Physics (SolarPipe.Training/Physics/DragBasedModel.cs)**

   > **AUDIT DECISION (Risk 3a — No adaptive ODE solver in MathNet.Numerics):** MathNet.Numerics 5.0.0 provides only fixed-step RK2/RK4 plus Adams-Bashforth — **no adaptive step-size, no error estimation, no implicit solvers**. Hand-code **Dormand-Prince RK4(5)** with adaptive stepping (~200 lines C#). The FSAL (First Same As Last) tableau provides embedded 4th/5th order error estimation for step-size control. This is required for Burton ODE stability under extreme events (τ < 0.36h → RK4 diverges).

   - ODE integration: dv/dt = -γ(v-w)|v-w|
   - **Dormand-Prince RK4(5) adaptive solver** (NOT fixed-step RK4 from MathNet)
     - FSAL tableau (7 stages, reuse last evaluation)
     - Embedded error estimation (4th vs 5th order)
     - Adaptive step-size control: `h_new = h * min(5, max(0.2, 0.9 * (tol/err)^0.2))`
     - Absolute + relative tolerance: `atol=1e-8, rtol=1e-6`
   - Input validation (speed ranges: v₀ ∈ [200, 3500] km/s, γ ∈ [0.2e-7, 2e-7] km⁻¹)
   - Output validation (arrival times 0.5-5 days, starting distance ≥ 20 R☉)
   - Tests: 10 test cases:
     - Slow/fast CMEs, edge cases
     - **Stability test**: verify solver produces valid results for τ < 1h (Carrington-class)
     - **Convergence test**: verify halving step size reduces error by ~2⁵ (5th order)
     - **Analytical solution comparison**: drag model with constant γ,w has closed-form solution
     - **MathNet RK4 comparison**: verify Dormand-Prince matches RK4 for well-conditioned cases

#### Week 6-7: Composition Models + gRPC Stub

1. **ChainedModel, ResidualModel, EnsembleModel, GatedModel implementations**
   - Each with comprehensive unit tests
   - Uncertainty propagation verified
   - Type checking at composition time
   - **Test with both in-process (ML.NET) and out-of-process (gRPC stub) backends** (ADR-011)
   - Use `Task.Factory.StartNew(..., TaskCreationOptions.LongRunning)` for ensemble parallel training (ADR-014)

2. **gRPC Sidecar Stub (Week 6, ADR-011)**

   > **ANALYSIS DECISION**: Validating composition algebra solely against ML.NET creates false confidence. The gRPC sidecar interface must be stubbed and integration-tested during Phase 2, not deferred to Phase 4. This catches serialization bottlenecks and out-of-process latency issues early.

   - Define proto schema (`solarpipe.proto`) — lock in Phase 2, reuse in Phase 4
   - Lightweight Python stub returning deterministic mock predictions (no PyTorch)
   - Arrow IPC schema enforcement test: C# writes Arrow file → passes path via gRPC → Python reads → returns predictions → C# validates types
   - Composition algebra integration test: `physics_model ^ grpc_stub_model` end-to-end
   - Estimated effort: ~2 days

#### Week 7-8: Data Transforms + Temporal Alignment

1. **TransformEngine (SolarPipe.Data/Transforms/TransformEngine.cs)**
   - Normalize, standardize, log_scale, lag, window_stats
   - Coupling functions: Newell, VBs, Borovsky
   - **Sentinel value conversion** (RULE-120): Convert `9999.9`, `-1e31` to `float.NaN` at load time
   - **NaN propagation guards** (RULE-121): Per-step validation in transform chains

2. **IDataFrame.ResampleAndAlign() (RULE-122, ADR-016)**
   - Temporal alignment primitive for mismatched cadences (1-min ACE, 1-hour OMNI)
   - Supports nearest-neighbor, linear interpolation, and forward-fill
   - Prevents ad-hoc merge script proliferation that wastes agent token context

---

### Phase 3: Mock Data & Validation (Weeks 10–14)

#### ParquetProvider

> **AUDIT DECISION (Risk 8 — Parquet.Net memory):** Parquet.Net loads entire row groups into memory; a 1 GB CSV→Parquet with one row group consumed 7+ GB RAM. A float integrity bug (issue #81) produced incorrect values starting at index 38 with Snappy compression. **Use ParquetSharp** (G-Research, wrapping Apache Parquet C++) instead — it is 4-10x faster and supports random-access row group reading. Carries P/Invoke risks: use-after-dispose produces access violations, not managed exceptions.

- Use **ParquetSharp** NuGet package (not Parquet.Net)
- Schema discovery via `ParquetFileReader.SchemaDescriptor`
- Read individual row groups via `ParquetReader.OpenRowGroupReader(index)`
- Chunk files into **≤64 MB row groups** during write
- Arrow IPC interop: use `Apache.Arrow` v22.1.0 `ArrowStreamWriter`/`ArrowStreamReader`
- **2 GiB buffer limit** in .NET Arrow — validate large binary/string arrays
- Tests with ENLIL ensemble data
- Tests: verify safe disposal patterns (wrap all native handles in `using`)

#### Cross-Validation Strategies

> **AUDIT DECISION (Risk 6a — No temporal CV in ML.NET):** `CrossValidate()` performs random k-fold only. SolarPipe must implement temporal CV from scratch. For N=300-500 CME events, use **expanding-window CV with 5 folds and 3-7 day gap buffers**. Each fold's test set must contain ≥50 events for RMSE stability. Do NOT use temporal LOOCV (high variance) or random k-fold (inflated metrics).
>
> **AUDIT DECISION (Risk 6b — ENLIL data dependency leakage):** ENLIL uses observed CME parameters as input. If ENLIL simulations for event X are in training while event X's observations are in testing, information leaks. Use ENLIL augmentation only from events strictly before the test period.

- **Expanding-window temporal CV** (primary) — train on events before time t, predict after t + gap
  - 5 folds, ≥50 events per test set
  - Gap buffer sizing: 3 days minimum (Dst recovery), 5 days (CME transit), 7-10 days (active-region repeats)
- **Purged cross-validation** (de Prado 2018) — purge all training samples whose event windows overlap with test events, plus 3-7 day embargo
- **Solar-cycle-aware splits** — ensure both SC24 and SC25 phases in each training fold
- K-fold CV (for non-temporal features only, with documented caveat)
- LOOCV only for final model comparison (not hyperparameter tuning)
- Temporal leakage guards:
  - Block L1 solar wind features that contain CME signature (circular logic)
  - Compute rolling statistics only within training fold, never on full dataset
  - Detect and purge overlapping storm events spanning train/test boundary

#### Uncertainty Quantification

> **ANALYSIS DECISION (ADR-012):** Standard split conformal prediction assumes exchangeability. Space weather data is non-stationary (solar cycle). Use **EnbPI (Ensemble Predictors with Prediction Intervals)** which adapts to distribution shifts via sliding residual windows.

- **EnbPI implementation** (~150 lines C#) — primary UQ method for time-series predictions
  - Sliding window of recent residuals (default: last 100 events)
  - Adapts interval width to solar cycle phase changes
  - Asymptotic coverage guarantees for non-exchangeable data
- **Standard split conformal** retained as fallback for non-temporal use cases
- **Data invariant tests** (RULE-120): verify NaN sentinel conversion, out-of-bounds rejection

#### Pipeline Checkpointing (ADR-015)

- Per-stage checkpoint serialization to `cache/{pipeline}/{stage}.checkpoint`
- Arrow IPC format for IDataFrame checkpoints (reusable by Python sidecar)
- SHA-256 fingerprint of stage config + input data hash — stale checkpoints auto-invalidated
- CLI flags: `--resume-from-stage <name>`, `--no-cache`
- ~200 lines for checkpoint manager

#### Mock Data Integration
- ResidualCalibrator: train on synthetic, then on (obs - synthetic) residuals
- Mixed training: blend synthetic + observational
- Pretrain/finetune: initial training on synthetic, continued on observational
- **ENLIL temporal isolation**: synthetic data for event X only used when event X is in training fold

---

### Phase 4: Python Sidecar & Advanced (Weeks 15–19)

#### gRPC Sidecar

> **AUDIT DECISION (Risk 4a — Orphan processes):** Spawning Python via `Process.Start` creates an independent OS process. If .NET crashes, Python keeps running (holding GPU memory, file locks). On Windows: use Job Objects via P/Invoke. On Linux: set `PR_SET_PDEATHSIG` via `prctl`. Implement `IHostedService` for lifecycle management.
>
> **AUDIT DECISION (Risk 4b — Long-running RPCs):** gRPC deadlines are inappropriate for multi-hour training jobs — server keeps running after client times out. Use **server-streaming RPCs** with periodic progress messages and cooperative `CancellationToken` propagation. Known: connections fail after ~9 hours idle (grpc-dotnet #2628).

- Proto definitions with explicit `float` (not `double`) schema for Arrow IPC compatibility
- Python server bound to `0.0.0.0:50051` (NOT `localhost` — unreachable from containers)
- **Health checking**: implement `grpc.health.v1.Health` using `grpcio-health-checking` package; integrate with `Microsoft.Extensions.Diagnostics.HealthChecks` on .NET side
- **Process lifecycle**: `IHostedService` with `IHostApplicationLifetime.ApplicationStopping` for graceful teardown; `Process.Exited` event for immediate crash detection
- TFT trainer, Neural ODE trainer
- **Arrow IPC type enforcement** (Risk 4 — float32/64 boundary): enforce explicit `pa.float32()` schema in Pandas→Arrow conversion; validate column types on receipt in .NET

#### Physics Completions

> **AUDIT DECISION (Risk 5a — Coordinate frame confusion):** GSE→GSM transformation requires Earth's dipole tilt angle (varies with season and UT). Using wrong frame for Bz is a **silent catastrophe**. Implement Hapgood 1992 transformations with unit tests against published test vectors. No shared standard exists — Russell 1971, Hapgood 1992, and Franz & Harper 2002 give slightly different results.

- BurtonOde: dDst*/dt = Q(t) - Dst*/τ(VBs) using **Dormand-Prince solver** (from Phase 2)
  - O'Brien & McPherron (2000) parameterization: τ = 0.060·Dst* + 16.65, floor 6.15 hr
  - Injection function Q activates only when VBs > 0.5 mV/m
  - Pressure correction: Dst* = Dst − b√Pdyn + c (handle missing DSCOVR data)
  - **Must use GSM-frame Bz** — add coordinate transform utility
- NewellCoupling: v^4/3 × B_T^2/3 × sin^8/3(θ)
- **CoordinateTransform utility class**: Hapgood 1992 GSE↔GSM, with dipole tilt angle calculation
  - Tests against published test vectors
  - Seasonal and UT variation verified

#### ONNX & REST

> **AUDIT DECISION (Risk 7 — Neural ODE ONNX export impossible):** torchdiffeq's adaptive solvers use dynamic control flow that ONNX's static graph cannot represent. The adjoint method compounds this. **Three workarounds, ranked:**
> 1. **Export only dynamics network f(y,t,θ) to ONNX; implement ODE solver in C#.** ~200 ORT calls per prediction with RK4/50 steps — feasible for SolarPipe batch sizes.
> 2. **Fixed-step Euler/RK4 unrolling** into static ONNX graph. Accuracy loss vs. adaptive.
> 3. **Keep Neural ODE inference in Python sidecar permanently.** Accept this backend never becomes pure-.NET.
>
> **Recommended**: Option 1 for production; Option 3 as interim.

- OnnxAdapter: `Microsoft.ML.OnnxRuntime` v1.24.4 with opset 21
  - For Neural ODE: load only dynamics network, run Dormand-Prince solver in C# calling ORT per step
  - For standard models (exported FastForest, TFT encoder): standard ONNX inference
- RestApiProvider for solar wind data
  - NOAA SWPC JSON at `services.swpc.noaa.gov/json/` (no auth, no rate limits documented, 7-day window)
  - NASA OMNI for historical validation data
  - Handle DSCOVR safe-hold gaps gracefully

---

## Code Organization & Naming Conventions

### File Structure
```
src/
├── SolarPipe.Core/
│   ├── Interfaces/
│   │   ├── IDataFrame.cs
│   │   ├── IDataSourceProvider.cs
│   │   ├── IFrameworkAdapter.cs
│   │   ├── ITrainedModel.cs
│   │   ├── IComposedModel.cs
│   │   └── IModelRegistry.cs
│   ├── Models/
│   │   ├── DataSchema.cs
│   │   ├── ColumnInfo.cs
│   │   ├── PredictionResult.cs
│   │   ├── ModelMetrics.cs
│   │   └── ModelArtifact.cs
│   └── Enums.cs

├── SolarPipe.Config/
│   ├── Parsing/
│   │   ├── PipelineConfigLoader.cs
│   │   └── ComposeExpressionParser.cs
│   ├── Graph/
│   │   ├── PipelineGraph.cs
│   │   └── PipelineGraphBuilder.cs
│   └── Models/
│       ├── PipelineConfig.cs
│       ├── StageConfig.cs
│       └── ComposeConfig.cs

├── SolarPipe.Data/
│   ├── Providers/
│   │   ├── IDataSourceProvider.cs (reference; defined in Core)
│   │   ├── SqliteProvider.cs
│   │   ├── CsvProvider.cs
│   │   ├── ParquetProvider.cs (Phase 3)
│   │   └── RestApiProvider.cs (Phase 4)
│   ├── Transforms/
│   │   ├── TransformEngine.cs
│   │   └── CouplingFunctions.cs
│   ├── DataSourceRegistry.cs
│   └── InMemoryDataFrame.cs

├── SolarPipe.Training/
│   ├── Adapters/
│   │   ├── MlNetAdapter.cs
│   │   ├── OnnxAdapter.cs (Phase 4)
│   │   ├── PythonSidecarAdapter.cs (Phase 4)
│   │   └── PhysicsAdapter.cs
│   ├── Physics/
│   │   ├── DragBasedModel.cs
│   │   ├── BurtonOde.cs (Phase 4)
│   │   └── NewellCoupling.cs (Phase 4)
│   ├── MockData/
│   │   ├── MockDataStrategy.cs
│   │   └── ResidualCalibrator.cs
│   └── Registry/
│       └── FileSystemModelRegistry.cs

├── SolarPipe.Prediction/
│   ├── Compose/
│   │   ├── ChainedModel.cs
│   │   ├── EnsembleModel.cs
│   │   ├── ResidualModel.cs
│   │   └── GatedModel.cs
│   └── PipelineExecutor.cs

└── SolarPipe.Host/
    ├── Program.cs
    └── Commands/
        ├── TrainCommand.cs
        ├── PredictCommand.cs
        ├── ValidateCommand.cs
        └── InspectCommand.cs
```

### Naming Conventions

**Classes**:
- Interface: `IPascalCase` (e.g., `IDataFrame`)
- Concrete: `PascalCase` (e.g., `InMemoryDataFrame`)
- Adapter: `{Framework}Adapter` (e.g., `MlNetAdapter`, `PythonSidecarAdapter`)
- Model: `{Type}Model` (e.g., `DragBasedModel`, `ResidualModel`)
- Configuration: `{Component}Config` (e.g., `PipelineConfig`, `StageConfig`)

**Methods**:
- Async methods: `{Action}Async` (e.g., `LoadAsync`, `TrainAsync`)
- Predicates: `Can{Action}` or `Is{Condition}` (e.g., `CanHandle`, `IsValid`)
- Getters: property-like (e.g., `GetColumn`, `GetHyperInt`)

**Variables**:
- Private fields: `_camelCase` (e.g., `_adapter`, `_config`)
- Local variables: `camelCase` (e.g., `trainingData`, `modelId`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_SEED`)

**Tests**:
- Unit test class: `{Class}Tests` (e.g., `InMemoryDataFrameTests`)
- Test method: `{Method}_{Scenario}_Does{Expected}` (e.g., `GetColumn_ValidName_ReturnsCorrectData`)

---

## Testing Strategy

> **AUDIT DECISIONS for testing (Risks 11, 12):**
> - **Fresh MLContext per test** (Risk 11): MLContext's internal RNG is shared state. Parallel xUnit test classes sharing a static `MLContext` produce flaky tests. Create `new MLContext(seed: 42)` in each test method.
> - **No Moq for Span types** (Risk 11a): Moq/NSubstitute cannot mock `ReadOnlySpan<T>` (ref struct cannot be generic arg). Use hand-written fakes/stubs for span-consuming internal APIs. Interfaces use `float[]` so Moq works there.
> - **Tolerance-based assertions** (Risk 11): Use `Assert.InRange` or custom tolerance helpers for all floating-point comparisons. Never use exact equality.
> - **Pin NuGet versions** (Risk 11): Model outputs change between ML.NET releases. Pin exact versions in `.csproj`.
> - **UQ via split conformal prediction** (Risk 12): ML.NET FastForest hides individual tree predictions. Use split conformal prediction (~50 lines C#) for coverage-guaranteed prediction intervals instead of tree variance (which is systematically overconfident).

### Unit Test Structure

```csharp
// AUDIT: Use xUnit (not MSTest) for better async support and parallel control
public class InMemoryDataFrameTests
{
    private InMemoryDataFrame _dataFrame;

    public InMemoryDataFrameTests()
    {
        // Create test data
        var schema = new DataSchema(new List<ColumnInfo> { /* ... */ });
        var data = new float[][] { /* ... */ };
        _dataFrame = new InMemoryDataFrame(data, schema);
    }

    [Fact]
    public void GetColumn_ValidName_ReturnsCorrectData()
    {
        // Arrange
        var columnName = "feature1";
        var expected = new[] { 1.0f, 2.0f, 3.0f };

        // Act
        var result = _dataFrame.GetColumn(columnName);

        // Assert
        Assert.Equal(expected, result);
    }

    [Fact]
    public void GetColumn_InvalidName_Throws()
    {
        // Act & Assert
        Assert.Throws<KeyNotFoundException>(() => _dataFrame.GetColumn("nonexistent"));
    }
}

// AUDIT: ML.NET tests get fresh MLContext per test, sequential execution for shared resources
[Collection("ML")]
public class MlNetAdapterTests
{
    [Fact]
    public async Task TrainAsync_FastForest_ProducesValidMetrics()
    {
        // AUDIT: Fresh MLContext per test — shared state causes flaky tests
        var mlContext = new MLContext(seed: 42);
        // ... test body ...

        // AUDIT: Tolerance-based assertions for floating-point
        Assert.InRange(metrics.RSquared, 0.0, 1.0);
        Assert.True(Math.Abs(metrics.MeanAbsoluteError - expectedMAE) < 0.01,
            $"MAE {metrics.MeanAbsoluteError} outside tolerance of {expectedMAE}");
    }
}
```

### Integration Test Structure

```csharp
[Trait("Category", "Integration")]
public class SqliteProviderTests : IDisposable
{
    private readonly SqliteProvider _provider;
    private readonly string _testDbPath;

    public SqliteProviderTests()
    {
        _testDbPath = Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid():N}.db");
        CreateTestDatabase(_testDbPath);
        _provider = new SqliteProvider();
    }

    [Fact]
    public async Task LoadAsync_ValidQuery_ReturnsAllRows()
    {
        // Arrange
        var config = new DataSourceConfig { Connection = $"Data Source={_testDbPath}" };
        var query = new DataQuery { Query = "SELECT * FROM test_table" };

        // Act
        var result = await _provider.LoadAsync(config, query, CancellationToken.None);

        // Assert
        result.RowCount.Should().Be(100);
        result.Schema.Columns.Should().HaveCount(5);
    }

    public void Dispose() => File.Delete(_testDbPath);
}
```

### Test Data Fixtures

Create `tests/fixtures/` directory with:
- `test.db` (SQLite database with sample data)
- `test.csv` (CSV file with headers)
- `test.parquet` (Parquet file, Phase 3)
- `sample_config.yaml` (YAML configuration)

### Coverage Targets

| Module | Target | Phase |
|--------|--------|-------|
| SolarPipe.Core | 90% | 1 |
| SolarPipe.Config | 85% | 1 |
| SolarPipe.Data | 85% | 1 |
| SolarPipe.Training | 80% | 2 |
| SolarPipe.Prediction | 80% | 2 |
| SolarPipe.Host | 75% | 1 |
| Overall | 80% | 4 |

---

## Development Workflow

### Daily Development Cycle

1. **Start of Day**
   - Pull latest changes
   - Review issue status
   - Identify blockers

2. **During Day**
   - Implement feature (write test first)
   - Run tests frequently (`/dotnet-cli test-all`)
   - Commit incrementally (not end-of-day mega-commits)
   - Document as you go

3. **End of Day**
   - All tests passing
   - Code formatted (`/dotnet-cli format-apply`)
   - Commit message clear and actionable
   - Update PROJECT_STATUS.md with progress

### Branching Strategy

- `main` — Stable, tested code (Phase N gate)
- `dev` — Integration branch (Phase N working branch)
- `feature/{name}` — Individual features (from `dev`, merge back via PR)

### Code Review Process

1. **Self-Review**
   - Run `/dotnet-cli test-all`
   - Run `dotnet format --verify-no-changes`
   - Review own code for obvious issues
   - Update CLAUDE.md if architectural changes

2. **Peer Review** (with @architecture-reviewer agent)
   - Submit feature branch as PR
   - Agent reviews against CLAUDE.md patterns
   - Address feedback (iterate if needed)
   - Merge to `dev` when approved

3. **Phase Gate Review**
   - At end of each phase, review all code
   - Verify success criteria met
   - Update PROJECT_STATUS.md
   - Approve phase completion

### Commit Message Format

```
[PHASE-X] [MODULE] Brief description

Longer explanation of what changed and why. Reference architecture
decisions from CLAUDE.md if relevant.

Issues: #123, #456 (if applicable)
Tests: Added 5 new unit tests for X
Coverage: +3% (now 82%)
```

Example:
```
[PHASE-1] [Data] Implement SqliteProvider with schema discovery

Added SqliteProvider class implementing IDataSourceProvider interface.
Uses PRAGMA table_info for schema discovery, parameterized queries for safety.
Handles type inference (int, float, string → float) and nullable columns.

Tests: 8 unit tests covering multiple tables, filters, type inference
Coverage: Data module now at 82%
```

---

## Debugging & Troubleshooting

### Common Issues

#### Issue: IDataFrame ToDataView() produces wrong schema
**Diagnosis**:
- Check ColumnInfo order matches float[][] order
- Verify ColumnRole assignments (Feature vs. Target)
- Confirm MLContext.LoadFromEnumerable type parameter

**Fix**:
- Debug InMemoryDataFrame schema construction
- Add logging to ToDataView()
- Unit test: create dataframe, convert to DataView, inspect

#### Issue: ML.NET model produces NaN predictions
**Diagnosis**:
- Check for missing/Inf values in training data
- Verify feature scaling (normalize if ranges very different)
- Check hyperparameters (learning rate, etc.)

**Fix**:
- Add pre-training data validation step
- Implement feature scaling in TransformEngine
- Log feature statistics before training

#### Issue: Physics equation produces unreasonable results
**Diagnosis**:
- Check units (km/s vs. m/s, etc.)
- Verify parameter ranges
- Compare to reference data
- **AUDIT CHECK**: Verify coordinate frame — GSE vs GSM for Bz (Risk 5a). Wrong frame = completely wrong geoeffectiveness
- **AUDIT CHECK**: Verify ODE solver stability — if τ < 0.5h, check |hλ| against RK4 stability bound of 2.785 (Risk 3a)

**Fix**:
- Add @physics-validator agent review
- Create reference test case with known output
- Step through ODE solver by hand
- Verify Hapgood 1992 GSE→GSM transform against published test vectors

#### Issue: YAML config values silently wrong (AUDIT Risk 2a)
**Diagnosis**:
- Check if any unquoted `YES`, `NO`, `ON`, `OFF`, `y`, `n` values exist — YAML 1.1 converts these to booleans
- Check if any null-like strings (`NULL`, `~`, empty) are used as non-nullable properties
- Run post-deserialization validation

**Fix**:
- Ensure `Yaml12BooleanConverter` is registered in DeserializerBuilder
- Quote all string values that could be misinterpreted
- Add explicit test cases for the "Norway problem" strings

### Logging Strategy

Use Microsoft.Extensions.Logging with structured fields:

```csharp
_logger.LogInformation("Training model {ModelId} on {RowCount} rows with {Features} features",
    config.Model, trainingData.RowCount, config.Features.Count);

_logger.LogDebug("Feature statistics: {FeatureName} min={Min:F2} max={Max:F2} mean={Mean:F2}",
    name, stats.Min, stats.Max, stats.Mean);

_logger.LogWarning("Model {ModelId} achieved R² of {R2:F3}, below typical 0.7 threshold",
    modelId, metrics.R2);

_logger.LogError(ex, "Failed to load model {ModelId} from {Path}", modelId, path);
```

Enable debug logging with:
```bash
LOGLEVEL=Debug dotnet run --project src/SolarPipe.Host -- train --config test.yaml
```

---

## Performance & Optimization

### Phase 1 Targets (Foundation)

- **Training**: Single stage (FastForest) on ~300 samples in <5 seconds
- **Prediction**: Inference on 100 samples in <500ms
- **Memory**: <500MB RAM for typical pipeline

### Phase 4 Targets (Full System)

- **Training**: Full pipeline with mock data (5000 samples) in <30 seconds
- **Prediction**: Inference with composition algebra in <1 second
- **Memory**: <2GB RAM for production use

### Optimization Opportunities

1. **Lazy Loading**: Load data columns only when needed
2. **Caching**: Cache compiled ML.NET models in memory
3. **Parallelization**: Train multiple stages in parallel (future)
4. **Batching**: Process predictions in batches for gRPC sidecar

### Benchmarking

Create `tests/performance/` with benchmarks:

```csharp
[SimpleJob(warmupCount: 3, targetCount: 5)]
public class InMemoryDataFrameBenchmarks
{
    private InMemoryDataFrame _dataFrame;

    [GlobalSetup]
    public void Setup()
    {
        // Create large dataframe (10k rows, 50 columns)
    }

    [Benchmark]
    public void GetColumn_ByName()
    {
        var col = _dataFrame.GetColumn("feature1");
    }
}
```

Run with:
```bash
dotnet run -c Release -p tests/SolarPipe.Tests.Performance
```

---

## Documentation Requirements

### Code Documentation

**Do NOT generate XML `<summary>`, `<param>`, or `<returns>` tags** (RULE-111). In a CLI/agent workflow with no IntelliSense, XML doc comments consume tokens while providing zero value. Write self-documenting code with clear names instead:

```csharp
// GOOD: Self-documenting name, no XML docs needed
public async Task<IDataFrame> LoadAsync(DataSourceConfig config, DataQuery query, CancellationToken ct)

// BAD: Verbose XML docs that restate the obvious — wastes agent context tokens
// /// <summary>Loads data from a data source</summary>
// /// <param name="config">The configuration</param>
```

If a method's purpose is not clear from its name and parameter types, rename it rather than adding documentation.

### Phase Completion Documentation

At end of each phase:
1. Update CLAUDE.md with any architectural changes
2. Update PROJECT_STATUS.md with actual vs. planned hours
3. Create phase summary document with:
   - Lines of code written
   - Test cases created
   - Coverage achieved
   - Known issues/technical debt
   - Lessons learned

### Architecture Decision Log

Record significant decisions:

**[2026-04-15] Use float[][] for internal data storage**

*Context*: Need efficient column-major storage for IDataFrame abstraction

*Decision*: Use float[][] (array of columns) instead of row-major array or custom matrix class

*Rationale*:
- Efficient for columnar operations (feature selection, transforms)
- Zero-copy access via ReadOnlySpan<float>
- Easy serialization for Python sidecar (direct array access)
- Simple schema mapping (index → column name)

*Consequences*:
- Requires transposition for row-wise operations
- No automatic type conversion (everything is float; use NaN for missing)

---

## Audit-Driven Architectural Decisions Summary

The following table summarizes all architectural decisions forced by the pre-implementation risk audit. Each decision is embedded in context throughout this document; this table serves as a quick reference.

| Risk ID | Decision | Rationale | Affects |
|---------|----------|-----------|---------|
| 1a | Validate column lengths match in `ToDataView()` | Mismatched columns cause silent wrong predictions | Week 1: IDataFrame |
| 1b | Default `FeatureFraction=0.7` (not 1.0) | 1.0 disables RF diversity mechanism | Week 3: MlNetAdapter |
| 1c | Use `float[]` in interfaces, not `ReadOnlySpan<float>` | ref struct banned from async methods | Week 1: All interfaces |
| 2a | Custom `Yaml12BooleanConverter` | YAML 1.1 converts NO/YES/ON/OFF to booleans | Week 3: Config loader |
| 2b | Post-deserialization null validation | YamlDotNet assigns null to non-nullable props silently | Week 3: Config loader |
| 3a | Hand-code Dormand-Prince RK4(5) | MathNet.Numerics has no adaptive/stiff solvers | Week 5-6: Physics |
| 4a | `IHostedService` + Job Objects for sidecar | Orphan processes hold GPU memory | Week 13-14: Sidecar |
| 4b | Server-streaming RPCs for training | gRPC deadlines cause server-side zombie execution | Week 13-14: Sidecar |
| 5a | Hapgood 1992 GSE→GSM transform with tests | Wrong coordinate frame = wrong Bz = catastrophe | Week 15-16: Physics |
| 6a | Custom expanding-window temporal CV | ML.NET has no temporal CV; random k-fold leaks | Week 9-10: Validation |
| 6b | Temporal isolation of ENLIL synthetic data | Shared input parameters leak through CME catalog | Week 11-12: Mock data |
| 7 | Export dynamics network only to ONNX; C# solver | Neural ODE adaptive solvers incompatible with ONNX | Week 15-16: ONNX |
| 8 | Use ParquetSharp instead of Parquet.Net | Parquet.Net loads full row groups; float bug at idx 38 | Week 9-10: Data |
| 9 | Atomic write (temp + File.Move) for registry | Linux advisory-only locking is unsafe | Week 4: Registry |
| 11a | `float[]` interfaces enable Moq; hand-write span fakes | Moq cannot intercept ref struct parameters | Week 1: Interfaces |
| 11 | Fresh `MLContext(seed: 42)` per test | Shared MLContext RNG causes flaky tests | All test phases |
| 12 | Split conformal prediction for UQ (~50 lines) | FastForest hides individual trees; tree variance overconfident | Week 9-12: UQ |

### Uncertainty Quantification Strategy

> **AUDIT DECISION (Risk 12):** ML.NET `FastForestRegressionModelParameters` does not expose individual tree predictions. Tree variance-based intervals are systematically overconfident (Wager et al. 2014). Use **split conformal prediction** instead:

```csharp
/// <summary>
/// Split conformal prediction: provides finite-sample coverage guarantees
/// regardless of underlying model. ~50 lines, no individual tree access needed.
/// </summary>
public class SplitConformalPredictor
{
    private float[] _sortedResiduals;  // |y_cal - ŷ_cal| sorted ascending

    /// <summary>
    /// Calibrate on held-out set: compute sorted absolute residuals.
    /// </summary>
    public void Calibrate(float[] actual, float[] predicted)
    {
        _sortedResiduals = actual.Zip(predicted, (a, p) => MathF.Abs(a - p))
            .OrderBy(r => r).ToArray();
    }

    /// <summary>
    /// Get prediction interval half-width for desired coverage level.
    /// Coverage guarantee: P(Y ∈ [ŷ ± width]) ≥ 1 - α for any distribution.
    /// </summary>
    public float GetIntervalWidth(float alpha = 0.1f)
    {
        // (1 - α) quantile of calibration residuals
        int index = (int)Math.Ceiling((1 - alpha) * (_sortedResiduals.Length + 1)) - 1;
        index = Math.Clamp(index, 0, _sortedResiduals.Length - 1);
        return _sortedResiduals[index];
    }
}
```

---

## Conclusion

This detailed implementation plan provides week-by-week guidance for building SolarPipe over 19 weeks (25% per-phase buffer). Key principles:

1. **Incremental delivery** — Each phase is independent; Phase N can ship alone
2. **Test-driven** — Every feature has tests; >80% coverage maintained
3. **Architecture-first** — Interfaces defined before implementations
4. **Documentation-as-code** — CLAUDE.md, AUTOMATION_*.md, ADRs guide future work

Next step: **Week 1 kickoff — Create solution and Core interfaces**

---

**Document Version History**:
- v1.3 (2026-04-06): Aligned to 19-week timeline (Weeks 1–5, 6–9, 10–14, 15–19); removed XML doc examples (contradicted RULE-111); corrected scope line from 16 to 19 weeks
- v1.2 (2026-04-06): CLI-agent workflow analysis — domain-driven types (GseVector, RadialSpeed), PhysicsTestFixtures, ArrayPool/LOH mitigation, gRPC stub in Phase 2, EnbPI for UQ, pipeline checkpointing, dry-run DAG validation, Central Package Management, TreatWarningsAsErrors, structured JSON logging, LongRunning tasks, temporal alignment primitive, sentinel value conversion, 25% phase buffers
- v1.1 (2026-04-06): Integrated pre-implementation risk audit (37 risks, 8 critical) — updated IDataFrame interface (float[] not Span), YAML 1.2 type converter, FeatureFraction fix, Dormand-Prince ODE solver, atomic registry writes, ParquetSharp, temporal CV, split conformal prediction, Neural ODE ONNX workaround, sidecar lifecycle, coordinate transforms
- v1.0 (2026-04-06): Initial creation from SolarPipe_Architecture_Plan.docx
