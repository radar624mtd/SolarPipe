# ARCHITECTURE.md

Detailed architecture reference for SolarPipe. Read CLAUDE.md first for agent rules and commands.

## Module Dependency Graph

```
SolarPipe.Config          (DSL parser, graph builder, YAML deserialization)
    ↓
SolarPipe.Data            (unified data abstraction, providers for SQLite/CSV/Parquet/REST)
    ↓
SolarPipe.Training        (framework dispatch: ML.NET, ONNX, Python gRPC sidecar, Physics)
    ↓
SolarPipe.Prediction      (composition engine: chain, ensemble, residual, gating)
```

**SolarPipe.Core** defines shared interfaces and models. All modules depend on Core; no horizontal dependencies.

## Framework Backends

| Framework | Use Case |
|-----------|---------|
| ML.NET | Tabular/tree models: FastForest, FastTree, LightGBM, GAM, SDCA |
| ONNX Runtime | Inference of exported models (model agnostic) |
| Physics Adapter | Analytical equations as first-class models: drag-based CME, Burton ODE, Newell coupling |
| Python gRPC Sidecar | Deep learning: TFT, Neural ODEs, custom PyTorch architectures |

## Model Composition Algebra

Models are composed via four operators:

| Operator | Symbol | Semantics |
|----------|--------|-----------|
| Chain | `->` | Output of left feeds input of right |
| Ensemble | `+` | Weighted average of outputs |
| Residual | `^` | Right learns residual (error) of left |
| Gate | `?` | Conditional routing: `classifier ? (model_if_true, model_if_false)` |

Example: `drag_physics ^ regression_forest` — RF learns to correct systematic errors in the physics baseline.

## Key Interfaces

### IDataFrame

Central data abstraction bridging all frameworks:

```csharp
public interface IDataFrame : IDisposable
{
    DataSchema Schema { get; }
    int RowCount { get; }
    float[] GetColumn(string name);       // float[], not ReadOnlySpan — see ADR-001/RULE-001
    float[] GetColumn(int index);
    IDataFrame Slice(int startRow, int count);
    IDataFrame SelectColumns(params string[] columns);
    IDataFrame AddColumn(string name, float[] values);
    IDataView ToDataView(MLContext mlContext);  // ML.NET bridge
    float[][] ToArray();                        // Python sidecar bridge
    IDataFrame ResampleAndAlign(TimeSpan cadence);
}
```

**float[] rationale**: ref structs cannot cross async boundaries and cannot be mocked by NSubstitute. Use `ReadOnlySpan<float>` only in synchronous internal methods.

**Memory**: Allocations >85KB land on the LOH. Use `ArrayPool<float>.Shared.Rent()/Return()` inside `InMemoryDataFrame` internals. See ADR-009.

**NaN policy**: `float.NaN` = missing. Sentinel values (`9999.9`, `-1e31`) must be converted to NaN at load time by providers. Provenance tracked in `DataSchema.ColumnInfo.MissingReason`.

**Partial classes**: `InMemoryDataFrame.Core.cs`, `InMemoryDataFrame.Transforms.cs`, `InMemoryDataFrame.IO.cs` — stay under ~300 lines per file.

### IFrameworkAdapter

```csharp
public interface IFrameworkAdapter
{
    FrameworkType FrameworkType { get; }
    IReadOnlyList<string> SupportedModels { get; }
    Task<ITrainedModel> TrainAsync(
        StageConfig config,
        IDataFrame trainingData,
        IDataFrame? validationData,
        CancellationToken ct);
}
```

### IDataSourceProvider

```csharp
public interface IDataSourceProvider
{
    string ProviderName { get; }
    Task<DataSchema> DiscoverSchemaAsync(DataSourceConfig config, CancellationToken ct);
    Task<IDataFrame> LoadAsync(DataSourceConfig config, DataQuery query, CancellationToken ct);
    bool CanHandle(DataSourceConfig config);
}
```

### IComposedModel

```csharp
public interface IComposedModel
{
    string Name { get; }
    Task<PredictionResult> PredictAsync(IDataFrame input, CancellationToken ct);
}
```

Implementations: `ChainedModel`, `EnsembleModel`, `ResidualModel`, `GatedModel`.

## Domain-Driven Type Safety

Use the type system to make invalid physics unrepresentable at compile time:

```csharp
// Coordinate frame safety — prevents GSE/GSM confusion (RULE-031)
public readonly record struct GseVector(float Bx, float By, float Bz);
public readonly record struct GsmVector(float Bx, float By, float Bz);
// No implicit conversion — must call CoordinateTransform.GseToGsm(gse, tiltAngle)

// Speed correction safety — prevents using uncorrected CDAW speeds (RULE-101)
public readonly record struct SkyPlaneSpeed(float KmPerSec);
public readonly record struct RadialSpeed(float KmPerSec);
// DragBasedModel.Solve() accepts RadialSpeed, not SkyPlaneSpeed

// Physics constants — always use PhysicalConstants.cs, never inline literals
public static class PhysicalConstants
{
    public const float EarthRadiusKm = 6371.0f;
    public const float SolarRadiusKm = 695700.0f;
    public const float ProtonMassKg = 1.6726219e-27f;
    public const float AuKm = 1.496e8f;
}
```

## Model Registry

```csharp
public record ModelArtifact
{
    public string ModelId { get; init; }
    public string Version { get; init; }          // Semantic versioning
    public string StageName { get; init; }
    public StageConfig Config { get; init; }      // Full reproducibility
    public ModelMetrics Metrics { get; init; }
    public string DataFingerprint { get; init; }  // SHA-256 hash of training data
    public DateTime TrainedAt { get; init; }
    public string ArtifactPath { get; init; }
}
```

Atomic writes required: temp file + `File.Move`. See RULE-040, ADR-008.

## YAML DSL Structure

```yaml
pipeline: <name>
  data_sources:    # provider, connection, schema
  stages:          # ordered processing stages
  compose:         # model composition expressions
  evaluation:      # validation strategy and metrics
  output:          # prediction output format
```

Each **stage** specifies: data source, framework, model type, feature/target columns, hyperparameters, validation method, optional mock data strategy.

## Data Transforms

| Transform | Syntax | Use Case |
|-----------|--------|---------|
| Normalize | `normalize(min, max)` | Scale to [0,1] |
| Standardize | `standardize` | Zero-mean, unit-variance |
| Log scale | `log_scale` | Handle skewed distributions |
| Lag | `lag(n=6, step=1)` | Lagged time-series features |
| Window stats | `window_stats(size=60, ops=[mean,std,min,max])` | Rolling aggregations |
| Coupling fn | `coupling(type=newell)` | Newell, VBs, Borovsky functions |
| Physics derived | `physics(equation=burton_tau, inputs=[v,bz])` | Physics equations as features |

## NuGet Dependencies

All versions pinned in `Directory.Packages.props`. Never floating ranges.

**Core runtime**: `Microsoft.ML`, `Microsoft.Extensions.ML`, `Microsoft.ML.OnnxRuntime`, `YamlDotNet` (YAML 1.1 + Yaml12BooleanConverter), `MathNet.Numerics` (matrix ops only — NOT ODE solving, see ADR-003)

**Data access**: `Microsoft.Data.Sqlite`, `CsvHelper`, `ParquetSharp` (NOT Parquet.Net — see ADR-005), `System.IO.Abstractions`

**Observability**: `Serilog`, `Serilog.Sinks.File`, `OpenTelemetry`, `grpc-dotnet`

**Testing**: `xUnit`, `FluentAssertions`, `NSubstitute`

## Cross-Cutting Concerns

- **Logging**: Serilog → `logs/dotnet_latest.json` (structured JSON). Console: progress + critical errors only. Python sidecar → `logs/python_latest.json`. Shared Trace ID for cross-process correlation.
- **DI**: `Microsoft.Extensions.DependencyInjection`; modules register via `IServiceCollection` extensions.
- **Config**: `Microsoft.Extensions.Configuration` + YamlDotNet. All YAML paths absolute or relative to `$SOLARPIPE_ROOT`. `validate` command catches all config errors before data loading.
- **Async/Cancellation**: `CancellationToken` throughout. Training timeout: 30 min. Prediction timeout: 60s. Throw `TimeoutException` with stage context — prevents terminal hangs.
- **Build**: `<TreatWarningsAsErrors>true</TreatWarningsAsErrors>` + `<NoWarn>CS1591</NoWarn>` in all `.csproj`.
- **CPU-bound training**: `Task.Factory.StartNew(..., TaskCreationOptions.LongRunning)` — not `Task.Run()`. See RULE-120, ADR-014.

## Physics as a First-Class Framework

PhysicsAdapter wraps analytical equations as "trained models" composable with ML models:
- **Drag-based CME kinematics** (Vršnak 2013) — arrival time from initial speed/mass + ambient solar wind
- **Burton ring-current ODE** — Dst index from solar wind coupling
- **Newell coupling function** — solar wind–magnetosphere energy transfer

ODE solving: Dormand-Prince RK4(5) only (~200 lines C#). MathNet.Numerics ODE solvers banned (no adaptive step-size). See RULE-030, ADR-003.

All physics equations operate in GSM-frame Bz. GSE Bz produces completely wrong geoeffectiveness. See RULE-031.

## Mock Data Strategies

| Strategy | When to Use |
|----------|-------------|
| `pretrain_then_finetune` | Train on synthetic, continue on observational (reduced LR) |
| `mixed_training` | Blend synthetic + observational with configurable weights |
| `residual_calibration` | Train on synthetic → learn correction on (observation - synthetic) residuals |

Residual calibration recommended for CME propagation: ENLIL simulations accurate in topology but systematically biased in magnitude.

## Validation Strategy

Small observational dataset (~300–500 CME events) — overfitting risk is high.

- **Primary: LOOCV** — Unbiased for small N; feasible for RF
- **Secondary: Expanding-window temporal CV** — Train before time t, predict after t + gap; prevents temporal leakage
- **Tertiary: Solar-cycle-aware splits** — Train on SC24 + SC25 phases for cycle generalization

Uncertainty quantification: **adaptive conformal prediction (EnbPI)** — standard split conformal assumes exchangeability which space weather violates. See ADR-007.

## Testing Approach

**Mandatory**: All new `IFrameworkAdapter`, `IDataSourceProvider`, `IComposedModel`, and `InMemoryDataFrame` code must ship with unit tests. Target >80% coverage per module.

- **Unit tests**: NSubstitute mocks. Fresh `MLContext(seed: 42)` per test method. `NumberOfThreads = 1` for determinism.
- **Data invariant tests**: Verify `InMemoryDataFrame` handles sentinel values, NaN propagation, domain bounds.
- **Integration tests**: Full data pipeline with test SQLite/Parquet files. `MockFileSystem` for registry tests.
- **Pipeline tests**: End-to-end composition algebra correctness, registry storage.
- **Physics tests**: Validated against published reference values (O'Brien & McPherron 2000, NASA OMNI, SuperMAG). Always use `PhysicsTestFixtures.cs`.

Test segmentation: `[Trait("Category", "Unit")]` and `[Trait("Category", "Integration")]`.

## Common Extension Patterns

### Adding a New Framework

1. `src/SolarPipe.Training/Adapters/YourFrameworkAdapter.cs` implementing `IFrameworkAdapter`
2. Register in DI container in Host project
3. Add `FrameworkType` enum value in Core
4. Add config parsing for framework-specific hyperparameters
5. Write integration tests

### Adding a New Data Provider

1. `src/SolarPipe.Data/Providers/YourProvider.cs` implementing `IDataSourceProvider`
2. Implement `DiscoverSchemaAsync` and `LoadAsync`
3. Register in `DataSourceRegistry`

### Adding a Physics Equation

1. Implement `IPhysicsEquation` in `SolarPipe.Training/Physics/`
2. Register in `PhysicsAdapter`'s equation dictionary
3. Reference by name in YAML config: `equation: your_equation_name`
4. Add `PhysicsTestFixtures.cs` test cases validated against published values

## Reference Use Case: CME Flux Rope Rotation Prediction

Pipeline predicts flux rope Bz orientation at Earth arrival:

1. **Drag-based physics baseline** (PhysicsAdapter) — CME arrival time from initial conditions
2. **Random Forest correction** (ML.NET FastForest) — Residual rotation from 15 features (chirality, axis angle, dimming morphology, HCS geometry, solar wind state)
3. **Residual composition** (`drag_baseline ^ rotation_predictor`)

Mock data: ENLIL MHD ensemble (5000 synthetic events) for pretraining; ~300 observational CME events for residual calibration.

## Known Gaps and Future Work

- Model registry: file-system MVP → MLflow/database for production
- Python sidecar gRPC must be **stubbed in Phase 2**, not deferred to Phase 4 (see ADR-012)
- Neural ODE ONNX: export dynamics network only — full adaptive solver export is fundamentally impossible
- Pipeline checkpointing (`--resume-from-stage`) with SHA-256 fingerprint invalidation
- Large-array IPC: file-based Arrow IPC (not inline Protobuf — gRPC 4MB limit)
- Feature importance stability analysis: stubbed
