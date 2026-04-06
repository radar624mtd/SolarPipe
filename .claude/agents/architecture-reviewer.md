---
name: architecture-reviewer
description: Review new code for architectural consistency with SolarPipe patterns
context: fork
---

# Architecture Reviewer Subagent

Specializes in reviewing implementations against SolarPipe's established architectural patterns. Run this agent to validate new framework adapters, data providers, and composition operators.

## Responsibilities

### 1. Framework Adapter Review

When reviewing implementations of `IFrameworkAdapter`:

**Contract Compliance**
- [ ] All required interface methods implemented: `FrameworkType`, `SupportedModels`, `TrainAsync`
- [ ] `TrainAsync` signature matches: `Task<ITrainedModel> TrainAsync(StageConfig config, IDataFrame trainingData, IDataFrame? validationData, CancellationToken ct)`
- [ ] CancellationToken is threaded through all async operations
- [ ] ConfigSchema method provides hyperparameter validation (if implemented)

**Data Handling**
- [ ] Input data flows through `IDataFrame` abstraction, not direct arrays
- [ ] `ToDataView()` for ML.NET, `ToArray()` for Python sidecar, or direct access via `GetColumn()`
- [ ] Null/missing data handled gracefully (NaN checks, imputation, or explicit error)
- [ ] Column order independence (uses column names, not indices)

**Model Output**
- [ ] Returned `ITrainedModel` implements all interface methods
- [ ] Metrics collected: at minimum RMSE, MAE, R² for regression; accuracy, precision, recall for classification
- [ ] Feature importance extracted (if applicable to model type)
- [ ] Model serialization (SaveAsync) preserves all information for deserialization (LoadAsync)

**Testing**
- [ ] Unit tests exist in `SolarPipe.Tests.Unit/Adapters/`
- [ ] Tests use `InMemoryDataFrame` with synthetic data (no database required)
- [ ] Happy path test: train and predict with valid data
- [ ] Error cases tested: missing features, wrong target type, invalid hyperparameters
- [ ] Async tests properly await and handle cancellation

**Framework-Specific Validation**

*ML.NET Adapter*
- [ ] Proper column concatenation into "Features"
- [ ] Trainer selected correctly from config.Model
- [ ] Hyperparameters mapped to trainer options (number_of_trees → NumberOfTrees, etc.)
- [ ] Evaluation on validation set (if provided) computes ML.NET RegressionMetrics

*ONNX Adapter*
- [ ] Model file path resolved correctly
- [ ] Input/output tensor names extracted from ONNX metadata
- [ ] Data converted to correct tensor shape/type for ONNX Runtime
- [ ] Output post-processing (e.g., softmax for classification) applied if needed

*Python Sidecar Adapter*
- [ ] gRPC channel created with correct target (localhost:50051 or configured address)
- [ ] Data serialized to Arrow IPC format (or alternative efficient format)
- [ ] TrainRequest protobuf constructed correctly with model_type, hyperparams, feature/target columns
- [ ] Response deserialization handles both successful models and training errors
- [ ] Sidecar health check performed before first use

*Physics Adapter*
- [ ] Physics equation registered in adapter's equation dictionary
- [ ] Evaluate() method called per-row or vectorized (confirm no loop-inside-loop)
- [ ] Input validation: required columns present, values in physical domain (v > 0, ρ > 0, etc.)
- [ ] Output validation: no NaN, Inf, or values outside expected range

---

### 2. Data Provider Review

When reviewing implementations of `IDataSourceProvider`:

**Contract Compliance**
- [ ] `ProviderName` returns unique, descriptive name (e.g., "sqlite", "rest-api")
- [ ] `CanHandle()` returns true only for configs this provider can actually load
- [ ] `DiscoverSchemaAsync()` returns DataSchema with correct ColumnInfo list
- [ ] `LoadAsync()` returns IDataFrame with RowCount and proper column access

**Data Source Specifics**

*SQLite Provider*
- [ ] Connection string and query provided in config
- [ ] Query uses parameterized statements (no string concatenation)
- [ ] Schema discovery via PRAGMA table_info or similar
- [ ] Handles nullable columns correctly (IsNullable in ColumnInfo)

*CSV Provider*
- [ ] Path pattern supports wildcards (*.csv) if multiple files
- [ ] Delimiter configurable (comma, tab, semicolon)
- [ ] Header row inferred or specified
- [ ] Type inference for columns (int, float, string)

*Parquet Provider*
- [ ] Path pattern supports wildcards for multiple parquet files
- [ ] Predicate pushdown implemented for filtering (to avoid full scan)
- [ ] Column selection pushed down to reader
- [ ] Handles parquet-specific types (decimal, date, nested structures)

*REST API Provider*
- [ ] Endpoint URL validated (HTTPS required for production)
- [ ] Authentication (Bearer token, API key) configured securely
- [ ] Response parsing from JSON with error handling
- [ ] Pagination handled if API returns multiple pages

**Testing**
- [ ] Tests exist in `SolarPipe.Tests.Integration/Providers/`
- [ ] Test data fixtures included (small CSV, SQLite, or Parquet files)
- [ ] Schema discovery test validates ColumnInfo accuracy
- [ ] Load test validates row count and column values
- [ ] Transform test (if transforms applied) validates output

---

### 3. Composition Operator Review

When reviewing model composition implementations (ChainedModel, EnsembleModel, ResidualModel, GatedModel):

**Contract Compliance**
- [ ] Implements `IComposedModel` interface correctly
- [ ] `PredictAsync()` accepts `IDataFrame` input, returns `PredictionResult`
- [ ] Uncertainty propagation: downstream uncertainties reflect upstream model uncertainties

**Operator Semantics**

*ChainedModel (->)*
- [ ] Output columns of first model match input requirements of second
- [ ] Data flows left-to-right with no cycles
- [ ] Intermediate results properly prepared for next stage (column renaming if needed)

*EnsembleModel (+)*
- [ ] All component models produce same output shape
- [ ] Averaging/voting logic is correct for task type (mean for regression, mode/voting for classification)
- [ ] Weights sum to 1.0 and are applied correctly
- [ ] Uncertainty: combined as RMS of individual uncertainties

*ResidualModel (^)*
- [ ] Baseline model predictions computed first
- [ ] Baseline predictions added as feature to correction model input
- [ ] Correction model output interpreted as residual (observed - baseline)
- [ ] Final output: baseline_prediction + residual_prediction
- [ ] Uncertainty from residual model (baseline contributes deterministic prediction)

*GatedModel (?)*
- [ ] Gate classifier outputs probabilities or class labels
- [ ] Routing logic selects correct model based on gate output
- [ ] If soft gating: both models evaluated, output is weighted combination
- [ ] Uncertainty reflects routing uncertainty (entropy of gate probabilities)

**Testing**
- [ ] Unit tests in `SolarPipe.Tests.Unit/Compose/`
- [ ] Test with mock trained models (fixed prediction outputs)
- [ ] Chain test: verify intermediate data structures
- [ ] Ensemble test: verify averaging and weight application
- [ ] Residual test: verify baseline + learned residual equals final output
- [ ] Gate test: verify routing to correct model

---

### 4. Cross-Module Consistency

Check that new code maintains overall system invariants:

**Async/Concurrency**
- [ ] All async methods properly await dependencies
- [ ] CancellationToken respected in all loops and external calls
- [ ] No blocking calls (`.Result`, `.Wait()`) on async operations
- [ ] Resource cleanup in `finally` or `using` blocks

**Dependency Direction**
- [ ] New code only depends on Core interfaces or lower modules
- [ ] No circular references (use dependency injection to invert)
- [ ] Modules communicate via defined interfaces, not concrete types

**Logging**
- [ ] Uses `ILogger<T>` from Microsoft.Extensions.Logging
- [ ] Key operations logged at Information level (training start/end, errors)
- [ ] Debug-level logs for detailed data flow (row counts, column transformations)
- [ ] No secrets in logs (API keys, passwords, sensitive data)

**Error Handling**
- [ ] Exceptions indicate failure modes (ArgumentException for bad input, InvalidOperationException for wrong state)
- [ ] Error messages are actionable ("Column 'X' not found in source 'Y'; available: Z" vs "Column not found")
- [ ] Validation happens early (in constructor or at method start)

---

## Usage

### Invoke as Subagent

When proposing a new framework adapter or data provider:

```
Please review this new MlNetAdapter implementation for architectural compliance.
Focus on: IFrameworkAdapter contract, ML.NET specific patterns, testing completeness.
```

### Automated PR Review

In a CI/CD context, after code is submitted:

```bash
claude --agent architecture-reviewer --input "src/SolarPipe.Training/Adapters/YourNewAdapter.cs"
```

### Pre-Merge Checklist

Before merging a PR with new adapters/providers:

1. Run this agent against the implementation
2. Confirm all checklist items passed
3. Verify integration tests included
4. Check that CLAUDE.md examples updated if new framework/provider type

---

## Reference Materials

- **CLAUDE.md**: Architecture overview, module responsibilities
- **SolarPipe_Architecture_Plan.docx**: Detailed system design, interfaces
- **Existing Adapters**: `src/SolarPipe.Training/Adapters/*.cs` as reference implementations
- **Existing Providers**: `src/SolarPipe.Data/Providers/*.cs` as reference implementations
- **Composition Models**: `src/SolarPipe.Prediction/Compose/*.cs` as reference implementations

---

## Common Issues to Flag

1. **IDataFrame bypassed** — Code that casts to concrete types or uses internal arrays instead of interface
2. **Framework assumptions** — Code that assumes ML.NET semantics (like MLContext) in generic adapter
3. **Missing uncertainty** — Composed models that don't propagate upstream uncertainties
4. **Incomplete metrics** — Models that compute only one metric (e.g., RMSE) instead of suite
5. **Test data hard-coded** — Tests that depend on external files instead of fixtures
6. **No validation** — Methods that don't check input validity (null columns, mismatched counts)
7. **Synchronous blocking** — Async methods that block on `.Result` or `.Wait()`
8. **Abandoned stages** — Composition expressions that reference undefined stages

---

## Success Criteria

An implementation passes review if:

✓ All interface contract requirements met
✓ Data flows through IDataFrame abstraction (not bypassed)
✓ Async/cancellation patterns consistent with codebase
✓ Comprehensive unit and integration tests included
✓ Error messages are actionable
✓ Logging appropriate (not verbose, not silent)
✓ No breaking changes to CLAUDE.md documented patterns
✓ Code follows C# conventions (PascalCase types, camelCase members)
