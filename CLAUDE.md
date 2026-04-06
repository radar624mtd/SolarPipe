# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Development Context

**Single developer, CLI-only, fully agentic AI workflow.** There is no IDE.
- You ARE the IDE. Run `dotnet build --no-restore` after every `.cs` file edit.
- Terminal output is your only feedback — errors must be parseable, not buried in noise.
- Token context is finite. Keep files small, avoid verbose boilerplate.
- When debugging, read structured log files (`logs/*.json`), not raw console output.

## Project Overview

**SolarPipe** is a .NET 8 declarative ML orchestration framework for CME propagation modeling and geomagnetic storm prediction. Configuration over code: pipeline topology, model selection, and composition rules are defined in YAML, not hardcoded.

For full architecture details, interfaces, design patterns, and reference use cases: read `ARCHITECTURE.md`.

## Solution Structure

```
SolarPipe.sln
├── src/
│   ├── SolarPipe.Core/       # Shared interfaces (IDataFrame, IFrameworkAdapter, etc.)
│   ├── SolarPipe.Config/     # DSL parser, graph builder, YAML deserialization
│   ├── SolarPipe.Data/       # Data providers (SQLite, CSV, Parquet, REST), transforms
│   ├── SolarPipe.Training/   # Framework adapters (ML.NET, ONNX, Physics, Python gRPC)
│   ├── SolarPipe.Prediction/ # Composition engine (chain, ensemble, residual, gate)
│   └── SolarPipe.Host/       # CLI entry point
├── tests/
│   ├── SolarPipe.Tests.Unit/
│   ├── SolarPipe.Tests.Integration/
│   └── SolarPipe.Tests.Pipeline/
├── python/                   # gRPC sidecar (PyTorch/TFT trainer)
├── configs/                  # Example YAML pipeline configurations
└── docs/
```

## Common Commands

```bash
# Build (run after every .cs edit)
dotnet build --no-restore

# Unit tests only during active coding
dotnet test --filter Category=Unit --no-build

# Full test suite
dotnet test

# Validate a pipeline config
dotnet run --project src/SolarPipe.Host -- validate --config configs/your_config.yaml

# Edit Directory.Packages.props, then restore (NEVER dotnet add package)
dotnet restore

# Python sidecar — always use explicit path
${SOLARPIPE_ROOT}/python/.venv/bin/python -m solarpipe_sidecar.server
```

## Session 0: Bootstrap (Run Before Any Coding)

**Shell**: Claude Code on Windows runs in Git Bash (MINGW64). All commands below use Git Bash syntax. If you are in PowerShell or cmd, substitute `C:\Users\radar\SolarPipe` for the working directory argument and use backslashes.

If `SolarPipe.sln` does not exist, run the bootstrap sequence. Run each command individually and verify it succeeds before continuing — a single silent failure will cascade:

```bash
# Verify SDK before anything else
dotnet --list-sdks   # Must show 8.0.x; if only 9.x, update global.json rollForward to "latestMajor"

# Create solution and projects (run from C:\Users\radar\SolarPipe)
dotnet new sln -n SolarPipe
dotnet new classlib -n SolarPipe.Core    -o src/SolarPipe.Core    -f net8.0
dotnet new classlib -n SolarPipe.Config  -o src/SolarPipe.Config  -f net8.0
dotnet new classlib -n SolarPipe.Data    -o src/SolarPipe.Data    -f net8.0
dotnet new classlib -n SolarPipe.Training    -o src/SolarPipe.Training    -f net8.0
dotnet new classlib -n SolarPipe.Prediction  -o src/SolarPipe.Prediction  -f net8.0
dotnet new console  -n SolarPipe.Host    -o src/SolarPipe.Host    -f net8.0
dotnet new xunit    -n SolarPipe.Tests.Unit        -o tests/SolarPipe.Tests.Unit        -f net8.0
dotnet new xunit    -n SolarPipe.Tests.Integration -o tests/SolarPipe.Tests.Integration -f net8.0
dotnet new xunit    -n SolarPipe.Tests.Pipeline    -o tests/SolarPipe.Tests.Pipeline    -f net8.0

# Add all projects to solution — verify each exits 0
dotnet sln SolarPipe.sln add src/SolarPipe.Core/SolarPipe.Core.csproj
dotnet sln SolarPipe.sln add src/SolarPipe.Config/SolarPipe.Config.csproj
dotnet sln SolarPipe.sln add src/SolarPipe.Data/SolarPipe.Data.csproj
dotnet sln SolarPipe.sln add src/SolarPipe.Training/SolarPipe.Training.csproj
dotnet sln SolarPipe.sln add src/SolarPipe.Prediction/SolarPipe.Prediction.csproj
dotnet sln SolarPipe.sln add src/SolarPipe.Host/SolarPipe.Host.csproj
dotnet sln SolarPipe.sln add tests/SolarPipe.Tests.Unit/SolarPipe.Tests.Unit.csproj
dotnet sln SolarPipe.sln add tests/SolarPipe.Tests.Integration/SolarPipe.Tests.Integration.csproj
dotnet sln SolarPipe.sln add tests/SolarPipe.Tests.Pipeline/SolarPipe.Tests.Pipeline.csproj

# Project references (strict unidirectional dependency order)
dotnet add src/SolarPipe.Config/SolarPipe.Config.csproj      reference src/SolarPipe.Core/SolarPipe.Core.csproj
dotnet add src/SolarPipe.Data/SolarPipe.Data.csproj          reference src/SolarPipe.Core/SolarPipe.Core.csproj
dotnet add src/SolarPipe.Training/SolarPipe.Training.csproj  reference src/SolarPipe.Core/SolarPipe.Core.csproj
dotnet add src/SolarPipe.Training/SolarPipe.Training.csproj  reference src/SolarPipe.Data/SolarPipe.Data.csproj
dotnet add src/SolarPipe.Prediction/SolarPipe.Prediction.csproj reference src/SolarPipe.Core/SolarPipe.Core.csproj
dotnet add src/SolarPipe.Prediction/SolarPipe.Prediction.csproj reference src/SolarPipe.Training/SolarPipe.Training.csproj
dotnet add src/SolarPipe.Host/SolarPipe.Host.csproj reference src/SolarPipe.Config/SolarPipe.Config.csproj
dotnet add src/SolarPipe.Host/SolarPipe.Host.csproj reference src/SolarPipe.Data/SolarPipe.Data.csproj
dotnet add src/SolarPipe.Host/SolarPipe.Host.csproj reference src/SolarPipe.Training/SolarPipe.Training.csproj
dotnet add src/SolarPipe.Host/SolarPipe.Host.csproj reference src/SolarPipe.Prediction/SolarPipe.Prediction.csproj
dotnet add tests/SolarPipe.Tests.Unit/SolarPipe.Tests.Unit.csproj               reference src/SolarPipe.Core/SolarPipe.Core.csproj
dotnet add tests/SolarPipe.Tests.Integration/SolarPipe.Tests.Integration.csproj reference src/SolarPipe.Data/SolarPipe.Data.csproj
dotnet add tests/SolarPipe.Tests.Pipeline/SolarPipe.Tests.Pipeline.csproj       reference src/SolarPipe.Host/SolarPipe.Host.csproj

# Write Directory.Packages.props — see IMPLEMENTATION_PLAN.md Task 1.1 for full contents
# Must exist before dotnet restore; do not skip this step

dotnet build   # Should succeed with zero warnings (TreatWarningsAsErrors is on)
```

After bootstrap, create `.gitignore` (see below), then commit: `git add -A && git commit -m "chore: bootstrap solution skeleton"`

**Required .gitignore** (create before first commit):
```
bin/
obj/
*.user
.vs/
*.db
logs/
models/registry/
python/.venv/
```

## Context Strategy (Read Per Phase)

Load only what you need. Each phase builds on the previous.

- **All phases**: `CLAUDE.md` (this file) + `DEVELOPMENT_RULES.md` sections 001–022, 040–041, 080–091, 110–115, 120, 130–133, 140–151
- **Phase 2+**: Add physics rules from `DEVELOPMENT_RULES.md`: RULE-030, 031, 032. Add composition section from `ARCHITECTURE.md`.
- **Phase 3+**: Add validation rules: RULE-050, 051, 052, 053. Add validation strategy from `ARCHITECTURE.md`.
- **Phase 4+**: Add sidecar rules: RULE-060, 061, 062, 063, 070. Add Python sidecar section from `ARCHITECTURE.md`.
- **When adding a model/framework/provider**: Read the relevant section from `ARCHITECTURE.md` and the corresponding ADR from `ARCHITECTURAL_DECISIONS.md`.
- **Never pre-load**: `IMPLEMENTATION_PLAN.md` (1600+ lines), `SolarPipe_Architecture_Plan.docx`. Reference only when planning a phase.

## Agent-Specific Rules (CLI-Only)

These rules exist because there is no IDE. Violating them causes expensive debugging loops.

1. **Build after every edit**: `dotnet build --no-restore` after any `.cs` file change. Catch breaks one at a time. (`dotnet format` runs only pre-commit, not after every edit — solution-wide formatting is slow.)
2. **No XML doc comments**: No `<summary>`, `<param>`, `<returns>` tags. Zero value without IntelliSense.
3. **No random physics test data**: Always use `PhysicsTestFixtures.cs`. Never `Random.NextDouble()` for speeds/densities/field values — domain violations cause NaN loops.
4. **Structured errors**: All exceptions must include stage name, input dimensions, and parameter values. Raw stack traces waste context.
5. **Git hygiene**: Run `git status` before any implementation step. Commit or stash uncommitted changes first.
6. **Small files**: No `.cs` file > ~400 lines. Use partial classes to split large implementations.
7. **Central package management**: Edit `Directory.Packages.props`, then `dotnet restore`. Never `dotnet add package`.
8. **Separate log streams**: C# → `logs/dotnet_latest.json`, Python → `logs/python_latest.json`. Shared Trace ID. Never interleave to stdout.
9. **Explicit Python path**: `${SOLARPIPE_ROOT}/python/.venv/bin/python` — never bare `python`.
10. **Translate exit codes**: `137 → "Out of Memory"`, `139 → "Segmentation fault — check ParquetSharp native handles"`.

## Critical Rules Quick Reference

Silent failures if violated — read full context in `DEVELOPMENT_RULES.md`:

| Rule | Summary |
|------|---------|
| RULE-001 | No `ReadOnlySpan<T>` in interfaces — use `float[]` |
| RULE-002 | Validate column lengths in `ToDataView()` |
| RULE-010 | Set `FeatureFraction=0.7` explicitly in ML.NET FastForest |
| RULE-020 | Register `Yaml12BooleanConverter` (YAML 1.1 Norway problem) |
| RULE-030 | Use Dormand-Prince RK4(5), not MathNet ODE solvers |
| RULE-031 | All physics equations use GSM-frame Bz, not GSE |
| RULE-040 | Atomic file write (temp + File.Move) for registry |
| RULE-051 | Temporal CV with gap buffers — never random k-fold |
| RULE-070 | Neural ODE ONNX: export dynamics network only |
| RULE-111 | No XML doc comments (no IntelliSense in CLI workflow) |
| RULE-120 | `Task.Factory.StartNew(..., LongRunning)` for ML.NET training |
| RULE-130 | Pipeline checkpointing: SHA-256 fingerprint per stage |

## Project Documentation Index

| Document | Purpose | When to Read |
|----------|---------|--------------|
| `CLAUDE.md` | Agent rules, commands, bootstrap (this file) | Always loaded |
| `ARCHITECTURE.md` | Interfaces, patterns, NuGet deps, design details | Per-phase as needed |
| `DEVELOPMENT_RULES.md` | 48 enforceable rules from risk audit | Per-phase sections |
| `ARCHITECTURAL_DECISIONS.md` | 16 ADRs documenting design choices | When implementing affected areas |
| `IMPLEMENTATION_PLAN.md` | Week-by-week coding tasks (19 weeks) | When planning a phase |
| `PROJECT_STATUS.md` | Timeline, task tracker, risk register | Status checks |
