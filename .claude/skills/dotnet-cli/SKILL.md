---
name: dotnet-cli
description: Build, test, and run SolarPipe .NET projects with common workflows
disable-model-invocation: true
tags: ["build", "test", "development", ".net"]
---

# dotnet-cli Skill

Common dotnet workflows for SolarPipe development. Run via `/dotnet-cli [command]`.

## Available Commands

### Build Operations

**build-all** — Build entire solution
```bash
dotnet build -c Release
```

**build-project [PROJECT]** — Build specific project (e.g., Core, Data, Training)
```bash
dotnet build src/SolarPipe.{PROJECT} -c Release
```

**clean** — Clean build artifacts
```bash
dotnet clean
dotnet build -c Release
```

### Testing

**test-all** — Run full test suite
```bash
dotnet test --configuration Release --logger "console;verbosity=detailed"
```

**test-unit** — Run unit tests only
```bash
dotnet test SolarPipe.Tests.Unit --configuration Release
```

**test-integration** — Run integration tests
```bash
dotnet test SolarPipe.Tests.Integration --configuration Release
```

**test-pipeline** — Run end-to-end pipeline tests
```bash
dotnet test SolarPipe.Tests.Pipeline --configuration Release
```

**test-class [CLASSNAME]** — Run tests for specific class
```bash
dotnet test --filter "ClassName={CLASSNAME}" --verbosity detailed
```

**test-method [METHODNAME]** — Run specific test method
```bash
dotnet test --filter "MethodName={METHODNAME}" --verbosity detailed
```

### CLI Operations

**validate [CONFIG]** — Validate pipeline configuration
```bash
dotnet run --project src/SolarPipe.Host -- validate --config {CONFIG}
```

**inspect-model [MODELID]** — Inspect trained model
```bash
dotnet run --project src/SolarPipe.Host -- inspect --model {MODELID}
```

**predict [CONFIG] [INPUT] [OUTPUT]** — Run prediction
```bash
dotnet run --project src/SolarPipe.Host -- predict \
  --config {CONFIG} --input {INPUT} --output {OUTPUT}
```

### Development Utilities

**format-check** — Check code formatting without applying
```bash
dotnet format --verify-no-changes --verbosity diagnostic
```

**format-apply** — Apply dotnet format to all files
```bash
dotnet format --verbosity diagnostic
```

**restore-deps** — Restore NuGet packages
```bash
dotnet restore
```

**list-projects** — List all projects in solution
```bash
dotnet sln list
```

## Examples

```
/dotnet-cli build-all
/dotnet-cli test-unit
/dotnet-cli test-method "TrainAsync should complete successfully"
/dotnet-cli validate configs/cme_propagation.yaml
/dotnet-cli format-check
```

## Configuration

All commands run from the SolarPipe root directory. Ensure:
- .NET 8 SDK is installed
- SolarPipe.sln exists in root
- Test projects have proper references to SolarPipe.Tests.* assemblies
