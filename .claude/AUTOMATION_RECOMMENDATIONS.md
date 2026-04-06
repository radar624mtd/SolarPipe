# Claude Code Automation Recommendations

## Codebase Profile

- **Language**: C# (.NET 8)
- **Project Type**: ML Orchestration Framework (greenfield implementation)
- **Key Architecture**: Multi-module clean architecture with framework dispatch pattern
- **Status**: Architecture planned, implementation roadmap defined
- **Key Systems**:
  - YAML DSL parser and pipeline graph builder
  - Framework adapters (ML.NET, ONNX, Physics, Python gRPC sidecar)
  - Model composition algebra engine
  - Multi-provider data abstraction
  - Model registry with versioning

---

## 🔌 Recommended MCP Servers

### 1. **GitHub MCP Server**
**Why**: SolarPipe is a research framework that will need GitHub issue tracking for implementation phases, pull request reviews for architecture changes, and actions for CI/CD validation. The multi-stage implementation roadmap (4 phases, 16 weeks) benefits from automated issue creation and workflow tracking.

**Install**:
```bash
claude mcp add github
```

**Use Cases**:
- Track implementation roadmap phases as GitHub issues/milestones
- Automate PR reviews for new framework adapters and data providers
- Link commits to architecture decisions documented in CLAUDE.md

---

### 2. **Playwright MCP Server**
**Why**: The Python gRPC sidecar requires integration testing for training/prediction requests and response validation. End-to-end pipeline tests need browser automation for any eventual web UI (dashboard, model inspection interface) or API testing.

**Install**:
```bash
claude mcp add playwright
```

**Use Cases**:
- Automated testing of gRPC sidecar request/response serialization
- Integration test generation for IDataFrame implementations
- Future: dashboard UI testing when web interface is built

---

## 🎯 Recommended Skills

### 1. **dotnet-cli** (Custom Skill)
**Why**: SolarPipe requires frequent dotnet build, test, and run operations across 6+ projects. A skill can encapsulate common workflows: single-test runs, multi-project builds, CLI command testing.

**Invocation**: User-only (runs external commands)

**Create**: `.claude/skills/dotnet-cli/SKILL.md`

```markdown
---
name: dotnet-cli
description: Build, test, and run SolarPipe .NET projects
disable-model-invocation: true
tags: ["build", "test", "development"]
---

# dotnet-cli Skill

## Commands

### Build single project
\`\`\`bash
dotnet build src/SolarPipe.Core -c Release
\`\`\`

### Run specific test
\`\`\`bash
dotnet test SolarPipe.Tests.Unit --filter "ClassName=MyTest" --verbosity detailed
\`\`\`

### Run full test suite
\`\`\`bash
dotnet test
\`\`\`

### CLI execution
\`\`\`bash
dotnet run --project src/SolarPipe.Host -- validate --config configs/cme_propagation.yaml
\`\`\`
```

---

### 2. **pipeline-config-validator** (Custom Skill)
**Why**: The YAML DSL is critical to SolarPipe's value proposition. A skill can automate validation of pipeline configurations: schema validation, reference resolution (data source names, feature columns), framework/model compatibility checks, and composition expression parsing.

**Invocation**: Both (Claude automates validation in analysis, user triggers ad-hoc)

**Create**: `.claude/skills/pipeline-config-validator/SKILL.md`

```markdown
---
name: pipeline-config-validator
description: Validate SolarPipe YAML pipeline configurations
tags: ["validation", "dsl", "configuration"]
---

# Pipeline Config Validator

Validates YAML pipeline configurations against the SolarPipe schema:

- Data source references exist and have required columns
- Stage feature columns exist in data sources
- Framework + model combinations are supported
- Composition expressions parse and are type-compatible
- Hyperparameters are valid for selected model
- Target columns exist in data sources

**Usage**: `solarpipe validate --config <yaml-file>`

Or programmatically via PipelineConfigLoader for integration with Claude analysis.
```

---

## ⚡ Recommended Hooks

### 1. **Auto-format on Edit** (C# formatting)
**Why**: Multi-developer framework benefits from consistent code style. .NET projects typically use editorconfig or VS formatting rules. Automate formatting on file save.

**Where**: `.claude/settings.json`

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "tool": "Edit",
        "actions": ["run-formatter"]
      }
    ]
  },
  "automations": {
    "run-formatter": {
      "command": "dotnet format --verify-no-changes --verbosity diagnostic",
      "allowFailure": true,
      "description": "Run dotnet format on edited files"
    }
  }
}
```

---

### 2. **Block Sensitive File Edits**
**Why**: SQLite database (10GB solar_data.db), gRPC proto definitions, and production model registry paths should not be accidentally edited or committed.

**Where**: `.claude/settings.json`

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "tool": "Edit",
        "patterns": [
          "*.db",
          "*.proto",
          "models/registry/**"
        ],
        "action": "require-confirmation",
        "message": "This file is sensitive. Confirm before editing."
      }
    ]
  }
}
```

---

### 3. **Run Related Tests on Edit**
**Why**: Fast feedback loop for framework adapter and data provider implementations. When editing MlNetAdapter, automatically run related unit and integration tests.

**Where**: `.claude/settings.json`

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "tool": "Edit",
        "patterns": [
          "src/SolarPipe.Training/Adapters/*.cs",
          "src/SolarPipe.Data/Providers/*.cs"
        ],
        "action": "run-tests",
        "command": "dotnet test SolarPipe.Tests.Unit --filter \"*{editedClass}*\""
      }
    ]
  }
}
```

---

## 🤖 Recommended Subagents

### 1. **architecture-reviewer** (Specialized Subagent)
**Why**: SolarPipe's architecture is novel (framework plurality + composition algebra + physics integration). New framework adapters, data providers, and composition operators should be reviewed for consistency with the established patterns: IFrameworkAdapter contract, data flow through IDataFrame, composition semantics.

**Create**: `.claude/agents/architecture-reviewer.md`

```markdown
---
name: architecture-reviewer
description: Review new code for architectural consistency
type: specialized
---

# Architecture Reviewer Agent

Reviews implementations of:
- New IFrameworkAdapter implementations for contract compliance
- New IDataSourceProvider implementations for schema discovery and transform patterns
- Composition operators (ChainedModel, EnsembleModel, etc.) for uncertainty propagation
- Mock data integration strategies for residual calibration correctness

Checklist:
- Does the adapter implement required interface methods?
- Are CancellationTokens threaded through all async operations?
- Is the framework-specific serialization/deserialization correct?
- Does the adapter handle missing data appropriately?
- Are metrics (RMSE, MAE, R², feature importance) collected correctly?
- Is the adapter tested with InMemoryDataFrame mock data?
```

**Usage**: Run before merging framework adapter PRs.

---

### 2. **physics-validator** (Specialized Subagent)
**Why**: PhysicsAdapter wraps analytical equations (drag-based kinematics, Burton ODE, Newell coupling). These are critical for correctness: wrong implementations lead to incorrect physics baselines and failed residual calibration.

**Create**: `.claude/agents/physics-validator.md`

```markdown
---
name: physics-validator
description: Validate physics equation implementations
type: specialized
---

# Physics Validator Agent

Validates implementations in SolarPipe.Training/Physics/:

- Drag-based model: dv/dt = -γ(v-w)|v-w| with correct drag coefficient γ
- Burton ODE: dDst*/dt = Q(t) - Dst*/τ(VBs) with O'Brien-McPherron decay timescale
- Newell coupling: dΦ/dt ∝ v^4/3 B_T^2/3 sin^8/3(θ/2) for magnetopause reconnection
- Numerical integration: ODE solvers preserve stability and accuracy
- Input/output ranges: Physics outputs are within domain of ML models (no NaN/Inf)

Cross-reference: SolarPipe_Architecture_Plan.docx Appendix 15 for equations.
```

**Usage**: Run when adding or modifying physics equations.

---

## 📦 Recommended Plugins

### **anthropic-agent-skills** Plugin
**Why**: General-purpose productivity. Includes `commit`, `code-review`, and development utilities useful for a framework with a 4-phase implementation roadmap.

**Install**: `claude plugin install anthropic-agent-skills`

---

## 🚀 Advanced Recommendations (If Needed Later)

Ask for more recommendations for any of these categories:

### Additional MCP Servers
- **Linear MCP** — If using Linear for issue/roadmap tracking instead of GitHub
- **Database MCP** — If needing direct queries on solar_data.db during testing/validation
- **Slack MCP** — For team notifications on training completions or long-running pipeline executions
- **Memory MCP** — For cross-session memory of framework design decisions and validation results

### Additional Skills
- **proto-generator** — Automate .proto file generation and grpc-dotnet binding updates
- **config-generator** — Create new pipeline YAML configurations from templates
- **migration-generator** — When transitioning model registry from file-system to MLflow/database

### Additional Hooks
- **Validate gRPC proto changes** — Ensure Python sidecar proto definitions stay in sync
- **Block uncommitted changes before training** — Prevent training models with dirty git state
- **Archive old model artifacts** — Periodically compress model registry versions older than N months

### Additional Subagents
- **performance-analyzer** — Profile Random Forest training and prediction latency; identify bottlenecks in data loading or composition
- **test-writer** — Generate unit tests for new framework adapters and data providers
- **security-reviewer** — Validate gRPC sidecar communication (TLS, authentication), database query parameterization

---

## Implementation Priority

1. **Immediate** (High value, low effort):
   - GitHub MCP Server (roadmap tracking)
   - dotnet-cli skill (workflow automation)
   - Auto-format hook (consistency)

2. **Short-term** (High value, moderate effort):
   - pipeline-config-validator skill (DSL correctness)
   - architecture-reviewer subagent (design consistency)
   - physics-validator subagent (correctness)

3. **Medium-term** (Moderate value, higher effort):
   - Playwright MCP Server (integration testing)
   - Block sensitive files hook (data protection)
   - Run-related-tests hook (fast feedback)

---

## Next Steps

- **Want to set up any of these?** Ask Claude to help implement (e.g., "Set up the GitHub MCP server integration" or "Create the dotnet-cli skill")
- **Want more MCP options?** Ask for additional server recommendations (e.g., "What other MCP servers would help with the Python sidecar?")
- **Want custom skills?** Ask Claude to build domain-specific skills (e.g., "Create a skill for generating ENLIL mock data configurations")
