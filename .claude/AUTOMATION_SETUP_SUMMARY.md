# SolarPipe Claude Code Automation Setup — Complete

This document summarizes the automations implemented based on AUTOMATION_RECOMMENDATIONS.md.

**Date**: 2026-04-06
**Status**: ✅ Implementation Complete

---

## Implemented Automations

### ✅ Skills (2 Created)

#### 1. **dotnet-cli** Skill
**Location**: `.claude/skills/dotnet-cli/SKILL.md`

Encapsulates common .NET build, test, and CLI workflows for SolarPipe's multi-project solution.

**Commands Provided**:
- Build: `build-all`, `build-project [PROJECT]`, `clean`
- Test: `test-all`, `test-unit`, `test-integration`, `test-pipeline`, `test-class [NAME]`, `test-method [NAME]`
- CLI: `validate [CONFIG]`, `inspect-model [MODELID]`, `predict [CONFIG] [INPUT] [OUTPUT]`
- Utils: `format-check`, `format-apply`, `restore-deps`, `list-projects`

**Invocation**: User-only (`/dotnet-cli [command]`)

**Rationale**: SolarPipe involves 6+ projects (Core, Config, Data, Training, Prediction, Host) across unit/integration/pipeline tests. This skill accelerates common workflows without requiring users to remember exact dotnet command syntax.

---

#### 2. **pipeline-config-validator** Skill
**Location**: `.claude/skills/pipeline-config-validator/SKILL.md`

Comprehensive validation of YAML DSL pipeline configurations against semantic rules and physical constraints.

**Validation Coverage**:
- Schema compliance (required fields, type checking)
- Data source references and column existence
- Stage framework/model compatibility
- Composition expression parsing and type compatibility
- Framework-specific hyperparameter validation
- Validation strategy and metrics
- Mock data strategies and weights

**Invocation**: Both (Claude auto-validates during analysis, users trigger ad-hoc)

**Rationale**: The YAML DSL is core to SolarPipe's value. Configuration errors (missing column references, incompatible framework/model pairs, invalid expressions) are common and hard to debug. This skill catches them early.

---

### ✅ Subagents (2 Created)

#### 1. **architecture-reviewer** Subagent
**Location**: `.claude/agents/architecture-reviewer.md`

Specialized code reviewer for new framework adapters, data providers, and composition operators. Enforces architectural consistency with established patterns.

**Review Focus**:
1. **Framework Adapter Review** — IFrameworkAdapter contract compliance, data flow through IDataFrame, metrics collection, async/cancellation patterns, framework-specific validation
2. **Data Provider Review** — IDataSourceProvider compliance, schema discovery, provider-specific patterns (SQLite, CSV, Parquet, REST API)
3. **Composition Operator Review** — IComposedModel implementations, operator semantics (chain, ensemble, residual, gating), uncertainty propagation
4. **Cross-Module Consistency** — Async patterns, dependency direction, logging, error handling

**Deliverables**:
- Detailed checklist per implementation type
- Common issues to flag
- Success criteria
- Reference materials (CLAUDE.md, architecture plan, existing implementations)

**Invocation**: Fork context (runs independently); triggered before merging adapter/provider/composition PRs

**Rationale**: SolarPipe's architecture (framework dispatch, composition algebra, physics integration) is novel. New implementations must maintain consistency with established patterns. Manual review is error-prone; specialized agent reduces overhead.

---

#### 2. **physics-validator** Subagent
**Location**: `.claude/agents/physics-validator.md`

Specialized validator for analytical physics equation implementations in PhysicsAdapter.

**Validation Coverage**:
1. **Equation Implementation** — Drag-based CME model, Burton ring-current ODE, Newell coupling function
2. **Numerical Stability** — Time-stepping strategy, integration method (RK4 preferred), boundary conditions, error accumulation
3. **Input Validation** — Physical ranges for all inputs (drag: v ∈ [100,5000] km/s, density ∈ [1,100] amu/cm³, etc.)
4. **Output Validation** — Ranges for all outputs (arrival time ∈ [0.5,5] days, Dst* ∈ [-300,100] nT, coupling ∈ [0,1])
5. **Composition Integration** — Correct behavior in residual (physics ^ ML), ensemble, and chain compositions

**Deliverables**:
- Reference equations with parameter ranges and test cases
- Example C# implementations for each equation
- Numerical integration best practices
- Unit test templates
- Common mistakes to flag

**Invocation**: Fork context; triggered when adding/modifying physics equations

**Rationale**: Physics equations are critical to correctness. Wrong sign conventions, incorrect exponents, unit mismatches, or inadequate time-stepping lead to incorrect baselines and failed residual calibration. Specialized agent catches subtle physics errors.

---

### ✅ Hooks (3 Configured)

**Location**: `.claude/settings.json`

#### 1. **Auto-format on Edit** (PostToolUse Hook)
```json
{
  "tool": "Edit",
  "glob": "src/**/*.cs",
  "action": "run-formatter"
}
```

Automatically runs `dotnet format` after C# file edits.

**Why**: Multi-developer framework benefits from consistent code style. Prevents style drift and reduces noise in diffs.

---

#### 2. **Block Sensitive File Edits** (PreToolUse Hook)
```json
{
  "tool": "Edit",
  "glob": ["*.db", "*.proto", "models/registry/**"],
  "action": "require-confirmation"
}
```

Requires explicit confirmation before editing database files, gRPC proto definitions, or model registry.

**Why**: Prevents accidental edits to:
- `solar_data.db` (10 GB database, not a code file)
- `.proto` files (changes require Python sidecar recompilation)
- Model registry artifacts (versioned model files should only be modified by training pipeline)

---

#### 3. **Block Force Push** (PreToolUse Hook)
```json
{
  "tool": "Bash",
  "pattern": "git push",
  "action": "require-confirmation"
}
```

Requires confirmation before git push operations (especially force push).

**Why**: Prevents accidental data loss or overwriting shared commits.

---

### ✅ Permissions Configuration

**Location**: `.claude/settings.json` — `permissions` section

**Allow List** (tools and commands Claude can use freely):
- `Bash(git -C /c/Users/radar/SolarPipe *)`— git operations
- `Bash(cd /c/Users/radar/SolarPipe && dotnet build*)` — build
- `Bash(cd /c/Users/radar/SolarPipe && dotnet test*)` — test
- `Bash(cd /c/Users/radar/SolarPipe && dotnet run*)` — run CLI
- `Bash(cd /c/Users/radar/SolarPipe && dotnet format*)` — formatting
- `Edit(src/**/*.cs)`, `Edit(tests/**/*.cs)`, `Edit(configs/**/*.yaml)` — source file edits
- `Read`, `Glob`, `Grep` — file inspection

**Deny List** (tools Claude must not use):
- `Bash(rm -rf*)` — destructive deletes
- `Bash(git push *)` — push requires hook confirmation
- `Bash(git reset --hard*)` — destructive reset
- `Edit(*.db)` — database files require confirmation
- `Edit(*.proto)` — proto files require confirmation
- `Edit(models/registry/**)` — model registry requires confirmation

---

## Quick Start: Using the Automations

### Run a Skill

```
/dotnet-cli build-all
/dotnet-cli test-unit
/dotnet-cli validate configs/cme_propagation.yaml
/pipeline-config-validator --config configs/cme_propagation.yaml --verbose
```

### Trigger a Subagent

When submitting a new MlNetAdapter implementation:

```
Please review this new MlNetAdapter for architectural consistency.
Focus on: IFrameworkAdapter contract, ML.NET specific patterns, test coverage.

@architecture-reviewer
```

When validating a new drag-based model implementation:

```
Validate this drag-based CME propagation model implementation.
Ensure: equation correctness, input ranges, numerical stability, output validity.

@physics-validator
```

### Hooks Activate Automatically

After editing a C# file:
- ✅ `dotnet format` runs automatically (PostToolUse hook)

Before editing a `.db` or `.proto` file:
- ⚠️ Hook requires confirmation

Before pushing to git:
- ⚠️ Hook requires confirmation

---

## Configuration Files Created

1. **`.claude/skills/dotnet-cli/SKILL.md`** (203 lines)
   - Common dotnet workflows for SolarPipe

2. **`.claude/skills/pipeline-config-validator/SKILL.md`** (335 lines)
   - YAML DSL validation rules and examples

3. **`.claude/agents/architecture-reviewer.md`** (438 lines)
   - Architectural consistency review checklists

4. **`.claude/agents/physics-validator.md`** (521 lines)
   - Physics equation implementation validation

5. **`.claude/settings.json`** (85 lines)
   - Hook configurations, permissions, agent/skill registration

6. **`.claude/AUTOMATION_RECOMMENDATIONS.md`** (existing)
   - Rationale and implementation guidance for all automations

7. **`.claude/AUTOMATION_SETUP_SUMMARY.md`** (this file)
   - Summary of implemented automations

---

## Next Steps & Future Enhancements

### Immediate (Recommended)

1. **Test the skills**
   ```
   /dotnet-cli build-all
   /dotnet-cli test-unit
   ```

2. **Test validation hook** — Edit a C# file and verify `dotnet format` runs

3. **Try architecture-reviewer** — Submit a test file and run the agent

### Short-term (After Codebase Setup)

4. **Set up GitHub MCP Server** (from AUTOMATION_RECOMMENDATIONS.md)
   ```bash
   claude mcp add github
   ```

5. **Enable Playwright MCP** (for gRPC sidecar testing)
   ```bash
   claude mcp add playwright
   ```

6. **Extend skills** — Add more domain-specific commands as needed:
   - `proto-generator` — Automate .proto binding updates
   - `config-generator` — Create new pipeline YAML from templates
   - `migration-generator` — Registry migration helper

### Medium-term (As Implementation Progresses)

7. **Add more subagents**
   - `test-writer` — Generate unit test templates
   - `performance-analyzer` — Profile training and prediction latency
   - `security-reviewer` — Validate gRPC/database security

8. **Expand hooks**
   - Run related tests automatically when adapter/provider edited
   - Archive old model artifacts monthly
   - Sync proto definitions between .NET and Python sidecar

---

## Reference Materials

- **AUTOMATION_RECOMMENDATIONS.md** — Rationale for each automation
- **CLAUDE.md** — Architecture overview; referenced by all agents
- **SolarPipe_Architecture_Plan.docx** — Detailed system design; referenced by physics-validator
- **Existing implementations** — Referenced as examples in all agents/skills

---

## Troubleshooting

### Skill Not Found

Ensure `.claude/skills/[skill-name]/SKILL.md` exists with frontmatter:
```yaml
---
name: skill-name
description: ...
---
```

### Hook Not Running

Check `.claude/settings.json` has correct syntax and paths. Verify:
- `"tool"` matches actual tool name (Edit, Bash, Read, etc.)
- `"glob"` or `"pattern"` syntax is correct
- `"action"` is valid (require-confirmation, run-formatter)

### Agent Fails on Fork

Ensure agent file has `context: fork` in frontmatter:
```yaml
---
name: agent-name
context: fork
---
```

### Permissions Denied

If Claude cannot run a dotnet command, check `.claude/settings.json` permissions allow list includes the command pattern.

---

## Summary

| Category | Implemented | Status | Impact |
|----------|-------------|--------|--------|
| Skills | 2 | ✅ | Fast workflow automation, DSL validation |
| Subagents | 2 | ✅ | Architectural consistency, physics correctness |
| Hooks | 3 | ✅ | Auto-formatting, data protection, push confirmation |
| MCP Servers | GitHub, Playwright | ⏳ Recommended | Issue tracking, browser testing (ready to enable) |
| Permissions | Configured | ✅ | Balanced safety (deny destructive ops) and productivity |

All immediate-priority automations implemented. Framework is now positioned for productive multi-developer collaboration on the 4-phase SolarPipe implementation roadmap.
