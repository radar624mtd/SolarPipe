# SolarPipe Claude Code Configuration

This directory contains Claude Code automation configuration for the SolarPipe .NET 8 ML orchestration framework.

## 📋 What's Here

### Configuration & Setup
- **`settings.json`** — Hook configurations, permissions, and automation registrations
- **`settings.local.json`** — Local development permissions (git, Python, dotnet)

### Documentation
- **`AUTOMATION_RECOMMENDATIONS.md`** — Analysis of codebase and recommendations for automations
- **`AUTOMATION_SETUP_SUMMARY.md`** — Summary of all implemented automations (this is the implementation receipt)

### Skills (User-Invocable Workflows)
Skills are packaged expertise and repeatable tasks invoked via `/skill-name` syntax.

- **`skills/dotnet-cli/`** — Common dotnet build, test, and CLI workflows
  - Build entire solution or specific projects
  - Run unit, integration, and end-to-end tests
  - Validate pipeline configs and inspect trained models
  - Format code and restore dependencies

- **`skills/pipeline-config-validator/`** — YAML DSL configuration validation
  - Schema compliance, data source references, column existence
  - Framework/model compatibility, composition expression parsing
  - Framework-specific hyperparameter validation
  - Validation strategy and mock data configuration

### Agents (Specialized Subagents)
Agents run in fork context (independent process) and specialize in code review tasks.

- **`agents/architecture-reviewer.md`** — Architectural consistency review
  - Reviews new IFrameworkAdapter implementations
  - Validates new IDataSourceProvider implementations
  - Checks composition operators (IComposedModel)
  - Enforces cross-module consistency patterns
  - Detailed checklists and reference materials included

- **`agents/physics-validator.md`** — Physics equation implementation validation
  - Validates drag-based CME propagation model
  - Validates Burton ring-current ODE
  - Validates Newell coupling function
  - Checks numerical stability, input/output ranges
  - Ensures composition integration correctness
  - Reference equations and test cases included

## 🚀 Quick Start

### Run a Skill

```bash
/dotnet-cli build-all
/dotnet-cli test-unit
/dotnet-cli validate configs/cme_propagation.yaml
/pipeline-config-validator --config configs/cme_propagation.yaml
```

### Trigger an Agent

When submitting new adapter code for review:

```
Please review this MlNetAdapter implementation for architectural consistency.
@architecture-reviewer
```

When adding or modifying physics equations:

```
Validate this drag-based CME model implementation.
@physics-validator
```

### Hooks (Automatic)

After editing a C# file:
- ✅ `dotnet format` runs automatically

Before editing sensitive files (`.db`, `.proto`, `models/registry/**`):
- ⚠️ Confirmation required

Before pushing to git:
- ⚠️ Confirmation required

## 📁 Directory Structure

```
.claude/
├── settings.json                    # Main configuration (hooks, permissions, agents/skills)
├── settings.local.json              # Local-only permissions
├── README.md                        # This file
├── AUTOMATION_RECOMMENDATIONS.md    # Rationale and analysis
├── AUTOMATION_SETUP_SUMMARY.md      # Implementation summary
├── skills/
│   ├── dotnet-cli/
│   │   └── SKILL.md                # dotnet workflow skill
│   └── pipeline-config-validator/
│       └── SKILL.md                # YAML validation skill
└── agents/
    ├── architecture-reviewer.md     # Architecture review agent
    └── physics-validator.md         # Physics equation validation agent
```

## 🔍 Configuration Details

### Permissions

**Allowed Operations**:
- Build, test, run dotnet commands
- Edit C#, YAML config files
- Read, glob, grep operations
- Git operations (with push confirmation)

**Denied Operations**:
- Destructive deletes (`rm -rf`)
- Force push to git
- Edit database files without confirmation
- Edit proto definitions without confirmation
- Edit model registry without confirmation

### Hooks

| Trigger | Action | Purpose |
|---------|--------|---------|
| PostToolUse (Edit *.cs) | run-formatter | Auto-format code |
| PreToolUse (Edit *.db, *.proto) | require-confirmation | Protect critical files |
| PreToolUse (Bash: git push) | require-confirmation | Prevent accidental pushes |

## 📚 Related Files

- **`../CLAUDE.md`** — Architecture overview, module responsibilities, common commands, reference use case
- **`../SolarPipe_Architecture_Plan.docx`** — Detailed system design with equations and implementation phases
- **`../AUTOMATION_RECOMMENDATIONS.md`** — This directory's AUTOMATION_RECOMMENDATIONS.md; includes future recommendations (GitHub MCP, Playwright MCP, additional skills)

## 🔄 Integration with Main Project Files

These configurations support the SolarPipe development workflow:

- **CLAUDE.md** describes architecture; agents/skills reference it
- **Architecture Plan** defines physics equations; physics-validator references it
- **Settings** govern which tools Claude can use during development
- **Skills** automate common workflows mentioned in CLAUDE.md (validate, train, predict, inspect)

## 🎯 What Works Now

✅ **Skills**: dotnet-cli, pipeline-config-validator (ready to use)
✅ **Agents**: architecture-reviewer, physics-validator (ready for code review)
✅ **Hooks**: Auto-formatting, file protection, push confirmation (active)
✅ **Permissions**: Balanced for safety + productivity (configured)

## ⏳ What's Next (Optional)

**Recommended Future Additions**:
1. **GitHub MCP Server** — Track implementation roadmap, manage PRs
2. **Playwright MCP Server** — Test gRPC sidecar and future UI
3. **Additional Skills**: proto-generator, config-generator, migration-generator
4. **More Hooks**: Run related tests on adapter edits, archive old models

See `AUTOMATION_RECOMMENDATIONS.md` for details and setup instructions.

## 📖 References

- Microsoft.Extensions.Logging documentation
- ML.NET documentation (for ML.NET adapter review)
- Vršnak et al. (2013) — CME drag propagation
- Burton et al. (1975) — Ring current ODE
- O'Brien & McPherron (2000) — Burton decay timescale
- Newell et al. (2007) — Magnetopause coupling function

## ❓ Troubleshooting

**Skill not found**: Ensure `.claude/skills/[name]/SKILL.md` exists with `name` in frontmatter

**Hook not running**: Check `settings.json` syntax and tool/glob patterns

**Agent fails**: Ensure agent has `context: fork` in frontmatter

**Permission denied**: Check `settings.json` allow list includes the command

For detailed help, see `AUTOMATION_SETUP_SUMMARY.md` troubleshooting section.
