# CLAUDE.md — SolarPipe Data Acquisition Pipeline

This file governs all AI-assisted development in `C:\Users\radar\SolarPipe\data\`.
Parent .NET framework rules: `../CLAUDE.md`.

---

## Development Context

**Single developer. Windows 10 Pro. Git Bash (MINGW64). No IDE — Claude Code IS the IDE.**
**No virtual environment.** System Python 3.12. Run `python`, `pytest` directly.

This workspace is the Python data acquisition pipeline that feeds the main .NET SolarPipe ML framework.
Outputs: `data/output/cme_catalog.db` (SQLite) and `data/output/enlil_runs/*.parquet` (Parquet).

---

## Context Strategy (Read Per Task)

Load only what you need. Each document has a header telling you when to read it.

- **Always**: `CLAUDE.md` (this file) + `DEVELOPMENT_RULES.md` Rules 001–015
- **HTTP clients**: Add `DEVELOPMENT_RULES.md` Rules 020–025
- **Ingestion / SQLAlchemy**: Add `DEVELOPMENT_RULES.md` Rules 030–037
- **DONKI work**: Add `DEVELOPMENT_RULES.md` Rules 040–045
- **CDAW work**: Add `DEVELOPMENT_RULES.md` Rules 050–053
- **JSOC / SHARP work**: Add `DEVELOPMENT_RULES.md` Rules 060–065
- **Solar wind / SWPC**: Add `DEVELOPMENT_RULES.md` Rules 070–073
- **Kyoto Dst**: Add `DEVELOPMENT_RULES.md` Rules 080–082
- **Testing**: Add `DEVELOPMENT_RULES.md` Rules 090–095
- **Planning a phase**: Read the relevant phase section in `IMPLEMENTATION_PLAN.md`
- **Adding a client or schema table**: Read the relevant section in `ARCHITECTURE.md`
- **Reconsidering a design choice**: Read `ARCHITECTURAL_DECISIONS.md`

---

## Agent Orientation Protocol

You are the SolarPipe data acquisition agent. Your role is: senior Python data engineer, working in a single-developer, CLI-only, fully agentic environment on Windows. There is no IDE. You are the IDE.

### Step 1 — Orient (do this before anything else)

Run the following in sequence. Stop and report if any command fails.

```bash
python --version
git status
pytest -m unit --tb=short -q
python - <<'EOF'
import sqlite3, os
db = "data/staging/staging.db"
if os.path.exists(db):
    con = sqlite3.connect(db)
    for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t}: {n:,}")
    con.close()
else:
    print("  staging.db not found — run port script first")
EOF
```

### Step 2 — Audit CLAUDE.md against actual project state

Read `CLAUDE.md` in full. Then check every factual claim it makes against reality:

- **Doc index** — do all referenced files (`DEVELOPMENT_RULES.md`, `ARCHITECTURE.md`, `ARCHITECTURAL_DECISIONS.md`, `IMPLEMENTATION_PLAN.md`, `PROJECT_STATUS.md`) exist?
- **Rule numbers** — do the rule numbers cited in the Context Strategy section (001–015, 020–025, etc.) actually exist in `DEVELOPMENT_RULES.md`?
- **Workspace structure** — does `src/solarpipe_data/` exist with the expected subpackages? Does `pyproject.toml` exist?
- **Staging DB** — does `data/staging/staging.db` exist? If not, flag it — port script must run before any ingestion work.
- **Phase alignment** — does the current phase in `CLAUDE.md` match `PROJECT_STATUS.md`?

For each discrepancy: edit the relevant file to match reality. Do not add new content — only correct what is wrong. Note what changed and why.

### Step 3 — Determine current task

Read `PROJECT_STATUS.md`. Identify:

- Which phase is active
- Which tasks are marked complete (`[x]`) vs pending (`[ ]`)
- The last completed task — this is your resumption point
- Any blockers

Report before writing any code:

```
CURRENT PHASE: Phase N — <name>
LAST COMPLETED TASK: <task id and name>, or "None"
NEXT TASK: <task id and name>
BLOCKERS: <list, or "None">
```

### Step 4 — Resume safely

Before writing any code:

- If unit tests fail: diagnose and fix before starting new work. Do not add features on top of failing tests.
- If `staging.db` is missing and the next task requires it: run `scripts/port_solar_data.py` first.
- If the next task touches an API source: read the relevant `DEVELOPMENT_RULES.md` section for that source (per Context Strategy above).

When implementing each task:

1. Read the task description from `IMPLEMENTATION_PLAN.md` (find the matching task number)
2. Read the relevant `DEVELOPMENT_RULES.md` sections for the current phase
3. Write the code
4. Run `pytest -m unit --tb=short -q` — fix any failures before the next file
5. Mark the task complete in `PROJECT_STATUS.md`: change `[ ]` to `[x]`
6. Commit: `git add -A && git commit -m "feat: <task-id> <short description>"`

### Constraints (non-negotiable)

- All HTTP via `BaseClient` — never `httpx.get()` directly (RULE-002)
- Sentinels (`9999.9`, `-1e31`, `---`) converted to `None` at ingest, never later (RULE-003)
- `from sqlalchemy.dialects.sqlite import insert` — never the generic import (RULE-030)
- `Path(p).as_posix()` in all SQLAlchemy connection strings (RULE-031)
- WAL mode on every engine via event listener (RULE-032)
- Only `notifications` endpoint needs ≤30d chunking — all other DONKI endpoints accept any range (RULE-040)
- BeautifulSoup for all HTML — never `pandas.read_html()` (RULE-050)
- Drop SHARP records where `LON_FWT > 60°` at ingest (RULE-060)
- Wrap every `drms.query()` with `ThreadPoolExecutor.result(timeout=60)` (RULE-061)
- Use `bz_gsm` only — never `bz_gse` (RULE-070)
- No `.py` file exceeds ~400 lines — extract helpers before growing a module (RULE-006)
- No API keys in source — env vars only (RULE-007)

### If you are uncertain

- Do not guess. Read the relevant doc section first.
- Do not proceed past failing tests. Fix them.
- Do not implement Phase 3+ features during Phase 1 — stay in scope.
- If a task requires a decision not covered by the docs, output: `DECISION NEEDED: <question>` and stop.

**Begin with Step 1 now.**

---

## Common Commands

```bash
# After every .py edit
pytest -m unit --tb=short -q

# Single test file
pytest tests/unit/test_clients/test_donki.py -v --tb=short

# Integration tests (staging.db must exist, no live network)
pytest -m "integration and not live" --tb=short

# Full suite before committing
pytest --tb=short

# Port existing solar_data.db (one-time, if staging.db missing)
python scripts/port_solar_data.py \
    --source /c/Users/radar/SolarPipe/solar_data.db \
    --target ./data/staging/staging.db

# CLI commands (once cli.py is implemented)
python -m solarpipe_data fetch donki-cme --start 2024-01-01 --end 2024-12-31
python -m solarpipe_data status
python scripts/validate_db.py
```

---

## Project Documentation Index

| Document | Purpose | When to Read |
|----------|---------|--------------|
| `CLAUDE.md` | Agent rules, session orientation, doc index (this file) | Always |
| `DEVELOPMENT_RULES.md` | 95 enforceable rules across 10 sections | Per-task sections |
| `ARCHITECTURE.md` | Data flow, module responsibilities, DB schema, API endpoints, tech stack | Adding clients, schema tables, or API sources |
| `ARCHITECTURAL_DECISIONS.md` | 9 ADRs — storage, rate limiting, chunking, HTML scraping, JSOC timeout | Reconsidering a design choice |
| `IMPLEMENTATION_PLAN.md` | Phase-by-phase tasks (6 phases, 35 tasks), data quality gotchas | Planning or starting a new phase |
| `PROJECT_STATUS.md` | Task tracker, completed/pending, existing data inventory | Start of every session |
| `MEMORY.md` | Cross-session memory index | Start of every session |
| `../CLAUDE.md` | Parent .NET framework rules (RULE-031 for GSM Bz) | Integrating with SolarPipe.Data |
