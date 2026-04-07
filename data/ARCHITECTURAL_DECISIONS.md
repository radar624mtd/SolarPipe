# ARCHITECTURAL_DECISIONS.md — SolarPipe Data Acquisition Pipeline

Architecture Decision Records for the Python data acquisition workspace.
Read when implementing the affected area or reconsidering the choice.

---

## ADR-D001: SQLite for Staging and Output Databases

**Status**: Accepted
**Date**: 2026-04-07

**Context**:
The C# SolarPipe ML framework has two data providers: `SqliteProvider` (reads `.db` files) and `ParquetProvider` (reads `.parquet` files). The output of this pipeline must be directly consumable by those providers without an intermediate conversion step.

**Decision**:
Use SQLite (via SQLAlchemy ORM) for both `data/staging/staging.db` and `data/output/cme_catalog.db`. Column names in `cme_catalog.db` are hardcoded in `configs/flux_rope_propagation_v1.yaml` — do not rename them.

**Consequences**:
- Do NOT switch staging to DuckDB, Parquet, or any other format.
- Do NOT use DuckDB for staging even though it supports vectorized operations — the C# side has no DuckDB reader.
- Schema changes require a `database/migrations.py` entry.

**When to re-read**: When considering a storage format change, or when adding new tables.

---

## ADR-D002: Parquet (PyArrow) for Synthetic ENLIL Ensemble Only

**Status**: Accepted
**Date**: 2026-04-07

**Context**:
The synthetic ENLIL ensemble output (`data/output/enlil_runs/*.parquet`) is consumed by the C# `ParquetProvider`, which wraps ParquetSharp (G-Research, Apache Parquet C++ via P/Invoke). ParquetSharp is 4–10x faster than Parquet.Net and supports random-access row group reading. It is already integrated on the C# side.

**Decision**:
Write the ENLIL ensemble exclusively as Parquet via PyArrow. Use ≤64 MB row groups (matching ParquetSharp's optimal read granularity). Include metadata: generation timestamp, parameter distributions, random seed.

**Consequences**:
- Parquet is ONLY for the synthetic ensemble output — not for staging, not for `cme_catalog.db`.
- PyArrow is the Python writer; ParquetSharp is the C# reader.
- All other pipeline data uses SQLite.

---

## ADR-D003: Token-Bucket Rate Limiting in BaseClient

**Status**: Accepted

**Context**:
NASA DONKI uses a rolling-window rate limit (not top-of-hour reset): registered key = 1,000 req/hr; DEMO_KEY = 30 req/hr, 50 req/day **per IP address** (not per key). Kyoto WDC has a fragile server. CDAW has no documented limit but is a single-server resource.

**Decision**:
All HTTP goes through `BaseClient` which implements a token-bucket rate limiter. Rate limits are configured per-source in `configs/default.yaml`. `BaseClient` exposes `X-RateLimit-Remaining` header to adjust dynamically for DONKI.

**Consequences**:
- Never call `httpx.get()` or instantiate `httpx.AsyncClient()` directly in any client module.
- Never bypass `BaseClient` for "just a quick test fetch" — rate limit state will be out of sync.

---

## ADR-D004: DONKI Chunking Applies Only to `notifications`

**Status**: Accepted — corrects prior documentation error
**Date**: 2026-04-07 (research confirmed)

**Context**:
Prior documentation incorrectly stated all DONKI endpoints require ≤30-day request windows. Live API research confirmed only the `/notifications` endpoint has this restriction. All other endpoints (CME, CMEAnalysis, FLR, GST, IPS, ENLIL, HSS, SEP) accept arbitrary date ranges.

**Decision**:
In `clients/donki.py`, use a single request covering the full date range for all endpoints except `/notifications`. `/notifications` splits into ≤30-day chunks.

**Consequences**:
- Chunking all endpoints wastes ~30x the rate limit quota for a full historical fetch.
- This is a breaking correction from earlier versions of CLAUDE.md.

---

## ADR-D005: BeautifulSoup for All HTML Scraping

**Status**: Accepted

**Context**:
`pandas.read_html()` misparsed the CDAW LASCO CME table — wrong row counts and misaligned columns due to the table's non-standard `<thead>`-less structure and footnote markers in numeric cells (`-54.7*1`).

**Decision**:
Use `BeautifulSoup(html, "lxml")` for all HTML table parsing. Strip footnote markers (`re.sub(r'[^0-9.\-].*$', '', cell)`) before numeric casting. The Apify Actor framework at `C:\Users\radar\FamilyTree\` is available as a reference for complex scraping patterns.

**Consequences**:
- No `pandas.read_html()` anywhere in the codebase.
- `lxml` is required (in `pyproject.toml` as `lxml>=5.0`).
- CDAW URL must use `UNIVERSAL_ver2/` path (changed May 2024, added 3,070 CMEs).

---

## ADR-D006: GSM-Frame Bz as Canonical Solar Wind Field

**Status**: Accepted — mirrors C# RULE-031

**Context**:
Solar wind magnetometer data is available in both GSE and GSM coordinate frames. The C# SolarPipe physics models (BurtonOde, NewellCoupling) use GSM-frame Bz exclusively (RULE-031 in `../DEVELOPMENT_RULES.md`). NOAA SWPC `rtsw_mag_1m.json` provides both `bz_gse` and `bz_gsm` in a single request.

**Decision**:
Store only `bz_gsm` as the canonical Bz field in `solar_wind_hourly`. Never use `bz_gse` for physics calculations or ML features. The OMNI dataset (already ported) includes `Bz_GSM` — use that column.

**Consequences**:
- Column is named `bz_gsm` in `solar_wind_hourly`.
- Any new solar wind data source must provide or compute GSM-frame Bz before ingest.

---

## ADR-D007: JSOC drms Timeout Wrapper

**Status**: Accepted

**Context**:
The `drms` Python package uses `urllib` internally with no configurable timeout. On JSOC server non-response (which occurs during maintenance and peak load), `c.query()` hangs indefinitely, blocking the entire async pipeline.

**Decision**:
Wrap every `drms.Client().query()` call with `concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(c.query, ...).result(timeout=60)`. The 60-second timeout is configurable via `configs/default.yaml` (`jsoc_timeout_s`).

**Consequences**:
- `drms` queries run in a thread pool, not in the async event loop.
- `sunpy 7.1.1` and `drms 0.9.0` are already installed — no install step needed.

---

## ADR-D008: Kyoto Dst Preference Cascade

**Status**: Accepted

**Context**:
Kyoto WDC publishes Dst in three quality tiers: final (gold standard, published ~3 years after), provisional (~3 months to 3 years lag), and real-time (current month). All three may be available for the same time period during transition.

**Decision**:
Upsert logic in `ingestion/ingest_dst.py` checks the existing `data_type` before replacing: `final` is never overwritten. `provisional` may replace `realtime`. `final` may replace `provisional`. This is enforced in the upsert SQL, not application logic.

**Consequences**:
- The `dst_hourly.data_type` column is always populated.
- Real-time data ingested today may be upgraded to provisional or final in a later run.

---

## ADR-D009: No Virtual Environment

**Status**: Accepted
**Date**: 2026-04-07

**Context**:
Single-developer workflow. Claude Code acts as the automated developer/IDE. System Python 3.12 is the runtime.

**Decision**:
No virtual environment. All commands use bare `python`, `pytest`, `pip`. Package dependencies managed via `pyproject.toml` with `pip install -e ".[dev]"` on the system Python.

**Consequences**:
- No `.venv/` prefixes in any command or documentation.
- System package state is the only environment — dependency conflicts surface immediately.
- `pyproject.toml` pins minimum versions, not exact versions.
