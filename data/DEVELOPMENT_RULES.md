# DEVELOPMENT_RULES.md — SolarPipe Data Acquisition Pipeline

Enforceable rules derived from API research, domain gotchas, and prior debugging loops.
**Read the sections relevant to your current task. Do not pre-load the entire file.**

- **All tasks**: Rules 001–015 (core agent rules)
- **HTTP clients**: Add Rules 020–025
- **Ingestion / SQLAlchemy**: Add Rules 030–037
- **DONKI specifically**: Add Rules 040–045
- **CDAW specifically**: Add Rules 050–053
- **JSOC / SHARP**: Add Rules 060–065
- **Solar wind / SWPC**: Add Rules 070–073
- **Kyoto Dst**: Add Rules 080–082
- **Testing**: Add Rules 090–095

---

## Section 001–015: Core Agent Rules

**RULE-001** — Test after every `.py` edit
`pytest --tb=short -q -m unit` after every file change. Catch one break at a time.

**RULE-002** — No raw httpx calls
ALL HTTP goes through `BaseClient`. Never call `httpx.get()` or instantiate `httpx.AsyncClient()` directly in a client module. `BaseClient` owns rate limiting, retry, caching, and logging.

**RULE-003** — Sentinel conversion is ingestion's job
Convert `9999.9`, `---`, `-1e31`, `"--"`, empty strings to Python `None` in the ingestion layer — never in transforms or crossmatch. Floats must not silently carry sentinel values into staging.db.

**RULE-004** — Upsert, never blind insert
Every ingestion module uses `INSERT OR REPLACE` (or SQLAlchemy `insert().on_conflict_do_update()`) keyed on the source's natural PK. Re-running ingestion must be idempotent.

**RULE-005** — Git hygiene before every task
`git status` before any implementation step. Commit or stash uncommitted changes first.

**RULE-006** — Small files
No `.py` file > ~400 lines. Extract helpers into submodules rather than growing a single file.

**RULE-007** — No API keys in source
All credentials via env vars or `configs/default.yaml` (gitignored). Never hardcode `NASA_API_KEY` or any token.

**RULE-008** — No schema changes without migrations
Any new table or column requires a corresponding entry in `database/migrations.py`. Never hand-edit `staging.db` directly.

**RULE-009** — Provenance columns on every staging table
`source_catalog TEXT`, `fetch_timestamp TEXT`, `data_version TEXT` on every table. Do not omit when adding new tables.

**RULE-010** — Port before fetching
`solar_data.db` at the repo root has 561K+ OMNI rows, 42K+ CDAW CMEs, 8K DONKI CMEs through 2026-04-05. Run `scripts/port_solar_data.py` first. Only fetch data that postdates or is missing from the port.

**RULE-011** — Type hints required
All public functions must have complete type annotations including return types. `Any` requires a comment explaining why.

**RULE-012** — No docstrings on private helpers
Google-style docstrings on public functions and CLI commands only. No docstrings on `_prefixed` helpers.

**RULE-013** — Click async bridge
Click has no native async. Bridge with a `run_async` decorator that calls `asyncio.run(f(*args, **kwargs))`. Never nest `asyncio.run()` inside an already-running event loop.

**RULE-014** — Delete placeholder tests
Remove any auto-generated test stubs immediately when real tests exist. They inflate counts and give false confidence.

**RULE-015** — Env vars are already set
`NASA_API_KEY`, `NASA_API_EMAIL`, `STANFORD_JSOC_EMAIL`, `STANFORD_JSOC_USING_SUNPY` are set in the shell environment. Read via `os.environ` or Pydantic settings. Do not prompt the user for them.

---

## Section 020–025: HTTP Client Rules

**RULE-020** — BaseClient owns all HTTP state
`httpx.AsyncClient` is created once per `BaseClient` instance, not per-request. Token-bucket rate limiter, retry-with-backoff, and file cache are all `BaseClient` responsibilities.

**RULE-021** — Cache before rate limiting
Check file cache before consuming rate limit quota. Cache key = `(source, url_hash, date_range)`. TTL from `configs/default.yaml`.

**RULE-022** — Structured error messages
All HTTP exceptions must include: source name, URL, response status code, and any API error message. Raw stack traces waste context.

**RULE-023** — Retry with exponential backoff
Retry on 429 and 5xx. Respect `Retry-After` header when present. Max retries from config. Do not retry on 400/401/403/404.

**RULE-024** — No timeout on outer asyncio loop
Use `httpx` timeout param, not `asyncio.wait_for`. `httpx.Timeout(connect=10, read=30)` is the pattern.

**RULE-025** — Windows asyncio event loop
Python 3.12 defaults to `ProactorEventLoop` on Windows. Use `asyncio.run()` — do not manually set event loop policy (deprecated in 3.14, removed in 3.16).

---

## Section 030–037: SQLAlchemy / SQLite Rules

**RULE-030** — Use sqlite dialect for upsert
`from sqlalchemy.dialects.sqlite import insert` — NOT `from sqlalchemy import insert`. The generic `insert` has no `.on_conflict_do_update()`.

**RULE-031** — Windows-safe connection strings
Always `Path(db_path).as_posix()` before building the SQLAlchemy URL. `str(WindowsPath)` produces backslashes that break the URL parser.

**RULE-032** — WAL mode on every engine
Register a `connect` event listener that runs `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL`. Do not rely on the default journal mode.

**RULE-033** — Session context manager
Always `with Session(engine) as s, s.begin(): ...`. Never create sessions without cleanup.

**RULE-034** — Nullable column upsert (SQLAlchemy bug #9702)
When upserting and a column is `None`, use `set_={col: insert_stmt.excluded.col}` explicitly — do not rely on `set_=dict(row)`. Silent data loss on null columns.

**RULE-035** — No hand-edits to staging.db
Schema changes go through `database/migrations.py`. Direct SQLite shell edits create untracked schema drift.

**RULE-036** — Commit in batches for large ports
When inserting >10K rows, commit every 5,000 rows to avoid holding a single massive transaction that can't be rolled back cleanly.

**RULE-037** — Check `len(body) > 100` before parsing HTML
Kyoto and CDAW return HTTP 200 with near-empty body during maintenance. Always check body length before BeautifulSoup parse.

---

## Section 040–045: DONKI Rules

**RULE-040** — Only `notifications` needs 30-day chunking
CME, CMEAnalysis, FLR, GST, IPS, ENLIL, HSS, SEP all accept arbitrary date ranges. Only `/notifications` is limited to 30 days. Do NOT chunk other endpoints — it wastes rate limit quota.

**RULE-041** — Always `mostAccurateOnly=true` on CMEAnalysis
Without it, multiple revision records per CME create join duplicates. Only omit when specifically studying revision history.

**RULE-042** — `time21_5` is not event start
In DONKI CMEAnalysis, `time21_5` is the CME front position at 21.5 solar radii — not liftoff. Use `associatedCMEstartTime` (or `cme_events.start_time`) for event origin time in all joins.

**RULE-043** — DONKI timestamp format has no seconds
All DONKI timestamps use `"2016-09-06T14:18Z"` (no seconds field). Parse with `datetime.fromisoformat(ts.replace("Z", "+00:00"))`.

**RULE-044** — `level_of_data` preference
0=real-time (avoid), 1=NRT, 2=definitive (prefer for training). Level 0 is explicitly labeled "prototyping quality" by CCMC. Reject pre-2012 data — ~43% sample rejection rate in quality-gated pipelines.

**RULE-045** — Broken `linkedEvents` references
Some DONKI `linkedEvents` entries reference events not in the DB. Handle with null FKs, not exceptions. WSAEnlilSimulations are not 1:1 with CMEs — deduplicate by taking the first simulation per CME.

---

## Section 050–053: CDAW Rules

**RULE-050** — BeautifulSoup only, never `pandas.read_html()`
`pandas.read_html()` misparsed the CDAW `<table>` — wrong row counts, misaligned columns. Always `BeautifulSoup(html, "lxml")`.

**RULE-051** — Use `UNIVERSAL_ver2/` URL path
Since May 2024: `https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL_ver2/{YYYY}_{MM}/univ{YYYY}_{MM}.html`. The old `UNIVERSAL/` path is stale. UNIVERSAL_ver2 added 3,070 CMEs in May 2024 — re-scrape 1996–2004 if cached from old path.

**RULE-052** — Strip footnote markers before numeric casting
Cells like `"-54.7*1"` appear in speed/angle columns. Strip trailing non-numeric characters before float conversion: `re.sub(r'[^0-9.\-].*$', '', cell)`.

**RULE-053** — `speed_20rs_kms` is the correct arrival-model speed
`linear_speed_kms` is a linear fit. `speed_20rs_kms` (quadratic at 20 R☉) is the correct speed for CME-arrival models. Never use linear speed for transit-time prediction.

---

## Section 060–065: JSOC / SHARP Rules

**RULE-060** — SHARP disk-passage filter at ingest
Drop records where `LON_FWT > 60°` from disk center. Projection effects make values unreliable near the limb. Apply at ingest, not in feature engineering.

**RULE-061** — JSOC has no timeout — always wrap
The `drms` package uses `urllib` internally with no timeout. Wrap every `c.query()` call:
```python
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=1) as pool:
    result = pool.submit(c.query, ...).result(timeout=60)
```

**RULE-062** — CEA series only
Use `hmi.sharp_cea_720s` for all 18 space-weather keywords. The CCD series (`hmi.sharp_720s`) does not compute the keywords correctly.

**RULE-063** — `NOAA_AR = 0` means no designation
Do not join on `NOAA_AR` when it is 0. Use `NOAA_ARS` tilde filter for multi-region HARPs: `[? NOAA_ARS ~ "12345" ?]`.

**RULE-064** — `DATE__OBS` double underscore
JSOC namespace uses `DATE__OBS` (double underscore). The `drms` package normalizes this to `DATE-OBS` in FITS export only — keyword queries return the double-underscore form.

**RULE-065** — NRT series degrades quality
`hmi.sharp_cea_720s_nrt` has larger errors and set `QUALITY` flags. Use the definitive series for training data. 12-minute cadence gaps are normal physics data gaps, not ingestion failures.

---

## Section 070–073: Solar Wind / SWPC Rules

**RULE-070** — GSM Bz only (RULE-031 mirror)
All solar wind Bz stored as GSM-frame. `rtsw_mag_1m.json` provides both `bz_gse` AND `bz_gsm` in one request — always use `bz_gsm`.

**RULE-071** — SWPC timestamp `Z` inconsistency
`/products/solar-wind/` files omit trailing `Z`; `/json/goes/` files include `Z`. Always: `datetime.fromisoformat(ts.rstrip("Z"))` before parsing.

**RULE-072** — `mag-5-minute.json` is a single record
Not a time series — it contains only the most recent 5-minute average. Use `mag-7-day.json` for time series.

**RULE-073** — L1 propagation delay is physical
L1 data arrives ~30–60 min before reaching Earth's magnetosphere. This is not a timing error. Lag L1 data by ~45 min when computing Dst correlation. ACE→DSCOVR transition occurred July 2016 — log it, do not create gap records.

---

## Section 080–082: Kyoto Dst Rules

**RULE-080** — Preference cascade: final > provisional > realtime
Track `data_type` column in `dst_hourly`. Never overwrite `final` with `provisional` or `realtime`. Upsert logic must check existing `data_type` before replacing.

**RULE-081** — Pre-2019 format differs
Since 2019-04-26, parse `index.html` HTML table (day row × 24 hourly columns). Before that date, files used WDC fixed-width binary format. Do not attempt to parse binary `.for.request` files as HTML.

**RULE-082** — Sentinel `9999` means missing
In WDC-format files, values > 500 nT or < −500 nT are sentinels → treat as `None`. Non-commercial use only clause applies — this pipeline is research use.

---

## Section 090–095: Testing Rules

**RULE-090** — `asyncio_mode = "auto"` in pyproject.toml
Set `asyncio_default_fixture_loop_scope = "function"` to suppress the 0.24+ deprecation warning. Do NOT add `@pytest.mark.asyncio` to individual tests in auto mode — it generates spurious warnings.

**RULE-091** — Fixture before test
Add `tests/fixtures/{source}_sample.json` (or `.html`) before writing the first test for a new client. Never write a unit test that requires network access to pass the first time.

**RULE-092** — Three markers only
`@pytest.mark.unit` — fully offline, pytest-httpx mocks all HTTP.
`@pytest.mark.integration` — staging.db must exist, no live network.
`@pytest.mark.live` — real API calls, never in CI.

**RULE-093** — CliRunner with `catch_exceptions=False`
During development: `runner.invoke(cmd, args, catch_exceptions=False)`. Silent swallowed exceptions waste debugging time.

**RULE-094** — Session-scoped async fixtures use sync wrapper
```python
@pytest.fixture(scope="session")
def shared_engine():
    engine = asyncio.run(_create_engine())
    yield engine
    asyncio.run(engine.dispose())
```

**RULE-095** — Delete placeholder tests
Remove auto-generated `test_*.py` stubs immediately when real tests exist in the same project.

---

## Critical Rules Quick Reference

Silent failures if violated:

| Rule | Summary |
|------|---------|
| RULE-030 | `from sqlalchemy.dialects.sqlite import insert` — generic insert has no upsert |
| RULE-031 | `Path(p).as_posix()` in all connection strings |
| RULE-032 | `PRAGMA journal_mode=WAL` on every new engine via event listener |
| RULE-002 | All HTTP via `BaseClient` — never `httpx.get()` directly |
| RULE-003 | Convert `9999.9`, `---`, `-1e31` to `None` at ingest, not later |
| RULE-040 | **Only `notifications` needs ≤30d chunks** — all other DONKI endpoints accept any range |
| RULE-041 | Always `mostAccurateOnly=True` on CMEAnalysis |
| RULE-042 | `time21_5` ≠ event start — use `associatedCMEstartTime` |
| RULE-050 | BeautifulSoup only for CDAW — never `pandas.read_html()` |
| RULE-051 | CDAW URL: `UNIVERSAL_ver2/` not legacy `UNIVERSAL/` |
| RULE-053 | Use `speed_20rs_kms` for arrival models, not `linear_speed_kms` |
| RULE-060 | Drop SHARP records where `LON_FWT > 60°` at ingest |
| RULE-061 | Wrap every `drms.query()` with `ThreadPoolExecutor.result(timeout=60)` |
| RULE-070 | Use `bz_gsm`, never `bz_gse` |
| RULE-071 | `ts.rstrip("Z")` before `fromisoformat()` on all SWPC timestamps |
| RULE-080 | Kyoto cascade: final > provisional > realtime, never overwrite downward |
| RULE-090 | `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` on individual tests |
| RULE-010 | Run port script before any live fetching |
| RULE-007 | No API keys in source — env vars only |
| RULE-008 | No DDL changes without `database/migrations.py` entry |
