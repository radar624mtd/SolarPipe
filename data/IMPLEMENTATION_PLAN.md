# IMPLEMENTATION_PLAN.md — SolarPipe Data Acquisition Pipeline

Phase-by-phase implementation plan. **Read only the current phase section.**
Full API/schema reference: `ARCHITECTURE.md`. Rules: `DEVELOPMENT_RULES.md`. Decisions: `ARCHITECTURAL_DECISIONS.md`.

---

## Phase Delivery Sequence

| Phase | Focus | Key Deliverable | Verification |
|-------|-------|-----------------|-------------|
| 1 | Foundation | staging.db seeded; CLI + BaseClient + DONKI client | `pytest -m integration -k donki` |
| 2 | CME & Flare Catalogs | CDAW, GOES flares, DONKI ancillary | `python -m solarpipe_data status` — all tables show records |
| 3 | Solar Wind & Indices | Incremental SWPC, Kyoto Dst, Kp, F10.7 | `pytest -m integration -k solar_wind` |
| 4 | SHARP Features | JSOC DRMS; 18 keywords; disk-passage filter | SHARP coverage ≥80% of Earth-directed CMEs |
| 5 | Cross-Matching | CME↔Flare, CME↔ICME, feature assembly, quality flags | Physical consistency checks pass |
| 6 | Synthetic & Export | ENLIL emulator, Parquet export, cme_catalog.db, validation | `python scripts/validate_db.py` exits 0 |

---

## Phase 1 — Foundation

**Goal**: Idempotent staging.db that can be seeded from both the existing solar_data.db port and live DONKI fetches.

### Task 1.0 — Port existing data ✅
`scripts/port_solar_data.py` — maps 9 tables from `solar_data.db` → staging.db with sentinel cleanup and provenance columns. Run once before any live fetching.

### Task 1.1 — Verify configuration loads
`python -c "from solarpipe_data.config import get_settings; s = get_settings(); print(s.nasa_api_key, s.staging_db_path)"`
Expected: key from env, posix path.

### Task 1.2 — `database/migrations.py`
Schema version table already created by `init_db()`. This module adds:
- `current_version()` — returns current schema version integer
- `apply_pending()` — runs any migration functions with version > current
- Migration function signature: `def migrate_v2(engine) -> None`

### Task 1.3 — `database/queries.py`
Common patterns used by ingestion and crossmatch:
- `upsert(engine, table_class, rows: list[dict])` — batch upsert using sqlite dialect
- `temporal_range(engine, table_class, start: str, end: str)` — query by datetime column
- `max_timestamp(engine, table_class, col: str) -> str | None` — for incremental fetch windows

### Task 1.4 — `cli.py` — Click harness
Six commands, all with async bridge via `run_async` decorator:
- `fetch <source> --start --end [--force]` — fetch to raw cache
- `ingest <source> --start --end` — raw cache → staging.db
- `crossmatch [--start --end]` — run all matchers
- `build --start [--no-fetch]` — full pipeline
- `validate` — run `scripts/validate_db.py` checks
- `status` — print record counts per table with temporal coverage

### Task 1.5 — `clients/base.py` — BaseClient
```python
class BaseClient:
    source_name: str
    rate_limit: float          # req/s from config
    cache_dir: Path
    client: httpx.AsyncClient  # single shared instance

    async def get(self, url: str, params: dict, cache_key: str) -> dict | list
    async def get_html(self, url: str, cache_key: str) -> str
    async def _rate_limit(self) -> None   # token bucket
    async def _request_with_retry(self, ...) -> httpx.Response
    def _cache_path(self, cache_key: str) -> Path
```
Token bucket: `asyncio.sleep` based. Check `X-RateLimit-Remaining` header for DONKI dynamic adjustment.

### Task 1.6 — `clients/donki.py` — DONKI client
Inherits `BaseClient`. One method per endpoint:
```python
async def fetch_cme(self, start: str, end: str) -> list[dict]
async def fetch_cme_analysis(self, start: str, end: str) -> list[dict]
async def fetch_flares(self, start: str, end: str) -> list[dict]
async def fetch_gst(self, start: str, end: str) -> list[dict]
async def fetch_ips(self, start: str, end: str) -> list[dict]
async def fetch_enlil(self, start: str, end: str) -> list[dict]
async def fetch_notifications(self, start: str, end: str) -> list[dict]  # chunks to ≤30d
```
All except `fetch_notifications` use a single request regardless of date range (ADR-D004).

### Task 1.7 — `ingestion/ingest_donki_cme.py`
- Parse `activity_id`, `start_time` (fromisoformat with Z replacement), all analysis fields
- Null `activeRegionNum` → `None` (not 0)
- `linked_event_ids` JSON list → TEXT column
- Upsert by `activity_id` using sqlite dialect insert

### Task 1.8 — Unit tests
Files: `tests/unit/test_clients/test_donki.py`, `tests/unit/test_ingestion/test_donki_cme.py`
Fixtures: `tests/fixtures/donki_cme_sample.json`, `tests/fixtures/donki_analysis_sample.json`
Cover: rate limiter token consumption, cache hit/miss, JSON parsing, sentinel conversion, upsert idempotency (insert twice, count once), CLI `status` command.

---

## Phase 2 — CME & Flare Catalogs

**Goal**: All CME and flare staging tables populated from CDAW, NOAA GOES, and DONKI ancillary endpoints.

### Task 2.1 — `clients/cdaw.py`
Fetch UNIVERSAL_ver2 monthly pages (1996–present) and halo catalog.
URL: `https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL_ver2/{YYYY}_{MM}/univ{YYYY}_{MM}.html`
Rate: ≤1 req/2s. Cache HTML response by month.

### Task 2.2 — `ingestion/ingest_cdaw_lasco.py`
Parse with BeautifulSoup. Skip first header row manually (no `<thead>`).
CPA column: `"Halo"` → `None`, `angular_width_deg = 360`.
Numeric columns: strip footnote markers before `float()`.
`speed_20rs_kms` (column `second_order_speed_20Rs`) is the canonical speed for arrival models.
Mass column: `"----"` → `None` — do not impute.
Remarks column: contains `"Poor Event"` / `"Very Poor Event"` → `quality_flag = 1` or `2`.
`cdaw_id` = `f"{date.replace('-','')}.{time_ut.replace(':','')}"`.

### Task 2.3 — `ingestion/ingest_donki_enlil.py`, `ingest_donki_gst.py`, `ingest_donki_ips.py`
ENLIL: not 1:1 with CMEs — store all, deduplicate at crossmatch by first simulation per CME.
GST: `kp_index_max`, `all_kp_values` as JSON text.
IPS: `location=Earth` filter already applied at API level.

### Task 2.4 — `clients/noaa_indices.py` + `ingestion/ingest_flares.py`
GOES flares from SWPC: `/json/goes/primary/xray-flares-7-day.json` (includes `Z` in timestamps).
NCEI archive for historical. Merge with DONKI FLR — tag `source_catalog` as `"GOES"` or `"DONKI"`.
`flare_id` for GOES: construct from `start_time + goes_satellite`.

### Task 2.5 — Merge GOES + DONKI flares
Same event may appear in both catalogs. Dedup by `begin_time ± 2min + active_region_num`.

### Task 2.6 — Integration tests
`pytest -m "integration and not live" -k "cdaw or flare"` — verify row counts match expected ranges.

---

## Phase 3 — Solar Wind & Indices

**Goal**: Incremental updates to solar wind and geomagnetic index tables (bulk already ported from solar_data.db).

### Task 3.1 — `clients/swpc.py`
Real-time endpoints for mag + plasma. NCEI CSV client for archival gaps.
Timestamp normalization: always `ts.rstrip("Z")` before `fromisoformat()`.

### Task 3.2 — `ingestion/ingest_solar_wind.py`
1-minute → hourly averages. ACE→DSCOVR transition July 2016: log, do not gap or duplicate.
Sentinel conversion: `99999.9`, `-1e31`, `9999.99` → `None`.
Only fetch records after `MAX(datetime)` in `solar_wind_hourly` (incremental).
L1 propagation: store as measured at L1 — do not adjust timestamps. Lag is applied at crossmatch.

### Task 3.3 — `clients/kyoto.py` + `ingestion/ingest_dst.py`
Try `final` → fallback to `provisional` → fallback to `realtime` for each month.
Preference cascade in upsert: check `data_type` of existing record before replacing (ADR-D008).
Pre-2019: WDC fixed-width format. Post-2019: HTML table parse.
Empty body check: `if len(html) < 100: return []`.

### Task 3.4 — `ingestion/ingest_kp.py`
GFZ API incremental only — bulk already ported (34,426 rows through 2026-04-02).
Fetch records after `MAX(datetime)` in `kp_3hr`.

### Task 3.5 — `ingestion/ingest_f107.py`
NOAA SWPC `/json/solar-cycle/observed-solar-cycle-indices.json`.

### Task 3.6 — Ambient solar wind context extraction
For each CME in `cme_events`, compute 6-hour pre-arrival window averages:
`sw_speed_ambient`, `sw_density_ambient`, `sw_bt_ambient` — stored back to staging as a context table.

### Task 3.7 — Unit tests
Cover: hourly averaging, ACE/DSCOVR transition handling, preference cascade logic, Kyoto HTML parser, empty-body guard.

---

## Phase 4 — SHARP Features

**Goal**: 18 JSOC space-weather keywords for every Earth-directed CME; ≥80% coverage.

### Task 4.1 — `clients/jsoc.py`
```python
import drms
from concurrent.futures import ThreadPoolExecutor

class JsocClient(BaseClient):
    def _query(self, ds: str, key: str) -> pd.DataFrame:
        c = drms.Client(email=cfg.jsoc_email)
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(c.query, ds, key=key).result(timeout=cfg.jsoc_timeout_s)
```
Series: `hmi.sharp_cea_720s` (not NRT, not CCD series).
Keywords: 18 space-weather + `LAT_FWT, LON_FWT, NOAA_AR, HARPNUM, T_REC`.

### Task 4.2 — `ingestion/ingest_sharps.py`
For each Earth-directed CME: query at `t_eruption`, `t-6h`, `t-12h`, `t+6h`.
Drop at ingest if `LON_FWT > 60°`.
`NOAA_AR = 0` → `None` (not a real AR designation).
`DATE__OBS` → `t_rec` column (normalize double-underscore).

### Task 4.3 — HARP↔NOAA mapping table
Separate query: `hmi.sharp_720s[][TIMERANGE]{HARPNUM,NOAA_AR,NOAA_ARS}`.
`NOAA_ARS` tilde filter for CMEs associated with multi-region HARPs.

### Task 4.4 — Feature selection: optimal snapshot per CME
Prefer `at_eruption` snapshot. Fall back to `minus_6h`, then `minus_12h`.
Log SHARP coverage fraction — alert if < 80% of Earth-directed CMEs.

### Task 4.5 — Integration test
`pytest -m "integration and not live" -k sharp` — verify coverage metric.

---

## Phase 5 — Cross-Matching & Feature Assembly

**Goal**: `feature_vectors` table with ~300–500 events at `quality_flag ≥ 3`.

### Task 5.1 — `crossmatch/cme_flare_matcher.py`
Priority: DONKI `linkedEvents` → temporal ±30min + spatial ±15° fallback → statistical (no match).
Null FKs for unmatched — never force an assignment.

### Task 5.2 — `crossmatch/cme_icme_matcher.py`
Priority: DONKI IPS `linkedEvents` → transit time estimate ±12hr.
Transit estimate: `t = launch_time + (1 AU / cme_speed)` with drag correction.
Matching is inherently ambiguous for interacting CMEs — flag with `match_confidence < 0.5` rather than force.

### Task 5.3 — `crossmatch/cme_sharp_matcher.py`
NOAA AR number match → fallback to source location proximity.

### Task 5.4 — `crossmatch/storm_matcher.py`
ICME arrival → Dst minimum within 24–48 hr post-arrival.
Threshold: `Dst < -30 nT`. L1 lag adjustment: shift L1 timestamps by +45 min before Dst correlation.

### Task 5.5 — `crossmatch/feature_assembler.py`
16+ column vector per event. Null-fill deferred features:
- `dimming_area`, `dimming_asymmetry` — null (AIA image processing deferred)
- `hcs_tilt_angle`, `hcs_distance` — null (requires PFSS model output; use pre-computed GONG maps if available)

### Task 5.6 — Quality flags (1–5)
| Flag | Meaning |
|------|---------|
| 5 | All features present; definitive data |
| 4 | Minor gaps (1–2 null features) |
| 3 | Moderate gaps; still usable (default filter) |
| 2 | Significant gaps or poor-quality source data |
| 1 | Poor event (CDAW "Poor Event" tag); avoid |

### Task 5.7 — Physical consistency validation
`transforms/validation.py`:
- Speed↔transit time correlation (Spearman r > 0.4 expected)
- Dst↔Bz_min correlation (Spearman r > 0.5 expected)
- Speed range: 100–3000 km/s (flag outliers)
- Density range: 0.1–100 cm⁻³ (flag sentinels that slipped through)

---

## Phase 6 — Synthetic Data & Export

**Goal**: `cme_catalog.db` and `enlil_runs/enlil_ensemble_v1.parquet` both pass validation.

### Task 6.1 — `synthetic/drag_model.py`
`scipy.integrate.solve_ivp` with Dormand-Prince (RK45) solver.
Drag equation: `dv/dt = -γ(v - w)|v - w|` where `w` is ambient solar wind speed.
γ calibrated to ENLIL ensemble mean from `enlil_simulations` table.

### Task 6.2 — `synthetic/rotation_model.py`
HCS alignment + coronal hole deflection model.
`hcs_tilt_angle` and `hcs_distance` from GONG synoptic maps (null-fill if unavailable).

### Task 6.3 — `synthetic/monte_carlo.py`
Physically-motivated distributions with enforced correlations:
- Speed: log-normal (μ=500, σ=200 km/s)
- Width: beta distribution (bounded 20°–360°)
- Correlation: speed ↔ flare class (Pearson r ~ 0.4 observed in training set)

### Task 6.4 — `synthetic/enlil_emulator.py`
Compose drag + rotation + systematic bias (from ENLIL simulation residuals) + Gaussian noise.
Output schema must match `ParquetProvider` expectations — check `ARCHITECTURE.md` output schema.

### Task 6.5 — `export/parquet_export.py`
Write to `data/output/enlil_runs/enlil_ensemble_v1.parquet`.
Row groups: ≤64 MB (ADR-D002). Include metadata dict with generation timestamp, seed, parameter distributions.

### Task 6.6 — `export/sqlite_export.py`
Assemble `cme_catalog.db` from `feature_vectors` + `flux_rope_fits` + `l1_arrivals`.
Column names must match `configs/flux_rope_propagation_v1.yaml` exactly — do not rename.

### Task 6.7 — `scripts/validate_db.py`
Exit 0 on success, non-zero on failure. Checks:
1. Three-table join query executes without error (see `ARCHITECTURE.md`)
2. Row count ≥ 100 events with `quality_flag ≥ 3` and `has_in_situ_fit = 1`
3. Speed range: 100–3000 km/s (no sentinels)
4. `bz_gsm` column present and has non-null values
5. `cme_catalog.db` schema matches expected column list

### Task 6.8 — End-to-end integration test
`pytest -m "integration and not live" -k e2e`
Covers: fetch from cache → ingest → crossmatch → build → validate — all in a temp directory.

---

## Data Quality Gotchas by Source

Reference during implementation. Read the section for the source you are currently working on.

### DONKI
- Pre-2012 data: ~43% sample rejection rate — use `donki_verified_start = "2012-01-01"` from config
- `level_of_data`: 0=real-time (avoid), 1=NRT, 2=definitive (prefer)
- `activeRegionNum` null in ~10–15% of entries — remediate via CDAW/GOES cross-reference at Phase 5
- `linkedEvents` may reference events not in DB — null FKs only, no exceptions
- `WSAEnlilSimulations`: one CME can appear in multiple sims — take first per CME at crossmatch

### CDAW
- SOHO vacation gap 1998–99: real data gap, do not retry
- Mass column: `"----"` → `None`. ~40% of events have null mass — never impute
- CPA for halo CMEs: `"Halo"` string → `None`, `angular_width_deg = 360`
- `speed_20rs_kms` is the correct column for arrival models (quadratic at 20 R☉)
- "Poor Event" / "Very Poor Event" in Remarks → `quality_flag = 1` or `2`

### SHARP / JSOC
- `NOAA_AR = 0`: no NOAA designation — do not join on 0
- NRT series (`hmi.sharp_cea_720s_nrt`): larger errors, set QUALITY flags — avoid for training data
- 12-minute cadence gaps are normal physics data gaps, not ingestion failures
- `DATE__OBS` double underscore — drms returns it as-is in keyword queries

### Solar Wind
- `active: false` records in SWPC: non-primary spacecraft — exclude or flag
- Alpha particle fields (`alpha_speed`, `alpha_density`): almost always null — do not depend on them
- OMNI hourly already ported (561K rows, 1963–2026): only fetch incremental updates

### Kyoto Dst
- Non-commercial use only — research use is acceptable
- Sentinel `9999` nT in WDC-format: values > 500 or < -500 → `None`
- Empty body (HTTP 200, `len(body) < 100`) during maintenance — return empty list

### Cross-Matching
- CME-ICME matching is inherently ambiguous for interacting/merging CMEs — flag ambiguous matches
- EUV dimming features (`dimming_area`, `dimming_asymmetry`) deferred — null-fill
- HCS parameters (`hcs_tilt_angle`, `hcs_distance`) require PFSS model — null-fill initially
