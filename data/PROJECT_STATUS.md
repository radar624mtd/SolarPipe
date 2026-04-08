# PROJECT_STATUS.md — SolarPipe Data Acquisition Pipeline

Last updated: 2026-04-07

## Current Phase: Phase 5 — Cross-Matching (in progress)

### Live Data Run — COMPLETE (2026-04-07)

All tables populated. Final row counts:

| Table | Rows | Notes |
|-------|------|-------|
| `cme_events` | 9,413 | Refreshed from kauai DONKI (1,376 new vs ported data) |
| `cme_analyses` | 8,038 | Full history 2010–2026 |
| `cdaw_cme_events` | 41,351 | Ported |
| `enlil_simulations` | 5,972 | Full history 2010–2026 |
| `flares` | 3,207 | Ported |
| `dst_hourly` | 143,136 | 2010-01 → 2026-04 (WDC fixed-width format + HTML fallback; RULE-081 updated) |
| `f107_daily` | 3,327 | Full NOAA history 1749–2026 |
| `sharp_keywords` | 102,305 | 95.6% coverage (1,456/1,523 earth-directed CMEs) |
| `sw_ambient_context` | 9,371 | Computed from all CMEs |
| `kp_3hr` | 34,426 | Ported |
| `solar_wind_hourly` | 561,024 | Ported |
| `symh_hourly` | 396,624 | Ported |
| `silso_daily_ssn` | 76,030 | Ported |
| `geomagnetic_storms` | 192 | Ported |
| `interplanetary_shocks` | 644 | Ported |
| `harp_noaa_map` | 0 | Not needed for Phase 5; bulk HARP era query deferred |

**Bug fixes applied this session (all committed):**
- `max_timestamp()`: dual calling convention fix (string-first vs engine-first)
- `_parse_cme_time()`: removed `clean[:len(fmt)]` slice causing silent None returns
- `ingest_sharps.py`: parallel fetch (8 workers), resume via `activity_id`, all-HARP fallback
- `ingest_donki_enlil.py`: batch upsert fix (SQLite 999 bind-var limit)
- DONKI endpoint: switched to `kauai.ccmc.gsfc.nasa.gov` (direct, no api_key, reliable)
- `http_timeout_s`: 30→120 for large DONKI payloads
- `scripts/run_live_ingest.py`: annual chunking for DONKI, full orchestration

---

## Phase 5 — Cross-Matching (in progress)

Last updated: 2026-04-07

- [x] **5.1** `crossmatch/cme_flare_matcher.py` — Priority: DONKI linkedEvents → temporal ±30 min + spatial ±15° fallback; null FKs for unmatched; `linked_missing` method when id referenced but absent from DB.
- [x] **5.2** `crossmatch/cme_icme_matcher.py` — Priority: DONKI IPS linkedEvents (both directions) → drag-corrected transit estimate ±12 hr; `match_confidence` halved when multiple candidates; `icme_match_confidence` stored.
- [x] **5.3** `crossmatch/cme_sharp_matcher.py` — NOAA AR match → spatial proximity fallback (|Δlat|≤15°, |Δlon|≤15°, Chebyshev); delegates snapshot preference to `select_sharp_features.get_best_sharp_snapshot()`; `_sharp_to_feature_dict()` flattens 18 keywords.
- [x] **5.4** `crossmatch/storm_matcher.py` — Dst min + Kp max in 0–48 hr post-ICME window; L1 lag +45 min; `storm_threshold_met` = Dst < -30 nT.
- [x] Unit tests: `test_cme_flare_matcher.py` (25 tests), `test_cme_icme_matcher.py` (20 tests), `test_cme_sharp_matcher.py` (10 tests), `test_storm_matcher.py` (12 tests) — 67 total, all passing (220 unit total).
- [x] Schema: `database/migrations.py` v5 → `feature_vectors` table with all Phase 5 columns + 2 indexes.
- [ ] **5.5** `crossmatch/feature_assembler.py` — 16+ column vector per event; null-fill deferred features (dimming, HCS).
- [ ] **5.6** Quality flags (1–5) in feature_assembler.
- [ ] **5.7** `transforms/validation.py` — physical consistency checks (speed↔transit, Dst↔Bz correlations).

---

## Current Phase: Phase 4 — SHARP Features ✅ COMPLETE

## Phase 4 — SHARP Features ✅ COMPLETE

Last updated: 2026-04-07

- [x] **4.1** `clients/jsoc.py` — `JsocClient` with `_query()` ThreadPoolExecutor timeout wrapper (RULE-061); CEA series (RULE-062); `_parse_sharp_df()`: LON_FWT>60° filter (RULE-060), NOAA_AR=0→None (RULE-063), DATE__OBS normalised (RULE-064); HARP↔NOAA mapping via CCD series (RULE-063).
- [x] **4.2** `ingestion/ingest_sharps.py` — 4-context snapshot query (at_eruption/±6h/±12h/+6h); Earth-directed proxy (|lat|≤45°, |lon|≤45°); HMI start guard (2010-05-01); batch insert 5K (RULE-036); coverage fraction logged with <80% warning.
- [x] **4.3** `database/schema.py` `HarpNoaaMap` + `migrations.py` v3 — `harp_noaa_map` table with indexes on harpnum + noaa_ar; migration applied (schema_version=3).
- [x] **4.4** `ingestion/select_sharp_features.py` — `get_best_sharp_snapshot()` with at_eruption>minus_6h>minus_12h preference; `compute_sharp_coverage()` logs fraction; `iter_best_snapshots()` generator for Phase 5.
- [x] **4.5** Integration tests: `test_phase4_sharps.py` — schema existence, parse→insert flow, LON_FWT boundary filter, NOAA_AR=0 storage, coverage metric on empty DB, optimal snapshot preference validation.
- [x] Unit tests: `test_jsoc.py` — time conversion, float helpers, parse_sharp_df (6 cases), earth-directed proxy, HMI start filter, snapshot preference order.

---

## Phase 3 — Solar Wind & Indices ✅ COMPLETE

Last updated: 2026-04-07

- [x] **3.1** `clients/swpc.py` — SWPC 7-day mag+plasma JSON feeds; `_parse_swpc_table()` with RULE-071 Z-strip + RULE-072 5min guard; sentinel 99999.9→None; bz_gsm canonical (RULE-070).
- [x] **3.2** `ingestion/ingest_solar_wind.py` — 1-min→hourly averages; ACE/DSCOVR transition July 2016 (RULE-073); incremental from MAX(datetime); RULE-003 sentinel conversion.
- [x] **3.3** `clients/kyoto.py` + `ingestion/ingest_dst.py` — Cascade final>provisional>realtime (RULE-080); HTML table parser (RULE-081 post-2019 only); sentinel >500/<-500→None (RULE-082); RULE-037 empty body guard.
- [x] **3.4** `ingestion/ingest_kp.py` — GFZ incremental from MAX(datetime); RULE-010 bulk already ported.
- [x] **3.5** `ingestion/ingest_f107.py` — NOAA SWPC observed solar cycle indices JSON; monthly records stored at YYYY-MM-01.
- [x] **3.6** `database/schema.py` `SwAmbientContext` table + `migrations.py` v2 + `ingestion/ingest_sw_ambient.py` — 6h pre-CME window averages of speed/density/Bt/Bz_GSM; batch upsert (RULE-036); migration applied to staging.db (schema_version=2).
- [x] **3.7** Tests: 145 total (130 unit + 15 integration), all passing. New tests: `test_swpc.py` (parse, sentinel, hourly averaging, ACE/DSCOVR), `test_solar_wind.py` (build_row, idempotent upsert), `test_kyoto.py` (HTML parser, pre-block parser, cascade logic), `test_kp_f107.py` (GFZ parsing, F10.7 parsing, ambient context helpers).

---

### Completed This Session (Phase 1-2)

- [x] **Workspace scaffold**: `pyproject.toml`, full directory tree (`src/`, `tests/`, `configs/`, `scripts/`, `data/raw/`, `data/staging/`, `data/output/`)
- [x] **Configuration system**: `src/solarpipe_data/config.py` — Pydantic settings with YAML + env-var overrides. All 7 API sources configured. `NASA_API_KEY`, `STANFORD_JSOC_EMAIL` read from env.
- [x] **Database schema**: `src/solarpipe_data/database/schema.py` — SQLAlchemy ORM for all 15 staging tables + `make_engine()` with WAL mode event listener + `init_db()`.
- [x] **Port script**: `scripts/port_solar_data.py` — maps `solar_data.db` → `staging.db` for 9 tables (donki_cme, cdaw_cme, donki_flare, omni_hourly, symh_hourly, gfz_kp_ap, silso_daily_ssn, donki_gst, donki_ips).
- [x] **CLAUDE.md rewrite**: 30 agent rules (corrected from 26), critical rules table (20 rows), complete API reference, existing data inventory, storage architecture clarification (SQLite staging + Parquet ENLIL output).
- [x] **1.1** Pydantic settings verified: `get_settings()` loads from env; posix path normalisation confirmed.
- [x] **1.2** `database/migrations.py` — `current_version()`, `apply_pending()`, `migrate()` with MIGRATIONS registry.
- [x] **1.3** `database/queries.py` — `upsert()`, `temporal_range()`, `max_timestamp()`, `row_count()`.
- [x] **1.4** `src/solarpipe_data/cli.py` — Click harness: `fetch`, `ingest`, `crossmatch`, `build`, `validate`, `status` with `run_async` bridge.
- [x] **1.5** `clients/base.py` — `BaseClient` with token-bucket rate limiter, file cache (check-before-rate-limit), retry on 429/5xx with `Retry-After` support.
- [x] **1.6** `clients/donki.py` — DONKI client: CME, CMEAnalysis (mostAccurateOnly=true), FLR, GST, IPS, ENLIL, notifications (30d chunks only).
- [x] **1.7** `ingestion/ingest_donki_cme.py` — parse DONKI CME JSON → upsert `cme_events`; AR=0→None, linked events extracted, best analysis by level_of_data.
- [x] **1.8** Unit tests: 30 tests passing. `test_donki.py` (date chunking, rate limiter, cache, fetch methods), `test_donki_cme.py` (parsing, sentinels, upsert idempotency).

---

## Phase 2 — CME & Flare Catalogs ✅ COMPLETE

Last updated: 2026-04-07

- [x] **2.1** `clients/cdaw.py` — `CdawClient` with `fetch_month()` and `fetch_range()`; UNIVERSAL_ver2 URL; per-month HTML cache; body-length guard (RULE-037, RULE-051).
- [x] **2.2** `ingestion/ingest_cdaw_lasco.py` — BeautifulSoup HTML parser (RULE-050); Halo→None/360; footnote stripping `re.sub(r'[^0-9.\-].*$', '', cell)` (RULE-052); quality_flag from remarks; `speed_20rs_kms` canonical (RULE-053); upsert idempotent.
- [x] **2.3** `ingestion/ingest_donki_enlil.py` + `ingest_donki_gst.py` + `ingest_donki_ips.py` — ENLIL linked_cme_ids from cmeInputs; GST kp_index_max computed; IPS instruments JSON; all upsert idempotent (RULE-045).
- [x] **2.4** `clients/noaa_indices.py` — NOAA SWPC client: `fetch_flares_recent()` (7-day) and `fetch_flares_year(satellite, year)`. `ingestion/ingest_flares.py` — GOES + DONKI FLR parsers; class letter/magnitude split; AR=0→None; upsert idempotent.
- [x] **2.5** `ingestion/dedup_flares.py` — SQL self-join dedup: begin_time ±2 min + same AR; stamps goes_satellite onto DONKI record; deletes GOES duplicate.
- [x] **2.6** Tests: 104 total (89 unit + 15 integration), all passing. New fixtures: `cdaw_month_sample.html`, `goes_flares_sample.json`, `donki_flares_sample.json`, `donki_gst_sample.json`, `donki_ips_sample.json`, `donki_enlil_sample.json`.

### Phase 1 Infrastructure Notes

- `pyproject.toml` build backend changed from `setuptools.backends.legacy:build` → `setuptools.build_meta` (legacy backend not available on this system).
- `pytest-asyncio` installed system-wide (not in venv — no venv policy).
- `pip install -e .` required for test imports; package is `solarpipe-data` from `src/`.

### Port Existing Data (one-time, if staging.db missing)

```bash
python scripts/port_solar_data.py \
    --source /c/Users/radar/SolarPipe/solar_data.db \
    --target ./data/staging/staging.db
```

---

## Existing Data Inventory (solar_data.db → staging.db)

| Source Table | Rows | Date Range | Status |
|-------------|------|------------|--------|
| `donki_cme` | 8,037 | 2010-04-03 → 2026-04-03 | Port ready |
| `cdaw_cme` | 42,424 | 1996-01-11 → 2025-12-31 | Port ready |
| `donki_flare` | 3,207 | 2010-04-03 → 2026-04-02 | Port ready |
| `gfz_kp_ap` | 34,426 | 1932-01-01 → 2026-04-02 | Port ready |
| `omni_hourly` | 561,024 | 1963-01-01 → 2026-12-31 | Port ready |
| `symh_hourly` | — | 1981-01-01 → 2026-03-31 | Port ready |
| `silso_daily_ssn` | — | 1818-01-01 → 2026-02-28 | Port ready |
| `donki_gst` | — | — | Port ready |
| `donki_ips` | — | — | Port ready |
| `goes_xrs_flares` | — | — | Merge into `flares` (Phase 2) |
| `omni2_daily` | 23,376 | — | Available for Phase 5 features |

**Not yet ported** (Phase 5/6 use): `mag_*`, `ml_*`, `planet_state`, `solar_event*`

---

## Key Architectural Decisions Made

| Decision | Choice | Reason |
|----------|--------|--------|
| Staging DB | SQLite (SQLAlchemy) | SolarPipe C# `SqliteProvider` reads it directly |
| Output DB | SQLite `cme_catalog.db` | Same — consumed by `SqliteProvider` |
| Synthetic ensemble | Parquet via PyArrow | C# `ParquetProvider` + ParquetSharp reads it; columnar/vectorized |
| HTML scraping | BeautifulSoup (never pandas.read_html) | pandas misparsed CDAW `<table>` |
| DONKI chunking | None except `notifications` | Only `notifications` endpoint has 30-day limit |
| Python runtime | System Python 3.12, no venv | User preference |
| Env vars | Already set in shell | `NASA_API_KEY`, `STANFORD_JSOC_EMAIL`, etc. |
| sunpy + drms | Already installed (7.1.1 / 0.9.0) | No install step needed for JSOC |

---

## Phase Roadmap

| Phase | Focus | Key Deliverable | Status |
|-------|-------|-----------------|--------|
| 1 | Foundation | staging.db seeded, CLI + BaseClient + DONKI client | Complete ✅ |
| 2 | CME & Flare Catalogs | CDAW, GOES flares, DONKI ancillary | Complete ✅ |
| 3 | Solar Wind & Indices | SWPC, Kyoto Dst, Kp, F10.7 (incremental only — bulk ported) | Complete ✅ |
| 4 | SHARP Features | JSOC DRMS client, 18 keywords, disk-passage filter | Complete ✅ |
| 5 | Cross-Matching | CME↔Flare, CME↔ICME, feature assembly, quality flags | In progress |
| 6 | Synthetic & Export | ENLIL emulator, Parquet export, cme_catalog.db, validation | Not started |
