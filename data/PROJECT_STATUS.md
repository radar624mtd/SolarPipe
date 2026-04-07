# PROJECT_STATUS.md — SolarPipe Data Acquisition Pipeline

Last updated: 2026-04-07

## Current Phase: Phase 2 — CME & Flare Catalogs ✅ COMPLETE

### Completed This Session

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
| 3 | Solar Wind & Indices | SWPC, Kyoto Dst, Kp, F10.7 (incremental only — bulk ported) | Not started |
| 4 | SHARP Features | JSOC DRMS client, 18 keywords, disk-passage filter | Not started |
| 5 | Cross-Matching | CME↔Flare, CME↔ICME, feature assembly, quality flags | Not started |
| 6 | Synthetic & Export | ENLIL emulator, Parquet export, cme_catalog.db, validation | Not started |
