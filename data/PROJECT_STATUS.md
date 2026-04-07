# PROJECT_STATUS.md — SolarPipe Data Acquisition Pipeline

Last updated: 2026-04-07

## Current Phase: Phase 1 — Foundation

### Completed This Session

- [x] **Workspace scaffold**: `pyproject.toml`, full directory tree (`src/`, `tests/`, `configs/`, `scripts/`, `data/raw/`, `data/staging/`, `data/output/`)
- [x] **Configuration system**: `src/solarpipe_data/config.py` — Pydantic settings with YAML + env-var overrides. All 7 API sources configured. `NASA_API_KEY`, `STANFORD_JSOC_EMAIL` read from env.
- [x] **Database schema**: `src/solarpipe_data/database/schema.py` — SQLAlchemy ORM for all 15 staging tables + `make_engine()` with WAL mode event listener + `init_db()`.
- [x] **Port script**: `scripts/port_solar_data.py` — maps `solar_data.db` → `staging.db` for 9 tables (donki_cme, cdaw_cme, donki_flare, omni_hourly, symh_hourly, gfz_kp_ap, silso_daily_ssn, donki_gst, donki_ips).
- [x] **CLAUDE.md rewrite**: 30 agent rules (corrected from 26), critical rules table (20 rows), complete API reference, existing data inventory, storage architecture clarification (SQLite staging + Parquet ENLIL output).

### Port Existing Data (one-time, if staging.db missing)

```bash
python scripts/port_solar_data.py \
    --source /c/Users/radar/SolarPipe/solar_data.db \
    --target ./data/staging/staging.db
```

### Pending Phase 1 Tasks

- [ ] **1.1** Verify Pydantic settings load correctly (`python -c "from solarpipe_data.config import get_settings; print(get_settings())"`)
- [ ] **1.2** `database/migrations.py` — version tracking + column-add migration helpers
- [ ] **1.3** `database/queries.py` — common query patterns (upsert helpers, temporal range queries)
- [ ] **1.4** `src/solarpipe_data/cli.py` — Click harness: `fetch`, `ingest`, `crossmatch`, `build`, `validate`, `status`
- [ ] **1.5** `clients/base.py` — BaseClient: httpx.AsyncClient, token-bucket rate limiter, retry with backoff, file cache
- [ ] **1.6** `clients/donki.py` — NASA DONKI client (CME, CMEAnalysis, FLR, GST, IPS, ENLIL, HSS, SEP)
- [ ] **1.7** `ingestion/ingest_donki_cme.py` — parse DONKI JSON → upsert cme_events by activity_id
- [ ] **1.8** Unit tests: `tests/unit/test_clients/test_donki.py`, `tests/unit/test_ingestion/test_donki_cme.py`

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
| 1 | Foundation | staging.db seeded, CLI + BaseClient + DONKI client | In Progress |
| 2 | CME & Flare Catalogs | CDAW, GOES flares, DONKI ancillary | Not started |
| 3 | Solar Wind & Indices | SWPC, Kyoto Dst, Kp, F10.7 (incremental only — bulk ported) | Not started |
| 4 | SHARP Features | JSOC DRMS client, 18 keywords, disk-passage filter | Not started |
| 5 | Cross-Matching | CME↔Flare, CME↔ICME, feature assembly, quality flags | Not started |
| 6 | Synthetic & Export | ENLIL emulator, Parquet export, cme_catalog.db, validation | Not started |
