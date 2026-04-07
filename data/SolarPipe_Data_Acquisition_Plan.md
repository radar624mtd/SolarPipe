# SOLARPIPE DATA ACQUISITION & PREPARATION

## Automated Pipeline for Space Weather Training Data

### Complete Architecture & Implementation Plan

**.NET 8 / Python — Data Ingestion, Cross-Matching, and Database Population**

**For CME Flux Rope Rotation Prediction & Geomagnetic Storm Forecasting**

**Version 1.0 — April 2026**

**Companion to: SolarPipe_Architecture_Plan.docx**

---

## Table of Contents

1. Executive Summary
2. Data Source Inventory
3. System Architecture
4. Phase 1: Foundation — Core Ingestion Framework (Weeks 1–3)
5. Phase 2: CME & Flare Catalogs (Weeks 4–6)
6. Phase 3: In-Situ Solar Wind & Geomagnetic Indices (Weeks 7–9)
7. Phase 4: SDO/HMI SHARP Magnetic Field Features (Weeks 10–12)
8. Phase 5: Cross-Matching & Feature Assembly (Weeks 13–15)
9. Phase 6: Synthetic Data Generation & Validation (Weeks 16–18)
10. Solution Structure
11. API Reference & Endpoints
12. Database Schema
13. Data Quality & Known Issues
14. Implementation Checklist
15. Appendix: Claude Code Workspace Configuration

---

## 1. Executive Summary

The SolarPipe framework (all 4 phases complete, 282 tests passing) provides the ML pipeline orchestration engine. What it lacks is data. The reference pipeline `flux_rope_propagation_v1` defined in Section 11 of the architecture plan requires a populated `cme_catalog.db` SQLite database containing cross-matched CME events with 16–20 features per event, observational target variables (rotation angle, Bz at Earth), and a synthetic ENLIL simulation ensemble in Parquet format.

This plan specifies a fully automated data acquisition and preparation pipeline that:

- Ingests CME event catalogs from NASA DONKI and CDAW/LASCO APIs
- Downloads solar flare associations from NOAA/GOES
- Retrieves SDO/HMI SHARP magnetic field parameters from Stanford JSOC
- Collects in-situ solar wind measurements from NOAA SWPC (DSCOVR/ACE at L1)
- Downloads geomagnetic indices (Dst, Kp) from Kyoto WDC and NOAA
- Cross-matches events across catalogs to build the unified feature table
- Generates synthetic training data via physics-based Monte Carlo simulation
- Populates the `cme_catalog.db` SQLite database in the schema expected by SolarPipe

The pipeline is implemented as a Python project in `C:\Users\radar\SolarPipe\data` with CLI tooling, rate-limited API clients, incremental caching, and full provenance tracking. It is designed for the same caliber of Claude Code automated development as the primary framework — every task is atomic, testable, and independently verifiable.

### 1.1 Design Principles

- **Idempotent ingestion:** Every data source can be re-fetched without duplicating records. Upsert semantics throughout.
- **Incremental by default:** Each fetcher tracks its last-fetched timestamp. Subsequent runs pull only new/updated data.
- **Rate-limit respectful:** All API clients honor published rate limits. NASA DONKI uses `DEMO_KEY` by default (30 req/hr, 50 req/day) with support for registered API keys (1000 req/hr).
- **Offline-capable:** Raw API responses are cached as JSON in `data/raw/`. The pipeline can rebuild the database entirely from cached files without network access.
- **Provenance-tracked:** Every record in the final database includes `source_catalog`, `fetch_timestamp`, and `data_version` fields.
- **Schema-first:** The SQLite schema is defined up front and enforced by migration scripts. All ingestion writes through a typed ORM layer.

### 1.2 Workspace Location

```
C:\Users\radar\SolarPipe\data\
```

This directory sits alongside the main SolarPipe .NET solution and is referenced by the framework's pipeline configurations via relative paths (e.g., `connection: "Data Source=./data/cme_catalog.db"`).

---

## 2. Data Source Inventory

### 2.1 Primary Sources

| Source | Provider | Coverage | Format | Rate Limit | SolarPipe Role |
|--------|----------|----------|--------|------------|----------------|
| DONKI CME Catalog | NASA CCMC | 2010–present | JSON REST | 1000/hr (API key) | CME events, speeds, half-angles, linked flares, ENLIL runs |
| DONKI CME Analysis | NASA CCMC | 2010–present | JSON REST | Same | Detailed cone model parameters per CME |
| DONKI WSA-ENLIL Sims | NASA CCMC | 2011–present | JSON REST | Same | Predicted arrival times, Kp estimates |
| DONKI Geomagnetic Storms | NASA CCMC | 2010–present | JSON REST | Same | Storm onset times, Kp index linkage |
| DONKI Interplanetary Shocks | NASA CCMC | 2010–present | JSON REST | Same | ICME shock arrival times at L1 |
| CDAW LASCO CME Catalog | NASA GSFC | 1996–present | HTML tables | None (scrape) | Linear/quadratic speeds, angular widths, mass estimates, position angles |
| CDAW Halo CME Catalog | NASA GSFC | 1996–present | HTML tables | None (scrape) | Space speeds, source locations, flare associations |
| GOES Solar Flare List | NOAA SWPC | 1975–present | JSON/text | None | Flare class, peak time, location, duration |
| SDO/HMI SHARPs | Stanford JSOC | 2010.05–present | DRMS/JSON | Courtesy (no hard limit) | 18 magnetic field keywords per active region per 12 min |
| SWPC Real-Time Solar Wind | NOAA DSCOVR/ACE | 1998–present | JSON | None | Bt, Bz_GSM, speed, density, temperature at L1 |
| SWPC Solar Wind (Archival) | NOAA NCEI | 1998–present | CSV/text | None | Historical 1-min and 1-hr averaged L1 data |
| Dst Index | Kyoto WDC | 1957–present | HTML/text | Courtesy | Hourly Dst for storm intensity ground truth |
| Kp Index | GFZ Potsdam / NOAA | 1932–present | JSON/text | None | 3-hourly Kp for geomagnetic activity level |
| F10.7 Solar Flux | NRCan / NOAA | 1947–present | text | None | Daily solar radio flux (solar cycle proxy) |

### 2.2 Derived / Synthetic Sources

| Source | Method | Output | SolarPipe Role |
|--------|--------|--------|----------------|
| ENLIL Ensemble (Synthetic) | Monte Carlo parameter sampling + drag model | Parquet | Mock data for residual calibration (5000+ events) |
| WSA/PFSS HCS Parameters | GONG synoptic maps via SunPy | Per-event float | HCS tilt angle, HCS distance features |
| Coronal Hole Maps | SDO/AIA 193Å + SPoCA algorithm | Per-event float | Coronal hole proximity, polarity features |

### 2.3 Event Counts (Estimated)

| Dataset | Estimated Records | Notes |
|---------|-------------------|-------|
| DONKI CMEs (all) | ~35,000+ | Since 2010; many are faint/narrow |
| DONKI CMEs (Earth-directed, quality ≥ 3) | ~800–1,200 | Filtered for geo-effectiveness |
| CMEs with in-situ counterpart at L1 | ~400–600 | The cross-matched training set |
| CMEs with full feature vector (all 16+ features) | ~300–500 | The usable training set for the RF model |
| CDAW Halo CMEs | ~700 | Since 1996; best-measured Earth-directed events |
| SHARP active regions | ~3,000+ HARPs | Since 2010 May; 12-min cadence keywords |
| Synthetic ENLIL events | 5,000–10,000 | Generated by Monte Carlo; tunable |

---

## 3. System Architecture

### 3.1 Directory Layout

```
C:\Users\radar\SolarPipe\data\
│
├── pyproject.toml                    # Project config, dependencies
├── README.md                         # Quick start
├── MEMORY.md                         # Claude Code memory file
├── PROJECT_STATUS.md                 # Phase tracking (same pattern as main project)
│
├── src/
│   └── solarpipe_data/
│       ├── __init__.py
│       ├── cli.py                    # Click-based CLI entry point
│       ├── config.py                 # Settings, API keys, paths
│       │
│       ├── clients/                  # API client modules (one per source)
│       │   ├── __init__.py
│       │   ├── base.py              # BaseClient with rate limiting, caching, retry
│       │   ├── donki.py             # NASA DONKI (CME, CMEAnalysis, GST, IPS, FLR, ENLIL)
│       │   ├── cdaw.py              # CDAW LASCO catalog scraper
│       │   ├── swpc.py              # NOAA SWPC solar wind JSON endpoints
│       │   ├── jsoc.py              # Stanford JSOC DRMS client for SHARPs
│       │   ├── kyoto.py             # Kyoto WDC Dst index
│       │   ├── noaa_indices.py      # Kp, F10.7, GOES flares
│       │   └── ncei.py              # NOAA NCEI archival solar wind
│       │
│       ├── ingestion/               # Raw data → staging tables
│       │   ├── __init__.py
│       │   ├── ingest_donki_cme.py
│       │   ├── ingest_donki_enlil.py
│       │   ├── ingest_cdaw_lasco.py
│       │   ├── ingest_flares.py
│       │   ├── ingest_solar_wind.py
│       │   ├── ingest_sharps.py
│       │   ├── ingest_dst.py
│       │   ├── ingest_kp.py
│       │   └── ingest_f107.py
│       │
│       ├── crossmatch/              # Event cross-matching logic
│       │   ├── __init__.py
│       │   ├── cme_flare_matcher.py       # CME ↔ Flare association
│       │   ├── cme_icme_matcher.py        # CME ↔ ICME arrival at L1
│       │   ├── cme_sharp_matcher.py       # CME ↔ SHARP active region
│       │   ├── storm_matcher.py           # ICME arrival ↔ Dst storm
│       │   └── feature_assembler.py       # Build final 16-col feature vector
│       │
│       ├── synthetic/               # Mock data generation
│       │   ├── __init__.py
│       │   ├── drag_model.py        # Drag-based CME propagation
│       │   ├── monte_carlo.py       # Parameter sampling engine
│       │   ├── rotation_model.py    # Simplified flux rope rotation physics
│       │   └── enlil_emulator.py    # Statistical ENLIL surrogate
│       │
│       ├── database/                # Schema, migrations, ORM
│       │   ├── __init__.py
│       │   ├── schema.py            # SQLAlchemy models
│       │   ├── migrations.py        # Schema versioning
│       │   └── queries.py           # Common query patterns
│       │
│       ├── transforms/              # Data cleaning & feature engineering
│       │   ├── __init__.py
│       │   ├── cleaning.py          # Outlier detection, missing value handling
│       │   ├── features.py          # Derived features (coupling functions, etc.)
│       │   └── validation.py        # Schema validation, range checks
│       │
│       └── export/                  # Output to SolarPipe-consumable formats
│           ├── __init__.py
│           ├── sqlite_export.py     # Final cme_catalog.db assembly
│           └── parquet_export.py    # ENLIL ensemble Parquet files
│
├── data/
│   ├── raw/                         # Cached API responses (JSON, HTML)
│   │   ├── donki/
│   │   ├── cdaw/
│   │   ├── swpc/
│   │   ├── jsoc/
│   │   ├── kyoto/
│   │   └── noaa/
│   ├── staging/                     # Intermediate SQLite (staging.db)
│   └── output/                      # Final deliverables
│       ├── cme_catalog.db           # THE database SolarPipe consumes
│       └── enlil_runs/              # Synthetic Parquet files
│           └── *.parquet
│
├── tests/
│   ├── unit/
│   │   ├── test_clients/
│   │   ├── test_ingestion/
│   │   ├── test_crossmatch/
│   │   ├── test_synthetic/
│   │   └── test_transforms/
│   ├── integration/
│   │   ├── test_donki_live.py       # Requires network
│   │   ├── test_full_pipeline.py
│   │   └── test_database.py
│   └── fixtures/                    # Sample API responses for offline testing
│       ├── donki_cme_sample.json
│       ├── cdaw_monthly_sample.html
│       ├── swpc_plasma_sample.json
│       └── sharp_keywords_sample.json
│
├── configs/
│   ├── default.yaml                 # Default configuration
│   ├── fetch_schedule.yaml          # Incremental fetch windows
│   └── crossmatch_rules.yaml       # Matching tolerances
│
├── scripts/
│   ├── bootstrap.py                 # First-time setup
│   ├── full_rebuild.py              # Rebuild DB from cached raw data
│   └── validate_db.py              # Verify DB integrity against SolarPipe schema
│
├── docs/
│   └── adr/                         # Architecture Decision Records
│       ├── ADR-D001-sqlite-staging.md
│       ├── ADR-D002-rate-limiting.md
│       └── ADR-D003-crossmatch-windows.md
│
└── .claude/
    └── settings.json                # Claude Code workspace configuration
```

### 3.2 Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | Python 3.12+ | API client ecosystem, SunPy/drms, pandas |
| Database | SQLite (via SQLAlchemy) | Matches SolarPipe's `SqliteProvider`; zero-deploy |
| HTTP Client | `httpx` | Async support, connection pooling, retry, rate limiting |
| HTML Scraping | `beautifulsoup4` + `lxml` | CDAW catalog HTML table extraction |
| Solar Data | `sunpy`, `drms`, `astropy` | JSOC SHARP queries, coordinate transforms |
| Data Processing | `pandas`, `numpy` | Tabular manipulation, feature engineering |
| Parquet I/O | `pyarrow` | Columnar storage for synthetic ENLIL data |
| ODE Solver | `scipy.integrate` | Drag-based model, Burton ODE for synthetic data |
| CLI | `click` | Command-line interface for all operations |
| Testing | `pytest`, `pytest-httpx` | Unit + integration tests with HTTP mocking |
| Config | `pyyaml`, `pydantic` | Typed settings with validation |

### 3.3 Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  EXTERNAL APIs                                                   │
│  DONKI · CDAW · SWPC · JSOC · Kyoto · NOAA                     │
└──────────────┬──────────────────────────────────────────────────┘
               │ HTTP/JSON/HTML
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  CLIENTS  (src/solarpipe_data/clients/)                          │
│  Rate-limited, cached, idempotent                                │
│  Output: JSON files in data/raw/                                 │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  INGESTION  (src/solarpipe_data/ingestion/)                      │
│  Parse raw → normalize → upsert into staging.db                  │
│  One ingestion module per source                                 │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGING DATABASE  (data/staging/staging.db)                     │
│  Normalized tables: cme_events, flares, solar_wind_hourly,       │
│  sharp_keywords, dst_hourly, kp_3hr, f107_daily, enlil_sims     │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  CROSS-MATCHING  (src/solarpipe_data/crossmatch/)                │
│  Temporal + spatial matching across catalogs                     │
│  CME↔Flare, CME↔ICME, CME↔SHARP, ICME↔Storm                    │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  FEATURE ASSEMBLY + TRANSFORMS                                   │
│  Build 16+ column feature vector per event                       │
│  Derived features: coupling functions, physics parameters        │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT                                                          │
│  data/output/cme_catalog.db  ← SolarPipe reads this             │
│  data/output/enlil_runs/*.parquet  ← Synthetic ensemble          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Phase 1: Foundation — Core Ingestion Framework (Weeks 1–3)

### 4.1 Objective

Stand up the project skeleton, database schema, base HTTP client with rate limiting and caching, and the CLI harness. By the end of Phase 1, `solarpipe-data fetch --source donki-cme --start 2024-01-01 --end 2024-01-31` fetches data and writes it to the staging database.

### 4.2 Tasks

**Task 1.1: Project scaffolding**
- Initialize `pyproject.toml` with all dependencies
- Create directory structure per Section 3.1
- Set up `pytest` configuration with markers (`unit`, `integration`, `live`)
- Create `MEMORY.md` and `PROJECT_STATUS.md`

**Task 1.2: Configuration system**
- Implement `config.py` with Pydantic settings model
- Support `default.yaml` config file + environment variable overrides
- Settings: `NASA_API_KEY`, `DATA_DIR`, `STAGING_DB_PATH`, `OUTPUT_DB_PATH`, rate limit parameters
- Defaults: `DEMO_KEY` for NASA, `data/` relative paths

**Task 1.3: Base HTTP client**
- Implement `clients/base.py` with `BaseClient` class
- Features: `httpx.AsyncClient` with connection pooling
- Rate limiting: token bucket per-domain (configurable req/sec, burst)
- Retry: exponential backoff on 429, 500, 502, 503, 504
- Caching: write raw responses to `data/raw/{source}/{date_range}.json`
- Cache-hit detection: skip network call if cached file exists and `--force` not set
- Logging: structured logging with source, URL, status code, cache hit/miss

**Task 1.4: Database schema (staging)**
- Implement `database/schema.py` with SQLAlchemy ORM models
- Core tables: `cme_events`, `cme_analyses`, `flares`, `solar_wind_hourly`, `sharp_keywords`, `dst_hourly`, `kp_3hr`, `f107_daily`, `enlil_simulations`, `interplanetary_shocks`, `geomagnetic_storms`
- Provenance columns on every table: `source_catalog TEXT`, `fetch_timestamp TEXT`, `data_version TEXT`
- Implement `database/migrations.py` with schema versioning (simple `CREATE TABLE IF NOT EXISTS` + version table)

**Task 1.5: CLI harness**
- Implement `cli.py` using Click
- Commands: `fetch`, `ingest`, `crossmatch`, `build`, `validate`, `status`
- `fetch` subcommands: `--source {donki-cme, donki-enlil, cdaw, swpc, jsoc, kyoto, noaa}`
- Global options: `--start`, `--end`, `--force` (ignore cache), `--dry-run`, `--verbose`

**Task 1.6: DONKI CME client**
- Implement `clients/donki.py` with methods for all DONKI endpoints
- `fetch_cme(start_date, end_date)` → calls `https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CME`
- `fetch_cme_analysis(start_date, end_date, most_accurate_only=True)` → `/CMEAnalysis`
- `fetch_gst(start_date, end_date)` → `/GST`
- `fetch_ips(start_date, end_date)` → `/IPS`
- `fetch_flr(start_date, end_date)` → `/FLR`
- `fetch_enlil(start_date, end_date)` → `/WSAEnlilSimulations`
- Date windowing: auto-chunk requests into 30-day windows (DONKI default limit)

**Task 1.7: DONKI CME ingestion**
- Implement `ingestion/ingest_donki_cme.py`
- Parse DONKI CME JSON into `cme_events` table
- Parse nested `cmeAnalyses` into `cme_analyses` table
- Parse `linkedEvents` to extract flare/IPS associations
- Handle known DONKI data quality issues (missing NOAA AR numbers, duplicate entries)
- Upsert by `activity_id` (DONKI's unique event identifier)

**Task 1.8: Unit tests for Phase 1**
- Test rate limiter behavior (token bucket, backoff)
- Test cache hit/miss logic
- Test DONKI JSON parsing with fixture files
- Test database upsert idempotency
- Test CLI argument parsing

### 4.3 Deliverable

`solarpipe-data fetch donki-cme --start 2024-01-01 --end 2024-12-31` downloads all 2024 CME data from DONKI, caches the raw JSON, and populates the `cme_events` and `cme_analyses` tables in `staging.db`.

---

## 5. Phase 2: CME & Flare Catalogs (Weeks 4–6)

### 5.1 Objective

Ingest all CME and flare catalog sources: CDAW LASCO (including halo CMEs), DONKI flares, DONKI geomagnetic storms, DONKI interplanetary shocks, DONKI ENLIL simulations, and NOAA GOES flare list. By end of phase, all CME-related staging tables are populated for 2010–present.

### 5.2 Tasks

**Task 2.1: CDAW LASCO catalog scraper**
- Implement `clients/cdaw.py`
- Scrape monthly HTML tables from `https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL_ver2/YYYY_MM/univ_all_YYYY_MM.html`
- Extract: date/time, CPA, angular width, linear speed, quadratic speed at 20 R☉, acceleration, mass, kinetic energy, MPA, remarks
- Scrape halo CME catalog from `https://cdaw.gsfc.nasa.gov/CME_list/halo/halo.html`
- Extract: space speed, source location, flare class, flare onset, remarks
- Handle: data gaps (SOHO vacation 1998–99), "Halo (S/BA/OA)" classification, "Poor Event" flags, missing mass values ("---")

**Task 2.2: CDAW ingestion**
- Implement `ingestion/ingest_cdaw_lasco.py`
- Parse scraped HTML into `cdaw_cme_events` staging table
- Columns: `first_c2_time`, `cpa_deg`, `angular_width_deg`, `linear_speed_kms`, `speed_20rs_kms`, `accel_ms2`, `mass_g`, `kinetic_energy_erg`, `mpa_deg`, `is_halo`, `halo_type`, `source_location`, `remarks`
- Parse halo catalog into enrichment columns (space speed, associated flare)

**Task 2.3: DONKI ancillary data ingestion**
- Implement `ingestion/ingest_donki_enlil.py` → `enlil_simulations` table
  - Fields: `simulation_id`, `model_completion_time`, `cme_inputs` (JSON), `estimated_shock_arrival`, `estimated_duration`, `kp_18`, `kp_90`, `kp_135`, `kp_180`, `is_earth_gb`, `impact_list` (JSON)
- Implement storm/shock ingestion from DONKI GST and IPS endpoints → `geomagnetic_storms`, `interplanetary_shocks` tables

**Task 2.4: NOAA GOES flare ingestion**
- Implement `clients/noaa_indices.py` with GOES flare list fetch
- Source: `https://services.swpc.noaa.gov/json/goes/primary/xray-flares-latest.json` for recent data
- Source: NCEI archived data for historical (`https://www.ncei.noaa.gov/data/goes-space-environment-monitor/`)
- Implement `ingestion/ingest_flares.py` → `flares` table
- Columns: `flare_id`, `begin_time`, `peak_time`, `end_time`, `class` (e.g., "X1.2"), `class_numeric` (float: X1.2 → 1.2e-4 W/m²), `source_location` (e.g., "N10E33"), `noaa_ar_number`, `duration_minutes`

**Task 2.5: DONKI flare ingestion**
- Fetch from DONKI FLR endpoint (overlaps with GOES but includes DONKI-specific linkages)
- Merge into `flares` table with source tagging
- Resolve conflicts: prefer GOES for timing, DONKI for event linkages

**Task 2.6: Integration tests — catalog completeness**
- Verify DONKI CME count for 2023 matches published statistics (~2500–3000 CMEs/year)
- Verify CDAW halo CME count for 2010–2023 in expected range (~40–60/year)
- Verify flare class distribution matches known solar cycle pattern
- Cross-check: every DONKI CME with a `linkedEvents` flare should match a flare in the `flares` table

### 5.3 Deliverable

All CME and flare staging tables populated for the SDO era (2010–present). `solarpipe-data status` reports record counts per table and per year.

---

## 6. Phase 3: In-Situ Solar Wind & Geomagnetic Indices (Weeks 7–9)

### 6.1 Objective

Ingest the ground-truth and context data: L1 solar wind measurements (for ambient conditions and ICME signatures), Dst index (storm intensity), Kp index (geomagnetic activity level), and F10.7 solar flux (solar cycle phase proxy).

### 6.2 Tasks

**Task 3.1: SWPC solar wind client**
- Implement `clients/swpc.py`
- Real-time endpoints (last 7 days):
  - `https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json` → Bt, Bx/By/Bz GSM, lon, lat
  - `https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json` → density, speed, temperature
- Archival data: NCEI bulk downloads for 1998–present
- Handle spacecraft transitions (ACE → DSCOVR primary, July 2016)

**Task 3.2: Solar wind ingestion**
- Implement `ingestion/ingest_solar_wind.py` → `solar_wind_hourly` table
- Hourly averages from 1-minute data (match Dst cadence)
- Columns: `timestamp`, `bt_nT`, `bx_gsm_nT`, `by_gsm_nT`, `bz_gsm_nT`, `speed_kms`, `density_cm3`, `temperature_K`, `spacecraft` (ACE/DSCOVR), `data_quality_flag`
- Handle data gaps (flag, don't interpolate — let the model handle missingness)

**Task 3.3: Kyoto Dst client & ingestion**
- Implement `clients/kyoto.py`
- Fetch from `https://wdc.kugi.kyoto-u.ac.jp/dst_final/index.html` (final), `dst_provisional/` (provisional), `dst_realtime/` (real-time)
- Parse HTML tables (Kyoto uses fixed-width HTML format, not JSON)
- Implement `ingestion/ingest_dst.py` → `dst_hourly` table
- Columns: `timestamp`, `dst_nT`, `data_type` (final/provisional/realtime)
- Preference cascade: final > provisional > realtime

**Task 3.4: Kp index ingestion**
- Source: GFZ Potsdam via `https://kp.gfz-potsdam.de/app/json/` or NOAA SWPC
- Implement `ingestion/ingest_kp.py` → `kp_3hr` table
- Columns: `timestamp`, `kp_value` (0.0–9.0), `ap_value`, `data_type`

**Task 3.5: F10.7 solar flux ingestion**
- Source: NRCan Penticton / NOAA SWPC `https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json`
- Implement `ingestion/ingest_f107.py` → `f107_daily` table
- Columns: `date`, `f107_observed_sfu`, `f107_adjusted_sfu`, `ssn` (sunspot number)

**Task 3.6: Ambient solar wind context extraction**
- Add utility: given a CME launch time, extract the "ambient" L1 solar wind state
- Logic: average the 6-hour window ending 1 hour before expected ICME arrival
- Produces: `sw_speed_ambient`, `sw_density_ambient`, `sw_bt_ambient` features for each CME event
- Handle: if no ICME arrival is known, use expected transit time from drag model estimate

**Task 3.7: Unit tests for time-series ingestion**
- Test Dst HTML parsing with fixture
- Test solar wind gap detection
- Test ambient extraction window logic
- Test timezone handling (all data stored UTC)

### 6.3 Deliverable

`solar_wind_hourly`, `dst_hourly`, `kp_3hr`, and `f107_daily` tables populated from 2010 to present. `solarpipe-data status` shows temporal coverage maps.

---

## 7. Phase 4: SDO/HMI SHARP Magnetic Field Features (Weeks 10–12)

### 7.1 Objective

Retrieve SHARP keywords from Stanford JSOC for every active region associated with a CME in our catalog. These 18 magnetic field parameters characterize the source region's complexity and stored energy — critical features for flare/CME prediction.

### 7.2 Tasks

**Task 4.1: JSOC DRMS client**
- Implement `clients/jsoc.py` using the `drms` Python package
- Query pattern: `hmi.sharp_cea_720s[HARPNUM][T_REC]`
- Keyword extraction (no image download needed): query keywords only via `drms.Client().query()`
- Key SHARP keywords to extract:
  - `USFLUX` — total unsigned flux (Mx)
  - `MEANGAM` — mean inclination angle (degrees)
  - `MEANGBT` — mean gradient of total field (G/Mm)
  - `MEANGBZ` — mean gradient of vertical field (G/Mm)
  - `MEANGBH` — mean gradient of horizontal field (G/Mm)
  - `MEANJZD` — mean vertical current density (mA/m²)
  - `TOTUSJZ` — total unsigned vertical current (A)
  - `MEANALP` — mean twist parameter α (1/Mm)
  - `MEANJZH` — mean current helicity (G²/m)
  - `TOTUSJH` — total unsigned current helicity (G²/m)
  - `ABSNJZH` — absolute value of net current helicity (G²/m)
  - `SAVNCPP` — sum of absolute value of net current per polarity (A)
  - `MEANPOT` — mean photospheric excess magnetic energy density (erg/cm³)
  - `TOTPOT` — total photospheric magnetic free energy density (erg/cm³)
  - `MEANSHR` — mean shear angle (degrees)
  - `SHRGT45` — fraction of area with shear > 45° (dimensionless)
  - `R_VALUE` — sum of flux near polarity inversion line (Mx)
  - `AREA_ACR` — area of strong field pixels (μHemispheres)
- Also extract: `NOAA_AR`, `HARPNUM`, `LAT_FWT`, `LON_FWT`, `T_REC`

**Task 4.2: HARP↔NOAA AR mapping**
- Build lookup table: NOAA AR number → HARPNUM
- Source: query `hmi.sharp_720s[][TIMERANGE]{HARPNUM,NOAA_AR,NOAA_ARS}` for full mission
- Handle: one-to-many mappings (multiple NOAAs in one HARP, vice versa)

**Task 4.3: SHARP ingestion**
- Implement `ingestion/ingest_sharps.py` → `sharp_keywords` table
- For each CME with an associated NOAA AR:
  1. Look up HARPNUM from NOAA AR
  2. Query SHARP keywords at the time closest to CME launch
  3. Also query at -12h, -6h, 0h, +6h relative to flare peak for temporal context
- Store per-record: `harpnum`, `noaa_ar`, `t_rec`, all 18+ keywords, `query_context` (at_eruption, minus_6h, etc.)

**Task 4.4: SHARP feature selection for CME events**
- For each CME event, select the SHARP snapshot closest to eruption onset
- If multiple SHARPs: pick the one with `LON_FWT` closest to disk center (best measurement)
- Add to feature vector: selected SHARP keywords as source region descriptors

**Task 4.5: Integration test — SHARP coverage**
- Verify: fraction of Earth-directed CMEs with SHARP data available (~80–90% post-2010)
- Verify: SHARP keyword values are physically reasonable (USFLUX > 0, MEANSHR in [0, 90], etc.)
- Verify: HARPNUM↔NOAA mapping covers >95% of known ARs

### 7.3 Deliverable

`sharp_keywords` table populated for all CME-associated active regions. The HARPNUM↔NOAA lookup is built. Feature selection logic picks the optimal SHARP snapshot per CME event.

---

## 8. Phase 5: Cross-Matching & Feature Assembly (Weeks 13–15)

### 8.1 Objective

Link records across all staging tables to build the final, unified training dataset. Each row represents one CME event with its full 16+ feature vector and target variables. This is where the data becomes useful to SolarPipe.

### 8.2 Cross-Matching Strategy

Cross-matching heliophysics catalogs is notoriously messy. Different catalogs use different event identifiers, different time conventions, and have different coverage. The matching is done in stages, from most reliable to least:

1. **DONKI internal links** (highest confidence): DONKI's `linkedEvents` field already associates CMEs with flares, shocks, and storms. Use these as ground truth where available.
2. **Temporal + spatial matching** (medium confidence): For CDAW CMEs not in DONKI, match by time window + position angle agreement.
3. **Statistical matching** (lower confidence): For events where spatial information is ambiguous, use speed/energy consistency as secondary criteria.

### 8.3 Tasks

**Task 5.1: CME ↔ Flare association**
- Implement `crossmatch/cme_flare_matcher.py`
- Primary: use DONKI `linkedEvents` where `activityID` contains "FLR"
- Secondary: temporal match within ±30 minutes of CME first appearance, spatial match within ±15° heliographic latitude/longitude
- Output: `cme_flare_links` table with `cme_id`, `flare_id`, `match_confidence`, `match_method`

**Task 5.2: CME ↔ ICME arrival at L1**
- Implement `crossmatch/cme_icme_matcher.py`
- Primary: DONKI IPS (interplanetary shocks) linked to CMEs
- Secondary: match CME launch time + estimated transit time (from speed) against observed shocks in solar wind data
  - Transit time estimate: `t_transit ≈ 1 AU / v_cme` with drag correction
  - Matching window: ±12 hours around estimated arrival
- Identify ICME boundaries in L1 data: elevated Bt, depressed Bz, enhanced speed, decreased density
- Output: `cme_icme_links` table with `cme_id`, `shock_arrival_time`, `icme_start`, `icme_end`, `observed_bz_min`, `observed_bt_max`, `match_confidence`

**Task 5.3: CME ↔ SHARP source region**
- Implement `crossmatch/cme_sharp_matcher.py`
- Match via NOAA AR number (both DONKI CMEs and SHARPs carry this)
- For CDAW CMEs without NOAA AR: use source location + time to find nearest HARP
- Output: `cme_sharp_links` table

**Task 5.4: ICME arrival ↔ Geomagnetic storm**
- Implement `crossmatch/storm_matcher.py`
- Match ICME arrival at L1 against Dst minimum within 24–48 hours
- Extract: `dst_min_nT`, `storm_onset_time`, `storm_duration_hours`, `kp_max`
- Output: `icme_storm_links` table

**Task 5.5: Feature assembler**
- Implement `crossmatch/feature_assembler.py`
- For each CME event with matched ICME arrival:
  1. CME kinematic features from DONKI/CDAW: `cme_speed`, `cme_mass`, `cme_angular_width`, `cme_half_angle`
  2. Source region features from SHARP: subset of 18 keywords (top predictors from literature: `USFLUX`, `TOTPOT`, `R_VALUE`, `MEANSHR`, `TOTUSJZ`)
  3. Flare features: `flare_class_numeric`
  4. Solar imagery features: `chirality` (from SHARP helicity sign), `initial_axis_angle` (from post-eruption arcade orientation — derived), `dimming_area`, `dimming_asymmetry` (placeholder: requires AIA analysis, deferred)
  5. Environmental features: `coronal_hole_proximity`, `coronal_hole_polarity`, `hcs_tilt_angle`, `hcs_distance` (placeholder: requires WSA/PFSS, partially deferred)
  6. Ambient context: `sw_speed_ambient`, `sw_density_ambient`, `sw_bt_ambient`, `f10_7`
  7. Target variables: `observed_rotation_angle` (from in-situ flux rope fitting — requires catalog), `observed_bz_at_earth` (from L1 data)
- Output: `feature_vectors` table — one row per training example

**Task 5.6: Quality flags and filtering**
- Add `quality_flag` column (1–5 scale):
  - 5: all features present, high-confidence matches throughout
  - 4: all features present, one medium-confidence match
  - 3: 1–2 features missing (imputable), matches solid
  - 2: multiple features missing or low-confidence matches
  - 1: significant data quality concerns
- Default filter for SolarPipe: `quality_flag >= 3`

**Task 5.7: Cross-match validation**
- Implement statistical sanity checks:
  - CME speed should correlate with transit time (anti-correlation, r < -0.3)
  - Dst minimum should correlate with southward Bz (correlation, r > 0.3)
  - Faster CMEs should associate with stronger flares (positive trend)
  - Feature distributions should be physically bounded (no negative speeds, etc.)
- Report any events that violate physical consistency as candidates for manual review

### 8.4 Deliverable

`feature_vectors` table populated with ~300–500 events (quality ≥ 3). The table schema matches what `flux_rope_propagation_v1` expects. `solarpipe-data validate` confirms schema compatibility with SolarPipe.

---

## 9. Phase 6: Synthetic Data Generation & Validation (Weeks 16–18)

### 9.1 Objective

Generate the synthetic ENLIL-surrogate dataset (~5,000–10,000 events) for mock data integration, and perform final validation and export of the `cme_catalog.db` and Parquet files that SolarPipe will consume.

### 9.2 Tasks

**Task 6.1: Drag-based CME propagation model**
- Implement `synthetic/drag_model.py`
- Solves: `dv/dt = -γ(v-w)|v-w|` via `scipy.integrate.solve_ivp`
- Parameters: `cme_speed`, `cme_mass`, `sw_density`, `sw_velocity`, `drag_coefficient`
- Output: arrival time at 1 AU, speed at arrival, transit time

**Task 6.2: Simplified flux rope rotation model**
- Implement `synthetic/rotation_model.py`
- Physics: HCS alignment tendency + deflection from coronal holes
- `rotation = f(initial_angle, hcs_tilt, ch_proximity, ch_polarity, transit_time)`
- Based on published parametric models (e.g., Kay et al. 2015 ForeCAT-simplified)
- Adds Gaussian noise calibrated to observational scatter

**Task 6.3: Monte Carlo parameter sampler**
- Implement `synthetic/monte_carlo.py`
- Sample from physically-motivated distributions:
  - `cme_speed`: log-normal, μ=500 km/s, σ=0.5 (in log space)
  - `cme_mass`: log-normal, μ=1e15 g, σ=0.8
  - `angular_width`: uniform 20°–360° with halo enrichment
  - `drag_coefficient`: uniform 0.2–2.0
  - `sw_speed`: bimodal (slow ~400 km/s, fast ~700 km/s)
  - `hcs_tilt`: uniform 10°–75° (solar cycle dependent)
  - `initial_axis_angle`: uniform 0°–180°
  - Coronal hole fields: sampled from observed distributions
- Correlations: enforce known physical correlations (faster CMEs → more massive, etc.)

**Task 6.4: ENLIL emulator**
- Implement `synthetic/enlil_emulator.py`
- Wraps drag model + rotation model to produce ENLIL-like output per synthetic event
- For each sampled parameter set:
  1. Run drag model → arrival time, arrival speed
  2. Run rotation model → rotation angle at Earth
  3. Compute Bz from rotation: `Bz = B0 * sin(rotation_angle) * f(chirality)`
  4. Add systematic bias (ENLIL is known to be ~6 hours early on average) + noise
- Output: DataFrame with same columns as observational `feature_vectors` + `is_synthetic=True`

**Task 6.5: Parquet export**
- Implement `export/parquet_export.py`
- Write synthetic ensemble to `data/output/enlil_runs/enlil_ensemble_v1.parquet`
- Schema matches SolarPipe's `ParquetProvider` expectations
- Include metadata: generation timestamp, parameter distributions used, random seed

**Task 6.6: Final SQLite export**
- Implement `export/sqlite_export.py`
- Build `data/output/cme_catalog.db` from `feature_vectors` + ancillary tables
- Schema must match what `flux_rope_propagation_v1.yaml` references:
  - `cme_events` table: one row per event, all features + targets
  - `flux_rope_fits` table: in-situ fitting results (Bz, rotation, etc.)
  - `l1_arrivals` table: ICME arrival details
  - Join keys: `event_id`
- Add indexes for SolarPipe query patterns

**Task 6.7: Database validation script**
- Implement `scripts/validate_db.py`
- Checks:
  - Schema matches SolarPipe expectations (table names, column types)
  - No NaN in required columns
  - Feature distributions are physically reasonable
  - Synthetic data is in separate Parquet, not mixed into SQLite
  - Can execute the exact SQL query from `flux_rope_propagation_v1.yaml` Section 11.4:
    ```sql
    SELECT * FROM cme_events e
    JOIN flux_rope_fits f ON e.event_id = f.event_id
    JOIN l1_arrivals a ON e.event_id = a.event_id
    WHERE e.quality_flag >= 3
    AND a.has_in_situ_fit = 1
    ```

**Task 6.8: End-to-end integration test**
- Test: fetch (from cache) → ingest → crossmatch → build → validate
- Verify the full pipeline runs without error from raw data to final DB
- Verify SolarPipe can load the generated database via `SqliteProvider`

### 9.3 Deliverable

`cme_catalog.db` and `enlil_runs/*.parquet` in `data/output/`, validated against SolarPipe schema. The `flux_rope_propagation_v1` pipeline configuration can execute `solarpipe train` against real data.

---

## 10. Solution Structure

### 10.1 Dependencies (`pyproject.toml`)

```toml
[project]
name = "solarpipe-data"
version = "1.0.0"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.27",
    "beautifulsoup4>=4.12",
    "lxml>=5.0",
    "sqlalchemy>=2.0",
    "pandas>=2.2",
    "numpy>=1.26",
    "pyarrow>=15.0",
    "scipy>=1.12",
    "sunpy>=5.1",
    "drms>=0.7",
    "astropy>=6.0",
    "click>=8.1",
    "pyyaml>=6.0",
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "rich>=13.7",         # CLI progress bars and tables
    "tenacity>=8.2",      # Retry logic
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-httpx>=0.30",
    "pytest-asyncio>=0.23",
    "pytest-cov>=4.1",
]

[project.scripts]
solarpipe-data = "solarpipe_data.cli:main"
```

---

## 11. API Reference & Endpoints

### 11.1 NASA DONKI

| Endpoint | URL | Parameters |
|----------|-----|------------|
| CME | `https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CME` | `startDate`, `endDate` (YYYY-MM-DD) |
| CME Analysis | `.../CMEAnalysis` | + `mostAccurateOnly`, `speed`, `halfAngle`, `catalog` |
| Solar Flare | `.../FLR` | `startDate`, `endDate` |
| Geomagnetic Storm | `.../GST` | `startDate`, `endDate` |
| Interplanetary Shock | `.../IPS` | `startDate`, `endDate`, `location`, `catalog` |
| WSA-ENLIL | `.../WSAEnlilSimulations` | `startDate`, `endDate` |
| Notifications | `.../notifications` | `startDate`, `endDate`, `type` |

Also available via `https://api.nasa.gov/DONKI/...?api_key=KEY` (preferred for rate limits).

### 11.2 NOAA SWPC Solar Wind

| Endpoint | URL | Content |
|----------|-----|---------|
| Mag 7-day | `https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json` | time_tag, bx_gsm, by_gsm, bz_gsm, lon_gsm, lat_gsm, bt |
| Plasma 7-day | `.../plasma-7-day.json` | time_tag, density, speed, temperature |
| Mag 1-day | `.../mag-1-day.json` | Same, shorter window |
| Plasma 1-day | `.../plasma-1-day.json` | Same, shorter window |
| Ephemerides | `.../ephemerides.json` | Spacecraft position |

### 11.3 Stanford JSOC (DRMS)

| Series | Query Pattern | Content |
|--------|--------------|---------|
| SHARP keywords | `hmi.sharp_cea_720s[HARPNUM][TIME]` | 18 space weather keywords |
| SHARP NRT | `hmi.sharp_cea_720s_nrt[HARPNUM][TIME]` | Near-real-time (within 3 hrs) |
| HARP↔NOAA | `hmi.sharp_720s[][TIME]{HARPNUM,NOAA_AR}` | Mapping table |

### 11.4 Kyoto WDC Dst

| Endpoint | URL | Content |
|----------|-----|---------|
| Final Dst | `https://wdc.kugi.kyoto-u.ac.jp/dst_final/YYYYMM/index.html` | Hourly Dst (nT), HTML table |
| Provisional Dst | `.../dst_provisional/YYYYMM/index.html` | Same, recent months |
| Real-time Dst | `.../dst_realtime/YYYYMM/index.html` | Same, current month |

---

## 12. Database Schema

### 12.1 Staging Database (`staging.db`)

```sql
-- CME events from DONKI
CREATE TABLE cme_events (
    activity_id TEXT PRIMARY KEY,       -- DONKI unique ID (e.g., "2024-01-01T12:00:00-CME-001")
    start_time TEXT NOT NULL,           -- ISO 8601
    source_location TEXT,               -- e.g., "N10E33"
    active_region_num INTEGER,          -- NOAA AR number
    note TEXT,
    catalog TEXT,                       -- M2M_CATALOG, etc.
    speed_kms REAL,                     -- from linked analysis
    half_angle_deg REAL,
    latitude REAL,
    longitude REAL,
    is_earth_directed INTEGER,          -- 0/1
    linked_flare_id TEXT,
    linked_ips_ids TEXT,                -- JSON array
    linked_gst_ids TEXT,                -- JSON array
    source_catalog TEXT DEFAULT 'DONKI',
    fetch_timestamp TEXT,
    data_version TEXT
);

-- CME analysis parameters (cone model fits)
CREATE TABLE cme_analyses (
    analysis_id TEXT PRIMARY KEY,
    cme_activity_id TEXT REFERENCES cme_events(activity_id),
    time21_5 TEXT,                      -- Time at 21.5 R☉
    latitude REAL,
    longitude REAL,
    half_angle_deg REAL,
    speed_kms REAL,
    is_most_accurate INTEGER,
    level_of_data INTEGER,              -- 0=real-time, 1=near-real-time, 2=definitive
    catalog TEXT,
    source_catalog TEXT DEFAULT 'DONKI',
    fetch_timestamp TEXT
);

-- Solar flares
CREATE TABLE flares (
    flare_id TEXT PRIMARY KEY,
    begin_time TEXT,
    peak_time TEXT,
    end_time TEXT,
    class TEXT,                         -- e.g., "X1.2", "M5.0"
    class_numeric REAL,                 -- W/m² (X1.0 = 1e-4)
    source_location TEXT,
    noaa_ar_number INTEGER,
    duration_minutes REAL,
    source_catalog TEXT,                -- 'DONKI' or 'GOES'
    fetch_timestamp TEXT
);

-- CDAW LASCO CME catalog (separate from DONKI)
CREATE TABLE cdaw_cme_events (
    cdaw_id TEXT PRIMARY KEY,           -- "YYYYMMDD.HHMMSS"
    first_c2_time TEXT,
    cpa_deg REAL,                       -- Central position angle
    angular_width_deg REAL,
    linear_speed_kms REAL,
    speed_20rs_kms REAL,                -- Quadratic speed at 20 R☉
    accel_ms2 REAL,
    mass_g REAL,
    kinetic_energy_erg REAL,
    mpa_deg REAL,                       -- Measurement position angle
    is_halo INTEGER,
    halo_type TEXT,                     -- S, BA, OA
    source_location TEXT,               -- For halo CMEs
    associated_flare_class TEXT,        -- For halo CMEs
    remarks TEXT,
    source_catalog TEXT DEFAULT 'CDAW',
    fetch_timestamp TEXT
);

-- Hourly solar wind at L1
CREATE TABLE solar_wind_hourly (
    timestamp TEXT PRIMARY KEY,         -- ISO 8601, hourly
    bt_nT REAL,
    bx_gsm_nT REAL,
    by_gsm_nT REAL,
    bz_gsm_nT REAL,
    speed_kms REAL,
    density_cm3 REAL,
    temperature_K REAL,
    spacecraft TEXT,                     -- 'ACE' or 'DSCOVR'
    data_quality_flag INTEGER
);

-- SHARP magnetic field keywords
CREATE TABLE sharp_keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    harpnum INTEGER,
    noaa_ar INTEGER,
    t_rec TEXT,                         -- TAI time string
    usflux REAL,
    meangam REAL,
    meangbt REAL,
    meangbz REAL,
    meangbh REAL,
    meanjzd REAL,
    totusjz REAL,
    meanalp REAL,
    meanjzh REAL,
    totusjh REAL,
    absnjzh REAL,
    savncpp REAL,
    meanpot REAL,
    totpot REAL,
    meanshr REAL,
    shrgt45 REAL,
    r_value REAL,
    area_acr REAL,
    lat_fwt REAL,
    lon_fwt REAL,
    query_context TEXT,                 -- 'at_eruption', 'minus_6h', etc.
    fetch_timestamp TEXT
);

-- Hourly Dst index
CREATE TABLE dst_hourly (
    timestamp TEXT PRIMARY KEY,
    dst_nT REAL,
    data_type TEXT                       -- 'final', 'provisional', 'realtime'
);

-- 3-hourly Kp index
CREATE TABLE kp_3hr (
    timestamp TEXT PRIMARY KEY,
    kp_value REAL,
    ap_value REAL,
    data_type TEXT
);

-- Daily F10.7 solar flux
CREATE TABLE f107_daily (
    date TEXT PRIMARY KEY,
    f107_observed_sfu REAL,
    f107_adjusted_sfu REAL,
    ssn INTEGER                          -- International sunspot number
);

-- DONKI WSA-ENLIL simulation results
CREATE TABLE enlil_simulations (
    simulation_id TEXT PRIMARY KEY,
    model_completion_time TEXT,
    cme_inputs TEXT,                     -- JSON
    estimated_shock_arrival TEXT,
    estimated_duration_hr REAL,
    kp_18 REAL,
    kp_90 REAL,
    kp_135 REAL,
    kp_180 REAL,
    is_earth_gb INTEGER,                -- Glancing blow flag
    impact_list TEXT,                    -- JSON
    fetch_timestamp TEXT
);

-- Geomagnetic storms from DONKI
CREATE TABLE geomagnetic_storms (
    gst_id TEXT PRIMARY KEY,
    start_time TEXT,
    kp_index REAL,
    linked_cme_ids TEXT,                -- JSON array
    source_catalog TEXT DEFAULT 'DONKI',
    fetch_timestamp TEXT
);

-- Interplanetary shocks from DONKI
CREATE TABLE interplanetary_shocks (
    ips_id TEXT PRIMARY KEY,
    event_time TEXT,
    location TEXT,                       -- 'Earth', 'STEREO-A', etc.
    instruments TEXT,                    -- JSON array
    linked_cme_ids TEXT,                -- JSON array
    source_catalog TEXT DEFAULT 'DONKI',
    fetch_timestamp TEXT
);

-- Schema version tracking
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT,
    description TEXT
);
```

### 12.2 Output Database (`cme_catalog.db`)

This is the database SolarPipe reads. Its schema is dictated by the `flux_rope_propagation_v1.yaml` pipeline configuration.

```sql
-- The unified CME event table with all features
CREATE TABLE cme_events (
    event_id TEXT PRIMARY KEY,
    launch_time TEXT NOT NULL,
    source_location TEXT,
    noaa_ar INTEGER,
    -- Kinematic features (from DONKI/CDAW)
    cme_speed REAL,
    cme_mass REAL,
    cme_angular_width REAL,
    flare_class_numeric REAL,
    -- Source region features (from SHARP)
    chirality INTEGER,                   -- +1 or -1 (from helicity sign)
    initial_axis_angle REAL,
    usflux REAL,
    totpot REAL,
    r_value REAL,
    meanshr REAL,
    totusjz REAL,
    -- Environmental features
    coronal_hole_proximity REAL,
    coronal_hole_polarity TEXT,
    hcs_tilt_angle REAL,
    hcs_distance REAL,
    -- Ambient L1 context
    sw_speed_ambient REAL,
    sw_density_ambient REAL,
    sw_bt_ambient REAL,
    f10_7 REAL,
    -- Quality
    quality_flag INTEGER DEFAULT 3
);

-- In-situ flux rope fitting results (target variables)
CREATE TABLE flux_rope_fits (
    event_id TEXT PRIMARY KEY REFERENCES cme_events(event_id),
    observed_rotation_angle REAL,        -- degrees (primary target)
    observed_bz_min REAL,                -- nT at L1
    bz_polarity TEXT,                    -- 'north' or 'south'
    fit_method TEXT,                      -- 'Lundquist', 'GH', 'MVA', etc.
    fit_quality REAL,
    has_in_situ_fit INTEGER DEFAULT 1
);

-- ICME arrival details at L1
CREATE TABLE l1_arrivals (
    event_id TEXT PRIMARY KEY REFERENCES cme_events(event_id),
    shock_arrival_time TEXT,
    icme_start_time TEXT,
    icme_end_time TEXT,
    transit_time_hours REAL,
    dst_min_nT REAL,
    kp_max REAL,
    has_in_situ_fit INTEGER DEFAULT 1
);
```

---

## 13. Data Quality & Known Issues

### 13.1 DONKI Known Issues

- **Missing NOAA AR numbers**: ~10–15% of CME entries have null `activeRegionNum`. Remediation: cross-reference with GOES flare catalog and CDAW source locations.
- **Duplicate CME entries**: Multiple analyses for the same CME appear as separate entries. Use `mostAccurateOnly=true` for CME Analysis queries.
- **Inconsistent time formats**: Mix of ISO 8601 and custom formats. Normalize all to UTC ISO 8601 during ingestion.
- **Broken `linkedEvents`**: Some links reference events that don't exist in the database. Handle gracefully with null foreign keys.

### 13.2 CDAW Known Issues

- **HTML format changes**: The catalog HTML has changed format multiple times since 1996. Scraper must handle multiple table layouts.
- **"Poor Event" entries**: Low-quality CME identifications flagged in remarks. Filter or downweight.
- **Mass estimates**: Available for only ~60% of CMEs. Missing values are marked "---".
- **No machine-readable API**: All data must be scraped from HTML. Rate-limit requests to avoid overloading NASA servers.

### 13.3 SHARP Known Issues

- **Disk-passage bias**: SHARP keywords are only reliable within ~60° of disk center. Measurements near the limb have projection effects.
- **HARP splitting/merging**: Complex active regions may be tracked as separate HARPs or merged. The NOAA↔HARP mapping handles most cases but edge cases exist.
- **Data gaps**: JSOC has occasional processing gaps. Not every 12-minute cadence is available.

### 13.4 Solar Wind Known Issues

- **Spacecraft transitions**: ACE was primary through July 2016; DSCOVR thereafter. Some overlap periods have both.
- **Data quality flags**: DSCOVR has known periods of degraded data. Always check quality flags.
- **Propagation delay**: L1 data arrives ~30–60 minutes before the solar wind reaches Earth's magnetosphere.

### 13.5 Cross-Matching Challenges

- **CME-ICME matching is inherently ambiguous**: Multiple CMEs can arrive in sequence, interact, or merge. The matching window must balance false positives vs. missed associations.
- **Dimming features are research-grade**: Automated extraction of EUV dimming area and asymmetry from SDO/AIA requires specialized image processing. These features are deferred to a future phase and filled with null/placeholder values initially.
- **HCS parameters require model output**: WSA/PFSS model runs are not readily available via API. Initial implementation uses pre-computed synoptic maps from GONG where available, with null fill otherwise.

---

## 14. Implementation Checklist

### Phase 1: Foundation (Weeks 1–3)
- [ ] Task 1.1: Project scaffolding (`pyproject.toml`, directories, pytest config, MEMORY.md)
- [ ] Task 1.2: Configuration system (Pydantic settings, YAML, env vars)
- [ ] Task 1.3: Base HTTP client (rate limiting, retry, caching)
- [ ] Task 1.4: Database schema — staging (SQLAlchemy models, migrations)
- [ ] Task 1.5: CLI harness (Click commands: fetch, ingest, crossmatch, build, validate, status)
- [ ] Task 1.6: DONKI CME client (all 6 endpoints, date windowing)
- [ ] Task 1.7: DONKI CME ingestion (parse → staging, upsert by activity_id)
- [ ] Task 1.8: Unit tests (rate limiter, cache, JSON parsing, upserts, CLI)

### Phase 2: CME & Flare Catalogs (Weeks 4–6)
- [ ] Task 2.1: CDAW LASCO catalog scraper (monthly HTML, halo catalog)
- [ ] Task 2.2: CDAW ingestion (HTML → `cdaw_cme_events` table)
- [ ] Task 2.3: DONKI ancillary ingestion (ENLIL, GST, IPS)
- [ ] Task 2.4: NOAA GOES flare ingestion (JSON + archival)
- [ ] Task 2.5: DONKI flare ingestion (merge with GOES, source tagging)
- [ ] Task 2.6: Integration tests — catalog completeness

### Phase 3: In-Situ Solar Wind & Indices (Weeks 7–9)
- [ ] Task 3.1: SWPC solar wind client (real-time JSON, archival CSV)
- [ ] Task 3.2: Solar wind ingestion (1-min → hourly averages)
- [ ] Task 3.3: Kyoto Dst client & ingestion (HTML parsing, preference cascade)
- [ ] Task 3.4: Kp index ingestion
- [ ] Task 3.5: F10.7 solar flux ingestion
- [ ] Task 3.6: Ambient solar wind context extraction (6-hr window logic)
- [ ] Task 3.7: Unit tests — time-series ingestion

### Phase 4: SDO/HMI SHARP Features (Weeks 10–12)
- [ ] Task 4.1: JSOC DRMS client (drms package, keyword queries)
- [ ] Task 4.2: HARP↔NOAA AR mapping table
- [ ] Task 4.3: SHARP ingestion (18 keywords per AR per time snapshot)
- [ ] Task 4.4: SHARP feature selection (optimal snapshot per CME)
- [ ] Task 4.5: Integration test — SHARP coverage

### Phase 5: Cross-Matching & Feature Assembly (Weeks 13–15)
- [ ] Task 5.1: CME ↔ Flare association (DONKI links + temporal/spatial)
- [ ] Task 5.2: CME ↔ ICME arrival matching (DONKI IPS + transit time)
- [ ] Task 5.3: CME ↔ SHARP source region matching (NOAA AR + location)
- [ ] Task 5.4: ICME ↔ Geomagnetic storm matching (Dst within 24–48 hr)
- [ ] Task 5.5: Feature assembler (16+ column vector per event)
- [ ] Task 5.6: Quality flags and filtering (5-tier quality scale)
- [ ] Task 5.7: Cross-match validation (physical consistency checks)

### Phase 6: Synthetic Data & Final Export (Weeks 16–18)
- [ ] Task 6.1: Drag-based CME propagation model (scipy ODE solver)
- [ ] Task 6.2: Simplified flux rope rotation model (parametric)
- [ ] Task 6.3: Monte Carlo parameter sampler (physically-motivated distributions)
- [ ] Task 6.4: ENLIL emulator (drag + rotation + bias + noise)
- [ ] Task 6.5: Parquet export (synthetic ensemble → `enlil_runs/`)
- [ ] Task 6.6: Final SQLite export (`cme_catalog.db` with SolarPipe schema)
- [ ] Task 6.7: Database validation script (schema match, query test)
- [ ] Task 6.8: End-to-end integration test (fetch → build → validate)

---

## 15. Appendix: Claude Code Workspace Configuration

### 15.1 `.claude/settings.json`

```json
{
  "project_name": "SolarPipe Data Acquisition",
  "project_root": "C:\\Users\\radar\\SolarPipe\\data",
  "memory_file": "MEMORY.md",
  "status_file": "PROJECT_STATUS.md",
  "orientation_steps": [
    "Read MEMORY.md for current state",
    "Read PROJECT_STATUS.md for phase/task tracking",
    "Check Python environment: python --version, pip list",
    "Run pytest --tb=short to verify test state",
    "Check data/staging/staging.db for record counts",
    "Check data/output/cme_catalog.db existence"
  ],
  "conventions": {
    "testing": "pytest with markers: @pytest.mark.unit, @pytest.mark.integration, @pytest.mark.live",
    "typing": "Full type hints on all public functions",
    "docstrings": "Google style",
    "imports": "isort, stdlib → third-party → local",
    "async": "httpx async for all HTTP clients, sync CLI wrapper"
  },
  "blocked_actions": [
    "Never commit API keys to source control",
    "Never make live API calls in unit tests (use fixtures)",
    "Never delete data/raw/ cache files without --force flag",
    "Never modify staging.db schema without a migration"
  ]
}
```

### 15.2 `MEMORY.md` (Initial State)

```markdown
# SolarPipe Data Acquisition — Memory

## Project Location
C:\Users\radar\SolarPipe\data

## Relationship to Main Project
This is the data preparation pipeline for the SolarPipe ML framework.
The main framework is at C:\Users\radar\SolarPipe (all 4 phases complete, 282 tests passing).
This pipeline produces the data files the framework consumes:
- data/output/cme_catalog.db (SQLite, referenced by pipeline YAML configs)
- data/output/enlil_runs/*.parquet (synthetic ensemble data)

## Current State
Phase: Not started
Last completed task: N/A
Next task: Task 1.1 — Project scaffolding

## Key Decisions
- Python 3.12+ (not .NET — API client ecosystem is Python-native)
- SQLite staging + SQLite output (matches SolarPipe SqliteProvider)
- httpx async client with token-bucket rate limiting
- All raw API responses cached in data/raw/ for offline rebuild

## SDK/Environment
- Python: (to be confirmed on first orientation)
- SolarPipe .NET: 8.0.419 + 9.0.310 (parent project)

## Blockers
- None
```

### 15.3 `PROJECT_STATUS.md` (Initial State)

```markdown
# SolarPipe Data Acquisition — Project Status

## Overall: 🟡 Not Started — Phase 1 pending

## Phase 1: Foundation (Weeks 1–3) — ⬜ Not Started
- [ ] Task 1.1: Project scaffolding
- [ ] Task 1.2: Configuration system
- [ ] Task 1.3: Base HTTP client
- [ ] Task 1.4: Database schema (staging)
- [ ] Task 1.5: CLI harness
- [ ] Task 1.6: DONKI CME client
- [ ] Task 1.7: DONKI CME ingestion
- [ ] Task 1.8: Unit tests

## Phase 2: CME & Flare Catalogs (Weeks 4–6) — ⬜ Not Started
- [ ] Task 2.1: CDAW LASCO catalog scraper
- [ ] Task 2.2: CDAW ingestion
- [ ] Task 2.3: DONKI ancillary ingestion
- [ ] Task 2.4: NOAA GOES flare ingestion
- [ ] Task 2.5: DONKI flare ingestion
- [ ] Task 2.6: Integration tests

## Phase 3: In-Situ Solar Wind & Indices (Weeks 7–9) — ⬜ Not Started
- [ ] Task 3.1: SWPC solar wind client
- [ ] Task 3.2: Solar wind ingestion
- [ ] Task 3.3: Kyoto Dst client & ingestion
- [ ] Task 3.4: Kp index ingestion
- [ ] Task 3.5: F10.7 ingestion
- [ ] Task 3.6: Ambient context extraction
- [ ] Task 3.7: Unit tests

## Phase 4: SDO/HMI SHARP Features (Weeks 10–12) — ⬜ Not Started
- [ ] Task 4.1: JSOC DRMS client
- [ ] Task 4.2: HARP↔NOAA mapping
- [ ] Task 4.3: SHARP ingestion
- [ ] Task 4.4: SHARP feature selection
- [ ] Task 4.5: Integration test

## Phase 5: Cross-Matching & Feature Assembly (Weeks 13–15) — ⬜ Not Started
- [ ] Task 5.1: CME ↔ Flare association
- [ ] Task 5.2: CME ↔ ICME arrival
- [ ] Task 5.3: CME ↔ SHARP
- [ ] Task 5.4: ICME ↔ Storm
- [ ] Task 5.5: Feature assembler
- [ ] Task 5.6: Quality flags
- [ ] Task 5.7: Validation

## Phase 6: Synthetic Data & Export (Weeks 16–18) — ⬜ Not Started
- [ ] Task 6.1: Drag model
- [ ] Task 6.2: Rotation model
- [ ] Task 6.3: Monte Carlo sampler
- [ ] Task 6.4: ENLIL emulator
- [ ] Task 6.5: Parquet export
- [ ] Task 6.6: SQLite export
- [ ] Task 6.7: Validation script
- [ ] Task 6.8: End-to-end test
```

---

*End of Document*
