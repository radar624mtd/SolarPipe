# ARCHITECTURE.md — SolarPipe Data Acquisition Pipeline

Reference for interfaces, data flow, API endpoints, database schemas, and technology choices.
**Read the section relevant to your current task. Do not pre-load the entire file.**

---

## Data Flow

Strictly one-directional. No module may import from a later stage.

```
External APIs  +  solar_data.db (existing bulk data)
      │                    │
      ▼  HTTP (BaseClient)  ▼  scripts/port_solar_data.py
  clients/  →  data/raw/{source}/  (JSON/HTML file cache)
      │
      ▼  parse + upsert (idempotent)
  ingestion/  →  data/staging/staging.db  (SQLAlchemy ORM)
      │
      ▼  temporal/spatial joins
  crossmatch/  →  feature_vectors table in staging.db
      │
      ▼  export + finalize
  export/  →  data/output/cme_catalog.db     (SQLite — consumed by C# SqliteProvider)
              data/output/enlil_runs/*.parquet (PyArrow — consumed by C# ParquetProvider)
```

`crossmatch/` never calls API clients. `export/` never writes back to staging.

---

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `clients/base.py` | httpx.AsyncClient, token-bucket rate limiter, retry/backoff, file cache |
| `clients/{source}.py` | Source-specific request construction and response parsing |
| `ingestion/ingest_{source}.py` | Raw → staging.db upsert; sentinel conversion; provenance columns |
| `crossmatch/` | Temporal/spatial event association across catalogs |
| `synthetic/` | ENLIL emulator, drag model, Monte Carlo sampler |
| `database/schema.py` | SQLAlchemy ORM for all 15 staging tables; `make_engine()`; `init_db()` |
| `database/migrations.py` | Version tracking; column-add helpers |
| `database/queries.py` | Common query patterns; temporal range helpers; upsert wrappers |
| `transforms/` | Cleaning, feature engineering, physical consistency validation |
| `export/sqlite_export.py` | Assemble `cme_catalog.db` from staging |
| `export/parquet_export.py` | Write `enlil_runs/enlil_ensemble_v1.parquet` |
| `cli.py` | Click harness: `fetch`, `ingest`, `crossmatch`, `build`, `validate`, `status` |
| `config.py` | Pydantic settings from `configs/default.yaml` + env vars |

---

## Storage Architecture

| Store | Format | Consumer | Notes |
|-------|--------|----------|-------|
| `data/staging/staging.db` | SQLite | Python ingestion/crossmatch | SQLAlchemy ORM; WAL mode |
| `data/output/cme_catalog.db` | SQLite | C# `SqliteProvider` | Column names hardcoded in YAML pipeline config |
| `data/output/enlil_runs/*.parquet` | Parquet (PyArrow) | C# `ParquetProvider` + ParquetSharp | Columnar; ≤64 MB row groups |
| `data/raw/{source}/` | JSON / HTML files | `BaseClient` cache layer | Never delete without `--force` |

---

## Staging Database Schema (`data/staging/staging.db`)

Provenance columns required on every table: `source_catalog TEXT`, `fetch_timestamp TEXT`, `data_version TEXT`.

### Event Tables

**`cme_events`** — PK: `activity_id TEXT`
Source: DONKI. Natural key from DONKI (e.g. `"2016-09-06T14:18Z-CME-001"`).
Key columns: `start_time, source_location, active_region_num, catalog, speed_kms, half_angle_deg, latitude, longitude, is_earth_directed, linked_flare_id, linked_ips_ids (JSON), linked_gst_ids (JSON)`
Note: ~10–15% of records have null `active_region_num` — remediate via CDAW/GOES cross-reference.

**`cme_analyses`** — PK: `analysis_id TEXT`
Source: DONKI CMEAnalysis. FK: `cme_activity_id → cme_events`.
Key columns: `time21_5` (CME front at 21.5 R☉ — NOT event start), `latitude, longitude, half_angle_deg, speed_kms, is_most_accurate, level_of_data (0=RT/1=NRT/2=definitive)`

**`cdaw_cme_events`** — PK: `cdaw_id TEXT` format `YYYYMMDD.HHMMSS`
Source: CDAW LASCO UNIVERSAL_ver2.
Key columns: `linear_speed_kms, speed_20rs_kms` (use this for arrival models), `angular_width_deg, mass_grams` (~40% null — do not impute), `remarks` ("Poor Event" → quality_flag=1)

**`flares`** — PK: `flare_id TEXT`
Source: DONKI FLR merged with NOAA GOES.
Key columns: `begin_time, peak_time, class_type, class_letter, class_magnitude, source_location, active_region_num`

**`enlil_simulations`** — PK: `simulation_id TEXT`
Source: DONKI WSAEnlilSimulations. Not 1:1 with CMEs — deduplicate by first simulation per CME.

**`geomagnetic_storms`** — PK: `gst_id TEXT`
Source: DONKI. Key columns: `start_time, kp_index_max, all_kp_values (JSON)`

**`interplanetary_shocks`** — PK: `ips_id TEXT`
Source: DONKI. Used for CME-ICME matching. Key columns: `event_time, location, catalog`

### Time-Series Tables

**`solar_wind_hourly`** — PK: `datetime TEXT` (ISO 8601 UTC hourly)
Source: OMNI (ported) + SWPC incremental.
Key columns: `bz_gsm` (canonical — never `bz_gse`), `flow_speed, proton_density, flow_pressure, dst_nt, kp_x10, f10_7_index, spacecraft` (ACE/DSCOVR — transition July 2016)
Existing data: 561,024 rows from 1963–2026-12-31 (ported from solar_data.db).

**`symh_hourly`** — PK: `datetime TEXT`
Source: WDC (ported). Key columns: `symh_nt, asyh_nt, symd_nt, asyd_nt`
Existing data: 1981–2026-03-31 (ported).

**`dst_hourly`** — PK: `datetime TEXT`
Source: Kyoto WDC (fetched). Key columns: `dst_nt, data_type` (final/provisional/realtime).
Preference cascade: final > provisional > realtime — never overwrite final with lower quality.

**`kp_3hr`** — PK: `datetime TEXT`
Source: GFZ Potsdam (ported bulk + incremental).
Key columns: `kp, ap, definitive, daily_f10_7_obs`
Existing data: 34,426 rows from 1932–2026-04-02 (ported).

**`f107_daily`** — PK: `date TEXT`
Source: NOAA SWPC.

**`silso_daily_ssn`** — PK: `date TEXT`
Source: SILSO (ported). Existing data: 1818–2026-02-28.

### Reference Tables

**`sharp_keywords`** — PK: `id INTEGER AUTOINCREMENT`
Source: JSOC DRMS (`hmi.sharp_cea_720s` only — not CCD series).
Unique constraint: `(harpnum, t_rec, query_context)`.
Key columns: 18 space-weather keywords (USFLUX, MEANGAM, MEANGBT, MEANGBZ, MEANGBH, MEANJZD, TOTUSJZ, MEANALP, MEANJZH, TOTUSJH, ABSNJZH, SAVNCPP, MEANPOT, TOTPOT, MEANSHR, SHRGT45, R_VALUE, AREA_ACR), `lat_fwt, lon_fwt` (drop at ingest if `lon_fwt > 60°`), `query_context` (at_eruption/minus_6h/minus_12h/plus_6h).

**`schema_version`** — PK: `version INTEGER`
Internal migration tracking.

---

## Output Database Schema (`data/output/cme_catalog.db`)

Column names are hardcoded in `configs/flux_rope_propagation_v1.yaml` — **do not rename**.

**`cme_events`** — PK: `event_id TEXT`

| Group | Columns |
|-------|---------|
| Identity | `event_id, launch_time, source_location, noaa_ar` |
| Kinematic | `cme_speed, cme_mass, cme_angular_width, flare_class_numeric` |
| Source region | `chirality, initial_axis_angle, usflux, totpot, r_value, meanshr, totusjz` |
| Environmental | `coronal_hole_proximity, coronal_hole_polarity, hcs_tilt_angle, hcs_distance` |
| Ambient L1 | `sw_speed_ambient, sw_density_ambient, sw_bt_ambient, f10_7` |
| Quality | `quality_flag INTEGER DEFAULT 3` |

**`flux_rope_fits`** — PK: `event_id` FK → `cme_events`
`observed_rotation_angle` (**primary prediction target**), `observed_bz_min, bz_polarity, fit_method, fit_quality, has_in_situ_fit`

**`l1_arrivals`** — PK: `event_id` FK → `cme_events`
`shock_arrival_time, icme_start_time, icme_end_time, transit_time_hours, dst_min_nT, kp_max, has_in_situ_fit`

Validation query (must execute without error — checked by `scripts/validate_db.py`):
```sql
SELECT * FROM cme_events e
JOIN flux_rope_fits f ON e.event_id = f.event_id
JOIN l1_arrivals a ON e.event_id = a.event_id
WHERE e.quality_flag >= 3 AND a.has_in_situ_fit = 1
```

---

## Complete API Endpoint Reference

### NASA DONKI
Base: `https://api.nasa.gov/DONKI/` — auth: `?api_key=${NASA_API_KEY}`

| Endpoint | Path | Chunking |
|----------|------|----------|
| CME | `/CME` | None — any date range |
| CME Analysis | `/CMEAnalysis?mostAccurateOnly=true&catalog=ALL` | None |
| Solar Flare | `/FLR` | None |
| Geomagnetic Storm | `/GST` | None |
| Interplanetary Shock | `/IPS?location=Earth` | None |
| WSA-ENLIL | `/WSAEnlilSimulations` | None |
| **Notifications** | `/notifications` | **≤30 days only** |

Rate: registered key = 1,000 req/hr. `DEMO_KEY` = 30 req/hr, 50 req/day **per IP address**.
Timestamp format: `"2016-09-06T14:18Z"` (no seconds). Parse: `datetime.fromisoformat(ts.replace("Z", "+00:00"))`.

### CDAW LASCO
| Resource | URL |
|----------|-----|
| Monthly (current) | `https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL_ver2/{YYYY}_{MM}/univ{YYYY}_{MM}.html` |
| Halo catalog | `https://cdaw.gsfc.nasa.gov/CME_list/halo/halo.html` |

No auth. Rate: ≤1 req/2s. Parse: BeautifulSoup only. SOHO gap 1998–99 is real — do not retry.

### NOAA SWPC
Base: `https://services.swpc.noaa.gov/` — no auth.

| Data | URL | Notes |
|------|-----|-------|
| Solar wind mag 7-day | `/products/solar-wind/mag-7-day.json` | Timestamps omit `Z` |
| Solar wind plasma 7-day | `/products/solar-wind/plasma-7-day.json` | |
| RTSW mag 1-min | `/json/rtsw/rtsw_mag_1m.json` | Has both `bz_gse` AND `bz_gsm` |
| RTSW plasma 1-min | `/json/rtsw/rtsw_wind_1m.json` | |
| GOES flares 7-day | `/json/goes/primary/xray-flares-7-day.json` | Timestamps include `Z` |
| Kp 1-min estimated | `/json/planetary_k_index_1m.json` | |
| F10.7 / solar cycle | `/json/solar-cycle/observed-solar-cycle-indices.json` | |

Always: `datetime.fromisoformat(ts.rstrip("Z"))` — `/products/` omit `Z`, `/json/goes/` include `Z`.

### Stanford JSOC (SHARP)
No auth for keyword queries. `STANFORD_JSOC_EMAIL` env var set.

| Query | Pattern |
|-------|---------|
| SHARP keywords (definitive) | `hmi.sharp_cea_720s[HARPNUM][T_REC]` |
| HARP↔NOAA mapping | `hmi.sharp_720s[][TIMERANGE]{HARPNUM,NOAA_AR,NOAA_ARS}` |
| Filter by NOAA AR | `hmi.sharp_cea_720s[][TIMERANGE][? NOAA_ARS ~ "12345" ?]` |

Required keywords: `USFLUX MEANGAM MEANGBT MEANGBZ MEANGBH MEANJZD TOTUSJZ MEANALP MEANJZH TOTUSJH ABSNJZH SAVNCPP MEANPOT TOTPOT MEANSHR SHRGT45 R_VALUE AREA_ACR LAT_FWT LON_FWT NOAA_AR HARPNUM T_REC`

### Kyoto WDC Dst
| Type | URL |
|------|-----|
| Final | `https://wdc.kugi.kyoto-u.ac.jp/dst_final/{YYYYMM}/index.html` |
| Provisional | `https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/{YYYYMM}/index.html` |
| Real-time | `https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/{YYYYMM}/index.html` |

Parse: `index.html` HTML table (day × 24 hourly columns). Check `len(body) > 100` before parsing.

### GFZ Potsdam Kp
Base: `https://kp.gfz-potsdam.de/app/json/` — no auth. Bulk data already ported (34,426 rows). Fetch incremental only.

---

## Cross-Matching Windows

| Match Type | Temporal Window | Spatial Constraint | Priority |
|------------|-----------------|-------------------|----------|
| CME ↔ Flare | ±30 minutes | ±15° lat/lon | DONKI `linkedEvents` → temporal+spatial → statistical |
| CME ↔ ICME | ±12 hours around estimated transit | None (L1 point) | DONKI IPS links → transit time estimate |
| ICME ↔ Geomagnetic storm | 24–48 h post-arrival | Dst minimum | Dst < −30 nT threshold |
| CME ↔ SHARP | At eruption time | NOAA AR match | Also query -12h, -6h, 0h, +6h |

Transit estimate: `t_transit ≈ 1 AU / v_cme` with drag correction. L1 propagation lag: ~45 min.

---

## Technology Stack

| Component | Library | Version | Notes |
|-----------|---------|---------|-------|
| HTTP client | `httpx` | ≥0.27 | AsyncClient only |
| HTML scraping | `beautifulsoup4` + `lxml` | ≥4.12 / ≥5.0 | Never `pandas.read_html()` |
| ORM + SQLite | `sqlalchemy` | ≥2.0 | sqlite dialect for upsert |
| Data processing | `pandas` + `numpy` | ≥2.2 / ≥1.26 | |
| Parquet I/O | `pyarrow` | ≥15.0 | ENLIL ensemble output only |
| ODE solver | `scipy` | ≥1.12 | Drag model (`solve_ivp`) |
| Solar data | `sunpy` + `drms` + `astropy` | 7.1.1 / 0.9.0 / ≥6.0 | Already installed |
| CLI | `click` | ≥8.1 | Async bridge via `asyncio.run()` |
| Config | `pydantic` + `pydantic-settings` | ≥2.6 / ≥2.2 | |
| UI | `rich` | ≥13.7 | Progress bars, status tables |
| Testing | `pytest` + `pytest-httpx` + `pytest-asyncio` | ≥8.0 / ≥0.30 / ≥0.24 | |

---

## Existing Data Inventory

Pre-populated in `C:\Users\radar\SolarPipe\solar_data.db` (10.8 GB). Port via `scripts/port_solar_data.py`.

| Source Table | Rows | Date Range | Staging Target |
|-------------|------|------------|---------------|
| `donki_cme` | 8,037 | 2010-04-03 → 2026-04-03 | `cme_events` |
| `cdaw_cme` | 42,424 | 1996-01-11 → 2025-12-31 | `cdaw_cme_events` |
| `donki_flare` | 3,207 | 2010-04-03 → 2026-04-02 | `flares` |
| `gfz_kp_ap` | 34,426 | 1932-01-01 → 2026-04-02 | `kp_3hr` |
| `omni_hourly` | 561,024 | 1963-01-01 → 2026-12-31 | `solar_wind_hourly` |
| `symh_hourly` | — | 1981-01-01 → 2026-03-31 | `symh_hourly` |
| `silso_daily_ssn` | — | 1818-01-01 → 2026-02-28 | `silso_daily_ssn` |
| `donki_gst` | — | — | `geomagnetic_storms` |
| `donki_ips` | — | — | `interplanetary_shocks` |
