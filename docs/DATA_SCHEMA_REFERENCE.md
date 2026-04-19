# SolarPipe Data Schema Reference

Auto-reference for all databases, tables, and CSV files in the project.  
Use this before writing any SQL queries, feature lists, or data-source configs.  
Last audited: 2026-04-18.

**⚠ Note:** The `staging.db` section below reflects pre-Tier-1/2 state. For current table schemas and row counts, see `data/src/solarpipe_data/database/schema.py` (ORM ground truth) and `docs/SOURCE_TO_SCHEMA_REFERENCE.md`. Key current additions: `rc_icme`, `hek_events`, `sep_events`, `pfss_topology`, `stereo_wind_hourly`, `goes_mag_hourly`, `magnetopause_crossings`, `supermag_hourly` (empty), `wind_waves_type2`, `hcs_tilt`, `pinn_expanded_flat` (1,974 × **148** cols — updated 2026-04-19 after EVE dimming ingest), `training_features` VIEW (9,418 × 133 cols).

---

## Quick-Reference: Feature Name Aliases

The sweep/pipeline YAML uses **logical names**. The actual DB column names differ.  
Always check this table before adding features to a stage config.

| Logical Name (YAML) | Source Table | Actual Column | Join Required? |
|---|---|---|---|
| `cme_speed_kms` | `cme_events` | `cme_speed` | No |
| `sw_speed_ambient_kms` | `cme_events` | `sw_speed_ambient` | No |
| `sw_density_n_cc` | `cme_events` | `sw_density_ambient` | No |
| `sw_bt_nt` | `cme_events` | `sw_bt_ambient` | No |
| `bz_gsm_proxy_nt` | `flux_rope_fits` | `observed_bz_min` | JOIN on `event_id` |
| `transit_hours_observed` | `l1_arrivals` | `transit_time_hours` | JOIN on `event_id` |
| `transit_time_hours` | `l1_arrivals` | `transit_time_hours` | JOIN on `event_id` |
| `delta_v_kms` | computed | `cme_speed - sw_speed_ambient` | No |
| `speed_ratio` | computed | `cme_speed / NULLIF(sw_speed_ambient, 0)` | No |
| `speed_x_bz` | computed | `cme_speed * observed_bz_min` | JOIN on `event_id` |
| `speed_x_density` | computed | `cme_speed * sw_density_ambient` | No |

**Training-ready flat view** (use instead of raw tables):  
`training_features_v3` in `cme_catalog.db` — 1,930 rows, already joined and renamed.  
This is the **canonical source** for supervised learning on CME transit time.

**PINN training flat view** (as of 2026-04-10):  
`pinn_training_flat` in `data/data/staging/staging.db` — 1,974 rows (1,884 train + 90 holdout).  
Source: full 2010–2025 DONKI IPS→CME catalog with OMNI backfill from SPDF FTP.

---

## Database Files

### 1. `solar_data.db` — 11 GB (root of project)
Primary data warehouse. All raw ingested observations + ML result tables.

#### Tables

| Table | Rows | Description |
|---|---:|---|
| `cdaw_cme` | 42,424 | CME observations from CDAW catalog |
| `donki_cme` | 8,037 | CME data from NASA DONKI |
| `donki_flare` | 3,207 | Solar flare events from DONKI |
| `donki_gst` | 192 | Geomagnetic storm events from DONKI |
| `donki_hss` | 759 | High-speed stream events |
| `donki_ips` | 644 | Interplanetary shock events |
| `donki_sep` | 468 | Solar energetic particle events |
| `gfz_kp_ap` | 34,426 | Kp and Ap geomagnetic indices |
| `goes_xrs_flares` | 70,524 | GOES X-ray flare detections |
| `omni2_daily` | 23,376 | OMNI2 daily solar wind summary |
| `omni_hourly` | 561,024 | OMNI hourly solar wind (65 cols) |
| `planet_pairs` | 1,824,720 | Planet pair state snapshots |
| `planet_state` | 1,824,720 | Planetary state vectors (193 cols) |
| `silso_daily_ssn` | 76,030 | Daily sunspot number (SILSO) |
| `solar_event` | 258,195 | Solar events with planetary alignment (30+ cols) |
| `solar_event_planet_link` | 0 | Event–planet linkage (empty) |
| `solar_events` | 0 | Events (empty) |
| `symh_hourly` | 396,624 | Hourly SYM-H geomagnetic index |
| `ml_comparison` | 66 | ML model comparison results |
| `ml_intensity_predictions` | 1,000 | Predicted storm intensities |
| `ml_intensity_results` | 344 | Intensity model results |
| `ml_ssn_predictions` | 10,950 | Sunspot number predictions |
| `ml_ssn_results` | 1,614 | Sunspot number model results |
| `ml_occurrence_results` | 264 | Occurrence prediction results |
| `ml_phase_results` | 211 | Phase analysis results |
| `ml_interaction_results` | 197 | Interaction term analysis |
| `ml_solar_wind_results` | 486 | Solar wind analysis results |
| `mag_results` | varies | Magnetosphere model results (17 cols) |
| `mag_predictions` | varies | Magnetosphere predictions |

#### Key Schema Details

**`omni_hourly`** (primary solar wind source, 561K rows, 65 cols)
```
datetime          TEXT  PK (ISO)
Bx_GSE, By_GSE, Bz_GSE          REAL  IMF components, GSE frame
By_GSM, Bz_GSM                  REAL  IMF components, GSM frame  ← use these for physics
B_scalar_avg, B_vector_mag       REAL
proton_temp_K, proton_density    REAL
flow_speed, flow_pressure        REAL
Dst_nT, AE_nT, Kp_x10, ap_index REAL/INT
F10_7_index                      REAL
electric_field, plasma_beta      REAL  (derived)
alfven_mach, magnetosonic_mach   REAL  (derived)
Bz_southward                     REAL  (derived, rectified southward component)
proton_flux_gt1MeV .. gt60MeV    REAL
```

**`planet_state`** (1.8M rows, 193 cols — one row per TDB day)
```
datetime_tdb      TEXT  PK
{planet}_x_km, {planet}_y_km, {planet}_z_km         REAL  (Mercury..Neptune)
{planet}_vx_kd, {planet}_vy_kd, {planet}_vz_kd      REAL  velocity (km/day)
{planet}_ax_kd2, {planet}_ay_kd2, {planet}_az_kd2   REAL  acceleration
{planet}_jx/y/z_kd3                                  REAL  jerk (3rd derivative)
{planet}_sx/y/z_kd4                                  REAL  snap (4th derivative)
{planet}_speed_km_s                                  REAL
{planet}_dist_sun_km, {planet}_dist_sun_au           REAL
{planet}_ang_vel_inv_deg, {planet}_ang_pos_inv_deg   REAL  inner-planet angles
```

**`donki_cme`** (8,037 rows)
```
activity_id       TEXT  PK
start_time, date  TEXT
speed_kms, half_angle_deg, latitude, longitude  REAL
active_region_num                               INT
is_most_accurate, catalog, analysis_type        TEXT/INT
n_linked_events, linked_event_ids               INT/TEXT
```

**`cdaw_cme`** (42,424 rows)
```
row_id            INT   PK
date, time_ut, datetime  TEXT
linear_speed_kms, second_order_speed_*, accel_kms2  REAL
angular_width_deg, central_pa_deg, mpa_deg          REAL
mass_grams, kinetic_energy_ergs                     REAL
```

**`gfz_kp_ap`** (34,426 rows)
```
date, hour_interval  TEXT  composite PK
Kp, ap, daily_Ap     REAL
daily_SN, daily_F10_7_obs, daily_F10_7_adj  REAL
```

**`symh_hourly`** (396,624 rows)
```
datetime   TEXT  PK
symh_nT, symh_min, symh_max  REAL
asyh_nT, symd_nT, asyd_nT    REAL
```

---

### 2. `data/data/output/cme_catalog.db` — 4.1 MB
Engineered CME dataset. **Primary source for supervised transit-time learning.**

#### Tables

| Table | Rows | Description |
|---|---:|---|
| `cme_events` | 9,418 | Raw CME observations |
| `flux_rope_fits` | 9,418 | In-situ flux rope fitting results |
| `l1_arrivals` | 9,418 | L1 arrival times and storm effects |
| `extended_features` | 2,018 | Extended SHARP-derived features |
| `enlil_features` | 3,623 | ENLIL ensemble simulation features |
| `cdaw_supplements` | 858 | CDAW supplemental CME parameters |
| `training_features` | 1,930 | Flat ML-ready view (v1) |
| `training_features_v2` | 1,930 | Flat ML-ready view (v2) |
| `training_features_v3` | 1,930 | **Flat ML-ready view (v3) ← preferred** |
| `schema_version` | 1 | Migration tracking |

#### Key Schema Details

**`cme_events`** (9,418 rows)
```
event_id          VARCHAR  PK
launch_time       VARCHAR
cme_speed         FLOAT    ← maps to cme_speed_kms in YAML
cme_mass, cme_angular_width, flare_class_numeric  FLOAT
chirality, source_location, noaa_ar               VARCHAR/INT
initial_axis_angle                                FLOAT
usflux, totpot, r_value, meanshr, totusjz         FLOAT  (SHARP)
coronal_hole_proximity, coronal_hole_polarity     FLOAT/INT
hcs_tilt_angle, hcs_distance                     FLOAT
sw_speed_ambient  FLOAT    ← maps to sw_speed_ambient_kms in YAML
sw_density_ambient FLOAT   ← maps to sw_density_n_cc in YAML
sw_bt_ambient      FLOAT   ← maps to sw_bt_nt in YAML
f10_7              FLOAT   (NULL for 2012 events — do not use)
quality_flag       INTEGER NOT NULL
```

**`flux_rope_fits`** (9,418 rows)
```
event_id              VARCHAR  PK  (JOIN key)
observed_bz_min       FLOAT    ← maps to bz_gsm_proxy_nt in YAML
observed_rotation_angle FLOAT
bz_polarity           INTEGER
fit_method            VARCHAR  NOT NULL
fit_quality           INTEGER
has_in_situ_fit       INTEGER  NOT NULL
```

**`l1_arrivals`** (9,418 rows)
```
event_id             VARCHAR  PK  (JOIN key)
shock_arrival_time   VARCHAR
icme_start_time      VARCHAR
icme_end_time        VARCHAR
transit_time_hours   FLOAT    ← maps to transit_hours_observed in YAML
dst_min_nT           FLOAT
kp_max               FLOAT
has_in_situ_fit      INTEGER  NOT NULL
```

**`training_features_v3`** (1,930 rows — already joined, use this for training)
```
event_id, launch_time                    VARCHAR/TEXT
transit_time_hours                       FLOAT   ← target column
quality_flag, has_in_situ_fit            INT
cme_speed_kms                            FLOAT   ← ready-to-use alias
cme_half_angle_deg, cme_latitude_deg,    FLOAT
cme_abs_longitude_deg, cme_speed_radial_kms
sw_speed_ambient_kms                     FLOAT   ← ready-to-use alias
sw_density_n_cc                          FLOAT   ← ready-to-use alias
sw_bt_nt                                 FLOAT   ← ready-to-use alias
bz_gsm_proxy_nt                          FLOAT   ← ready-to-use alias
delta_v_kms, speed_ratio                 FLOAT   (pre-computed)
speed_x_bz, speed_x_density             FLOAT   (pre-computed)
delta_v_radial_kms, speed_ratio_radial  FLOAT   (pre-computed)
```

**`enlil_features`** (3,623 rows)
```
event_id                  TEXT  (JOIN key)
enlil_transit_mean, enlil_transit_std  REAL
enlil_transit_p10, enlil_transit_p90   REAL
enlil_arrival_speed_mean/std           REAL
enlil_ambient_wind_mean                REAL
enlil_lat_mean, enlil_lon_mean, enlil_width_mean  REAL
enlil_speed_initial_mean               REAL
enlil_bias                             REAL
enlil_n_members                        INTEGER
```

**Standard JOIN for full training row:**
```sql
SELECT
    e.event_id,
    e.launch_time,
    la.transit_time_hours,
    e.cme_speed           AS cme_speed_kms,
    e.sw_speed_ambient    AS sw_speed_ambient_kms,
    e.sw_density_ambient  AS sw_density_n_cc,
    e.sw_bt_ambient       AS sw_bt_nt,
    f.observed_bz_min     AS bz_gsm_proxy_nt,
    (e.cme_speed - e.sw_speed_ambient) AS delta_v_kms,
    (e.cme_speed / NULLIF(e.sw_speed_ambient, 0)) AS speed_ratio,
    (e.cme_speed * f.observed_bz_min)     AS speed_x_bz,
    (e.cme_speed * e.sw_density_ambient)  AS speed_x_density
FROM cme_events e
JOIN l1_arrivals la    ON e.event_id = la.event_id
JOIN flux_rope_fits f  ON e.event_id = f.event_id
WHERE e.cme_speed IS NOT NULL
  AND la.transit_time_hours IS NOT NULL
  AND la.transit_time_hours > 0
  AND la.transit_time_hours < 300
  AND e.quality_flag >= 3
```

---

### 3. `data/data/staging/staging.db` — 283 MB
Intermediate processing layer. Joined and denormalized tables for ML pipelines.

#### Tables

| Table | Rows | Description |
|---|---:|---|
| `feature_vectors` | 9,418 | **Full 57-col ML feature vector** ← richest source |
| `cme_events` | 9,418 | DONKI-format CME events |
| `cdaw_cme_events` | 41,351 | CDAW format CME events |
| `solar_wind_hourly` | 561,024 | Hourly solar wind (matches `omni_hourly`) |
| `symh_hourly` | 396,624 | Hourly SYM-H |
| `dst_hourly` | 143,136 | Hourly Dst index |
| `kp_3hr` | 34,426 | 3-hourly Kp index |
| `silso_daily_ssn` | 76,030 | Daily sunspot number |
| `flares` | 3,207 | Solar flares |
| `enlil_simulations` | 5,972 | ENLIL simulation metadata |
| `sharp_keywords` | 102,305 | SHARP magnetic field keywords |
| `geomagnetic_storms` | 192 | Geomagnetic storm catalog |
| `interplanetary_shocks` | 645 | Interplanetary shock events |
| `f107_daily` | 3,327 | Daily F10.7 solar flux index |
| `sw_ambient_context` | 9,371 | Ambient solar wind at CME launch |
| `harp_noaa_map` | 0 | HARP-to-NOAA AR mapping (empty) |
| `schema_version` | 5 | Migration tracking |

**`feature_vectors`** (9,418 rows, 57 cols — most comprehensive single table)
```
activity_id         TEXT  PK
launch_time, icme_arrival_time, dst_min_time  TEXT
cme_speed_kms, cme_half_angle_deg, cme_latitude, cme_longitude  REAL
cme_mass_grams, cme_angular_width_deg                           REAL
linked_flare_id, flare_class_letter, flare_class_numeric        TEXT/REAL
flare_peak_time, flare_active_region, flare_match_method        TEXT
sharp_harpnum, sharp_noaa_ar                                    INT
usflux, meangam, meangbt, meangbz, meangbh                     REAL (SHARP)
meanjzd, totusjz, meanalp, meanjzh, totusjh                    REAL (SHARP)
absnjzh, savncpp, meanpot, totpot, meanshr, shrgt45, r_value   REAL (SHARP)
area_acr, lat_fwt, lon_fwt                                     REAL (SHARP)
sw_speed_ambient, sw_density_ambient, sw_bt_ambient, sw_bz_ambient  REAL
linked_ips_id, transit_time_hours, icme_match_method            TEXT/REAL
icme_match_confidence                                           REAL
dst_min_nt, kp_max, storm_threshold_met                         REAL/INT
hcs_tilt_angle, hcs_distance, f10_7, sunspot_number            REAL
dimming_area, dimming_asymmetry                                 REAL
quality_flag                                                    INT (default 3)
source_catalog, fetch_timestamp                                 TEXT
```

---

### 4. `data/cme_catalog.db` — 0 bytes (empty, ignore)

---

## Parquet Files

### `data/data/output/enlil_runs/enlil_ensemble_v1.parquet` — 13 MB
ENLIL ensemble simulation data. Schema mirrors `enlil_features` in `cme_catalog.db`:
- `enlil_transit_mean/std/p10/p90`, `enlil_arrival_speed_mean/std`
- `enlil_ambient_wind_mean`, `enlil_lat/lon/width_mean`
- `enlil_speed_initial_mean`, `enlil_bias`, `enlil_n_members`

---

## Validation / CSV Files

| File | Rows | Key Columns |
|---|---|---|
| `data/validation/observed_transits_2026.csv` | ~10 | `cme_id, departure_utc, arrival_utc, transit_hours_observed, cme_speed_kms, bz_gsm_min_nt, arrival_speed_kms, storm_level` |
| `data/validation/predict_input_2026_apr.csv` | ~5 | `cme_speed_kms, bz_gsm_proxy_nt, sw_density_n_cc, sw_speed_ambient_kms, sw_bt_nt` |
| `data/validation/historical_predict_input_valid.csv` | varies | `cme_speed_kms, bz_gsm_proxy_nt, sw_density_n_cc, sw_speed_ambient_kms, sw_bt_nt` |
| `data/validation/historical_events_meta.csv` | varies | `event_id, launch_time, transit_hours_observed` |
| `data/validation/ccmc_scoreboard_2026.csv` | varies | `cme_id, model, predicted_arrival_utc, actual_arrival_utc, error_hours, mae_published` |

---

## Data Source Config Snippets

Use these in pipeline YAML `data_sources:` blocks:

```yaml
# CME training data — preferred flat view
cme_events:
  provider: sqlite
  connection_string: "Data Source=data/data/output/cme_catalog.db"
  options:
    table: training_features_v3

# CME raw tables (when JOINs needed)
cme_catalog:
  provider: sqlite
  connection_string: "Data Source=data/data/output/cme_catalog.db"
  options:
    table: cme_events   # + JOIN l1_arrivals, flux_rope_fits

# Full solar wind / geomagnetic index data
solar_data:
  provider: sqlite
  connection_string: "Data Source=solar_data.db"
  options:
    table: omni_hourly  # or symh_hourly, gfz_kp_ap, etc.

# ENLIL ensemble
enlil_ensemble:
  provider: parquet
  connection_string: data/data/output/enlil_runs/enlil_ensemble_v1.parquet
```

---

## PINN Training Tables (staging.db)

Location: `data/data/staging/staging.db`  
Built by: `scripts/build_pinn_feature_matrix.py`  
OMNI backfill source: SPDF FTP `omni2_YYYY.dat` via `scripts/backfill_omni_hourly.py`

### pinn_training_flat — 1,974 rows (primary model input)

| Column | Type | NULL rate (train) | Source |
|---|---|---|---|
| `activity_id` | TEXT PK | 0% | donki_cme |
| `linked_ips_id` | TEXT | — | donki_ips |
| `launch_time` | TEXT | 0% | feature_vectors |
| `icme_arrival_time` | TEXT | — | feature_vectors |
| `split` | TEXT | — | `'train'` if launch<2026, else `'holdout'` |
| `transit_time_hours` | REAL | 0% | IPS.event_time − CME.start_time |
| `omni_24h_bz_mean` | REAL | 0% | omni_hourly 24h pre-launch |
| `omni_24h_bz_std` | REAL | 0% | omni_hourly 24h pre-launch |
| `omni_24h_bz_min` | REAL | 0% | omni_hourly 24h pre-launch |
| `omni_24h_speed_mean` | REAL | 0% | omni_hourly 24h pre-launch |
| `omni_24h_density_mean` | REAL | 0% | omni_hourly 24h pre-launch |
| `omni_24h_pressure_mean` | REAL | 0% | omni_hourly 24h pre-launch |
| `omni_24h_ae_max` | REAL | 0% | omni_hourly 24h pre-launch |
| `omni_24h_dst_min` | REAL | 0% | omni_hourly 24h pre-launch |
| `omni_24h_kp_max` | REAL | 0% | omni_hourly 24h pre-launch |
| `f10_7` | REAL | 0% | gfz_kp_ap daily_F10_7_obs |
| `cluster_id_k5` | INTEGER | 0% | ml_clusters kmeans k=5 (NNN-imputed for 192 train rows) |
| `cluster_assigned` | INTEGER | — | 0=direct join, 1=NNN imputed |
| `preceding_cme_count_48h` | INTEGER | 0% | donki_cme 48h pre-launch |
| `preceding_cme_speed_max` | REAL | — | donki_cme 48h pre-launch |
| `preceding_cme_speed_mean` | REAL | — | donki_cme 48h pre-launch |
| `preceding_cme_angular_sep_min` | REAL | — | angular separation to nearest preceding CME |
| `is_multi_cme` | INTEGER | 0% | donki_ips.n_linked_events > 1 (39% of train) |
| `omni_48h_density_spike_max` | REAL | 0% | omni_hourly 48h pre-launch |
| `omni_48h_speed_gradient` | REAL | 0% | flow_speed[-1] − flow_speed[-48] |
| `cme_speed_kms` | REAL | 0% | donki_cme |
| `cme_half_angle_deg` | REAL | 0% | donki_cme |
| `cme_latitude` | REAL | 0% | donki_cme |
| `cme_longitude` | REAL | 0% | donki_cme |
| `cme_angular_width_deg` | REAL | — | feature_vectors |
| `cdaw_linear_speed_kms` | REAL | 2.0% | cdaw_cme ±6h match |
| `cdaw_angular_width_deg` | REAL | 2.0% | cdaw_cme ±6h match |
| `cdaw_mass_log10` | REAL | — | log10(cdaw_cme.mass_grams), ~58% NULL raw |
| `cdaw_ke_log10` | REAL | — | log10(cdaw_cme.kinetic_energy_ergs), ~58% NULL raw |
| `cdaw_matched` | INTEGER | 0% | 1 if CDAW match found within ±6h |
| `flare_class_numeric` | REAL | 72.8% | donki_flare linked to CME |
| `has_flare` | INTEGER | 0% | binary: 1 if any flare linked |
| `flare_source_longitude` | REAL | — | parsed from donki_flare.source_location |
| `omni_150h_density_median` | REAL | 0% | omni_hourly 150h pre-launch |
| `omni_150h_speed_median` | REAL | 0% | omni_hourly 150h pre-launch |
| `sw_bz_ambient` | REAL | — | feature_vectors.sw_bz_ambient |
| `delta_v_kms` | REAL | 0% | cme_speed_kms − omni_150h_speed_median |
| `usflux` | REAL | 59% NULL | SHARP magnetic flux |
| `sharp_available` | INTEGER | 0% | binary: 1 if SHARP harpnum present; **41% of events = 1** (809/1974, measured 2026-04-18) |

### Other PINN tables (staging.db)

| Table | Rows | Description |
|---|---|---|
| `pinn_events` | 1,974 | Base identity + transit label + split |
| `pinn_regime_features` | 1,974 | Stage 1 only: 24h OMNI + F10.7 + cluster |
| `pinn_interaction_features` | 1,974 | Stage 2 only: 48h CME catalog + OMNI spikes |
| `pinn_physics_features` | 1,974 | Stage 3 only: 150h OMNI + CDAW + flare |

---

## Sentinel Value Rules

Per RULE-120: convert these to `NaN` at data load time.

| Value | Meaning |
|---|---|
| `9999.9` | Missing / fill value (OMNI convention) |
| `-1e31` | Missing / fill value (OMNI convention) |
| `NULL` | Missing in SQLite — handled natively |
| `f10_7 = NULL` for 2012 events | Known gap — exclude `f10_7` feature for those years |
