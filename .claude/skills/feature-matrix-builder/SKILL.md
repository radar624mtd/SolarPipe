---
name: feature-matrix-builder
description: Expand pinn_training_flat with CDAW acceleration/curvature, 14 SHARP magnetic parameters, multi-cluster labels (k=8, k=12, dbscan), and existing model predictions as features. Use before writing build_expanded_feature_matrix.py or modifying the feature extraction pipeline.
---

You are building or reviewing the expanded feature matrix for SolarPipe's neural ensemble. The goal is to add all structured signal that `build_pinn_feature_matrix.py` deliberately omitted.

## What to add (and where to find it)

### 1. CDAW kinematic fields — `solar_data.db:cdaw_cme`
Join to `pinn_events` via CDAW match logic (±6h window on `start_time`).

Fields to add:
- `second_order_speed_init` — initial speed from 2nd-order fit (km/s)
- `second_order_speed_final` — final speed at 20 Rs (km/s)
- `second_order_speed_20Rs` — speed at exactly 20 solar radii (km/s)
- `accel_kms2` — integrated acceleration (km/s²)
- `mpa_deg` — mid-point angle (CME trajectory direction, degrees)

These fields encode the deceleration curve — the physical drag signature. They are directly predictive of transit time and must not be omitted.

### 2. SHARP magnetic keywords — `staging.db:sharp_keywords`
Join to `pinn_events` via `active_region_num` (NOAA AR number) from `donki_flare` or `donki_cme`.

Fields to add (all 14, plus the ones already used):
- `meangam` — mean inclination angle (deg)
- `meangbt`, `meangbz`, `meangbh` — mean field components (G)
- `meanjzd`, `totusjz` — current density (mA/m²)
- `meanjzh`, `totusjh`, `absnjzh` — current helicity metrics
- `meanalp`, `savncpp` — alpha / non-potential field
- `meanpot`, `totpot` — potential field proxy
- `meanshr`, `shrgt45` — shear angle metrics
- `r_value` — correlation metric
- `area_acr` — active region area

Use the SHARP record closest in time to `launch_time` (within 24h). If no SHARP match, fill with NaN (do not drop the event).

### 3. Multi-cluster labels — `solar_data.db:ml_clusters`
Add cluster IDs for k=8 and k=12 (k=5 already in pinn_training_flat):
- `cluster_id_k8` — kmeans k=8 cluster assignment
- `cluster_id_k12` — kmeans k=12 cluster assignment
- `cluster_id_dbscan` — DBSCAN cluster ID (96 clusters; -1 = noise)

Join on `event_id` (= `activity_id`) with `cluster_method='kmeans'` and respective k.

### 4. Existing model predictions as features
Add per-event predictions from completed models (these become meta-learner inputs):
- `phase8_pred_transit_hours` — from `output/phase8_domain_results.json`
- `pinn_v1_pred_transit_hours` — from `output/pinn_v1/pinn_v1_results.json`
- `physics_ode_transit_hours` — physics-only ODE baseline (already in pinn_training_flat as `transit_physics`)
- `phase9_pred_transit_hours` — from `output/phase9_m8*/phase9_m8*_comparison.json` (use progressive result where available, static fallback)

For holdout events, predictions are available. For train events in CV, these must be out-of-fold predictions only — never in-fold predictions as features.

## Output table: `pinn_expanded_flat` in `staging.db`
- Same 1,974 rows as `pinn_training_flat`
- All original 45 columns retained
- ~30 new columns added above
- Null fill: NaN for missing SHARP/CDAW matches, not 0
- `split` column preserved for temporal CV

## Validation checks before writing
1. Row count must equal `pinn_training_flat` row count (1,974)
2. `activity_id` must be identical set (no new rows, no missing rows)
3. `transit_time_hours` must be bit-identical to pinn_training_flat (no accidental join pollution)
4. CDAW match rate: log how many events got `accel_kms2` (expect ~70–80% match)
5. SHARP match rate: log how many events got `meangam` (expect ~30–50% match — many events have no AR)
6. Cluster match rate: log how many events got `cluster_id_k8` (expect ~100%)

## Rules
- Do NOT drop events with partial feature coverage. Use NaN and let the model handle it.
- Do NOT rebuild `pinn_training_flat` — write to a new `pinn_expanded_flat` table.
- Do NOT include in-fold model predictions as features for the same fold's events.
- Read `docs/DATA_SCHEMA_REFERENCE.md` before writing any SQL join to verify column names.
- Use `/db-schema-lookup` if any column name is uncertain.
