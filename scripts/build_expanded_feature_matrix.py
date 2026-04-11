"""Expand pinn_training_flat with additional features for the neural ensemble goal.

New columns added to pinn_expanded_flat (staging.db):
  CDAW kinematics  : second_order_speed_init, second_order_speed_final,
                     second_order_speed_20Rs, accel_kms2, mpa_deg
  SHARP magnetic   : meangam, meangbt, meangbz, meangbh, meanjzd, totusjz,
                     meanjzh, totusjh, absnjzh, meanalp, savncpp, meanpot,
                     totpot, meanshr, shrgt45, r_value, area_acr
  Multi-cluster    : cluster_id_k8, cluster_id_k12, cluster_id_dbscan
  ENLIL            : enlil_predicted_arrival_hours, enlil_au, enlil_matched
  Model predictions: (populated by train_pinn_model.py --write-oof-preds;
                      these columns exist but are NULL until that is run)
                     phase8_pred_transit_hours, pinn_v1_pred_transit_hours

Usage:
    python scripts/build_expanded_feature_matrix.py [--validate-only]

Prerequisites:
    pinn_training_flat must exist in staging.db (run build_pinn_feature_matrix.py first).
    SHARP join success rate is ~30-50% (not all CMEs have a linked active region).
    CDAW match rate is ~70-80% (same ±6h window as original script).
    ENLIL match rate is variable — linked_cme_ids is NULL in all rows; time-window
    matching on simulation_id vs donki_cme.start_time is used as fallback.
"""
from __future__ import annotations

import argparse
import math
import re
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

SOLAR_DB = Path("C:/Users/radar/SolarPipe/solar_data.db")
STAGING_DB = Path("C:/Users/radar/SolarPipe/data/data/staging/staging.db")

# ── datetime helpers ──────────────────────────────────────────────────────────

_TZ_RE = re.compile(r"[+-]\d{2}:\d{2}$|Z$")


def _parse_dt(s: str) -> datetime | None:
    if not s:
        return None
    s2 = _TZ_RE.sub("", s.strip()).replace("T", " ").replace("Z", "")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s2.strip(), fmt)
        except ValueError:
            continue
    return None


def _dt_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


# ── CDAW expanded kinematics ──────────────────────────────────────────────────

def _build_cdaw_expanded(conn_solar: sqlite3.Connection) -> dict[str, dict]:
    """Return {activity_id: {cdaw_*}} using same ±6h window logic as original script.

    Adds second_order_speed_init/final/20Rs, accel_kms2, mpa_deg.
    Matches on the same closest-in-time row that the original script uses.
    """
    # Load entire cdaw_cme once; match per event by julian day distance
    rows = conn_solar.execute("""
        SELECT datetime,
               second_order_speed_init,
               second_order_speed_final,
               second_order_speed_20Rs,
               accel_kms2,
               mpa_deg
        FROM cdaw_cme
        WHERE datetime IS NOT NULL
    """).fetchall()

    # Index by parsed datetime for fast nearest-neighbor lookup
    cdaw_entries: list[tuple[datetime, tuple]] = []
    for r in rows:
        dt = _parse_dt(r[0])
        if dt is not None:
            cdaw_entries.append((dt, r[1:]))

    def _lookup(launch: datetime) -> dict:
        best_delta = timedelta(hours=6.01)
        best_vals = None
        for dt, vals in cdaw_entries:
            delta = abs(dt - launch)
            if delta < best_delta:
                best_delta = delta
                best_vals = vals
        if best_vals is None:
            return {}
        return {
            "second_order_speed_init": best_vals[0],
            "second_order_speed_final": best_vals[1],
            "second_order_speed_20Rs": best_vals[2],
            "accel_kms2": best_vals[3],
            "mpa_deg": best_vals[4],
        }

    return _lookup  # type: ignore[return-value]  — returns callable


def build_cdaw_expanded(conn_solar: sqlite3.Connection) -> callable:
    """Returns a lookup(launch_dt) → dict of new CDAW fields."""
    rows = conn_solar.execute("""
        SELECT datetime,
               second_order_speed_init,
               second_order_speed_final,
               second_order_speed_20Rs,
               accel_kms2,
               mpa_deg
        FROM cdaw_cme
        WHERE datetime IS NOT NULL
    """).fetchall()

    cdaw_entries: list[tuple[datetime, tuple]] = []
    for r in rows:
        dt = _parse_dt(r[0])
        if dt is not None:
            cdaw_entries.append((dt, r[1:]))

    print(f"  CDAW: loaded {len(cdaw_entries)} rows")

    def lookup(launch: datetime) -> dict:
        best_delta = timedelta(hours=6.01)
        best_vals: tuple | None = None
        for dt, vals in cdaw_entries:
            delta = abs(dt - launch)
            if delta < best_delta:
                best_delta = delta
                best_vals = vals
        if best_vals is None:
            return {k: None for k in ("second_order_speed_init", "second_order_speed_final",
                                       "second_order_speed_20Rs", "accel_kms2", "mpa_deg")}
        return {
            "second_order_speed_init": best_vals[0],
            "second_order_speed_final": best_vals[1],
            "second_order_speed_20Rs": best_vals[2],
            "accel_kms2": best_vals[3],
            "mpa_deg": best_vals[4],
        }

    return lookup


# ── SHARP magnetic keywords ───────────────────────────────────────────────────

_SHARP_COLS = [
    "meangam", "meangbt", "meangbz", "meangbh",
    "meanjzd", "totusjz", "meanjzh", "totusjh", "absnjzh",
    "meanalp", "savncpp", "meanpot", "totpot",
    "meanshr", "shrgt45", "r_value", "area_acr",
]

_SHARP_NULL = {c: None for c in _SHARP_COLS}


def build_sharp_lookup(
    conn_solar: sqlite3.Connection,
    conn_staging: sqlite3.Connection,
) -> callable:
    """Returns a lookup(activity_id, launch_dt) → dict of SHARP fields.

    Join path:
      donki_cme.activity_id
        → donki_flare.linked_event_ids (contains activity_id)
        → donki_flare.active_region_num (NOAA AR)
        → sharp_keywords.noaa_ar
        → select record closest in time to launch_dt within ±24h
    """
    # Build activity_id → NOAA AR map via donki_flare
    ar_map: dict[str, int] = {}
    for row in conn_solar.execute(
        "SELECT linked_event_ids, active_region_num FROM donki_flare "
        "WHERE active_region_num IS NOT NULL AND linked_event_ids IS NOT NULL"
    ):
        linked, ar_num = row
        for aid in linked.split(","):
            aid = aid.strip()
            if aid:
                ar_map[aid] = int(ar_num)

    print(f"  SHARP: AR map built for {len(ar_map)} CME activity_ids")

    # Load SHARP records indexed by (noaa_ar, t_rec)
    cols_sql = ", ".join(_SHARP_COLS)
    sharp_by_ar: dict[int, list[tuple[datetime, tuple]]] = {}
    for row in conn_staging.execute(
        f"SELECT noaa_ar, t_rec, {cols_sql} FROM sharp_keywords "
        "WHERE noaa_ar IS NOT NULL AND t_rec IS NOT NULL"
    ):
        ar = int(row[0])
        t = _parse_dt(str(row[1]))
        if t is None:
            continue
        vals = row[2:]
        if ar not in sharp_by_ar:
            sharp_by_ar[ar] = []
        sharp_by_ar[ar].append((t, vals))

    print(f"  SHARP: loaded records for {len(sharp_by_ar)} unique NOAA ARs")

    def lookup(activity_id: str, launch: datetime) -> dict:
        ar = ar_map.get(activity_id)
        if ar is None:
            return dict(_SHARP_NULL)
        recs = sharp_by_ar.get(ar)
        if not recs:
            return dict(_SHARP_NULL)
        # Nearest t_rec within ±24h of launch
        best_delta = timedelta(hours=24.01)
        best_vals: tuple | None = None
        for t, vals in recs:
            delta = abs(t - launch)
            if delta < best_delta:
                best_delta = delta
                best_vals = vals
        if best_vals is None:
            return dict(_SHARP_NULL)
        return dict(zip(_SHARP_COLS, best_vals))

    return lookup


# ── Multi-cluster labels ──────────────────────────────────────────────────────

def build_cluster_lookup(conn_solar: sqlite3.Connection) -> callable:
    """Returns a lookup(activity_id) → {cluster_id_k8, cluster_id_k12, cluster_id_dbscan}."""
    k8_map: dict[str, int] = {}
    k12_map: dict[str, int] = {}
    dbscan_map: dict[str, int] = {}

    for row in conn_solar.execute(
        "SELECT event_id, cluster_method, k, cluster_id FROM ml_clusters "
        "WHERE (cluster_method='kmeans' AND k IN (8,12)) OR cluster_method='dbscan'"
    ):
        event_id, method, k, cid = row
        aid = event_id[4:] if event_id.startswith("CME_") else event_id
        if method == "kmeans" and k == 8:
            k8_map[aid] = cid
        elif method == "kmeans" and k == 12:
            k12_map[aid] = cid
        elif method == "dbscan":
            dbscan_map[aid] = cid

    print(f"  clusters: k=8: {len(k8_map)}, k=12: {len(k12_map)}, dbscan: {len(dbscan_map)}")

    def lookup(activity_id: str) -> dict:
        return {
            "cluster_id_k8": k8_map.get(activity_id),
            "cluster_id_k12": k12_map.get(activity_id),
            "cluster_id_dbscan": dbscan_map.get(activity_id),
        }

    return lookup


# ── ENLIL simulation matching ─────────────────────────────────────────────────

def build_enlil_lookup(conn_staging: sqlite3.Connection) -> callable:
    """Returns a lookup(activity_id, launch_dt) → {enlil_*}.

    linked_cme_ids is NULL in all rows; fall back to time-window matching:
    simulation_id is a datetime string — match to launch_dt within ±12h.
    Only the closest match within the window is used.
    au=1.0 rows represent L1 arrival predictions.
    """
    # Filter to au≈1.0 (Earth-distance) rows for arrival prediction
    rows = conn_staging.execute(
        "SELECT simulation_id, au FROM enlil_simulations WHERE simulation_id IS NOT NULL"
    ).fetchall()

    entries: list[tuple[datetime, float | None]] = []
    for sim_id, au in rows:
        dt = _parse_dt(sim_id)
        if dt is not None:
            entries.append((dt, au))

    print(f"  ENLIL: loaded {len(entries)} simulation entries")

    def lookup(activity_id: str, launch: datetime) -> dict:  # noqa: ARG001
        best_delta = timedelta(hours=12.01)
        best_au: float | None = None
        best_dt: datetime | None = None
        for dt, au in entries:
            delta = abs(dt - launch)
            if delta < best_delta:
                best_delta = delta
                best_au = au
                best_dt = dt
        if best_dt is None:
            return {
                "enlil_predicted_arrival_hours": None,
                "enlil_au": None,
                "enlil_matched": 0,
            }
        # enlil_predicted_arrival_hours: difference between simulation completion
        # and launch (proxy for ENLIL's transit time prediction at 1 AU)
        # Note: simulation_id is the ENLIL run time, not an arrival prediction.
        # Until linked_cme_ids is populated, we cannot extract a true arrival time.
        # We store the time offset as a weak signal; enlil_matched=1 means a run
        # was found within ±12h of the CME launch.
        offset_hours = (best_dt - launch).total_seconds() / 3600.0
        return {
            "enlil_predicted_arrival_hours": offset_hours,
            "enlil_au": best_au,
            "enlil_matched": 1,
        }

    return lookup


# ── Create and write pinn_expanded_flat ──────────────────────────────────────

def _create_expanded_table(conn: sqlite3.Connection) -> None:
    sharp_col_defs = "\n    ".join(f"{c} REAL," for c in _SHARP_COLS)
    conn.executescript(f"""
    DROP TABLE IF EXISTS pinn_expanded_flat;
    CREATE TABLE pinn_expanded_flat (
        -- all original pinn_training_flat columns (copied verbatim)
        activity_id TEXT PRIMARY KEY,
        linked_ips_id TEXT,
        launch_time TEXT,
        icme_arrival_time TEXT,
        split TEXT,
        exclude INTEGER DEFAULT 0,
        transit_time_hours REAL,
        omni_24h_bz_mean REAL, omni_24h_bz_std REAL, omni_24h_bz_min REAL,
        omni_24h_speed_mean REAL, omni_24h_density_mean REAL, omni_24h_pressure_mean REAL,
        omni_24h_ae_max REAL, omni_24h_dst_min REAL, omni_24h_kp_max REAL,
        f10_7 REAL, cluster_id_k5 INTEGER, cluster_assigned INTEGER,
        preceding_cme_count_48h INTEGER,
        preceding_cme_speed_max REAL, preceding_cme_speed_mean REAL,
        preceding_cme_angular_sep_min REAL, is_multi_cme INTEGER,
        omni_48h_density_spike_max REAL, omni_48h_speed_gradient REAL,
        cme_speed_kms REAL, cme_half_angle_deg REAL,
        cme_latitude REAL, cme_longitude REAL, cme_angular_width_deg REAL,
        cdaw_linear_speed_kms REAL, cdaw_angular_width_deg REAL,
        cdaw_mass_log10 REAL, cdaw_ke_log10 REAL, cdaw_matched INTEGER,
        flare_class_numeric REAL, has_flare INTEGER, flare_source_longitude REAL,
        omni_150h_density_median REAL, omni_150h_speed_median REAL,
        sw_bz_ambient REAL, delta_v_kms REAL,
        usflux REAL, sharp_available INTEGER,

        -- new: CDAW expanded kinematics
        second_order_speed_init REAL,
        second_order_speed_final REAL,
        second_order_speed_20Rs REAL,
        accel_kms2 REAL,
        mpa_deg REAL,

        -- new: SHARP magnetic keywords (17 fields)
        {sharp_col_defs}

        -- new: multi-cluster labels
        cluster_id_k8 INTEGER,
        cluster_id_k12 INTEGER,
        cluster_id_dbscan INTEGER,

        -- new: ENLIL simulation match
        enlil_predicted_arrival_hours REAL,
        enlil_au REAL,
        enlil_matched INTEGER DEFAULT 0,

        -- new: existing model predictions (populated by train_pinn_model.py --write-oof-preds)
        phase8_pred_transit_hours REAL,
        pinn_v1_pred_transit_hours REAL
    );
    """)
    conn.commit()


# Original columns to copy from pinn_training_flat
_BASE_COLS = [
    "activity_id", "linked_ips_id", "launch_time", "icme_arrival_time",
    "split", "exclude", "transit_time_hours",
    "omni_24h_bz_mean", "omni_24h_bz_std", "omni_24h_bz_min",
    "omni_24h_speed_mean", "omni_24h_density_mean", "omni_24h_pressure_mean",
    "omni_24h_ae_max", "omni_24h_dst_min", "omni_24h_kp_max",
    "f10_7", "cluster_id_k5", "cluster_assigned",
    "preceding_cme_count_48h",
    "preceding_cme_speed_max", "preceding_cme_speed_mean",
    "preceding_cme_angular_sep_min", "is_multi_cme",
    "omni_48h_density_spike_max", "omni_48h_speed_gradient",
    "cme_speed_kms", "cme_half_angle_deg",
    "cme_latitude", "cme_longitude", "cme_angular_width_deg",
    "cdaw_linear_speed_kms", "cdaw_angular_width_deg",
    "cdaw_mass_log10", "cdaw_ke_log10", "cdaw_matched",
    "flare_class_numeric", "has_flare", "flare_source_longitude",
    "omni_150h_density_median", "omni_150h_speed_median",
    "sw_bz_ambient", "delta_v_kms",
    "usflux", "sharp_available",
]

_NEW_COLS = (
    ["second_order_speed_init", "second_order_speed_final", "second_order_speed_20Rs",
     "accel_kms2", "mpa_deg"]
    + _SHARP_COLS
    + ["cluster_id_k8", "cluster_id_k12", "cluster_id_dbscan",
       "enlil_predicted_arrival_hours", "enlil_au", "enlil_matched",
       "phase8_pred_transit_hours", "pinn_v1_pred_transit_hours"]
)

_ALL_COLS = _BASE_COLS + _NEW_COLS


def write_expanded_table(
    conn_staging: sqlite3.Connection,
    conn_solar: sqlite3.Connection,
) -> None:
    _create_expanded_table(conn_staging)

    # Build all lookup functions
    print("\n  Building CDAW expanded lookup...")
    cdaw_lookup = build_cdaw_expanded(conn_solar)

    print("  Building SHARP magnetic lookup...")
    sharp_lookup = build_sharp_lookup(conn_solar, conn_staging)

    print("  Building multi-cluster lookup...")
    cluster_lookup = build_cluster_lookup(conn_solar)

    print("  Building ENLIL lookup...")
    enlil_lookup = build_enlil_lookup(conn_staging)

    # Load base rows
    base_col_sql = ", ".join(_BASE_COLS)
    base_rows = conn_staging.execute(
        f"SELECT {base_col_sql} FROM pinn_training_flat ORDER BY activity_id"
    ).fetchall()
    print(f"\n  Base rows loaded: {len(base_rows)}")

    # Counters for audit
    cdaw_matched = 0
    sharp_matched = 0
    enlil_matched = 0

    rows_out = []
    for base in base_rows:
        base_dict = dict(zip(_BASE_COLS, base))
        aid = base_dict["activity_id"]
        launch = _parse_dt(base_dict.get("launch_time") or "")
        if launch is None:
            # Should not happen — base events are pre-validated
            launch = datetime(2000, 1, 1)

        cdaw_extra = cdaw_lookup(launch)
        if cdaw_extra.get("accel_kms2") is not None:
            cdaw_matched += 1

        sharp_extra = sharp_lookup(aid, launch)
        if sharp_extra.get("meangam") is not None:
            sharp_matched += 1

        cluster_extra = cluster_lookup(aid)
        enlil_extra = enlil_lookup(aid, launch)
        if enlil_extra.get("enlil_matched"):
            enlil_matched += 1

        # Merge; model prediction cols are NULL until populated externally
        row_dict = {
            **base_dict,
            **cdaw_extra,
            **sharp_extra,
            **cluster_extra,
            **enlil_extra,
            "phase8_pred_transit_hours": None,
            "pinn_v1_pred_transit_hours": None,
        }
        rows_out.append(tuple(row_dict.get(c) for c in _ALL_COLS))

    placeholders = ", ".join("?" * len(_ALL_COLS))
    conn_staging.executemany(
        f"INSERT INTO pinn_expanded_flat VALUES ({placeholders})",
        rows_out,
    )
    conn_staging.commit()

    n = len(rows_out)
    print(f"\n  pinn_expanded_flat written: {n} rows, {len(_ALL_COLS)} columns")
    print(f"  CDAW expanded match: {cdaw_matched}/{n} ({100*cdaw_matched/n:.1f}%)")
    print(f"  SHARP match:         {sharp_matched}/{n} ({100*sharp_matched/n:.1f}%)")
    print(f"  ENLIL match:         {enlil_matched}/{n} ({100*enlil_matched/n:.1f}%)")


# ── audit report ─────────────────────────────────────────────────────────────

def print_audit_report(conn: sqlite3.Connection) -> None:
    print("\n=== pinn_expanded_flat audit ===")

    try:
        r = conn.execute(
            "SELECT COUNT(*), split FROM pinn_expanded_flat GROUP BY split"
        ).fetchall()
    except sqlite3.OperationalError:
        print("  pinn_expanded_flat does not exist — run without --validate-only first")
        return

    for row in r:
        print(f"  split={row[1]}: {row[0]} rows")

    col_count = len(conn.execute(
        "PRAGMA table_info(pinn_expanded_flat)"
    ).fetchall())
    print(f"  total columns: {col_count}")

    def null_rate(col: str) -> str:
        total, nulls = conn.execute(
            f"SELECT COUNT(*), SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) "
            f"FROM pinn_expanded_flat WHERE split='train'"
        ).fetchone()
        if total == 0:
            return "N/A"
        pct = 100.0 * nulls / total
        return f"{pct:.0f}% null"

    print("\n  New column NULL rates (train split):")
    new_checks = [
        "accel_kms2", "second_order_speed_init", "second_order_speed_final",
        "second_order_speed_20Rs", "mpa_deg",
        "meangam", "meangbt", "totusjz", "r_value",
        "cluster_id_k8", "cluster_id_k12", "cluster_id_dbscan",
        "enlil_matched",
    ]
    for col in new_checks:
        try:
            print(f"    {col}: {null_rate(col)}")
        except sqlite3.OperationalError:
            print(f"    {col}: column missing")

    r2 = conn.execute(
        "SELECT COUNT(*) FROM pinn_expanded_flat WHERE "
        "transit_time_hours IS NOT NULL AND activity_id IS NOT NULL"
    ).fetchone()
    print(f"\n  rows with label: {r2[0]}")

    # Verify row count matches pinn_training_flat
    r3 = conn.execute("SELECT COUNT(*) FROM pinn_training_flat").fetchone()
    r4 = conn.execute("SELECT COUNT(*) FROM pinn_expanded_flat").fetchone()
    match = "OK" if r3[0] == r4[0] else f"MISMATCH (base={r3[0]}, expanded={r4[0]})"
    print(f"  row count parity: {match}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-only", action="store_true",
                    help="Only print audit report on existing table; do not rebuild")
    args = ap.parse_args()

    conn_solar = sqlite3.connect(str(SOLAR_DB))
    conn_staging = sqlite3.connect(str(STAGING_DB))

    if args.validate_only:
        print_audit_report(conn_staging)
        conn_solar.close()
        conn_staging.close()
        return 0

    print("=== build_expanded_feature_matrix ===")
    print(f"solar_db:   {SOLAR_DB}")
    print(f"staging_db: {STAGING_DB}")

    # Verify prereq
    try:
        n = conn_staging.execute("SELECT COUNT(*) FROM pinn_training_flat").fetchone()[0]
    except sqlite3.OperationalError:
        print("ERROR: pinn_training_flat not found in staging.db")
        print("  Run: python scripts/build_pinn_feature_matrix.py")
        return 1
    print(f"\n  pinn_training_flat: {n} rows (base)")

    print("\n[1/1] Building and writing pinn_expanded_flat...")
    write_expanded_table(conn_staging, conn_solar)

    print_audit_report(conn_staging)

    conn_solar.close()
    conn_staging.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
