"""Build PINN feature matrix from full 2010-2025 CME/IPS catalog.

Outputs 5 tables to staging.db (non-destructive, does not touch feature_vectors):
  pinn_events             - base: one row per CME with transit label + split flag
  pinn_regime_features    - Stage 1: 24h pre-launch OMNI + F10.7 + cluster
  pinn_interaction_features - Stage 2: 48h preceding CME catalog + OMNI spikes
  pinn_physics_features   - Stage 3: 150h OMNI + CDAW + flare
  pinn_training_flat      - all features joined, ready for model input

Usage:
    python scripts/build_pinn_feature_matrix.py [--validate-only]
"""
from __future__ import annotations

import argparse
import math
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SOLAR_DB = Path("C:/Users/radar/SolarPipe/solar_data.db")
STAGING_DB = Path("C:/Users/radar/SolarPipe/data/data/staging/staging.db")

HOLDOUT_START = "2026-01-01"
TRANSIT_MIN = 10.0
TRANSIT_MAX = 200.0

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


# ── OMNI window helper ────────────────────────────────────────────────────────

_SENTINEL_LARGE = 9990.0


def _omni_window(
    conn_solar: sqlite3.Connection,
    launch: datetime,
    hours_before: int,
    cols: list[str],
) -> dict[str, list[float | None]]:
    """Return {col: [hourly values]} for the N hours before launch.

    Hours with NULL or sentinel (>=9990 or <=-1e30) stay as None.
    Gap interpolation: linear fill within runs of <=6 consecutive Nones.
    """
    t_start = launch - timedelta(hours=hours_before)
    t_end = launch
    start_str = _dt_str(t_start)
    end_str = _dt_str(t_end)
    col_sql = ", ".join(f'"{c}"' for c in cols)
    rows = conn_solar.execute(
        f'SELECT datetime, {col_sql} FROM omni_hourly '
        f'WHERE datetime >= ? AND datetime < ? ORDER BY datetime',
        (start_str, end_str),
    ).fetchall()

    result: dict[str, list[float | None]] = {c: [] for c in cols}
    for row in rows:
        for i, c in enumerate(cols):
            v = row[i + 1]
            if v is not None and (abs(v) >= _SENTINEL_LARGE or v <= -1e29):
                v = None
            result[c].append(v)

    # Linear interpolation for runs of <=6 consecutive None
    for c in cols:
        vals = result[c]
        n = len(vals)
        i = 0
        while i < n:
            if vals[i] is None:
                run_start = i
                while i < n and vals[i] is None:
                    i += 1
                run_len = i - run_start
                if run_len <= 6:
                    left = vals[run_start - 1] if run_start > 0 else None
                    right = vals[i] if i < n else None
                    if left is not None and right is not None:
                        for j in range(run_len):
                            frac = (j + 1) / (run_len + 1)
                            vals[run_start + j] = left + frac * (right - left)
            else:
                i += 1
    return result


def _safe_mean(vals: list[float | None]) -> float | None:
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def _safe_std(vals: list[float | None]) -> float | None:
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return None
    mu = sum(v) / len(v)
    return math.sqrt(sum((x - mu) ** 2 for x in v) / len(v))


def _safe_min(vals: list[float | None]) -> float | None:
    v = [x for x in vals if x is not None]
    return min(v) if v else None


def _safe_max(vals: list[float | None]) -> float | None:
    v = [x for x in vals if x is not None]
    return max(v) if v else None


def _safe_median(vals: list[float | None]) -> float | None:
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    n = len(v)
    mid = n // 2
    return (v[mid - 1] + v[mid]) / 2 if n % 2 == 0 else v[mid]


# ── Step 1: load base events ──────────────────────────────────────────────────

_SHARP_FIELDS = [
    "usflux", "meangam", "meangbt", "meangbz", "meangbh",
    "meanjzd", "totusjz", "meanalp", "meanjzh", "totusjh",
    "absnjzh", "savncpp", "meanpot", "totpot", "meanshr",
    "shrgt45", "r_value", "area_acr",
]


def load_base_events(conn_staging: sqlite3.Connection) -> list[dict]:
    sharp_cols = ", ".join(_SHARP_FIELDS)
    rows = conn_staging.execute(f"""
        SELECT activity_id, launch_time, icme_arrival_time, transit_time_hours,
               linked_ips_id, cme_speed_kms, cme_half_angle_deg,
               cme_latitude, cme_longitude, cme_angular_width_deg,
               sw_speed_ambient, sw_density_ambient, sw_bz_ambient,
               flare_class_numeric, flare_class_letter,
               linked_flare_id, sharp_harpnum,
               f10_7,
               {sharp_cols}
        FROM feature_vectors
        WHERE transit_time_hours BETWEEN ? AND ?
    """, (TRANSIT_MIN, TRANSIT_MAX)).fetchall()

    events = []
    for r in rows:
        launch = _parse_dt(r[1])
        if launch is None:
            continue
        tt = r[3]
        if tt is None or tt < 0:
            continue  # exclude data errors
        split = "holdout" if (launch >= _parse_dt(HOLDOUT_START)) else "train"
        ev: dict = {
            "activity_id": r[0],
            "launch_time": _dt_str(launch),
            "launch_dt": launch,
            "icme_arrival_time": r[2],
            "transit_time_hours": tt,
            "linked_ips_id": r[4],
            "cme_speed_kms": r[5],
            "cme_half_angle_deg": r[6],
            "cme_latitude": r[7],
            "cme_longitude": r[8],
            "cme_angular_width_deg": r[9],
            "sw_speed_ambient": r[10],
            "sw_density_ambient": r[11],
            "sw_bz_ambient": r[12],
            "flare_class_numeric": r[13],
            "flare_class_letter": r[14],
            "linked_flare_id": r[15],
            "sharp_available": 1 if r[16] else 0,
            "f10_7_fv": r[17],
            "split": split,
        }
        # SHARP magnetic fields (columns 18+)
        for i, field in enumerate(_SHARP_FIELDS):
            ev[field] = r[18 + i]
        events.append(ev)
    return events


# ── Step 2: regime features (24h OMNI + cluster + gfz) ───────────────────────

def build_regime_features(
    conn_solar: sqlite3.Connection,
    events: list[dict],
) -> None:
    """Adds regime_ keys in-place."""
    # Preload ml_clusters for k=5, k=8, k=12
    cluster_map: dict[str, int] = {}       # k=5
    cluster_map_k8: dict[str, int] = {}    # k=8
    cluster_map_k12: dict[str, int] = {}   # k=12
    pca_pts: list[tuple[float, float, int]] = []  # (x, y, cluster_id) for k=5 NNN fallback
    for row in conn_solar.execute(
        'SELECT event_id, cluster_id, k, pca_x, pca_y FROM ml_clusters '
        'WHERE cluster_method="kmeans" AND k IN (5, 8, 12)'
    ):
        event_id, cid, k, px, py = row
        aid = event_id[4:] if event_id.startswith("CME_") else event_id
        if k == 5:
            cluster_map[aid] = cid
            if px is not None and py is not None:
                pca_pts.append((px, py, cid))
        elif k == 8:
            cluster_map_k8[aid] = cid
        elif k == 12:
            cluster_map_k12[aid] = cid

    print(f"  ml_clusters loaded: k5={len(cluster_map)}, k8={len(cluster_map_k8)}, k12={len(cluster_map_k12)}, pca_pts={len(pca_pts)}")

    for ev in events:
        launch = ev["launch_dt"]
        cols = [
            "Bz_GSM", "flow_speed", "proton_density", "flow_pressure",
            "AE_nT", "Dst_nT", "Kp_x10",
            "Bx_GSE", "By_GSM", "sigma_Bz",
            "electric_field", "plasma_beta", "alfven_mach", "magnetosonic_mach",
        ]
        w = _omni_window(conn_solar, launch, 24, cols)

        ev["omni_24h_bz_mean"] = _safe_mean(w["Bz_GSM"])
        ev["omni_24h_bz_std"] = _safe_std(w["Bz_GSM"])
        ev["omni_24h_bz_min"] = _safe_min(w["Bz_GSM"])
        ev["omni_24h_speed_mean"] = _safe_mean(w["flow_speed"])
        ev["omni_24h_density_mean"] = _safe_mean(w["proton_density"])
        ev["omni_24h_pressure_mean"] = _safe_mean(w["flow_pressure"])
        ev["omni_24h_ae_max"] = _safe_max(w["AE_nT"])
        ev["omni_24h_dst_min"] = _safe_min(w["Dst_nT"])
        ev["omni_24h_kp_max"] = _safe_max(w["Kp_x10"])
        # Expanded OMNI channels
        ev["omni_24h_bx_mean"] = _safe_mean(w["Bx_GSE"])
        ev["omni_24h_by_mean"] = _safe_mean(w["By_GSM"])
        ev["omni_24h_bz_sigma"] = _safe_mean(w["sigma_Bz"])
        ev["omni_24h_efield_mean"] = _safe_mean(w["electric_field"])
        ev["omni_24h_plasma_beta_mean"] = _safe_mean(w["plasma_beta"])
        ev["omni_24h_alfven_mach_mean"] = _safe_mean(w["alfven_mach"])
        ev["omni_24h_ms_mach_mean"] = _safe_mean(w["magnetosonic_mach"])

        # F10.7 — prefer gfz daily, fall back to feature_vectors
        date_str = launch.strftime("%Y-%m-%d")
        f107_row = conn_solar.execute(
            'SELECT daily_F10_7_obs FROM gfz_kp_ap WHERE date=? LIMIT 1', (date_str,)
        ).fetchone()
        if f107_row and f107_row[0] is not None:
            ev["f10_7"] = f107_row[0]
        else:
            ev["f10_7"] = ev.get("f10_7_fv")

        # cluster IDs — k5, k8, k12
        aid = ev["activity_id"]
        if aid in cluster_map:
            ev["cluster_id_k5"] = cluster_map[aid]
            ev["cluster_assigned"] = 0
        else:
            ev["cluster_id_k5"] = None
            ev["cluster_assigned"] = None
        ev["cluster_id_k8"] = cluster_map_k8.get(aid)
        ev["cluster_id_k12"] = cluster_map_k12.get(aid)

    # NNN imputation for unmatched training events (k=5 only; k8/k12 have full coverage)
    train_unmatched = [e for e in events if e["split"] == "train" and e["cluster_id_k5"] is None]
    if train_unmatched and pca_pts:
        print(f"  NNN cluster imputation for {len(train_unmatched)} unmatched training events")
        for ev in train_unmatched:
            # Use ambient speed/density as proxy PCA coords (normalized)
            spd = ev.get("sw_speed_ambient") or ev.get("omni_24h_speed_mean") or 400.0
            den = ev.get("sw_density_ambient") or ev.get("omni_24h_density_mean") or 5.0
            # Normalize to rough PCA scale
            px_query = (spd - 400.0) / 150.0
            py_query = (den - 5.0) / 5.0
            best_dist = float("inf")
            best_cid = 0
            for px, py, cid in pca_pts:
                d = (px - px_query) ** 2 + (py - py_query) ** 2
                if d < best_dist:
                    best_dist = d
                    best_cid = cid
            ev["cluster_id_k5"] = best_cid
            ev["cluster_assigned"] = 1


# ── Step 3: interaction features (48h preceding CME catalog) ─────────────────

def _angular_sep(lat1: float | None, lon1: float | None, lat2: float | None, lon2: float | None) -> float | None:
    if any(x is None for x in (lat1, lon1, lat2, lon2)):
        return None
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return math.degrees(2 * math.asin(min(1, math.sqrt(a))))


def build_interaction_features(
    conn_solar: sqlite3.Connection,
    events: list[dict],
) -> None:
    """Adds interaction_ keys in-place."""
    # Build IPS n_linked_events map
    ips_multi: dict[str, int] = {}
    for row in conn_solar.execute("SELECT ips_id, n_linked_events FROM donki_ips"):
        ips_multi[row[0]] = row[1]

    for ev in events:
        launch = ev["launch_dt"]
        t_start = _dt_str(launch - timedelta(hours=48))
        t_end = _dt_str(launch)
        aid = ev["activity_id"]

        # Preceding CMEs (exclude self)
        prec = conn_solar.execute("""
            SELECT speed_kms, latitude, longitude FROM donki_cme
            WHERE start_time >= ? AND start_time < ? AND activity_id != ?
        """, (t_start, t_end, aid)).fetchall()

        ev["preceding_cme_count_48h"] = len(prec)
        speeds = [r[0] for r in prec if r[0] is not None]
        ev["preceding_cme_speed_max"] = max(speeds) if speeds else None
        ev["preceding_cme_speed_mean"] = (sum(speeds) / len(speeds)) if speeds else None

        my_lat = ev.get("cme_latitude")
        my_lon = ev.get("cme_longitude")
        seps = [
            _angular_sep(my_lat, my_lon, r[1], r[2])
            for r in prec
        ]
        seps = [s for s in seps if s is not None]
        ev["preceding_cme_angular_sep_min"] = min(seps) if seps else None

        linked_ips = ev.get("linked_ips_id") or ""
        n_linked = ips_multi.get(linked_ips, 0)
        ev["is_multi_cme"] = 1 if n_linked > 1 else 0

        # 48h OMNI density spike + speed gradient
        cols_48 = ["proton_density", "flow_speed"]
        w48 = _omni_window(conn_solar, launch, 48, cols_48)
        ev["omni_48h_density_spike_max"] = _safe_max(w48["proton_density"])
        spd48 = [v for v in w48["flow_speed"] if v is not None]
        if len(spd48) >= 2:
            ev["omni_48h_speed_gradient"] = spd48[-1] - spd48[0]
        else:
            ev["omni_48h_speed_gradient"] = None


# ── Step 4: physics features (150h OMNI + CDAW + flare) ──────────────────────

_LOC_RE = re.compile(r"([NS])(\d+)([EW])(\d+)", re.IGNORECASE)


def _parse_source_location(loc: str | None) -> float | None:
    if not loc:
        return None
    m = _LOC_RE.search(loc)
    if not m:
        return None
    ew, deg = m.group(3).upper(), int(m.group(4))
    return deg if ew == "W" else -deg


def build_physics_features(
    conn_solar: sqlite3.Connection,
    conn_staging: sqlite3.Connection,
    events: list[dict],
) -> None:
    """Adds physics_ keys in-place."""
    # Preload flare→CME link map (donki_flare.linked_event_ids → activity_id)
    flare_map: dict[str, tuple[float | None, float | None]] = {}
    for row in conn_solar.execute(
        "SELECT flare_id, class_type, source_location, linked_event_ids FROM donki_flare"
    ):
        fid, cls, src, linked = row
        if not linked:
            continue
        # class_type → numeric
        cls_num = _flare_class_numeric(cls)
        src_lon = _parse_source_location(src)
        for aid in linked.split(","):
            aid = aid.strip()
            if aid:
                flare_map[aid] = (cls_num, src_lon)

    for ev in events:
        launch = ev["launch_dt"]
        aid = ev["activity_id"]

        # 150h OMNI
        cols_150 = ["proton_density", "flow_speed"]
        w150 = _omni_window(conn_solar, launch, 150, cols_150)
        ev["omni_150h_density_median"] = _safe_median(w150["proton_density"])
        ev["omni_150h_speed_median"] = _safe_median(w150["flow_speed"])

        spd_cme = ev.get("cme_speed_kms")
        spd_sw = ev.get("omni_150h_speed_median")
        ev["delta_v_kms"] = (spd_cme - spd_sw) if (spd_cme is not None and spd_sw is not None) else None

        # CDAW match ±6h
        t_str = _dt_str(launch)
        cdaw = conn_solar.execute("""
            SELECT linear_speed_kms, angular_width_deg, mass_grams, kinetic_energy_ergs,
                   second_order_speed_init, second_order_speed_final,
                   second_order_speed_20Rs, accel_kms2
            FROM cdaw_cme
            WHERE abs(julianday(datetime) - julianday(?)) * 24 <= 6
            ORDER BY abs(julianday(datetime) - julianday(?))
            LIMIT 1
        """, (t_str, t_str)).fetchone()
        if cdaw:
            ev["cdaw_linear_speed_kms"] = cdaw[0]
            ev["cdaw_angular_width_deg"] = cdaw[1]
            ev["cdaw_mass_log10"] = math.log10(cdaw[2]) if cdaw[2] and cdaw[2] > 0 else None
            ev["cdaw_ke_log10"] = math.log10(cdaw[3]) if cdaw[3] and cdaw[3] > 0 else None
            ev["cdaw_2nd_speed_init"] = cdaw[4]
            ev["cdaw_2nd_speed_final"] = cdaw[5]
            ev["cdaw_2nd_speed_20rs"] = cdaw[6]
            ev["cdaw_accel_kms2"] = cdaw[7]
            ev["cdaw_matched"] = 1
        else:
            ev["cdaw_linear_speed_kms"] = None
            ev["cdaw_angular_width_deg"] = None
            ev["cdaw_mass_log10"] = None
            ev["cdaw_ke_log10"] = None
            ev["cdaw_2nd_speed_init"] = None
            ev["cdaw_2nd_speed_final"] = None
            ev["cdaw_2nd_speed_20rs"] = None
            ev["cdaw_accel_kms2"] = None
            ev["cdaw_matched"] = 0

        # Flare
        fl = flare_map.get(aid)
        if fl:
            ev["flare_class_numeric_db"] = fl[0]
            ev["flare_source_longitude"] = fl[1]
            ev["has_flare"] = 1
        else:
            ev["flare_class_numeric_db"] = ev.get("flare_class_numeric")
            ev["flare_source_longitude"] = None
            ev["has_flare"] = 1 if ev.get("flare_class_letter") else 0


def _flare_class_numeric(cls: str | None) -> float | None:
    if not cls:
        return None
    cls = cls.strip().upper()
    m = re.match(r"([ABCMX])(\d+\.?\d*)", cls)
    if not m:
        return None
    letter = m.group(1)
    num = float(m.group(2))
    base = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
    b = base.get(letter)
    return b * num if b else None


# ── Step 5: write tables ──────────────────────────────────────────────────────

def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    DROP TABLE IF EXISTS pinn_events;
    CREATE TABLE pinn_events (
        activity_id TEXT PRIMARY KEY,
        linked_ips_id TEXT,
        launch_time TEXT,
        icme_arrival_time TEXT,
        transit_time_hours REAL,
        split TEXT,
        exclude INTEGER DEFAULT 0
    );

    DROP TABLE IF EXISTS pinn_regime_features;
    CREATE TABLE pinn_regime_features (
        activity_id TEXT PRIMARY KEY,
        omni_24h_bz_mean REAL, omni_24h_bz_std REAL, omni_24h_bz_min REAL,
        omni_24h_speed_mean REAL, omni_24h_density_mean REAL, omni_24h_pressure_mean REAL,
        omni_24h_ae_max REAL, omni_24h_dst_min REAL, omni_24h_kp_max REAL,
        omni_24h_bx_mean REAL, omni_24h_by_mean REAL, omni_24h_bz_sigma REAL,
        omni_24h_efield_mean REAL, omni_24h_plasma_beta_mean REAL,
        omni_24h_alfven_mach_mean REAL, omni_24h_ms_mach_mean REAL,
        f10_7 REAL, cluster_id_k5 INTEGER, cluster_assigned INTEGER,
        cluster_id_k8 INTEGER, cluster_id_k12 INTEGER
    );

    DROP TABLE IF EXISTS pinn_interaction_features;
    CREATE TABLE pinn_interaction_features (
        activity_id TEXT PRIMARY KEY,
        preceding_cme_count_48h INTEGER,
        preceding_cme_speed_max REAL,
        preceding_cme_speed_mean REAL,
        preceding_cme_angular_sep_min REAL,
        is_multi_cme INTEGER,
        omni_48h_density_spike_max REAL,
        omni_48h_speed_gradient REAL
    );

    DROP TABLE IF EXISTS pinn_physics_features;
    CREATE TABLE pinn_physics_features (
        activity_id TEXT PRIMARY KEY,
        cme_speed_kms REAL, cme_half_angle_deg REAL,
        cme_latitude REAL, cme_longitude REAL, cme_angular_width_deg REAL,
        cdaw_linear_speed_kms REAL, cdaw_angular_width_deg REAL,
        cdaw_mass_log10 REAL, cdaw_ke_log10 REAL,
        cdaw_2nd_speed_init REAL, cdaw_2nd_speed_final REAL,
        cdaw_2nd_speed_20rs REAL, cdaw_accel_kms2 REAL,
        cdaw_matched INTEGER,
        flare_class_numeric REAL, has_flare INTEGER, flare_source_longitude REAL,
        omni_150h_density_median REAL, omni_150h_speed_median REAL,
        sw_bz_ambient REAL, delta_v_kms REAL,
        usflux REAL, meangam REAL, meangbt REAL, meangbz REAL, meangbh REAL,
        meanjzd REAL, totusjz REAL, meanalp REAL, meanjzh REAL, totusjh REAL,
        absnjzh REAL, savncpp REAL, meanpot REAL, totpot REAL, meanshr REAL,
        shrgt45 REAL, r_value REAL, area_acr REAL,
        sharp_available INTEGER
    );

    DROP TABLE IF EXISTS pinn_training_flat;
    CREATE TABLE pinn_training_flat (
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
        omni_24h_bx_mean REAL, omni_24h_by_mean REAL, omni_24h_bz_sigma REAL,
        omni_24h_efield_mean REAL, omni_24h_plasma_beta_mean REAL,
        omni_24h_alfven_mach_mean REAL, omni_24h_ms_mach_mean REAL,
        f10_7 REAL, cluster_id_k5 INTEGER, cluster_assigned INTEGER,
        cluster_id_k8 INTEGER, cluster_id_k12 INTEGER,
        preceding_cme_count_48h INTEGER,
        preceding_cme_speed_max REAL, preceding_cme_speed_mean REAL,
        preceding_cme_angular_sep_min REAL, is_multi_cme INTEGER,
        omni_48h_density_spike_max REAL, omni_48h_speed_gradient REAL,
        cme_speed_kms REAL, cme_half_angle_deg REAL,
        cme_latitude REAL, cme_longitude REAL, cme_angular_width_deg REAL,
        cdaw_linear_speed_kms REAL, cdaw_angular_width_deg REAL,
        cdaw_mass_log10 REAL, cdaw_ke_log10 REAL,
        cdaw_2nd_speed_init REAL, cdaw_2nd_speed_final REAL,
        cdaw_2nd_speed_20rs REAL, cdaw_accel_kms2 REAL,
        cdaw_matched INTEGER,
        flare_class_numeric REAL, has_flare INTEGER, flare_source_longitude REAL,
        omni_150h_density_median REAL, omni_150h_speed_median REAL,
        sw_bz_ambient REAL, delta_v_kms REAL,
        usflux REAL, meangam REAL, meangbt REAL, meangbz REAL, meangbh REAL,
        meanjzd REAL, totusjz REAL, meanalp REAL, meanjzh REAL, totusjh REAL,
        absnjzh REAL, savncpp REAL, meanpot REAL, totpot REAL, meanshr REAL,
        shrgt45 REAL, r_value REAL, area_acr REAL,
        sharp_available INTEGER
    );
    """)
    conn.commit()


def _g(ev: dict, key: Any, default: Any = None) -> Any:
    return ev.get(key, default)


def write_tables(conn: sqlite3.Connection, events: list[dict]) -> None:
    _create_tables(conn)

    conn.executemany(
        "INSERT INTO pinn_events VALUES (?,?,?,?,?,?,?)",
        [(_g(e, "activity_id"), _g(e, "linked_ips_id"), _g(e, "launch_time"),
          _g(e, "icme_arrival_time"), _g(e, "transit_time_hours"), _g(e, "split"), 0)
         for e in events],
    )

    conn.executemany(
        f"INSERT INTO pinn_regime_features VALUES ({','.join(['?']*22)})",
        [(_g(e, "activity_id"),
          _g(e, "omni_24h_bz_mean"), _g(e, "omni_24h_bz_std"), _g(e, "omni_24h_bz_min"),
          _g(e, "omni_24h_speed_mean"), _g(e, "omni_24h_density_mean"), _g(e, "omni_24h_pressure_mean"),
          _g(e, "omni_24h_ae_max"), _g(e, "omni_24h_dst_min"), _g(e, "omni_24h_kp_max"),
          _g(e, "omni_24h_bx_mean"), _g(e, "omni_24h_by_mean"), _g(e, "omni_24h_bz_sigma"),
          _g(e, "omni_24h_efield_mean"), _g(e, "omni_24h_plasma_beta_mean"),
          _g(e, "omni_24h_alfven_mach_mean"), _g(e, "omni_24h_ms_mach_mean"),
          _g(e, "f10_7"), _g(e, "cluster_id_k5"), _g(e, "cluster_assigned"),
          _g(e, "cluster_id_k8"), _g(e, "cluster_id_k12"))
         for e in events],
    )

    conn.executemany(
        "INSERT INTO pinn_interaction_features VALUES (?,?,?,?,?,?,?,?)",
        [(_g(e, "activity_id"),
          _g(e, "preceding_cme_count_48h"), _g(e, "preceding_cme_speed_max"),
          _g(e, "preceding_cme_speed_mean"), _g(e, "preceding_cme_angular_sep_min"),
          _g(e, "is_multi_cme"),
          _g(e, "omni_48h_density_spike_max"), _g(e, "omni_48h_speed_gradient"))
         for e in events],
    )

    conn.executemany(
        f"INSERT INTO pinn_physics_features VALUES ({','.join(['?']*41)})",
        [(_g(e, "activity_id"),
          _g(e, "cme_speed_kms"), _g(e, "cme_half_angle_deg"),
          _g(e, "cme_latitude"), _g(e, "cme_longitude"), _g(e, "cme_angular_width_deg"),
          _g(e, "cdaw_linear_speed_kms"), _g(e, "cdaw_angular_width_deg"),
          _g(e, "cdaw_mass_log10"), _g(e, "cdaw_ke_log10"),
          _g(e, "cdaw_2nd_speed_init"), _g(e, "cdaw_2nd_speed_final"),
          _g(e, "cdaw_2nd_speed_20rs"), _g(e, "cdaw_accel_kms2"),
          _g(e, "cdaw_matched"),
          _g(e, "flare_class_numeric_db"), _g(e, "has_flare"), _g(e, "flare_source_longitude"),
          _g(e, "omni_150h_density_median"), _g(e, "omni_150h_speed_median"),
          _g(e, "sw_bz_ambient"), _g(e, "delta_v_kms"),
          _g(e, "usflux"), _g(e, "meangam"), _g(e, "meangbt"), _g(e, "meangbz"), _g(e, "meangbh"),
          _g(e, "meanjzd"), _g(e, "totusjz"), _g(e, "meanalp"), _g(e, "meanjzh"), _g(e, "totusjh"),
          _g(e, "absnjzh"), _g(e, "savncpp"), _g(e, "meanpot"), _g(e, "totpot"), _g(e, "meanshr"),
          _g(e, "shrgt45"), _g(e, "r_value"), _g(e, "area_acr"),
          _g(e, "sharp_available"))
         for e in events],
    )

    conn.executemany(
        f"INSERT INTO pinn_training_flat VALUES ({','.join(['?']*75)})",
        [(_g(e, "activity_id"), _g(e, "linked_ips_id"), _g(e, "launch_time"),
          _g(e, "icme_arrival_time"), _g(e, "split"), 0, _g(e, "transit_time_hours"),
          # regime OMNI (16)
          _g(e, "omni_24h_bz_mean"), _g(e, "omni_24h_bz_std"), _g(e, "omni_24h_bz_min"),
          _g(e, "omni_24h_speed_mean"), _g(e, "omni_24h_density_mean"), _g(e, "omni_24h_pressure_mean"),
          _g(e, "omni_24h_ae_max"), _g(e, "omni_24h_dst_min"), _g(e, "omni_24h_kp_max"),
          _g(e, "omni_24h_bx_mean"), _g(e, "omni_24h_by_mean"), _g(e, "omni_24h_bz_sigma"),
          _g(e, "omni_24h_efield_mean"), _g(e, "omni_24h_plasma_beta_mean"),
          _g(e, "omni_24h_alfven_mach_mean"), _g(e, "omni_24h_ms_mach_mean"),
          # cluster (5)
          _g(e, "f10_7"), _g(e, "cluster_id_k5"), _g(e, "cluster_assigned"),
          _g(e, "cluster_id_k8"), _g(e, "cluster_id_k12"),
          # interaction (7)
          _g(e, "preceding_cme_count_48h"), _g(e, "preceding_cme_speed_max"),
          _g(e, "preceding_cme_speed_mean"), _g(e, "preceding_cme_angular_sep_min"),
          _g(e, "is_multi_cme"),
          _g(e, "omni_48h_density_spike_max"), _g(e, "omni_48h_speed_gradient"),
          # CME geometry (5)
          _g(e, "cme_speed_kms"), _g(e, "cme_half_angle_deg"),
          _g(e, "cme_latitude"), _g(e, "cme_longitude"), _g(e, "cme_angular_width_deg"),
          # CDAW (9)
          _g(e, "cdaw_linear_speed_kms"), _g(e, "cdaw_angular_width_deg"),
          _g(e, "cdaw_mass_log10"), _g(e, "cdaw_ke_log10"),
          _g(e, "cdaw_2nd_speed_init"), _g(e, "cdaw_2nd_speed_final"),
          _g(e, "cdaw_2nd_speed_20rs"), _g(e, "cdaw_accel_kms2"),
          _g(e, "cdaw_matched"),
          # flare (3)
          _g(e, "flare_class_numeric_db"), _g(e, "has_flare"), _g(e, "flare_source_longitude"),
          # OMNI150 + ambient (4)
          _g(e, "omni_150h_density_median"), _g(e, "omni_150h_speed_median"),
          _g(e, "sw_bz_ambient"), _g(e, "delta_v_kms"),
          # SHARP magnetic (18)
          _g(e, "usflux"), _g(e, "meangam"), _g(e, "meangbt"), _g(e, "meangbz"), _g(e, "meangbh"),
          _g(e, "meanjzd"), _g(e, "totusjz"), _g(e, "meanalp"), _g(e, "meanjzh"), _g(e, "totusjh"),
          _g(e, "absnjzh"), _g(e, "savncpp"), _g(e, "meanpot"), _g(e, "totpot"), _g(e, "meanshr"),
          _g(e, "shrgt45"), _g(e, "r_value"), _g(e, "area_acr"),
          # sharp flag (1)
          _g(e, "sharp_available"))
         for e in events],
    )

    conn.commit()


# ── Step 6: audit report ──────────────────────────────────────────────────────

def print_audit_report(conn: sqlite3.Connection) -> None:
    print("\n=== pinn_training_flat audit ===")

    r = conn.execute(
        "SELECT COUNT(*), split FROM pinn_training_flat GROUP BY split"
    ).fetchall()
    for row in r:
        print(f"  split={row[1]}: {row[0]} rows")

    def null_rate(col: str) -> str:
        total, nulls = conn.execute(
            f"SELECT COUNT(*), SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) "
            f"FROM pinn_training_flat WHERE split='train'"
        ).fetchone()
        if total == 0:
            return "N/A"
        pct = 100.0 * nulls / total
        return f"{nulls}/{total} ({pct:.1f}%)"

    checks = [
        "transit_time_hours", "omni_150h_density_median", "omni_150h_speed_median",
        "omni_24h_bz_mean", "cluster_id_k5", "preceding_cme_count_48h",
        "cdaw_linear_speed_kms", "flare_class_numeric", "has_flare", "delta_v_kms",
        "f10_7",
    ]
    print("\n  NULL rates in train split:")
    for col in checks:
        print(f"    {col}: {null_rate(col)}")

    r2 = conn.execute(
        "SELECT MIN(launch_time), MAX(launch_time) FROM pinn_training_flat WHERE split='holdout'"
    ).fetchone()
    print(f"\n  holdout range: {r2[0]} → {r2[1]}")

    r3 = conn.execute(
        "SELECT COUNT(*) FROM pinn_training_flat WHERE split='holdout'"
    ).fetchone()
    print(f"  holdout rows: {r3[0]}")

    r4 = conn.execute(
        "SELECT MIN(transit_time_hours), MAX(transit_time_hours), AVG(transit_time_hours) "
        "FROM pinn_training_flat WHERE split='train'"
    ).fetchone()
    print(f"\n  train transit_time_hours: min={r4[0]:.1f}h max={r4[1]:.1f}h mean={r4[2]:.1f}h")

    r5 = conn.execute(
        "SELECT SUM(is_multi_cme), COUNT(*) FROM pinn_training_flat WHERE split='train'"
    ).fetchone()
    print(f"  is_multi_cme: {r5[0]}/{r5[1]} ({100*r5[0]/r5[1]:.1f}%)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-only", action="store_true",
                    help="Only print audit report on existing tables; do not rebuild")
    args = ap.parse_args()

    conn_solar = sqlite3.connect(str(SOLAR_DB))
    conn_staging = sqlite3.connect(str(STAGING_DB))

    if args.validate_only:
        print_audit_report(conn_staging)
        return 0

    print("=== build_pinn_feature_matrix ===")
    print(f"solar_db:   {SOLAR_DB}")
    print(f"staging_db: {STAGING_DB}")

    print("\n[1/5] Loading base events...")
    events = load_base_events(conn_staging)
    print(f"  loaded {len(events)} events (transit 10–200h, exclude negatives)")
    n_train = sum(1 for e in events if e["split"] == "train")
    n_hold = sum(1 for e in events if e["split"] == "holdout")
    print(f"  train={n_train}, holdout={n_hold}")

    print("\n[2/5] Building regime features (24h OMNI + cluster + F10.7)...")
    build_regime_features(conn_solar, events)
    print("  done")

    print("\n[3/5] Building interaction features (48h preceding CME + OMNI spikes)...")
    build_interaction_features(conn_solar, events)
    print("  done")

    print("\n[4/5] Building physics features (150h OMNI + CDAW + flare)...")
    build_physics_features(conn_solar, conn_staging, events)
    print("  done")

    print("\n[5/5] Writing tables to staging.db...")
    write_tables(conn_staging, events)
    print("  5 tables written")

    print_audit_report(conn_staging)

    conn_solar.close()
    conn_staging.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
