"""Backfill omni_hourly in solar_data.db from NASA SPDF OMNI2 low-res files.

Source: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_YYYY.dat
Format: fixed-column ASCII, 55 whitespace-separated fields per hourly row.

Usage:
    python scripts/backfill_omni_hourly.py [--years 2026] [--force]
    python scripts/backfill_omni_hourly.py --since 2026-03-01
"""
from __future__ import annotations

import argparse
import sqlite3
import ssl
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

SOLAR_DB = Path("C:/Users/radar/SolarPipe/solar_data.db")
CACHE_DIR = Path("C:/Users/radar/SolarPipe/data/cache/omni")
BASE_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni"

# OMNI2 fill values (magnitude >= these, with tolerance, treated as missing).
# Using "abs(v) >= threshold" captures 999.9, 9999., 99999.99, 9999999., etc.
_FILL_REAL_THRESHOLDS = {
    "small": 999.0,      # e.g. 999.9
    "mid":   9999.0,     # e.g. 9999.
    "large": 99999.0,    # e.g. 99999.99
    "huge":  999999.0,   # e.g. 999999.99, 9999999.
}

# Column order in omni2_YYYY.dat — confirmed from SPDF format doc and
# cross-checked against solar_data.db:omni_hourly known row.
OMNI2_COLS = [
    ("year",                 int,   None),
    ("doy",                  int,   None),
    ("hour",                 int,   None),
    ("bartels_rotation",     int,   9999),
    ("imf_spacecraft_id",    int,   99),
    ("plasma_spacecraft_id", int,   99),
    ("n_imf_points",         int,   999),
    ("n_plasma_points",      int,   999),
    ("B_scalar_avg",         float, "small"),
    ("B_vector_mag",         float, "small"),
    ("B_lat_GSE",            float, "small"),
    ("B_lon_GSE",            float, "small"),
    ("Bx_GSE",               float, "small"),
    ("By_GSE",               float, "small"),
    ("Bz_GSE",               float, "small"),
    ("By_GSM",               float, "small"),
    ("Bz_GSM",               float, "small"),
    ("sigma_B_scalar",       float, "small"),
    ("sigma_B_vector",       float, "small"),
    ("sigma_Bx",             float, "small"),
    ("sigma_By",             float, "small"),
    ("sigma_Bz",             float, "small"),
    ("proton_temp_K",        float, "huge"),   # 9999999.
    ("proton_density",       float, "small"),  # 999.9
    ("flow_speed",           float, "mid"),    # 9999.
    ("flow_longitude",       float, "small"),
    ("flow_latitude",        float, "small"),
    ("alpha_proton_ratio",   float, 9.999),
    ("flow_pressure",        float, 99.99),
    ("sigma_T",              float, "huge"),
    ("sigma_N",              float, "small"),
    ("sigma_V",              float, "mid"),
    ("sigma_phi_V",          float, "small"),
    ("sigma_theta_V",        float, "small"),
    ("sigma_alpha_ratio",    float, 9.999),
    ("electric_field",       float, 999.99),
    ("plasma_beta",          float, 999.99),
    ("alfven_mach",          float, "small"),
    ("Kp_x10",               int,   99),
    ("sunspot_number_R",     int,   999),
    ("Dst_nT",               int,   99999),
    ("AE_nT",                int,   9999),
    ("proton_flux_gt1MeV",   float, 999999.99),
    ("proton_flux_gt2MeV",   float, 99999.99),
    ("proton_flux_gt4MeV",   float, 99999.99),
    ("proton_flux_gt10MeV",  float, 99999.99),
    ("proton_flux_gt30MeV",  float, 99999.99),
    ("proton_flux_gt60MeV",  float, 99999.99),
    ("magnetosphere_flag",   int,   None),    # 0 = no flag
    ("ap_index",             int,   999),
    ("F10_7_index",          float, "small"),
    ("PC_N_index",           float, "small"),
    ("AL_nT",                int,   99999),
    ("AU_nT",                int,   99999),
    ("magnetosonic_mach",    float, 99.9),
]
assert len(OMNI2_COLS) == 55


def _is_fill(val: float, marker) -> bool:
    if marker is None:
        return False
    if isinstance(marker, str):
        thresh = _FILL_REAL_THRESHOLDS[marker]
        return abs(val) >= thresh
    # numeric marker — compare with tolerance
    if isinstance(marker, int):
        return int(val) == marker
    return abs(val - marker) < 1e-6


def _parse_line(line: str) -> dict | None:
    parts = line.split()
    if len(parts) != 55:
        return None
    out = {}
    for i, (name, typ, fill) in enumerate(OMNI2_COLS):
        raw = parts[i]
        try:
            val = typ(raw) if typ is int else float(raw)
        except ValueError:
            out[name] = None
            continue
        if _is_fill(val, fill):
            out[name] = None
        else:
            out[name] = val
    return out


def _download(year: int, force: bool) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local = CACHE_DIR / f"omni2_{year}.dat"
    if local.exists() and not force:
        # Always re-download current year (rolling file)
        if year < datetime.now(timezone.utc).year:
            return local
    url = f"{BASE_URL}/omni2_{year}.dat"
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "SolarPipe/1.0"})
    with urllib.request.urlopen(req, context=ssl.create_default_context(), timeout=120) as r:
        local.write_bytes(r.read())
    return local


def _derive_row(fields: dict) -> dict:
    """Add datetime/date/year/doy/hour and derived phase-9 columns."""
    y, d, h = fields["year"], fields["doy"], fields["hour"]
    dt = datetime(y, 1, 1, h, tzinfo=timezone.utc) + timedelta(days=d - 1)
    dt_str = dt.strftime("%Y-%m-%d %H:%M")

    row = dict(fields)
    row["datetime"] = dt_str
    row["date"] = dt.date().isoformat()

    # Derived columns present in schema but not in OMNI2 file
    bz = row.get("Bz_GSM")
    row["Bz_southward"] = (-bz) if (bz is not None and bz < 0) else 0.0 if bz is not None else None

    v = row.get("flow_speed")
    n = row.get("proton_density")
    if v is not None and n is not None:
        # dynamic pressure impulse proxy: rho*v^2 normalized constant
        # keep None — schema has it but upstream uses it as delta; leave None to preserve semantics
        row["pressure_impulse"] = None
    else:
        row["pressure_impulse"] = None

    # sw_electric_field = -v*Bz (mV/m) when both present
    if v is not None and bz is not None:
        row["sw_electric_field"] = -v * bz * 1e-3
    else:
        row["sw_electric_field"] = None

    row["storm_label"] = None
    return row


# All 65 columns of omni_hourly
_ALL_COLS = [
    "datetime", "date", "year", "doy", "hour",
    "bartels_rotation", "imf_spacecraft_id", "plasma_spacecraft_id",
    "n_imf_points", "n_plasma_points",
    "B_scalar_avg", "B_vector_mag", "B_lat_GSE", "B_lon_GSE",
    "Bx_GSE", "By_GSE", "Bz_GSE", "By_GSM", "Bz_GSM",
    "sigma_B_scalar", "sigma_B_vector", "sigma_Bx", "sigma_By", "sigma_Bz",
    "proton_temp_K", "proton_density", "flow_speed",
    "flow_longitude", "flow_latitude", "alpha_proton_ratio", "flow_pressure",
    "sigma_T", "sigma_N", "sigma_V", "sigma_phi_V", "sigma_theta_V", "sigma_alpha_ratio",
    "electric_field", "plasma_beta", "alfven_mach",
    "Kp_x10", "sunspot_number_R", "Dst_nT", "AE_nT",
    "proton_flux_gt1MeV", "proton_flux_gt2MeV", "proton_flux_gt4MeV",
    "proton_flux_gt10MeV", "proton_flux_gt30MeV", "proton_flux_gt60MeV",
    "magnetosphere_flag", "ap_index", "F10_7_index", "PC_N_index",
    "AL_nT", "AU_nT", "magnetosonic_mach", "lyman_alpha", "proton_QI",
    "magnetopause_r0_Re", "magnetopause_alpha",
    "storm_label", "Bz_southward", "pressure_impulse", "sw_electric_field",
]
assert len(_ALL_COLS) == 65


def upsert_year(conn: sqlite3.Connection, year: int, since: str | None, force: bool) -> tuple[int, int]:
    path = _download(year, force=force)
    placeholders = ", ".join(f":{c}" for c in _ALL_COLS)
    cols = ", ".join(f'"{c}"' for c in _ALL_COLS)
    sql = f'INSERT OR REPLACE INTO omni_hourly ({cols}) VALUES ({placeholders})'

    updated = 0
    skipped = 0
    batch: list[dict] = []
    with path.open() as f:
        for line in f:
            fields = _parse_line(line)
            if fields is None:
                skipped += 1
                continue
            row = _derive_row(fields)
            if since and row["datetime"] < since:
                continue
            # Skip rows that are entirely fill (no useful payload)
            if row.get("flow_speed") is None and row.get("Bz_GSM") is None and row.get("B_scalar_avg") is None:
                # Still upsert as skeleton? No — only write if we have any useful value.
                # But we also want to preserve IMF-only or plasma-only rows.
                continue
            # Fill in any missing columns with None
            for c in _ALL_COLS:
                row.setdefault(c, None)
            batch.append(row)
    if batch:
        conn.executemany(sql, batch)
        conn.commit()
        updated = len(batch)
    return updated, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=None,
                    help="Years to fetch (default: infer from --since or current year)")
    ap.add_argument("--since", type=str, default=None,
                    help="Only upsert rows with datetime >= this (e.g. 2026-03-01)")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if cached")
    args = ap.parse_args()

    if args.years is None:
        if args.since:
            start_year = int(args.since[:4])
        else:
            start_year = datetime.now(timezone.utc).year
        years = list(range(start_year, datetime.now(timezone.utc).year + 1))
    else:
        years = args.years

    print(f"backfill_omni_hourly: years={years} since={args.since} force={args.force}")
    print(f"target db: {SOLAR_DB}")

    conn = sqlite3.connect(str(SOLAR_DB))
    try:
        total_up = 0
        for y in years:
            print(f"year {y}:")
            up, skipped = upsert_year(conn, y, since=args.since, force=args.force)
            total_up += up
            print(f"  upserted {up} rows, skipped {skipped} malformed lines")

        # Report new coverage
        cur = conn.execute(
            "SELECT MAX(datetime) FROM omni_hourly WHERE flow_speed IS NOT NULL"
        )
        print(f"new MAX(datetime WHERE flow_speed NOT NULL) = {cur.fetchone()[0]}")
        cur = conn.execute(
            "SELECT MAX(datetime) FROM omni_hourly WHERE Bz_GSM IS NOT NULL"
        )
        print(f"new MAX(datetime WHERE Bz_GSM NOT NULL)    = {cur.fetchone()[0]}")
        print(f"total upserted: {total_up}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
