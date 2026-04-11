"""Close the OMNI gap using SWPC ACE 1-hour rolling feeds.

The NASA OMNI2 low-res dataset lags real-time by 2-4 weeks. SWPC publishes
rolling 30-day ACE 1-hour averages which cover the gap between OMNI's end
and "now". We upsert these into:
  - solar_data.db:omni_hourly            (upstream master)
  - data/data/staging/staging.db:solar_wind_hourly (project mirror)

Sources:
  https://services.swpc.noaa.gov/json/ace/mag/ace_mag_1h.json
  https://services.swpc.noaa.gov/json/ace/swepam/ace_swepam_1h.json

Only rows where at least one of (Bz_GSM, flow_speed, proton_density) is
non-null are upserted. Existing non-null cells are preserved by only
UPDATE-ing columns where the new value is not null.
"""
from __future__ import annotations

import json
import sqlite3
import ssl
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SOLAR_DB   = Path("C:/Users/radar/SolarPipe/solar_data.db")
STAGING_DB = Path("C:/Users/radar/SolarPipe/data/data/staging/staging.db")

MAG_URL = "https://services.swpc.noaa.gov/json/ace/mag/ace_mag_1h.json"
SWE_URL = "https://services.swpc.noaa.gov/json/ace/swepam/ace_swepam_1h.json"

FETCH_TS = datetime.now(timezone.utc).isoformat()


def _fetch_json(url: str) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "SolarPipe/1.0"})
    with urllib.request.urlopen(req, context=ssl.create_default_context(), timeout=60) as r:
        return json.loads(r.read())


def _hour_key(tag: str) -> str:
    # "2026-03-10T20:00:00" -> "2026-03-10 20:00"
    return tag[:10] + " " + tag[11:16]


def _flag_ok(d: dict) -> bool:
    # dsflag: 0 = nominal, 1 = bad, 2 = no data (SWPC convention)
    return d.get("dsflag", 0) == 0


def _clean(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # ACE SWPC sentinels
    if abs(f) >= 99999.0:
        return None
    if f in (-1e31, 9999.9, -9999.9):
        return None
    return f


def _clean_density(v):
    """Like _clean, but also rejects physically implausible proton density values.

    Solar wind density at 1 AU is very rarely below 2 cm⁻³ even during extended
    coronal hole fast-stream intervals. ACE SWEPAM reported values from 0.01–1.9
    cm⁻³ with dsflag=0 during the March–April 2026 sensor anomaly period. The
    OMNI2 archive stores density rounded to 1 decimal; ACE raw data has full float
    precision. Rather than relying on a magnitude threshold alone, reject any value
    that:
      (a) is below 2.0 cm⁻³, OR
      (b) has fractional precision beyond 1 decimal (ACE raw signature)
    so that degraded ACE readings do not overwrite valid OMNI2 data.
    """
    f = _clean(v)
    if f is None:
        return None
    if f < 2.0:
        return None
    # Reject values with fractional precision beyond 1 decimal (ACE raw signature)
    if abs(f - round(f, 1)) > 1e-6:
        return None
    return f


def build_rows() -> dict[str, dict]:
    mag = _fetch_json(MAG_URL)
    swe = _fetch_json(SWE_URL)
    print(f"  fetched {len(mag)} mag rows, {len(swe)} swepam rows")

    rows: dict[str, dict] = {}

    for d in mag:
        if not _flag_ok(d):
            continue
        hk = _hour_key(d["time_tag"])
        r = rows.setdefault(hk, {})
        r["Bx_GSE"] = _clean(d.get("gse_bx"))
        r["By_GSE"] = _clean(d.get("gse_by"))
        r["Bz_GSE"] = _clean(d.get("gse_bz"))
        r["By_GSM"] = _clean(d.get("gsm_by"))
        r["Bz_GSM"] = _clean(d.get("gsm_bz"))
        r["B_scalar_avg"] = _clean(d.get("bt"))
        r["B_vector_mag"] = _clean(d.get("bt"))

    for d in swe:
        if not _flag_ok(d):
            continue
        hk = _hour_key(d["time_tag"])
        r = rows.setdefault(hk, {})
        r["flow_speed"]     = _clean(d.get("speed"))
        r["proton_density"] = _clean_density(d.get("dens"))
        r["proton_temp_K"]  = _clean(d.get("temperature"))

    # Drop rows with nothing useful
    useful = {}
    for hk, r in rows.items():
        if any(r.get(c) is not None for c in ("Bz_GSM", "flow_speed", "proton_density", "B_scalar_avg")):
            # Add derived columns
            bz = r.get("Bz_GSM")
            v  = r.get("flow_speed")
            if bz is not None:
                r["Bz_southward"] = -bz if bz < 0 else 0.0
            if v is not None and bz is not None:
                r["sw_electric_field"] = -v * bz * 1e-3
            # Datetime keys
            dt = datetime.strptime(hk, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            r["datetime"] = hk
            r["date"] = dt.date().isoformat()
            r["year"] = dt.year
            r["doy"]  = dt.timetuple().tm_yday
            r["hour"] = dt.hour
            useful[hk] = r
    return useful


def _upsert_preserve(conn: sqlite3.Connection, table: str, pk: str, row: dict) -> None:
    """Upsert that preserves non-null existing values.

    Strategy: UPDATE non-null columns; if no row exists, INSERT new row.
    """
    cur = conn.cursor()
    cur.execute(f'SELECT 1 FROM {table} WHERE "{pk}" = ?', (row[pk],))
    exists = cur.fetchone() is not None

    payload = {k: v for k, v in row.items() if v is not None}

    if exists:
        set_cols = [k for k in payload if k != pk]
        if not set_cols:
            return
        sets = ", ".join(f'"{k}" = ?' for k in set_cols)
        params = [payload[k] for k in set_cols] + [row[pk]]
        cur.execute(f'UPDATE {table} SET {sets} WHERE "{pk}" = ?', params)
    else:
        cols = list(payload.keys())
        placeholders = ", ".join("?" for _ in cols)
        col_list = ", ".join(f'"{c}"' for c in cols)
        cur.execute(
            f'INSERT INTO {table} ({col_list}) VALUES ({placeholders})',
            [payload[c] for c in cols],
        )


def _staging_row(src_row: dict) -> dict:
    """Map omni-style row to staging solar_wind_hourly columns (lowercase with underscores)."""
    hk = src_row["datetime"]
    dt = datetime.strptime(hk, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    spacecraft = "DSCOVR" if dt >= datetime(2016, 7, 27, tzinfo=timezone.utc) else "ACE"
    r = {
        "datetime": hk,
        "date": src_row.get("date"),
        "year": src_row.get("year"),
        "doy":  src_row.get("doy"),
        "hour": src_row.get("hour"),
        "b_scalar_avg":  src_row.get("B_scalar_avg"),
        "bx_gse":        src_row.get("Bx_GSE"),
        "by_gse":        src_row.get("By_GSE"),
        "bz_gse":        src_row.get("Bz_GSE"),
        "bz_gsm":        src_row.get("Bz_GSM"),
        "flow_speed":    src_row.get("flow_speed"),
        "proton_density": src_row.get("proton_density"),
        "proton_temp_k": src_row.get("proton_temp_K"),
        "spacecraft":    spacecraft,
        "source_catalog": "ACE_SWPC_1h",
        "fetch_timestamp": FETCH_TS,
        "data_version": "ace_1h_backfill",
    }
    return r


def main() -> int:
    print("ace_1h gap backfill")
    rows = build_rows()
    print(f"  usable rows: {len(rows)}")
    if not rows:
        return 0
    keys = sorted(rows.keys())
    print(f"  range: {keys[0]} -> {keys[-1]}")

    # --- Upstream master (omni_hourly) ---
    con = sqlite3.connect(str(SOLAR_DB))
    try:
        n = 0
        for hk in keys:
            _upsert_preserve(con, "omni_hourly", "datetime", rows[hk])
            n += 1
        con.commit()
        print(f"  omni_hourly upserts: {n}")
    finally:
        con.close()

    # --- Project staging (solar_wind_hourly) ---
    if STAGING_DB.exists():
        con = sqlite3.connect(str(STAGING_DB))
        try:
            n = 0
            for hk in keys:
                _upsert_preserve(con, "solar_wind_hourly", "datetime", _staging_row(rows[hk]))
                n += 1
            con.commit()
            print(f"  solar_wind_hourly upserts: {n}")
        finally:
            con.close()
    else:
        print(f"  staging db not found at {STAGING_DB}, skipping")

    return 0


if __name__ == "__main__":
    sys.exit(main())
