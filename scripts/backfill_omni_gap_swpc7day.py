"""Fill remaining OMNI gaps with SWPC 7-day high-resolution feeds.

Uses SWPC 1-minute solar wind mag + plasma JSON, averages to hourly,
and upserts into solar_data.db:omni_hourly and staging.db:solar_wind_hourly
where values are currently NULL.

Complements backfill_omni_gap_ace1h.py. Run this AFTER the ACE 1h backfill
to densify the last 7 days using higher-quality DSCOVR real-time data.
"""
from __future__ import annotations

import json
import sqlite3
import ssl
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SOLAR_DB   = Path("C:/Users/radar/SolarPipe/solar_data.db")
STAGING_DB = Path("C:/Users/radar/SolarPipe/data/data/staging/staging.db")

MAG_URL    = "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json"
PLASMA_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"

FETCH_TS = datetime.now(timezone.utc).isoformat()

_SENTINELS = {99999.9, -99999.9, 9999.99, -9999.99, 999.9, -9999.9, -1e31}


def _clean(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if abs(f) >= 99999.0:
        return None
    if f in _SENTINELS:
        return None
    return f


def _fetch(url: str) -> list[list]:
    req = urllib.request.Request(url, headers={"User-Agent": "SolarPipe/1.0"})
    with urllib.request.urlopen(req, context=ssl.create_default_context(), timeout=60) as r:
        return json.loads(r.read())


def _hour_key(ts: str) -> str:
    # "2026-04-03 19:18:00.000"
    return ts[:13] + ":00"


def _avg(buckets: dict[str, list[float]]) -> dict[str, float | None]:
    return {k: (sum(v) / len(v) if v else None) for k, v in buckets.items()}


def build_hourly() -> dict[str, dict]:
    mag = _fetch(MAG_URL)
    plasma = _fetch(PLASMA_URL)
    mag_hdr, mag_data = mag[0], mag[1:]
    pla_hdr, pla_data = plasma[0], plasma[1:]
    print(f"  mag header: {mag_hdr}")
    print(f"  plasma header: {pla_hdr}")
    print(f"  mag rows: {len(mag_data)}, plasma rows: {len(pla_data)}")

    mag_idx = {c: i for i, c in enumerate(mag_hdr)}
    pla_idx = {c: i for i, c in enumerate(pla_hdr)}

    # Group by hour
    mag_by_hour: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {k: [] for k in ("bx_gsm", "by_gsm", "bz_gsm", "bt", "lon_gsm", "lat_gsm")}
    )
    pla_by_hour: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {k: [] for k in ("density", "speed", "temperature")}
    )

    for row in mag_data:
        ts = row[mag_idx["time_tag"]]
        hk = _hour_key(ts)
        for k in ("bx_gsm", "by_gsm", "bz_gsm", "bt"):
            if k in mag_idx:
                v = _clean(row[mag_idx[k]])
                if v is not None:
                    mag_by_hour[hk][k].append(v)

    for row in pla_data:
        ts = row[pla_idx["time_tag"]]
        hk = _hour_key(ts)
        for k in ("density", "speed", "temperature"):
            if k in pla_idx:
                v = _clean(row[pla_idx[k]])
                if v is not None:
                    pla_by_hour[hk][k].append(v)

    # Merge
    out: dict[str, dict] = {}
    all_hours = set(mag_by_hour.keys()) | set(pla_by_hour.keys())
    for hk in sorted(all_hours):
        m = _avg(mag_by_hour.get(hk, {}))
        p = _avg(pla_by_hour.get(hk, {}))
        dt = datetime.strptime(hk, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        row = {
            "datetime": hk,
            "date": dt.date().isoformat(),
            "year": dt.year,
            "doy": dt.timetuple().tm_yday,
            "hour": dt.hour,
            "B_scalar_avg":   m.get("bt"),
            "B_vector_mag":   m.get("bt"),
            "Bz_GSM":         m.get("bz_gsm"),
            "By_GSM":         m.get("by_gsm"),
            "flow_speed":     p.get("speed"),
            "proton_density": p.get("density"),
            "proton_temp_K":  p.get("temperature"),
        }
        bz = row["Bz_GSM"]
        v  = row["flow_speed"]
        if bz is not None:
            row["Bz_southward"] = -bz if bz < 0 else 0.0
        if bz is not None and v is not None:
            row["sw_electric_field"] = -v * bz * 1e-3
        # Only keep rows with at least one payload
        if any(row.get(c) is not None for c in ("Bz_GSM", "flow_speed", "proton_density", "B_scalar_avg")):
            out[hk] = row
    return out


def _upsert_preserve(conn: sqlite3.Connection, table: str, pk: str, row: dict) -> None:
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


def _staging_row(r: dict) -> dict:
    hk = r["datetime"]
    dt = datetime.strptime(hk, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    return {
        "datetime": hk,
        "date": r.get("date"),
        "year": r.get("year"),
        "doy":  r.get("doy"),
        "hour": r.get("hour"),
        "b_scalar_avg":   r.get("B_scalar_avg"),
        "bz_gsm":         r.get("Bz_GSM"),
        "flow_speed":     r.get("flow_speed"),
        "proton_density": r.get("proton_density"),
        "proton_temp_k":  r.get("proton_temp_K"),
        "spacecraft":     "DSCOVR",
        "source_catalog": "SWPC_7day",
        "fetch_timestamp": FETCH_TS,
        "data_version":   "swpc_7day_backfill",
    }


def main() -> int:
    print("swpc 7-day gap backfill")
    rows = build_hourly()
    print(f"  usable hourly rows: {len(rows)}")
    if not rows:
        return 0
    keys = sorted(rows.keys())
    print(f"  range: {keys[0]} -> {keys[-1]}")

    con = sqlite3.connect(str(SOLAR_DB))
    try:
        for hk in keys:
            _upsert_preserve(con, "omni_hourly", "datetime", rows[hk])
        con.commit()
        print(f"  omni_hourly upserts: {len(keys)}")
    finally:
        con.close()

    if STAGING_DB.exists():
        con = sqlite3.connect(str(STAGING_DB))
        try:
            for hk in keys:
                _upsert_preserve(con, "solar_wind_hourly", "datetime", _staging_row(rows[hk]))
            con.commit()
            print(f"  solar_wind_hourly upserts: {len(keys)}")
        finally:
            con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
