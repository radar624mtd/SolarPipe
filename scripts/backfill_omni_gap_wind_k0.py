"""Backfill OMNI gap hours using Wind spacecraft k0 (real-time) CDFs.

Wind's SWE_k0 (plasma) and MFI_k0 (magnetometer) real-time CDFs are
published on SPDF within ~24 hours of observation. For the 2026 OMNI gap
(Mar-28 through Apr-03+), Wind coverage is typically 90%+ vs ACE SWEPAM's
~70%.

Wind is upstream of L1 by ~200 Earth radii and its observations require a
solar-wind advection delay correction, but for hourly averages the delay
is a small fraction of an hour and we accept it as-is for drag-based
prediction (the hour-scale resolution absorbs minute-scale shifts).

Pulls SWE_k0 (Np, V_GSE, V_GSM) and MFI_k0 (B_GSE, B_GSM) CDFs, averages
to hourly, and non-destructively upserts into:
  - solar_data.db:omni_hourly
  - data/data/staging/staging.db:solar_wind_hourly

Run AFTER backfill_omni_gap_ace1h.py and backfill_omni_gap_swpc7day.py.
Only fills cells that are still NULL.
"""
from __future__ import annotations

import re
import sqlite3
import ssl
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

try:
    import cdflib
except ImportError:
    print("cdflib not installed. pip install cdflib", file=sys.stderr)
    sys.exit(2)

SOLAR_DB   = Path("C:/Users/radar/SolarPipe/solar_data.db")
STAGING_DB = Path("C:/Users/radar/SolarPipe/data/data/staging/staging.db")
CACHE_DIR  = Path("C:/Users/radar/SolarPipe/data/cache/wind")

SWE_BASE = "https://spdf.gsfc.nasa.gov/pub/data/wind/swe/swe_k0"
MFI_BASE = "https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/mfi_k0"

FETCH_TS = datetime.now(timezone.utc).isoformat()

# Wind k0 fill value: -1e31
def _clean_scalar(v: float) -> float | None:
    if v is None:
        return None
    fv = float(v)
    if not np.isfinite(fv):
        return None
    if abs(fv) >= 1e30:
        return None
    return fv


def _list_year_dir(base: str, year: int) -> list[str]:
    url = f"{base}/{year}/"
    req = urllib.request.Request(url, headers={"User-Agent": "SolarPipe/1.0"})
    with urllib.request.urlopen(req, context=ssl.create_default_context(), timeout=30) as r:
        body = r.read().decode("utf-8", "ignore")
    return sorted(set(re.findall(r'(wi_k0_[a-z]+_\d{8}_v\d{2}\.cdf)', body)))


def _download_cdf(base: str, year: int, fname: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local = CACHE_DIR / fname
    if local.exists():
        return local
    url = f"{base}/{year}/{fname}"
    req = urllib.request.Request(url, headers={"User-Agent": "SolarPipe/1.0"})
    with urllib.request.urlopen(req, context=ssl.create_default_context(), timeout=120) as r:
        local.write_bytes(r.read())
    return local


def _find_file_for_date(files: list[str], prefix: str, yyyymmdd: str) -> str | None:
    """Pick the highest-version file matching a date."""
    pattern = re.compile(rf"^{re.escape(prefix)}_{yyyymmdd}_v(\d{{2}})\.cdf$")
    matches = [(int(m.group(1)), f) for f in files if (m := pattern.match(f))]
    if not matches:
        return None
    return sorted(matches)[-1][1]


def _read_swe_cdf(path: Path) -> list[tuple[datetime, float | None, float | None, float | None]]:
    """Return list of (dt, np_cm3, v_mag_kms, v_gse_x_kms)."""
    cdf = cdflib.CDF(str(path))
    epoch = cdf.varget("Epoch")
    if epoch is None or len(epoch) == 0:
        return []
    np_arr = cdf.varget("Np")
    v_gse  = cdf.varget("V_GSE")
    # Convert epoch to datetime
    out: list[tuple] = []
    for i in range(len(epoch)):
        t_iso = cdflib.cdfepoch.encode(epoch[i])
        # t_iso like '2026-03-30T00:02:22.861'
        try:
            dt = datetime.fromisoformat(t_iso[:19]).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        dens = _clean_scalar(np_arr[i])
        if v_gse is not None and v_gse.ndim == 2:
            vx = _clean_scalar(v_gse[i, 0])
            vy = _clean_scalar(v_gse[i, 1])
            vz = _clean_scalar(v_gse[i, 2])
            if vx is None or vy is None or vz is None:
                vmag = None
            else:
                vmag = float(np.sqrt(vx * vx + vy * vy + vz * vz))
                if vmag > 2500 or vmag < 100:
                    vmag = None
        else:
            vmag = None
            vx = None
        out.append((dt, dens, vmag, vx))
    return out


def _read_mfi_cdf(path: Path) -> list[tuple[datetime, float | None, float | None, float | None, float | None, float | None, float | None, float | None]]:
    """Return list of (dt, bt, bx_gse, by_gse, bz_gse, bx_gsm, by_gsm, bz_gsm)."""
    cdf = cdflib.CDF(str(path))
    info = cdf.cdf_info()
    varlist = info.rVariables + info.zVariables
    # Find epoch and B variables — Wind MFI k0 typical: Epoch, BGSE, BGSM, BF1 (magnitude)
    epoch_name = "Epoch" if "Epoch" in varlist else None
    if not epoch_name:
        return []
    epoch = cdf.varget(epoch_name)
    if epoch is None or len(epoch) == 0:
        return []

    # Try common names
    # Wind MFI k0 uses BGSEc/BGSEa (centered/averaged) and BGSMc/BGSMa variants;
    # prefer the centered (instantaneous) variant, fall back to averaged.
    def pick(*names):
        for n in names:
            if n in varlist:
                return cdf.varget(n)
        return None
    bgse = pick("BGSEc", "BGSE", "BGSEa")
    bgsm = pick("BGSMc", "BGSM", "BGSMa")
    bt   = pick("BF1", "B1F1", "BT")

    out: list[tuple] = []
    for i in range(len(epoch)):
        t_iso = cdflib.cdfepoch.encode(epoch[i])
        try:
            dt = datetime.fromisoformat(t_iso[:19]).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        btv = _clean_scalar(bt[i]) if bt is not None else None
        bxg = _clean_scalar(bgse[i, 0]) if bgse is not None and bgse.ndim == 2 else None
        byg = _clean_scalar(bgse[i, 1]) if bgse is not None and bgse.ndim == 2 else None
        bzg = _clean_scalar(bgse[i, 2]) if bgse is not None and bgse.ndim == 2 else None
        bxm = _clean_scalar(bgsm[i, 0]) if bgsm is not None and bgsm.ndim == 2 else None
        bym = _clean_scalar(bgsm[i, 1]) if bgsm is not None and bgsm.ndim == 2 else None
        bzm = _clean_scalar(bgsm[i, 2]) if bgsm is not None and bgsm.ndim == 2 else None
        out.append((dt, btv, bxg, byg, bzg, bxm, bym, bzm))
    return out


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def build_hourly(date_start: datetime, date_end: datetime) -> dict[str, dict]:
    """Build hourly rows for all dates in [date_start, date_end] inclusive."""
    year = date_start.year
    assert date_end.year == year, "one year at a time"

    print(f"  listing Wind k0 files for {year}")
    swe_files = _list_year_dir(SWE_BASE, year)
    mfi_files = _list_year_dir(MFI_BASE, year)

    hourly: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    d = date_start
    while d <= date_end:
        ymd = d.strftime("%Y%m%d")
        swe_fn = _find_file_for_date(swe_files, "wi_k0_swe", ymd)
        mfi_fn = _find_file_for_date(mfi_files, "wi_k0_mfi", ymd)

        # SWE — plasma
        if swe_fn:
            p = _download_cdf(SWE_BASE, year, swe_fn)
            try:
                samples = _read_swe_cdf(p)
            except Exception as e:
                print(f"  SWE {ymd}: read error {e}")
                samples = []
            for dt, dens, vmag, vx in samples:
                hk = dt.strftime("%Y-%m-%d %H:00")
                if dens is not None:
                    hourly[hk]["proton_density"].append(dens)
                if vmag is not None:
                    hourly[hk]["flow_speed"].append(vmag)

        # MFI — mag
        if mfi_fn:
            p = _download_cdf(MFI_BASE, year, mfi_fn)
            try:
                samples = _read_mfi_cdf(p)
            except Exception as e:
                print(f"  MFI {ymd}: read error {e}")
                samples = []
            for dt, btv, bxg, byg, bzg, bxm, bym, bzm in samples:
                hk = dt.strftime("%Y-%m-%d %H:00")
                if btv is not None: hourly[hk]["B_scalar_avg"].append(btv)
                if bxg is not None: hourly[hk]["Bx_GSE"].append(bxg)
                if byg is not None: hourly[hk]["By_GSE"].append(byg)
                if bzg is not None: hourly[hk]["Bz_GSE"].append(bzg)
                if bxm is not None: hourly[hk]["Bx_GSM"].append(bxm)
                if bym is not None: hourly[hk]["By_GSM"].append(bym)
                if bzm is not None: hourly[hk]["Bz_GSM"].append(bzm)

        d += timedelta(days=1)

    # Build rows — clip to requested date window (CDF files occasionally
    # contain stray records with corrupt epochs that decode to wild dates)
    out: dict[str, dict] = {}
    window_start = date_start.strftime("%Y-%m-%d 00:00")
    window_end   = (date_end + timedelta(days=1)).strftime("%Y-%m-%d 00:00")
    for hk, buckets in sorted(hourly.items()):
        if hk < window_start or hk >= window_end:
            continue
        dt = datetime.strptime(hk, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        row = {
            "datetime": hk,
            "date": dt.date().isoformat(),
            "year": dt.year,
            "doy":  dt.timetuple().tm_yday,
            "hour": dt.hour,
            "B_scalar_avg":   _mean(buckets.get("B_scalar_avg", [])),
            "B_vector_mag":   _mean(buckets.get("B_scalar_avg", [])),
            "Bx_GSE":         _mean(buckets.get("Bx_GSE", [])),
            "By_GSE":         _mean(buckets.get("By_GSE", [])),
            "Bz_GSE":         _mean(buckets.get("Bz_GSE", [])),
            "By_GSM":         _mean(buckets.get("By_GSM", [])),
            "Bz_GSM":         _mean(buckets.get("Bz_GSM", [])),
            "flow_speed":     _mean(buckets.get("flow_speed", [])),
            "proton_density": _mean(buckets.get("proton_density", [])),
        }
        bz = row["Bz_GSM"]; v = row["flow_speed"]
        if bz is not None:
            row["Bz_southward"] = -bz if bz < 0 else 0.0
        if bz is not None and v is not None:
            row["sw_electric_field"] = -v * bz * 1e-3
        # Keep only if we got *something*
        if any(row.get(k) is not None for k in ("Bz_GSM", "flow_speed", "proton_density", "B_scalar_avg")):
            out[hk] = row
    return out


def _upsert_preserve(conn: sqlite3.Connection, table: str, pk: str, row: dict) -> int:
    """Returns 1 if any cell was actually filled, 0 if no-op."""
    cur = conn.cursor()
    # Fetch current row
    cur.execute(f'PRAGMA table_info({table})')
    valid_cols = {r[1] for r in cur.fetchall()}
    payload = {k: v for k, v in row.items() if v is not None and k in valid_cols}
    if not payload:
        return 0
    cur.execute(f'SELECT 1 FROM {table} WHERE "{pk}" = ?', (row[pk],))
    exists = cur.fetchone() is not None
    if exists:
        # Only UPDATE cells that are currently NULL
        cols = [k for k in payload if k != pk]
        if not cols:
            return 0
        set_clause = ", ".join(f'"{c}" = COALESCE("{c}", ?)' for c in cols)
        params = [payload[c] for c in cols] + [row[pk]]
        cur.execute(f'UPDATE {table} SET {set_clause} WHERE "{pk}" = ?', params)
        return cur.rowcount
    else:
        cols = list(payload.keys())
        placeholders = ", ".join("?" for _ in cols)
        col_list = ", ".join(f'"{c}"' for c in cols)
        cur.execute(
            f'INSERT INTO {table} ({col_list}) VALUES ({placeholders})',
            [payload[c] for c in cols],
        )
        return 1


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
        "bx_gse":         r.get("Bx_GSE"),
        "by_gse":         r.get("By_GSE"),
        "bz_gse":         r.get("Bz_GSE"),
        "bz_gsm":         r.get("Bz_GSM"),
        "flow_speed":     r.get("flow_speed"),
        "proton_density": r.get("proton_density"),
        "spacecraft":     "Wind",
        "source_catalog": "Wind_k0",
        "fetch_timestamp": FETCH_TS,
        "data_version":   "wind_k0_backfill",
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default="2026-03-28")
    ap.add_argument("--end",   type=str, default="2026-04-08")
    args = ap.parse_args()

    d0 = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    d1 = datetime.strptime(args.end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
    print(f"wind k0 backfill: {d0.date()} -> {d1.date()}")

    rows = build_hourly(d0, d1)
    print(f"  built {len(rows)} hourly rows from Wind k0")
    if not rows:
        return 0
    keys = sorted(rows.keys())
    print(f"  range: {keys[0]} -> {keys[-1]}")

    con = sqlite3.connect(str(SOLAR_DB))
    try:
        n = sum(_upsert_preserve(con, "omni_hourly", "datetime", rows[hk]) for hk in keys)
        con.commit()
        print(f"  omni_hourly touched: {n}")
    finally:
        con.close()

    if STAGING_DB.exists():
        con = sqlite3.connect(str(STAGING_DB))
        try:
            n = sum(_upsert_preserve(con, "solar_wind_hourly", "datetime", _staging_row(rows[hk])) for hk in keys)
            con.commit()
            print(f"  solar_wind_hourly touched: {n}")
        finally:
            con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
