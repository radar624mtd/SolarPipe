"""Ingest SWPC real-time solar wind → solar_wind_hourly (incremental).

Rules enforced:
- RULE-003: Sentinels 99999.9, -1e31, 9999.99 → None at ingest
- RULE-004: Upsert idempotent on datetime PK
- RULE-030: sqlite dialect insert
- RULE-031: Path.as_posix() connection strings (via make_engine)
- RULE-032: WAL mode via schema.make_engine
- RULE-033: Session context manager
- RULE-070: bz_gsm only, never bz_gse
- RULE-073: Store L1 timestamps as-is; lag applied at crossmatch
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from ..clients.swpc import SwpcClient
from ..database.queries import max_timestamp
from ..database.schema import SolarWindHourly, make_engine

logger = logging.getLogger(__name__)

# SWPC 7-day feed sentinels (distinct from OMNI -1e31)
_SWPC_SENTINELS = {99999.9, -99999.9, 9999.99, -9999.99, 999.9}

# ACE → DSCOVR transition (RULE-073)
_DSCOVR_START = datetime(2016, 7, 27, tzinfo=timezone.utc)


def _sentinel(val: float | None) -> float | None:
    """RULE-003: convert SWPC sentinel floats to None."""
    if val is None:
        return None
    if abs(val) >= 99999.0:
        return None
    if val in _SWPC_SENTINELS:
        return None
    return val


def _spacecraft_for(dt: datetime) -> str:
    """RULE-073: ACE before July 2016, DSCOVR after."""
    if dt >= _DSCOVR_START:
        return "DSCOVR"
    return "ACE"


def _hourly_key(time_tag: str) -> str:
    """Truncate 'YYYY-MM-DD HH:MM:SS' → 'YYYY-MM-DD HH:00'."""
    return time_tag[:13] + ":00"


def _average_records(
    records: list[dict[str, Any]], keys: list[str]
) -> dict[str, float | None]:
    """Mean of non-None values per key across records. Always returns all keys."""
    buckets: dict[str, list[float]] = {k: [] for k in keys}
    for rec in records:
        for k in keys:
            v = rec.get(k)
            if v is not None:
                buckets[k].append(v)
    return {k: (sum(v) / len(v) if v else None) for k, v in buckets.items()}


def _build_row(
    hour_key: str,
    mag_avg: dict[str, float | None],
    plasma_avg: dict[str, float | None],
    fetch_ts: str,
) -> dict[str, Any]:
    """Assemble a solar_wind_hourly row dict from averaged mag + plasma."""
    dt = datetime.strptime(hour_key, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    spacecraft = _spacecraft_for(dt)

    return {
        "datetime": hour_key,
        "date": dt.date().isoformat(),
        "year": dt.year,
        "doy": dt.timetuple().tm_yday,
        "hour": dt.hour,
        # Magnetic — RULE-070: bz_gsm is canonical
        "b_scalar_avg": _sentinel(mag_avg.get("bt")),
        "bx_gse": _sentinel(mag_avg.get("bx_gse")),
        "by_gse": _sentinel(mag_avg.get("by_gse")),
        "bz_gse": _sentinel(mag_avg.get("bz_gse")),
        "bz_gsm": _sentinel(mag_avg.get("bz_gsm")),   # canonical field
        # Plasma
        "flow_speed": _sentinel(plasma_avg.get("speed")),
        "proton_density": _sentinel(plasma_avg.get("density")),
        "proton_temp_k": _sentinel(plasma_avg.get("temperature")),
        # Spacecraft
        "spacecraft": spacecraft,
        # Provenance
        "source_catalog": "SWPC",
        "fetch_timestamp": fetch_ts,
        "data_version": "7day-rt",
    }


def _merge_by_hour(
    mag_records: list[dict[str, Any]],
    plasma_records: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Group mag and plasma records by hour key."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for rec in mag_records:
        hk = _hourly_key(rec["time_tag"])
        grouped.setdefault(hk, {"mag": [], "plasma": []})["mag"].append(rec)

    for rec in plasma_records:
        hk = _hourly_key(rec["time_tag"])
        grouped.setdefault(hk, {"mag": [], "plasma": []})["plasma"].append(rec)

    return grouped


async def ingest_solar_wind(db_path: str, force: bool = False) -> int:
    """Fetch SWPC 7-day feeds and upsert new hourly rows.

    Returns number of rows upserted.
    """
    engine = make_engine(db_path)
    fetch_ts = datetime.now(timezone.utc).isoformat()

    # Determine incremental cutoff
    max_dt = max_timestamp("solar_wind_hourly", "datetime", engine)
    logger.info("solar_wind_hourly MAX(datetime) = %s", max_dt)

    async with SwpcClient(rate_limit=2.0, cache_ttl_hours=1) as client:
        mag_records = await client.fetch_mag_7day()
        plasma_records = await client.fetch_plasma_7day()

    logger.info(
        "SWPC fetched: %d mag records, %d plasma records",
        len(mag_records), len(plasma_records),
    )

    grouped = _merge_by_hour(mag_records, plasma_records)

    mag_keys = ["bx_gse", "by_gse", "bz_gse", "bz_gsm", "bt"]
    plasma_keys = ["density", "speed", "temperature"]

    rows: list[dict[str, Any]] = []
    for hour_key in sorted(grouped.keys()):
        # Incremental: skip hours already in DB (unless force)
        if not force and max_dt and hour_key <= max_dt:
            continue

        bucket = grouped[hour_key]
        mag_avg = _average_records(bucket["mag"], mag_keys)
        plasma_avg = _average_records(bucket["plasma"], plasma_keys)

        # Skip rows where we have no useful data at all
        if not bucket["mag"] and not bucket["plasma"]:
            continue

        rows.append(_build_row(hour_key, mag_avg, plasma_avg, fetch_ts))

    if not rows:
        logger.info("No new solar wind rows to upsert.")
        return 0

    with Session(engine) as s, s.begin():
        stmt = insert(SolarWindHourly)
        update_cols = {
            c: stmt.excluded[c]
            for c in [
                "b_scalar_avg", "bx_gse", "by_gse", "bz_gse", "bz_gsm",
                "flow_speed", "proton_density", "proton_temp_k",
                "spacecraft", "source_catalog", "fetch_timestamp", "data_version",
            ]
        }
        stmt = stmt.on_conflict_do_update(index_elements=["datetime"], set_=update_cols)
        s.execute(stmt, rows)

    logger.info("Upserted %d solar wind hourly rows.", len(rows))
    return len(rows)
