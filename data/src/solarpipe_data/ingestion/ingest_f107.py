"""Ingest NOAA SWPC F10.7 solar radio flux → f107_daily.

Source: /json/solar-cycle/observed-solar-cycle-indices.json
Returns monthly averages; we store them keyed by YYYY-MM-01.

Rules enforced:
- RULE-002: HTTP via BaseClient (SwpcClient)
- RULE-003: Sentinel conversion
- RULE-004: Upsert idempotent on date PK
- RULE-030: sqlite dialect insert
- RULE-033: Session context manager
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from ..clients.base import BaseClient
from ..database.queries import max_timestamp
from ..database.schema import F107Daily, make_engine

logger = logging.getLogger(__name__)

_URL = "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"

_SENTINELS = {-1.0, -999.9, 999.9, 9999.9}


def _sentinel(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if f in _SENTINELS or f < 0:
        return None
    return f


class F107Client(BaseClient):
    source_name = "swpc_f107"

    async def fetch_observed_indices(self) -> list[dict[str, Any]]:
        """Fetch NOAA observed solar cycle indices JSON."""
        return await self.get(_URL, cache_key="noaa_observed_solar_cycle")


def _parse_record(rec: dict[str, Any], fetch_ts: str) -> dict[str, Any] | None:
    """Convert NOAA solar cycle JSON record to f107_daily row."""
    # Format: {"time-tag": "1749-02", "ssn": 0.0, "smoothed_ssn": null,
    #          "observed_swpc_solar_flux": 999.9, "smoothed_swpc_solar_flux": null, ...}
    raw_date = rec.get("time-tag") or rec.get("time_tag")
    if not raw_date:
        return None

    # YYYY-MM → YYYY-MM-01
    try:
        if len(str(raw_date)) == 7:
            date_str = f"{raw_date}-01"
            datetime.strptime(date_str, "%Y-%m-%d")
        else:
            date_str = str(raw_date)[:10]
    except ValueError:
        return None

    f107_obs = _sentinel(
        rec.get("observed_swpc_solar_flux")
        or rec.get("f107_obs")
        or rec.get("flux")
    )
    f107_adj = _sentinel(
        rec.get("smoothed_swpc_solar_flux")
        or rec.get("f107_adj")
    )
    ssn = _sentinel(rec.get("ssn") or rec.get("sunspot_number"))

    return {
        "date": date_str,
        "f10_7_obs": f107_obs,
        "f10_7_adj": f107_adj,
        "sunspot_number": ssn,
        "source_catalog": "NOAA",
        "fetch_timestamp": fetch_ts,
        "data_version": "1.0",
    }


async def ingest_f107(db_path: str, force: bool = False) -> int:
    """Fetch NOAA F10.7 observed solar cycle indices and upsert.

    Returns number of rows upserted.
    The NOAA endpoint returns all historical data; we upsert everything
    (idempotent) unless force=False, in which case only new dates are inserted.
    """
    engine = make_engine(db_path)
    fetch_ts = datetime.now(timezone.utc).isoformat()

    max_dt = max_timestamp("f107_daily", "date", engine)
    logger.info("f107_daily MAX(date) = %s", max_dt)

    async with F107Client(rate_limit=1.0, cache_ttl_hours=24) as client:
        raw = await client.fetch_observed_indices()

    if not raw:
        logger.info("No F10.7 records returned.")
        return 0

    rows: list[dict[str, Any]] = []
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        parsed = _parse_record(rec, fetch_ts)
        if parsed is None:
            continue
        # Incremental: skip dates already present unless force
        if not force and max_dt and parsed["date"] <= max_dt:
            continue
        rows.append(parsed)

    if not rows:
        logger.info("No new F10.7 rows to upsert.")
        return 0

    with Session(engine) as s, s.begin():
        stmt = insert(F107Daily)
        update_cols = {
            c: stmt.excluded[c]
            for c in ["f10_7_obs", "f10_7_adj", "sunspot_number",
                      "source_catalog", "fetch_timestamp", "data_version"]
        }
        stmt = stmt.on_conflict_do_update(index_elements=["date"], set_=update_cols)
        s.execute(stmt, rows)

    logger.info("Upserted %d F10.7 daily rows.", len(rows))
    return len(rows)
