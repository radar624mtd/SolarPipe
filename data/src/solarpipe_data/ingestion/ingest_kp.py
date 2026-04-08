"""Ingest GFZ Kp 3-hourly index → kp_3hr (incremental).

Bulk data already ported (34,426 rows through 2026-04-02).
Only fetches records after MAX(datetime) in kp_3hr.

Rules enforced:
- RULE-003: Sentinel conversion
- RULE-004: Upsert idempotent on datetime PK
- RULE-010: Incremental only — bulk is ported
- RULE-030: sqlite dialect insert
- RULE-033: Session context manager
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from ..clients.base import BaseClient
from ..config import get_settings
from ..database.queries import max_timestamp
from ..database.schema import Kp3hr, make_engine

logger = logging.getLogger(__name__)

_SENTINELS = {-1.0, 999.0, 9999.0}


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


def _sentinel_int(val: Any) -> int | None:
    f = _sentinel(val)
    return int(f) if f is not None else None


class GfzKpClient(BaseClient):
    """GFZ Potsdam Kp/ap JSON API."""

    source_name = "gfz"

    async def fetch_range(self, start: date, end: date) -> list[dict[str, Any]]:
        """Fetch Kp data as JSON for [start, end] inclusive."""
        settings = get_settings()
        url = settings.gfz_kp_url
        params = {
            "startdate": start.isoformat(),
            "enddate": end.isoformat(),
            "index": "Kp",
        }
        cache_key = f"gfz_kp_{start}_{end}"
        return await self.get(url, params=params, cache_key=cache_key)


def _parse_record(rec: dict[str, Any], fetch_ts: str) -> dict[str, Any] | None:
    """Convert GFZ JSON record to kp_3hr row dict."""
    # GFZ returns datetime as "2024-01-01 00:00:00" or ISO with T
    raw_dt = rec.get("datetime") or rec.get("time_tag") or rec.get("date_time")
    if not raw_dt:
        return None

    # Normalise
    clean_dt = str(raw_dt).replace("T", " ").rstrip("Z")
    try:
        dt = datetime.strptime(clean_dt[:16], "%Y-%m-%d %H:%M")
    except ValueError:
        try:
            dt = datetime.fromisoformat(clean_dt)
        except ValueError:
            logger.debug("GFZ: could not parse datetime %r", raw_dt)
            return None

    return {
        "datetime": dt.strftime("%Y-%m-%d %H:%M"),
        "kp": _sentinel(rec.get("Kp") or rec.get("kp")),
        "ap": _sentinel_int(rec.get("ap") or rec.get("Ap")),
        "definitive": bool(rec.get("definitive", True)),
        "daily_ap": _sentinel(rec.get("ap_daily") or rec.get("Ap_daily")),
        "daily_f10_7_obs": _sentinel(rec.get("f107_obs") or rec.get("F107_obs")),
        "daily_f10_7_adj": _sentinel(rec.get("f107_adj") or rec.get("F107_adj")),
        "source_catalog": "GFZ",
        "fetch_timestamp": fetch_ts,
        "data_version": "1.0",
    }


async def ingest_kp(
    db_path: str,
    start: date | None = None,
    end: date | None = None,
    force: bool = False,
) -> int:
    """Fetch GFZ Kp and upsert new rows.

    Returns number of rows upserted.
    """
    engine = make_engine(db_path)
    fetch_ts = datetime.now(timezone.utc).isoformat()

    if start is None:
        max_dt = max_timestamp("kp_3hr", "datetime", engine)
        if max_dt:
            try:
                start = datetime.strptime(max_dt[:10], "%Y-%m-%d").date()
            except ValueError:
                start = date(2026, 1, 1)
        else:
            start = date(2010, 1, 1)

    if end is None:
        end = date.today()

    if not force and start >= end:
        logger.info("kp_3hr already up to date (max=%s).", start)
        return 0

    logger.info("Fetching GFZ Kp %s → %s", start, end)

    async with GfzKpClient(rate_limit=0.5, cache_ttl_hours=6) as client:
        raw = await client.fetch_range(start, end)

    # GFZ API returns either a list of records or a dict with a data key
    if isinstance(raw, dict):
        raw = raw.get("data") or raw.get("Kp") or []

    if not raw:
        logger.info("No Kp records returned for %s → %s.", start, end)
        return 0

    rows: list[dict[str, Any]] = []
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        parsed = _parse_record(rec, fetch_ts)
        if parsed:
            rows.append(parsed)

    if not rows:
        logger.info("No valid Kp rows parsed.")
        return 0

    with Session(engine) as s, s.begin():
        stmt = insert(Kp3hr)
        update_cols = {
            c: stmt.excluded[c]
            for c in [
                "kp", "ap", "definitive", "daily_ap",
                "daily_f10_7_obs", "daily_f10_7_adj",
                "source_catalog", "fetch_timestamp", "data_version",
            ]
        }
        stmt = stmt.on_conflict_do_update(index_elements=["datetime"], set_=update_cols)
        s.execute(stmt, rows)

    logger.info("Upserted %d Kp 3-hourly rows.", len(rows))
    return len(rows)
