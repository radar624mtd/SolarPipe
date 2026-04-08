"""Ingest Kyoto WDC Dst hourly index → dst_hourly (incremental).

Rules enforced:
- RULE-003: Sentinel conversion in kyoto client
- RULE-004: Upsert idempotent on datetime PK
- RULE-030: sqlite dialect insert
- RULE-033: Session context manager
- RULE-080: Preference cascade final > provisional > realtime; never overwrite downward
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from ..clients.kyoto import KyotoClient
from ..database.queries import max_timestamp
from ..database.schema import DstHourly, make_engine

logger = logging.getLogger(__name__)

# Data type rank: higher = better quality (RULE-080)
_RANK: dict[str, int] = {"final": 3, "provisional": 2, "realtime": 1, "none": 0, "unsupported": -1}


def _should_update(existing_type: str | None, incoming_type: str) -> bool:
    """RULE-080: only update if incoming quality >= existing quality."""
    existing_rank = _RANK.get(existing_type or "none", 0)
    incoming_rank = _RANK.get(incoming_type, 0)
    return incoming_rank >= existing_rank


async def ingest_dst(
    db_path: str,
    start: date | None = None,
    end: date | None = None,
    force: bool = False,
) -> int:
    """Fetch Kyoto Dst and upsert with cascade preference logic.

    Returns number of rows upserted.
    If start/end not provided, fetches incremental from MAX(datetime) to today.
    """
    engine = make_engine(db_path)
    fetch_ts = datetime.now(timezone.utc).isoformat()

    if start is None:
        max_dt = max_timestamp("dst_hourly", "datetime", engine)
        if max_dt:
            start = datetime.strptime(max_dt[:10], "%Y-%m-%d").date()
        else:
            # Default: start 6 months back for initial fill
            from datetime import timedelta
            start = date.today().replace(day=1) - _months(6)

    if end is None:
        end = date.today()

    logger.info("Fetching Kyoto Dst %s → %s", start, end)

    async with KyotoClient(rate_limit=1.0, cache_ttl_hours=12) as client:
        records = await client.fetch_range(start, end)

    if not records:
        logger.info("No Dst records returned.")
        return 0

    # RULE-080: Load existing data_type for affected datetimes to enforce cascade
    datetimes = [r["datetime"] for r in records]

    with Session(engine) as s, s.begin():
        existing = {
            row[0]: row[1]
            for row in s.execute(
                DstHourly.__table__.select().where(
                    DstHourly.__table__.c.datetime.in_(datetimes)
                ).with_only_columns(
                    DstHourly.__table__.c.datetime,
                    DstHourly.__table__.c.data_type,
                )
            )
        }

        rows_to_upsert = [
            r for r in records
            if force or _should_update(existing.get(r["datetime"]), r["data_type"])
        ]

        if not rows_to_upsert:
            logger.info("No Dst rows qualify for upsert (cascade protection).")
            return 0

        stmt = insert(DstHourly)
        update_cols = {
            c: stmt.excluded[c]
            for c in ["dst_nt", "data_type", "source_catalog", "fetch_timestamp", "data_version"]
        }
        stmt = stmt.on_conflict_do_update(index_elements=["datetime"], set_=update_cols)
        s.execute(stmt, rows_to_upsert)

    logger.info("Upserted %d Dst hourly rows.", len(rows_to_upsert))
    return len(rows_to_upsert)


def _months(n: int):
    """Return a timedelta approximating n months (30 days each)."""
    from datetime import timedelta
    return timedelta(days=30 * n)
