"""Ingest DONKI interplanetary shocks into interplanetary_shocks.

Rules enforced:
- RULE-003: Sentinel conversion at ingest
- RULE-004: Upsert idempotent on ips_id
- RULE-030: from sqlalchemy.dialects.sqlite import insert
- RULE-033: Session context manager
- RULE-034: Explicit set_ for nullable columns
- RULE-043: DONKI timestamp format — no seconds field

Note: DonkiClient.fetch_ips() already filters location=Earth at the API
level, so no location filter needed here.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from solarpipe_data.database.schema import InterplanetaryShock

logger = logging.getLogger(__name__)


def ingest_ips(engine: Engine, records: list[dict[str, Any]]) -> int:
    """Parse DONKI IPS records and upsert into interplanetary_shocks.

    Returns number of rows upserted.
    """
    if not records:
        return 0

    fetch_ts = datetime.now(tz=timezone.utc).isoformat()
    rows = [_parse_record(r, fetch_ts) for r in records]
    rows = [r for r in rows if r is not None]

    if not rows:
        return 0

    _upsert_rows(engine, rows)
    logger.info("donki_ips: upserted %d rows", len(rows))
    return len(rows)


def _parse_record(r: dict[str, Any], fetch_ts: str) -> dict[str, Any] | None:
    ips_id = _str_or_none(r.get("activityID"))
    if ips_id is None:
        logger.debug("donki_ips: skipping record with no activityID")
        return None

    instruments = r.get("instruments") or []
    inst_names = [i.get("displayName") or i.get("displayname") for i in instruments]
    linked = r.get("linkedEvents") or []

    return {
        "ips_id": ips_id,
        "event_time": _str_or_none(r.get("eventTime")),
        "location": _str_or_none(r.get("location")),
        "catalog": _str_or_none(r.get("catalog")),
        "instruments": json.dumps(inst_names) if inst_names else None,
        "link": _str_or_none(r.get("link")),
        "n_linked_events": len(linked),
        "linked_event_ids": json.dumps([e.get("activityID") for e in linked]) if linked else None,
        "source_catalog": "DONKI",
        "fetch_timestamp": fetch_ts,
        "data_version": "v1",
    }


def _upsert_rows(engine: Engine, rows: list[dict[str, Any]]) -> None:
    stmt = insert(InterplanetaryShock).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["ips_id"],
        set_={
            "event_time": stmt.excluded.event_time,
            "location": stmt.excluded.location,
            "catalog": stmt.excluded.catalog,
            "instruments": stmt.excluded.instruments,
            "link": stmt.excluded.link,
            "n_linked_events": stmt.excluded.n_linked_events,
            "linked_event_ids": stmt.excluded.linked_event_ids,
            "source_catalog": stmt.excluded.source_catalog,
            "fetch_timestamp": stmt.excluded.fetch_timestamp,
            "data_version": stmt.excluded.data_version,
        },
    )
    with Session(engine) as s, s.begin():
        s.execute(stmt)


def _str_or_none(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return None if not s else s
