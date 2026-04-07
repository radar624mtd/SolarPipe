"""Ingest DONKI geomagnetic storms into geomagnetic_storms.

Rules enforced:
- RULE-003: Sentinel conversion at ingest
- RULE-004: Upsert idempotent on gst_id
- RULE-030: from sqlalchemy.dialects.sqlite import insert
- RULE-033: Session context manager
- RULE-034: Explicit set_ for nullable columns
- RULE-043: DONKI timestamp format — no seconds field

Schema:
  gst_id          — "gstID" from DONKI response
  start_time      — "startTime" (ISO string)
  kp_index_max    — max over allKpIndex[].kpIndex
  all_kp_values   — JSON dump of allKpIndex list
  link            — "link"
  n_linked_events — count of linkedEvents entries
  linked_event_ids — JSON dump of event activity IDs
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from solarpipe_data.database.schema import GeomagneticStorm

logger = logging.getLogger(__name__)


def ingest_gst(engine: Engine, records: list[dict[str, Any]]) -> int:
    """Parse DONKI GST records and upsert into geomagnetic_storms.

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
    logger.info("donki_gst: upserted %d rows", len(rows))
    return len(rows)


def _parse_record(r: dict[str, Any], fetch_ts: str) -> dict[str, Any] | None:
    gst_id = _str_or_none(r.get("gstID"))
    if gst_id is None:
        logger.debug("donki_gst: skipping record with no gstID")
        return None

    kp_list = r.get("allKpIndex") or []
    kp_max = _kp_max(kp_list)
    linked = r.get("linkedEvents") or []

    return {
        "gst_id": gst_id,
        "start_time": _str_or_none(r.get("startTime")),
        "kp_index_max": kp_max,
        "all_kp_values": json.dumps(kp_list) if kp_list else None,
        "link": _str_or_none(r.get("link")),
        "n_linked_events": len(linked),
        "linked_event_ids": json.dumps([e.get("activityID") for e in linked]) if linked else None,
        "source_catalog": "DONKI",
        "fetch_timestamp": fetch_ts,
        "data_version": "v1",
    }


def _kp_max(kp_list: list[dict[str, Any]]) -> float | None:
    """Return the maximum kpIndex from allKpIndex array."""
    values: list[float] = []
    for entry in kp_list:
        v = entry.get("kpIndex")
        if v is not None:
            try:
                values.append(float(v))
            except (ValueError, TypeError):
                pass
    return max(values) if values else None


def _upsert_rows(engine: Engine, rows: list[dict[str, Any]]) -> None:
    stmt = insert(GeomagneticStorm).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["gst_id"],
        set_={
            "start_time": stmt.excluded.start_time,
            "kp_index_max": stmt.excluded.kp_index_max,
            "all_kp_values": stmt.excluded.all_kp_values,
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
