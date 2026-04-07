"""Ingest DONKI WSA-ENLIL simulations into enlil_simulations.

Rules enforced:
- RULE-003: Sentinel conversion at ingest
- RULE-004: Upsert idempotent on simulation_id
- RULE-030: from sqlalchemy.dialects.sqlite import insert
- RULE-033: Session context manager
- RULE-034: Explicit set_ for nullable columns
- RULE-043: DONKI timestamp format — no seconds field
- RULE-045: Not 1:1 with CMEs — store all; dedup at crossmatch

simulation_id: DONKI does not always provide a stable ID. Use
  "modelCompletionTime" as the primary key. If missing, skip.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from solarpipe_data.database.schema import EnlilSimulation

logger = logging.getLogger(__name__)

_SENTINEL_STRINGS = frozenset({"", "null", "None"})


def ingest_enlil(engine: Engine, records: list[dict[str, Any]]) -> int:
    """Parse DONKI ENLIL records and upsert into enlil_simulations.

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
    logger.info("donki_enlil: upserted %d rows", len(rows))
    return len(rows)


def _parse_record(r: dict[str, Any], fetch_ts: str) -> dict[str, Any] | None:
    """Map one DONKI ENLIL record to DB row dict."""
    sim_id = _str_or_none(r.get("modelCompletionTime"))
    if sim_id is None:
        logger.debug("donki_enlil: skipping record with no modelCompletionTime")
        return None

    linked_cme_ids = _extract_linked_cme_ids(r)

    return {
        "simulation_id": sim_id,
        "model_completion_time": sim_id,
        "au": _float_or_none(r.get("au")),
        "linked_cme_ids": json.dumps(linked_cme_ids) if linked_cme_ids else None,
        "link": _str_or_none(r.get("link")),
        "source_catalog": "DONKI",
        "fetch_timestamp": fetch_ts,
        "data_version": "v1",
    }


def _extract_linked_cme_ids(r: dict[str, Any]) -> list[str]:
    """Extract linked CME activity IDs from an ENLIL record."""
    linked: list[str] = []
    for entry in r.get("cmeInputs", []) or []:
        act_id = entry.get("associatedCMEActivityID") or entry.get("cmeActivityID")
        if act_id:
            linked.append(str(act_id))
    return linked


def _upsert_rows(engine: Engine, rows: list[dict[str, Any]]) -> None:
    stmt = insert(EnlilSimulation).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["simulation_id"],
        set_={
            "model_completion_time": stmt.excluded.model_completion_time,
            "au": stmt.excluded.au,
            "linked_cme_ids": stmt.excluded.linked_cme_ids,
            "link": stmt.excluded.link,
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
    return None if s in _SENTINEL_STRINGS else s


def _float_or_none(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None
