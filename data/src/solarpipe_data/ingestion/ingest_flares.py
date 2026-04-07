"""Ingest solar flares from DONKI FLR and NOAA GOES into the flares table.

Rules enforced:
- RULE-003: Sentinel conversion at ingest
- RULE-004: Upsert idempotent on flare_id
- RULE-030: from sqlalchemy.dialects.sqlite import insert
- RULE-033: Session context manager
- RULE-034: Explicit set_ for nullable columns
- RULE-043: DONKI timestamps have no seconds — strip "Z", fromisoformat()

Source tagging:
  source_catalog = "GOES"  for NOAA SWPC records
  source_catalog = "DONKI" for DONKI FLR records

flare_id construction:
  GOES:  "{begin_time_compact}_{goes_satellite}"
         e.g. "20231028T1500_G16"
  DONKI: flrID field from API (already unique)

Class parsing:
  "X1.5" → class_letter="X", class_magnitude=1.5
  "M2.3" → class_letter="M", class_magnitude=2.3
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from solarpipe_data.database.schema import Flare

logger = logging.getLogger(__name__)

_SENTINEL_STRINGS = frozenset({"", "null", "None", "--", "---"})
_CLASS_RE = re.compile(r"^([A-Z])(\d+\.?\d*)$")


# ---------------------------------------------------------------------------
# GOES ingestor
# ---------------------------------------------------------------------------

def ingest_goes_flares(
    engine: Engine,
    records: list[dict[str, Any]],
    satellite: str = "unknown",
) -> int:
    """Parse NOAA GOES flare records and upsert into flares table.

    Args:
        records: list of dicts from SWPC JSON endpoint
        satellite: satellite label e.g. "G16" for tagging flare_id

    Returns number of rows upserted.
    """
    if not records:
        return 0

    fetch_ts = datetime.now(tz=timezone.utc).isoformat()
    rows = [_parse_goes_record(r, satellite, fetch_ts) for r in records]
    rows = [r for r in rows if r is not None]

    if not rows:
        return 0

    _upsert_rows(engine, rows)
    logger.info("goes_flares (%s): upserted %d rows", satellite, len(rows))
    return len(rows)


def _parse_goes_record(
    r: dict[str, Any], satellite: str, fetch_ts: str
) -> dict[str, Any] | None:
    begin_raw = _str_or_none(r.get("begin_time") or r.get("beginTime"))
    if begin_raw is None:
        return None

    # Normalise timestamp: strip trailing "Z", replace space with "T"
    begin_norm = begin_raw.rstrip("Z").replace(" ", "T")
    compact = begin_norm.replace("-", "").replace(":", "").replace("T", "T")
    flare_id = f"{compact}_{satellite}"

    peak_raw = r.get("peak_time") or r.get("peakTime")
    end_raw = r.get("end_time") or r.get("endTime")
    class_raw = _str_or_none(r.get("max_class") or r.get("classType") or r.get("class"))

    letter, magnitude = _parse_class(class_raw)
    ar_num = _int_or_none(r.get("noaa_ar") or r.get("active_region") or r.get("activeRegionNum"))
    location = _str_or_none(r.get("source_location") or r.get("location"))
    date_part = begin_norm[:10] if len(begin_norm) >= 10 else None

    return {
        "flare_id": flare_id,
        "begin_time": begin_norm + "Z" if begin_norm else None,
        "peak_time": _norm_ts(peak_raw),
        "end_time": _norm_ts(end_raw),
        "date": date_part,
        "class_type": class_raw,
        "class_letter": letter,
        "class_magnitude": magnitude,
        "source_location": location,
        "active_region_num": ar_num,
        "catalog": "GOES",
        "instruments": None,
        "note": None,
        "link": None,
        "n_linked_events": None,
        "linked_event_ids": None,
        "goes_satellite": satellite,
        "source_catalog": "GOES",
        "fetch_timestamp": fetch_ts,
        "data_version": "v1",
    }


# ---------------------------------------------------------------------------
# DONKI FLR ingestor
# ---------------------------------------------------------------------------

def ingest_donki_flares(engine: Engine, records: list[dict[str, Any]]) -> int:
    """Parse DONKI FLR records and upsert into flares table.

    Returns number of rows upserted.
    """
    if not records:
        return 0

    fetch_ts = datetime.now(tz=timezone.utc).isoformat()
    rows = [_parse_donki_flare(r, fetch_ts) for r in records]
    rows = [r for r in rows if r is not None]

    if not rows:
        return 0

    _upsert_rows(engine, rows)
    logger.info("donki_flares: upserted %d rows", len(rows))
    return len(rows)


def _parse_donki_flare(r: dict[str, Any], fetch_ts: str) -> dict[str, Any] | None:
    # RULE-043: DONKI timestamps have no seconds — "2016-09-06T14:18Z"
    flare_id = _str_or_none(r.get("flrID"))
    if flare_id is None:
        return None

    begin_raw = _str_or_none(r.get("beginTime"))
    class_raw = _str_or_none(r.get("classType"))
    letter, magnitude = _parse_class(class_raw)
    ar_num = _int_or_none(r.get("activeRegionNum"))
    if ar_num == 0:
        ar_num = None  # RULE-003 analogue: 0 is sentinel for "no AR"

    linked = r.get("linkedEvents") or []
    instruments_raw = r.get("instruments") or []
    inst_names = [i.get("displayName") or i.get("displayname") for i in instruments_raw]

    begin_norm = _norm_ts(begin_raw)
    date_part = begin_norm[:10] if begin_norm and len(begin_norm) >= 10 else None

    return {
        "flare_id": flare_id,
        "begin_time": begin_norm,
        "peak_time": _norm_ts(_str_or_none(r.get("peakTime"))),
        "end_time": _norm_ts(_str_or_none(r.get("endTime"))),
        "date": date_part,
        "class_type": class_raw,
        "class_letter": letter,
        "class_magnitude": magnitude,
        "source_location": _str_or_none(r.get("sourceLocation")),
        "active_region_num": ar_num,
        "catalog": _str_or_none(r.get("catalog")),
        "instruments": json.dumps(inst_names) if inst_names else None,
        "note": _str_or_none(r.get("note")),
        "link": _str_or_none(r.get("link")),
        "n_linked_events": len(linked),
        "linked_event_ids": json.dumps([e.get("activityID") for e in linked]) if linked else None,
        "goes_satellite": None,
        "source_catalog": "DONKI",
        "fetch_timestamp": fetch_ts,
        "data_version": "v1",
    }


# ---------------------------------------------------------------------------
# DB upsert
# ---------------------------------------------------------------------------

def _upsert_rows(engine: Engine, rows: list[dict[str, Any]]) -> None:
    stmt = insert(Flare).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["flare_id"],
        set_={
            "begin_time": stmt.excluded.begin_time,
            "peak_time": stmt.excluded.peak_time,
            "end_time": stmt.excluded.end_time,
            "date": stmt.excluded.date,
            "class_type": stmt.excluded.class_type,
            "class_letter": stmt.excluded.class_letter,
            "class_magnitude": stmt.excluded.class_magnitude,
            "source_location": stmt.excluded.source_location,
            "active_region_num": stmt.excluded.active_region_num,
            "catalog": stmt.excluded.catalog,
            "instruments": stmt.excluded.instruments,
            "note": stmt.excluded.note,
            "link": stmt.excluded.link,
            "n_linked_events": stmt.excluded.n_linked_events,
            "linked_event_ids": stmt.excluded.linked_event_ids,
            "goes_satellite": stmt.excluded.goes_satellite,
            "source_catalog": stmt.excluded.source_catalog,
            "fetch_timestamp": stmt.excluded.fetch_timestamp,
            "data_version": stmt.excluded.data_version,
        },
    )
    with Session(engine) as s, s.begin():
        s.execute(stmt)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_class(class_str: str | None) -> tuple[str | None, float | None]:
    """Parse GOES class string into (letter, magnitude).

    "X1.5" → ("X", 1.5), "M2.3" → ("M", 2.3), None → (None, None).
    """
    if not class_str:
        return None, None
    m = _CLASS_RE.match(class_str.strip().upper())
    if m:
        return m.group(1), float(m.group(2))
    # Single-letter class like "X" without magnitude
    if len(class_str.strip()) == 1 and class_str.strip().isalpha():
        return class_str.strip().upper(), None
    return class_str.strip() if class_str.strip() else None, None


def _norm_ts(ts: str | None) -> str | None:
    """Normalise a timestamp by stripping trailing Z and adding it back cleanly."""
    if not ts:
        return None
    norm = ts.strip().rstrip("Z").replace(" ", "T")
    return norm + "Z" if norm else None


def _str_or_none(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return None if s in _SENTINEL_STRINGS else s


def _int_or_none(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None
