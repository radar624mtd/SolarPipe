"""Ingest DONKI CME JSON responses into staging.db cme_events table.

Rules enforced:
- RULE-003: Sentinels (None, 0 for AR, missing keys) → Python None at this layer
- RULE-004: Upsert by activity_id — re-running is idempotent
- RULE-043: DONKI timestamp format "2016-09-06T14:18Z" (no seconds)
- RULE-044: level_of_data preference tracked in CmeAnalysis table
- RULE-045: Broken linkedEvents → null FKs, not exceptions
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import sqlalchemy as sa

from solarpipe_data.database.queries import upsert
from solarpipe_data.database.schema import CmeEvent

logger = logging.getLogger(__name__)

_FETCH_TS = datetime.now(timezone.utc).isoformat()


def ingest_cme_batch(engine: sa.Engine, raw: list[dict[str, Any]]) -> int:
    """Parse DONKI CME JSON list and upsert into cme_events.

    Returns number of rows processed.
    """
    rows = [_parse_cme_record(r) for r in raw]
    rows = [r for r in rows if r is not None]
    if not rows:
        return 0
    return upsert(engine, CmeEvent, rows)


def _parse_cme_record(rec: dict[str, Any]) -> dict[str, Any] | None:
    activity_id = rec.get("activityID")
    if not activity_id:
        logger.warning("CME record missing activityID — skipping: %s", rec)
        return None

    # Extract most-accurate analysis fields from nested cmeAnalyses list
    speed_kms = None
    half_angle = None
    latitude = None
    longitude = None
    is_most_accurate = None
    analysis_type = None
    is_earth_directed = None

    analyses = rec.get("cmeAnalyses") or []
    best = _pick_best_analysis(analyses)
    if best:
        speed_kms = _float_or_none(best.get("speed"))
        half_angle = _float_or_none(best.get("halfAngle"))
        latitude = _float_or_none(best.get("latitude"))
        longitude = _float_or_none(best.get("longitude"))
        is_most_accurate = best.get("isMostAccurate")
        analysis_type = best.get("type")
        is_earth_directed = best.get("isEarthDirected")

    # Extract linked event IDs (RULE-045: ignore broken references gracefully)
    linked = rec.get("linkedEvents") or []
    linked_event_ids = json.dumps([e["activityID"] for e in linked if e.get("activityID")])

    linked_flare_id = None
    linked_ips_ids_list: list[str] = []
    linked_gst_ids_list: list[str] = []

    for event in linked:
        eid = event.get("activityID", "")
        if "-FLR-" in eid and linked_flare_id is None:
            linked_flare_id = eid
        elif "-IPS-" in eid:
            linked_ips_ids_list.append(eid)
        elif "-GST-" in eid:
            linked_gst_ids_list.append(eid)

    # activeRegionNum: 0 means unknown (RULE per task spec), convert to None
    ar_num = rec.get("activeRegionNum")
    if ar_num == 0:
        ar_num = None

    return {
        "activity_id": activity_id,
        "start_time": _normalise_ts(rec.get("startTime")),
        "source_location": rec.get("sourceLocation") or None,
        "active_region_num": ar_num,
        "catalog": rec.get("catalog") or None,
        "note": rec.get("note") or None,
        "instruments": json.dumps([i.get("displayName") for i in (rec.get("instruments") or [])]),
        "link": rec.get("link") or None,
        "speed_kms": speed_kms,
        "half_angle_deg": half_angle,
        "latitude": latitude,
        "longitude": longitude,
        "is_earth_directed": is_earth_directed,
        "analysis_type": analysis_type,
        "is_most_accurate": is_most_accurate,
        "linked_flare_id": linked_flare_id,
        "linked_ips_ids": json.dumps(linked_ips_ids_list),
        "linked_gst_ids": json.dumps(linked_gst_ids_list),
        "n_linked_events": len(linked),
        "linked_event_ids": linked_event_ids,
        "source_catalog": "DONKI",
        "fetch_timestamp": _FETCH_TS,
        "data_version": None,
    }


def _pick_best_analysis(analyses: list[dict]) -> dict | None:
    """Return most-accurate analysis, preferring higher level_of_data (RULE-044)."""
    accurate = [a for a in analyses if a.get("isMostAccurate")]
    if not accurate:
        accurate = analyses
    if not accurate:
        return None
    return max(accurate, key=lambda a: a.get("levelOfData") or 0)


def _normalise_ts(ts: str | None) -> str | None:
    """Parse DONKI timestamp 'YYYY-MM-DDTHH:MMZ' → ISO with timezone (RULE-043)."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).isoformat()
    except (ValueError, AttributeError):
        logger.warning("Unparseable DONKI timestamp: %r", ts)
        return ts


def _float_or_none(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
