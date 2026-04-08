"""Task 5.2 — CME ↔ ICME (Interplanetary Shock) cross-matching.

Priority order:
1. DONKI IPS linkedEvents — if the IPS record references this CME activity_id,
   it is an authoritative match.
2. Transit time estimate: t_arrival = launch_time + (1 AU / cme_speed) with
   simple drag correction.  Accept IPS events within ±12 hours.
3. No match — linked_ips_id left null.

For interacting CMEs (multiple CMEs in same window) match_confidence < 0.5.
Never force an assignment when ambiguous.

Transit time drag correction:
    Empirical formula:  TT_hours = (1 AU in km) / speed_kms / 3600
                                   × (1 + alpha * (speed_kms - w_kms))
    alpha = 0.0005  (calibrated to ENLIL ensemble residuals — updated in Phase 6)
    w_kms = 400     (ambient solar wind speed; replaced by sw_speed_ambient when available)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from ..database.schema import CmeEvent, InterplanetaryShock, make_engine

logger = logging.getLogger(__name__)

# 1 AU in km
_AU_KM = 1.496e8
# Drag calibration constant
_ALPHA = 0.0005
# Default ambient solar wind speed km/s
_W_DEFAULT = 400.0
# Transit window for IPS matching
_TRANSIT_WINDOW = timedelta(hours=12)


# ---------------------------------------------------------------------------
# Transit time estimation
# ---------------------------------------------------------------------------

def estimate_arrival_time(
    launch_time: str | None,
    speed_kms: float | None,
    sw_speed_ambient: float | None = None,
) -> datetime | None:
    """Estimate ICME arrival time at L1 using drag-corrected transit model.

    Returns None if speed or launch_time unavailable.
    """
    if not launch_time or not speed_kms or speed_kms <= 0:
        return None

    w = sw_speed_ambient if (sw_speed_ambient and sw_speed_ambient > 0) else _W_DEFAULT

    # Simple empirical transit-time correction
    raw_tt_hours = _AU_KM / speed_kms / 3600.0
    correction_factor = 1.0 + _ALPHA * (speed_kms - w)
    tt_hours = raw_tt_hours / correction_factor

    launch_dt = _parse_dt(launch_time)
    if launch_dt is None:
        return None
    return launch_dt + timedelta(hours=tt_hours)


def _parse_dt(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        ts = ts.strip()
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _linked_cme_ids_from_ips(linked_event_ids: str | None) -> list[str]:
    """Extract CME activity IDs linked from an IPS record."""
    if not linked_event_ids:
        return []
    try:
        items = json.loads(linked_event_ids)
        return [i for i in items if isinstance(i, str) and "CME" in i.upper()]
    except (json.JSONDecodeError, TypeError):
        return []


def _linked_ips_ids_from_cme(linked_ips_ids: str | None) -> list[str]:
    """Extract IPS ids from cme_events.linked_ips_ids JSON."""
    if not linked_ips_ids:
        return []
    try:
        items = json.loads(linked_ips_ids)
        return [i for i in items if isinstance(i, str) and "IPS" in i.upper()]
    except (json.JSONDecodeError, TypeError):
        return []


# ---------------------------------------------------------------------------
# Core matcher
# ---------------------------------------------------------------------------

def match_cme_to_icme(
    cme: dict[str, Any],
    ips_events: list[dict[str, Any]],
    sw_speed_ambient: float | None = None,
) -> dict[str, Any]:
    """Match a single CME to an IPS (ICME proxy) event.

    Returns a dict with keys:
        linked_ips_id, icme_arrival_time, transit_time_hours,
        icme_match_method, icme_match_confidence
    """
    result: dict[str, Any] = {
        "linked_ips_id": None,
        "icme_arrival_time": None,
        "transit_time_hours": None,
        "icme_match_method": "none",
        "icme_match_confidence": 0.0,
    }

    if not ips_events:
        return result

    activity_id = cme.get("activity_id", "")

    # --- Priority 1: DONKI linked events (both directions) ---
    # Forward link: cme_events.linked_ips_ids
    cme_linked_ips = _linked_ips_ids_from_cme(cme.get("linked_ips_ids"))
    for ips in ips_events:
        if ips["ips_id"] in cme_linked_ips:
            return _fill_icme(result, cme, ips, "linked", 1.0)

    # Reverse link: ips.linked_event_ids references this CME
    for ips in ips_events:
        linked_cmes = _linked_cme_ids_from_ips(ips.get("linked_event_ids"))
        if activity_id in linked_cmes:
            return _fill_icme(result, cme, ips, "linked", 1.0)

    # --- Priority 2: transit time estimate ± 12 hr ---
    arrival_est = estimate_arrival_time(
        cme.get("start_time"), cme.get("speed_kms"), sw_speed_ambient
    )
    if arrival_est is None:
        return result

    candidates: list[tuple[timedelta, dict[str, Any]]] = []
    for ips in ips_events:
        ips_dt = _parse_dt(ips.get("event_time"))
        if ips_dt is None:
            continue
        gap = abs(arrival_est - ips_dt)
        if gap <= _TRANSIT_WINDOW:
            candidates.append((gap, ips))

    if not candidates:
        return result

    candidates.sort(key=lambda x: x[0])

    # Confidence: 1 candidate in window → high; multiple → flag as ambiguous
    n = len(candidates)
    closest_gap, best_ips = candidates[0]

    # Confidence decays with gap and number of candidates
    confidence = max(0.1, 1.0 - closest_gap.total_seconds() / _TRANSIT_WINDOW.total_seconds())
    if n > 1:
        confidence *= 0.5   # ambiguous — multiple CMEs possible
        logger.debug(
            "CME %s has %d IPS candidates in window — confidence halved (%.2f)",
            activity_id, n, confidence,
        )

    return _fill_icme(result, cme, best_ips, "transit_estimate", confidence)


def _fill_icme(
    result: dict[str, Any],
    cme: dict[str, Any],
    ips: dict[str, Any],
    method: str,
    confidence: float,
) -> dict[str, Any]:
    launch_dt = _parse_dt(cme.get("start_time"))
    arrival_dt = _parse_dt(ips.get("event_time"))
    transit_hours: float | None = None
    if launch_dt and arrival_dt:
        gap = arrival_dt - launch_dt
        transit_hours = gap.total_seconds() / 3600.0

    result["linked_ips_id"] = ips["ips_id"]
    result["icme_arrival_time"] = ips.get("event_time")
    result["transit_time_hours"] = transit_hours
    result["icme_match_method"] = method
    result["icme_match_confidence"] = round(confidence, 3)
    return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_cme_icme_matching(
    db_path: str,
    ambient_speeds: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Match all CMEs to IPS events. Returns {activity_id: match_dict}.

    ambient_speeds: optional {activity_id: sw_speed_ambient} from sw_ambient_context.
    """
    engine = make_engine(db_path)
    results: dict[str, dict[str, Any]] = {}

    with Session(engine) as s:
        cmes = [dict(r._mapping) for r in s.execute(
            sa.select(CmeEvent.__table__)
        ).fetchall()]
        ips_events = [dict(r._mapping) for r in s.execute(
            sa.select(InterplanetaryShock.__table__)
        ).fetchall()]

    if ambient_speeds is None:
        ambient_speeds = {}

    logger.info("Matching %d CMEs against %d IPS events", len(cmes), len(ips_events))
    linked = transit = none = 0

    for cme in cmes:
        sw_speed = ambient_speeds.get(cme["activity_id"])
        m = match_cme_to_icme(cme, ips_events, sw_speed)
        results[cme["activity_id"]] = m
        method = m["icme_match_method"]
        if method == "linked":
            linked += 1
        elif method == "transit_estimate":
            transit += 1
        else:
            none += 1

    logger.info(
        "CME-ICME match summary: linked=%d transit_estimate=%d none=%d",
        linked, transit, none,
    )
    return results
