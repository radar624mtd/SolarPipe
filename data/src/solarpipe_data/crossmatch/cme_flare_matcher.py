"""Task 5.1 — CME ↔ Flare cross-matching.

Priority order:
1. DONKI linkedEvents (linked_flare_id on cme_events) — authoritative
2. Temporal ±30 min: flare begin_time within 30 min of CME start_time
3. Spatial ±15°: flare source_location within 15° lat AND 15° lon of CME source_location
   (only applied after temporal window match)
4. No match — linked_flare_id left null; flare_match_method = "none"

Null FKs for unmatched — never force an assignment.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from ..database.schema import CmeEvent, Flare, make_engine

logger = logging.getLogger(__name__)

_LOCATION_RE = re.compile(
    r"([NS])(\d{1,2})[^\d]*([EW])(\d{1,2})", re.IGNORECASE
)

TEMPORAL_WINDOW = timedelta(minutes=30)
SPATIAL_THRESHOLD_DEG = 15.0


# ---------------------------------------------------------------------------
# Location helpers
# ---------------------------------------------------------------------------

def _parse_location(loc: str | None) -> tuple[float, float] | None:
    """Parse 'S09W11' → (lat, lon) with sign conventions.

    N → positive lat, S → negative. E → negative lon, W → positive lon
    (heliographic convention: west of central meridian is positive).
    Returns None if parse fails.
    """
    if not loc:
        return None
    m = _LOCATION_RE.search(loc)
    if not m:
        return None
    ns, lat_mag, ew, lon_mag = m.group(1), m.group(2), m.group(3), m.group(4)
    lat = float(lat_mag) * (1 if ns.upper() == "N" else -1)
    lon = float(lon_mag) * (1 if ew.upper() == "W" else -1)
    return lat, lon


def _angular_sep_deg(loc_a: str | None, loc_b: str | None) -> float | None:
    """Return max(|Δlat|, |Δlon|) between two source location strings."""
    pa = _parse_location(loc_a)
    pb = _parse_location(loc_b)
    if pa is None or pb is None:
        return None
    return max(abs(pa[0] - pb[0]), abs(pa[1] - pb[1]))


def _parse_dt(ts: str | None) -> datetime | None:
    """Parse ISO timestamp (with or without Z/offset) → UTC datetime."""
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


# ---------------------------------------------------------------------------
# Linked-events extraction
# ---------------------------------------------------------------------------

def _linked_flare_from_json(linked_event_ids: str | None) -> list[str]:
    """Extract FLR activity IDs from cme_events.linked_event_ids JSON."""
    if not linked_event_ids:
        return []
    try:
        items = json.loads(linked_event_ids)
        return [i for i in items if isinstance(i, str) and "FLR" in i.upper()]
    except (json.JSONDecodeError, TypeError):
        return []


# ---------------------------------------------------------------------------
# Core matcher
# ---------------------------------------------------------------------------

def match_cme_to_flare(
    cme: dict[str, Any],
    flares: list[dict[str, Any]],
) -> dict[str, Any]:
    """Match a single CME dict to the best available flare.

    Returns a dict with keys:
        linked_flare_id, flare_class_letter, flare_class_numeric,
        flare_peak_time, flare_active_region, flare_match_method
    """
    result: dict[str, Any] = {
        "linked_flare_id": None,
        "flare_class_letter": None,
        "flare_class_numeric": None,
        "flare_peak_time": None,
        "flare_active_region": None,
        "flare_match_method": "none",
    }

    if not flares:
        return result

    # --- Priority 1: DONKI linked_flare_id ---
    direct_id = cme.get("linked_flare_id")
    if direct_id:
        for f in flares:
            if f["flare_id"] == direct_id:
                return _fill_flare(result, f, "linked")
        # ID referenced but not in DB — still record the id, no other data
        result["linked_flare_id"] = direct_id
        result["flare_match_method"] = "linked_missing"
        return result

    # Also check linked_event_ids JSON array
    linked_ids = _linked_flare_from_json(cme.get("linked_event_ids"))
    if linked_ids:
        for f in flares:
            if f["flare_id"] in linked_ids:
                return _fill_flare(result, f, "linked")

    # --- Priority 2 & 3: temporal ± 30 min + optional spatial ± 15° ---
    cme_dt = _parse_dt(cme.get("start_time"))
    if cme_dt is None:
        return result

    candidates: list[tuple[timedelta, dict[str, Any]]] = []
    for f in flares:
        f_dt = _parse_dt(f.get("begin_time"))
        if f_dt is None:
            continue
        dt_gap = abs(cme_dt - f_dt)
        if dt_gap <= TEMPORAL_WINDOW:
            candidates.append((dt_gap, f))

    if not candidates:
        return result

    # Sort by temporal proximity; pick closest
    candidates.sort(key=lambda x: x[0])

    # Try spatial refinement first — if any candidate within ±15°, prefer it
    cme_loc = cme.get("source_location")
    for _, f in candidates:
        sep = _angular_sep_deg(cme_loc, f.get("source_location"))
        if sep is not None and sep <= SPATIAL_THRESHOLD_DEG:
            return _fill_flare(result, f, "spatial")

    # Fall back to closest temporal match (no spatial filter available)
    _, best = candidates[0]
    return _fill_flare(result, best, "temporal")


def _fill_flare(
    result: dict[str, Any],
    flare: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    result["linked_flare_id"] = flare.get("flare_id")
    result["flare_class_letter"] = flare.get("class_letter")
    result["flare_class_numeric"] = flare.get("class_magnitude")
    result["flare_peak_time"] = flare.get("peak_time")
    result["flare_active_region"] = flare.get("active_region_num")
    result["flare_match_method"] = method
    return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_cme_flare_matching(db_path: str) -> dict[str, dict[str, Any]]:
    """Match all CMEs in cme_events to flares. Returns {activity_id: match_dict}.

    Loads all flares into memory (< 10K rows — fine for SQLite).
    """
    engine = make_engine(db_path)
    results: dict[str, dict[str, Any]] = {}

    with Session(engine) as s:
        cmes = [dict(r._mapping) for r in s.execute(
            sa.select(CmeEvent.__table__)
        ).fetchall()]
        flares = [dict(r._mapping) for r in s.execute(
            sa.select(Flare.__table__)
        ).fetchall()]

    logger.info("Matching %d CMEs against %d flares", len(cmes), len(flares))
    linked = temporal = spatial = none = missing = 0

    for cme in cmes:
        m = match_cme_to_flare(cme, flares)
        results[cme["activity_id"]] = m
        method = m["flare_match_method"]
        if method == "linked":
            linked += 1
        elif method == "linked_missing":
            missing += 1
        elif method == "temporal":
            temporal += 1
        elif method == "spatial":
            spatial += 1
        else:
            none += 1

    logger.info(
        "CME-flare match summary: linked=%d temporal=%d spatial=%d "
        "linked_missing=%d none=%d",
        linked, temporal, spatial, missing, none,
    )
    return results
