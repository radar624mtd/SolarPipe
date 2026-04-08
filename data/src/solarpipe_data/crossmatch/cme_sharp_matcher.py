"""Task 5.3 — CME ↔ SHARP cross-matching.

Priority order:
1. NOAA AR number match: cme_events.active_region_num == sharp_keywords.noaa_ar
2. Source location proximity: |Δlat| ≤ 15° AND |Δlon| ≤ 15° between CME source_location
   and best SHARP snapshot lat_fwt/lon_fwt (disk-centre-facing coordinates)
3. No match — SHARP features null in feature_vectors.

Delegates snapshot preference (at_eruption > minus_6h > minus_12h) to
ingestion.select_sharp_features.get_best_sharp_snapshot().
"""
from __future__ import annotations

import logging
import re
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from ..database.schema import SharpKeyword, make_engine
from ..ingestion.select_sharp_features import get_best_sharp_snapshot

logger = logging.getLogger(__name__)

_LOCATION_RE = re.compile(
    r"([NS])(\d{1,2})[^\d]*([EW])(\d{1,2})", re.IGNORECASE
)
_SHARP_SPATIAL_THRESHOLD = 15.0  # degrees

_SHARP_COLUMNS = [
    "harpnum", "noaa_ar", "query_context",
    "usflux", "meangam", "meangbt", "meangbz", "meangbh",
    "meanjzd", "totusjz", "meanalp", "meanjzh", "totusjh",
    "absnjzh", "savncpp", "meanpot", "totpot", "meanshr",
    "shrgt45", "r_value", "area_acr", "lat_fwt", "lon_fwt",
]


def _parse_location(loc: str | None) -> tuple[float, float] | None:
    """Parse 'S09W11' → (lat, lon) with sign conventions."""
    if not loc:
        return None
    m = _LOCATION_RE.search(loc)
    if not m:
        return None
    ns, lat_mag, ew, lon_mag = m.group(1), m.group(2), m.group(3), m.group(4)
    lat = float(lat_mag) * (1 if ns.upper() == "N" else -1)
    lon = float(lon_mag) * (1 if ew.upper() == "W" else -1)
    return lat, lon


def _sharp_to_feature_dict(snap: dict[str, Any] | None) -> dict[str, Any]:
    """Extract SHARP feature columns into flat dict for feature_vectors."""
    if snap is None:
        return {
            "sharp_harpnum": None,
            "sharp_noaa_ar": None,
            "sharp_snapshot_context": None,
            "sharp_match_method": "none",
            **{col: None for col in _SHARP_COLUMNS[3:]},  # all feature columns null
        }
    return {
        "sharp_harpnum": snap.get("harpnum"),
        "sharp_noaa_ar": snap.get("noaa_ar"),
        "sharp_snapshot_context": snap.get("query_context"),
        **{col: snap.get(col) for col in _SHARP_COLUMNS[3:]},
    }


def match_cme_to_sharp(
    cme: dict[str, Any],
    engine: sa.Engine,
) -> dict[str, Any]:
    """Match a single CME to the best SHARP snapshot.

    Returns a dict merging SHARP feature columns + sharp_match_method.
    """
    noaa_ar = cme.get("active_region_num")
    source_loc = cme.get("source_location")
    t_eruption = cme.get("start_time") or ""

    # --- Priority 1: NOAA AR number match ---
    if noaa_ar and noaa_ar > 0:
        snap = get_best_sharp_snapshot(engine, noaa_ar=noaa_ar, harpnum=None, t_eruption=t_eruption)
        if snap:
            result = _sharp_to_feature_dict(snap)
            result["sharp_match_method"] = "noaa_ar"
            return result

    # --- Priority 2: spatial proximity via HARP lat_fwt / lon_fwt ---
    cme_pos = _parse_location(source_loc)
    if cme_pos is not None:
        snap = _find_by_location(engine, cme_pos, t_eruption)
        if snap:
            result = _sharp_to_feature_dict(snap)
            result["sharp_match_method"] = "location"
            return result

    # --- No match ---
    result = _sharp_to_feature_dict(None)
    return result


def _find_by_location(
    engine: sa.Engine,
    cme_pos: tuple[float, float],
    t_eruption: str,
) -> dict[str, Any] | None:
    """Find the SHARP snapshot spatially closest to (cme_lat, cme_lon).

    Searches at_eruption snapshots first, then minus_6h.
    Returns None if no snapshot within threshold.
    """
    cme_lat, cme_lon = cme_pos
    lat_lo = cme_lat - _SHARP_SPATIAL_THRESHOLD
    lat_hi = cme_lat + _SHARP_SPATIAL_THRESHOLD
    lon_lo = cme_lon - _SHARP_SPATIAL_THRESHOLD
    lon_hi = cme_lon + _SHARP_SPATIAL_THRESHOLD

    with Session(engine) as s:
        for context in ["at_eruption", "minus_6h", "minus_12h"]:
            rows = s.execute(
                sa.select(SharpKeyword.__table__)
                .where(
                    sa.and_(
                        SharpKeyword.__table__.c.query_context == context,
                        SharpKeyword.__table__.c.lat_fwt.between(lat_lo, lat_hi),
                        SharpKeyword.__table__.c.lon_fwt.between(lon_lo, lon_hi),
                    )
                )
                .order_by(
                    # Order by Chebyshev distance proxy (no SQL abs combo — compute in Python)
                    SharpKeyword.__table__.c.id.desc()
                )
                .limit(20)
            ).fetchall()

            if not rows:
                continue

            # Pick closest by Chebyshev distance
            best = min(
                rows,
                key=lambda r: max(
                    abs((r._mapping.get("lat_fwt") or 0) - cme_lat),
                    abs((r._mapping.get("lon_fwt") or 0) - cme_lon),
                ),
            )
            sep = max(
                abs((best._mapping.get("lat_fwt") or 0) - cme_lat),
                abs((best._mapping.get("lon_fwt") or 0) - cme_lon),
            )
            if sep <= _SHARP_SPATIAL_THRESHOLD:
                return dict(best._mapping)

    return None


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_cme_sharp_matching(db_path: str) -> dict[str, dict[str, Any]]:
    """Match all CMEs to SHARP snapshots. Returns {activity_id: match_dict}."""
    engine = make_engine(db_path)
    results: dict[str, dict[str, Any]] = {}

    with Session(engine) as s:
        cmes = [dict(r._mapping) for r in s.execute(
            sa.select(
                sa.text("activity_id, start_time, source_location, active_region_num")
            ).select_from(sa.text("cme_events"))
        ).fetchall()]

    logger.info("Matching %d CMEs to SHARP snapshots", len(cmes))
    noaa_match = location_match = none = 0

    for cme in cmes:
        m = match_cme_to_sharp(cme, engine)
        results[cme["activity_id"]] = m
        method = m.get("sharp_match_method", "none")
        if method == "noaa_ar":
            noaa_match += 1
        elif method == "location":
            location_match += 1
        else:
            none += 1

    logger.info(
        "CME-SHARP match summary: noaa_ar=%d location=%d none=%d",
        noaa_match, location_match, none,
    )
    return results
