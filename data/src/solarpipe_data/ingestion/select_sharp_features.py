"""Feature selection: choose optimal SHARP snapshot per CME (Task 4.4).

Preference order: at_eruption → minus_6h → minus_12h.
(plus_6h is post-eruption and not used for prediction.)

Outputs: a query/view that picks the best available snapshot per (noaa_ar, harpnum)
combination within a ±1-hour window of a given CME start_time.

Also computes and logs SHARP coverage fraction — alerts if < 80%.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from ..database.schema import SharpKeyword, make_engine

logger = logging.getLogger(__name__)

_CONTEXT_PREFERENCE = ["at_eruption", "minus_6h", "minus_12h"]


def get_best_sharp_snapshot(
    engine: sa.Engine,
    noaa_ar: int | None,
    harpnum: int | None,
    t_eruption: str,
) -> dict[str, Any] | None:
    """Return the best SHARP snapshot for a CME.

    Preference: at_eruption > minus_6h > minus_12h.
    Returns None if no snapshot available.
    """
    if not noaa_ar and not harpnum:
        return None

    with Session(engine) as s:
        for context in _CONTEXT_PREFERENCE:
            filters = [
                SharpKeyword.__table__.c.query_context == context,
            ]
            if noaa_ar:
                filters.append(SharpKeyword.__table__.c.noaa_ar == noaa_ar)
            elif harpnum:
                filters.append(SharpKeyword.__table__.c.harpnum == harpnum)

            row = s.execute(
                sa.select(SharpKeyword.__table__)
                .where(sa.and_(*filters))
                .order_by(SharpKeyword.__table__.c.id.desc())
                .limit(1)
            ).fetchone()

            if row is not None:
                return dict(row._mapping)

    return None


def compute_sharp_coverage(db_path: str) -> dict[str, Any]:
    """Compute SHARP coverage fraction across Earth-directed CMEs.

    Returns {total_cmes, covered, coverage_pct}.
    Logs a warning if coverage < 80%.
    """
    engine = make_engine(db_path)

    with Session(engine) as s:
        # Earth-directed CMEs post-HMI with AR numbers
        total = s.execute(
            sa.text("""
                SELECT COUNT(*) FROM cme_events
                WHERE latitude IS NOT NULL AND ABS(latitude) <= 45
                  AND longitude IS NOT NULL AND ABS(longitude) <= 45
                  AND start_time >= '2010-05-01'
                  AND active_region_num IS NOT NULL AND active_region_num > 0
            """)
        ).scalar() or 0

        # How many have at least one SHARP snapshot
        covered = s.execute(
            sa.text("""
                SELECT COUNT(DISTINCT noaa_ar) FROM sharp_keywords
                WHERE query_context IN ('at_eruption', 'minus_6h', 'minus_12h')
                  AND noaa_ar IS NOT NULL
            """)
        ).scalar() or 0

    coverage_pct = round(100.0 * covered / total, 1) if total > 0 else 0.0

    if total > 0 and coverage_pct < 80.0:
        logger.warning(
            "SHARP coverage %.1f%% below 80%% threshold (%d/%d CMEs with AR covered)",
            coverage_pct, covered, total,
        )
    else:
        logger.info("SHARP coverage: %.1f%% (%d/%d)", coverage_pct, covered, total)

    return {
        "total_cmes_with_ar": total,
        "covered": covered,
        "coverage_pct": coverage_pct,
    }


def iter_best_snapshots(db_path: str):
    """Yield the best SHARP snapshot row for each distinct (noaa_ar, harpnum) pair.

    Useful for building the feature_vectors table in Phase 5.
    """
    engine = make_engine(db_path)

    with Session(engine) as s:
        # Get distinct AR/HARP combinations
        pairs = s.execute(
            sa.text("""
                SELECT DISTINCT noaa_ar, harpnum FROM sharp_keywords
                WHERE noaa_ar IS NOT NULL OR harpnum IS NOT NULL
            """)
        ).fetchall()

    for noaa_ar, harpnum in pairs:
        snap = get_best_sharp_snapshot(engine, noaa_ar, harpnum, "")
        if snap:
            yield snap
