"""Deduplicate flares table — merge GOES and DONKI records for the same event.

Strategy (per Implementation Plan Task 2.5):
  Same event = begin_time within ±2 minutes AND same active_region_num.
  When a GOES record and DONKI record match:
    - Keep the DONKI record as canonical (has note, link, linkedEvents)
    - Stamp the DONKI record's goes_satellite from the GOES record
    - Delete the GOES duplicate
  When active_region_num is NULL for both: match on begin_time ±2 min only
  (higher false-positive risk — only applies to B/C class flares).

This is run after both ingest_donki_flares() and ingest_goes_flares() complete.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_MATCH_WINDOW_S = 120  # ±2 minutes in seconds


def dedup_flares(engine: Engine) -> int:
    """Find GOES/DONKI duplicate pairs and merge them.

    Returns the number of GOES duplicates removed.
    """
    pairs = _find_duplicate_pairs(engine)
    if not pairs:
        logger.info("dedup_flares: no duplicates found")
        return 0

    removed = _merge_pairs(engine, pairs)
    logger.info("dedup_flares: removed %d GOES duplicates (merged into DONKI records)", removed)
    return removed


def _find_duplicate_pairs(engine: Engine) -> list[tuple[str, str, str | None]]:
    """Find (donki_flare_id, goes_flare_id, goes_satellite) pairs that are the same event.

    Returns list of (donki_id, goes_id, goes_satellite).
    """
    sql = text("""
        SELECT
            d.flare_id  AS donki_id,
            g.flare_id  AS goes_id,
            g.goes_satellite
        FROM flares d
        JOIN flares g
            ON g.source_catalog = 'GOES'
            AND d.source_catalog = 'DONKI'
            AND (
                d.active_region_num IS NOT NULL
                AND g.active_region_num IS NOT NULL
                AND d.active_region_num = g.active_region_num
                AND ABS(
                    CAST(strftime('%s', REPLACE(d.begin_time, 'Z', '')) AS INTEGER)
                    - CAST(strftime('%s', REPLACE(g.begin_time, 'Z', '')) AS INTEGER)
                ) <= :window_s
            )
        UNION
        SELECT
            d.flare_id,
            g.flare_id,
            g.goes_satellite
        FROM flares d
        JOIN flares g
            ON g.source_catalog = 'GOES'
            AND d.source_catalog = 'DONKI'
            AND d.active_region_num IS NULL
            AND g.active_region_num IS NULL
            AND ABS(
                CAST(strftime('%s', REPLACE(d.begin_time, 'Z', '')) AS INTEGER)
                - CAST(strftime('%s', REPLACE(g.begin_time, 'Z', '')) AS INTEGER)
            ) <= :window_s
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, {"window_s": _MATCH_WINDOW_S}).fetchall()

    return [(r[0], r[1], r[2]) for r in rows]


def _merge_pairs(
    engine: Engine,
    pairs: list[tuple[str, str, str | None]],
) -> int:
    """For each duplicate pair: stamp goes_satellite onto DONKI record, delete GOES record."""
    removed = 0
    with Session(engine) as s, s.begin():
        for donki_id, goes_id, goes_satellite in pairs:
            if goes_satellite:
                s.execute(
                    text(
                        "UPDATE flares SET goes_satellite = :sat WHERE flare_id = :fid"
                    ),
                    {"sat": goes_satellite, "fid": donki_id},
                )
            s.execute(
                text("DELETE FROM flares WHERE flare_id = :fid"),
                {"fid": goes_id},
            )
            removed += 1

    return removed
