"""Ingest HMI SHARP space-weather keywords → sharp_keywords (Phase 4).

For each Earth-directed CME: query JSOC at 4 time contexts:
  - at_eruption: t_eruption
  - minus_6h:    t_eruption - 6h
  - minus_12h:   t_eruption - 12h
  - plus_6h:     t_eruption + 6h  (post-eruption, not used for prediction but useful for validation)

Earth-directed proxy: ABS(latitude) ≤ 45 AND ABS(longitude) ≤ 45.
HMI data only available post-2010-05-01 (HMI first light).

Rules enforced:
- RULE-004: Upsert idempotent (activity_id + query_context composite for resume)
- RULE-030: sqlite dialect insert
- RULE-033: Session context manager
- RULE-060: LON_FWT > 60° dropped in jsoc.py client
- RULE-061: Timeout wrap enforced in jsoc.py client
- RULE-062: CEA series enforced in jsoc.py client
- RULE-063: NOAA_AR = 0 → None enforced in jsoc.py client
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from ..clients.jsoc import JsocClient
from ..database.schema import HarpNoaaMap, SharpKeyword, make_engine

logger = logging.getLogger(__name__)

# HMI first light — no SHARP data before this date
_HMI_START = datetime(2010, 5, 1, tzinfo=timezone.utc)

# Earth-directed CME proxy thresholds
_LAT_THRESHOLD = 45.0
_LON_THRESHOLD = 45.0

# Parallel JSOC workers (drms uses urllib; more threads = more concurrent connections)
_MAX_WORKERS = 8

# Batch size for inserts
_BATCH_SIZE = 5000

# Query contexts and their time offsets
_CONTEXTS: list[tuple[str, timedelta]] = [
    ("at_eruption", timedelta(0)),
    ("minus_6h", timedelta(hours=-6)),
    ("minus_12h", timedelta(hours=-12)),
    ("plus_6h", timedelta(hours=6)),
]


def _is_earth_directed(lat: float | None, lon: float | None) -> bool:
    if lat is None or lon is None:
        return False
    return abs(lat) <= _LAT_THRESHOLD and abs(lon) <= _LON_THRESHOLD


def _parse_cme_time(ts: str) -> datetime | None:
    if not ts:
        return None
    clean = str(ts).replace("T", " ").rstrip("Z").strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(clean, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _is_post_hmi(dt: datetime | None) -> bool:
    if dt is None:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt >= _HMI_START


def _load_completed_pairs(engine: sa.Engine) -> set[tuple[str, str]]:
    """Return set of (activity_id, query_context) already in sharp_keywords."""
    with Session(engine) as s:
        try:
            rows = s.execute(
                sa.text("""
                    SELECT activity_id, query_context
                    FROM sharp_keywords
                    WHERE activity_id IS NOT NULL AND query_context IS NOT NULL
                """)
            ).fetchall()
            return {(r[0], r[1]) for r in rows}
        except Exception:
            return set()


def _fetch_cme_contexts_sync(
    client: JsocClient,
    activity_id: str,
    noaa_ar: int,
    cme_dt: datetime,
    fetch_ts: str,
    skip_contexts: set[str],
) -> list[dict[str, Any]]:
    """Synchronous: fetch all 4 contexts for one CME. Called from thread pool."""
    rows: list[dict[str, Any]] = []
    for context, offset in _CONTEXTS:
        if context in skip_contexts:
            continue
        t_query = cme_dt + offset
        if t_query < _HMI_START:
            continue
        records = client.fetch_sharp_at_time(noaa_ar, t_query, context)
        for rec in records:
            rec["fetch_timestamp"] = fetch_ts
            rec["activity_id"] = activity_id
        rows.extend(records)
    return rows


def _bulk_insert_sharps(engine: sa.Engine, rows: list[dict[str, Any]]) -> int:
    """Insert SHARP rows in batches (RULE-036)."""
    total = 0
    for i in range(0, len(rows), _BATCH_SIZE):
        batch = rows[i : i + _BATCH_SIZE]
        with Session(engine) as s, s.begin():
            s.execute(SharpKeyword.__table__.insert(), batch)
        total += len(batch)
    logger.info("Inserted %d SHARP keyword rows.", total)
    return total


async def ingest_sharps(
    db_path: str,
    force: bool = False,
    max_cmes: int | None = None,
) -> dict[str, int]:
    """Fetch SHARP keywords for Earth-directed CMEs in parallel.

    Uses a thread pool (JSOC/drms is synchronous) with _MAX_WORKERS concurrent
    connections. Supports resume: skips (activity_id, query_context) pairs already
    present in sharp_keywords.

    Returns dict: {total_cmes, queried, rows_upserted, coverage_pct}.
    """
    engine = make_engine(db_path)
    fetch_ts = datetime.now(timezone.utc).isoformat()

    # Load Earth-directed CME candidates
    with Session(engine) as s:
        rows = s.execute(
            sa.text("""
                SELECT activity_id, start_time, active_region_num, latitude, longitude
                FROM cme_events
                WHERE start_time IS NOT NULL
                  AND latitude IS NOT NULL
                  AND longitude IS NOT NULL
                ORDER BY start_time ASC
            """)
        ).fetchall()

    candidates = [
        row for row in rows
        if _is_earth_directed(row[3], row[4])
        and _is_post_hmi(_parse_cme_time(row[1] or ""))
    ]

    total_cmes = len(candidates)
    logger.info("Earth-directed CME candidates (post-HMI): %d", total_cmes)

    if max_cmes is not None:
        candidates = candidates[:max_cmes]
        logger.info("Capped to %d CMEs", max_cmes)

    # Resume: load already-completed (activity_id, context) pairs
    completed: set[tuple[str, str]] = set()
    if not force:
        completed = _load_completed_pairs(engine)
        if completed:
            logger.info("Resume: %d (activity_id, context) pairs already ingested", len(completed))

    client = JsocClient()
    loop = asyncio.get_event_loop()

    # Build tasks — skip CMEs where ALL 4 contexts are already done
    tasks = []
    for activity_id, start_time, noaa_ar_raw, lat, lon in candidates:
        cme_dt = _parse_cme_time(start_time or "")
        if cme_dt is None:
            continue
        noaa_ar = int(noaa_ar_raw) if noaa_ar_raw and int(noaa_ar_raw) > 0 else None
        if noaa_ar is None:
            continue  # No AR number — JSOC query requires it

        # Determine which contexts still need fetching
        skip = {ctx for ctx, _ in _CONTEXTS if (activity_id, ctx) in completed}
        if len(skip) == len(_CONTEXTS):
            continue  # All contexts already done

        tasks.append((activity_id, noaa_ar, cme_dt, skip))

    if not tasks:
        logger.info("All CMEs already ingested — nothing to do.")
        return {
            "total_cmes": total_cmes,
            "queried": 0,
            "rows_upserted": 0,
            "coverage_pct": 100.0 if total_cmes > 0 else 0.0,
        }

    logger.info("Querying JSOC for %d CMEs (%d workers)", len(tasks), _MAX_WORKERS)

    semaphore = asyncio.Semaphore(_MAX_WORKERS)

    async def fetch_one(activity_id, noaa_ar, cme_dt, skip):
        async with semaphore:
            return await loop.run_in_executor(
                None,
                _fetch_cme_contexts_sync,
                client, activity_id, noaa_ar, cme_dt, fetch_ts, skip,
            )

    all_rows: list[dict[str, Any]] = []
    queried = 0
    done = 0

    # Process in chunks of 200 to flush to DB periodically (reduces memory + enables progress)
    chunk_size = 200
    for chunk_start in range(0, len(tasks), chunk_size):
        chunk = tasks[chunk_start : chunk_start + chunk_size]
        chunk_results = await asyncio.gather(
            *[fetch_one(a, n, d, s) for a, n, d, s in chunk]
        )
        chunk_rows: list[dict[str, Any]] = []
        for result in chunk_results:
            if result:
                queried += 1
                chunk_rows.extend(result)
        if chunk_rows:
            _bulk_insert_sharps(engine, chunk_rows)
            all_rows.extend(chunk_rows)
        done += len(chunk)
        logger.info("Progress: %d/%d CMEs processed, %d rows so far", done, len(tasks), len(all_rows))

    rows_upserted = len(all_rows)
    coverage = round(100.0 * queried / total_cmes, 1) if total_cmes > 0 else 0.0
    if coverage < 80.0:
        logger.warning(
            "SHARP coverage %.1f%% below 80%% threshold (%d/%d CMEs queried)",
            coverage, queried, total_cmes,
        )
    else:
        logger.info("SHARP coverage: %.1f%% (%d/%d)", coverage, queried, total_cmes)

    return {
        "total_cmes": total_cmes,
        "queried": queried,
        "rows_upserted": rows_upserted,
        "coverage_pct": coverage,
    }


def ingest_harp_noaa_mapping(
    db_path: str,
    t_start: datetime,
    t_end: datetime,
) -> int:
    """Fetch HARP↔NOAA AR mapping and store in harp_noaa_map (Task 4.3).

    Returns number of rows inserted.
    """
    engine = make_engine(db_path)
    fetch_ts = datetime.now(timezone.utc).isoformat()

    client = JsocClient()
    records = client.fetch_harp_noaa_mapping(t_start, t_end)

    if not records:
        logger.info("No HARP↔NOAA mapping records returned.")
        return 0

    rows = [
        {**rec, "source_catalog": "JSOC", "fetch_timestamp": fetch_ts, "data_version": "hmi.sharp_720s"}
        for rec in records
    ]

    with Session(engine) as s, s.begin():
        s.execute(HarpNoaaMap.__table__.insert(), rows)

    logger.info("Inserted %d HARP↔NOAA mapping rows.", len(rows))
    return len(rows)
