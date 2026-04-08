"""Compute ambient solar wind context for each CME (Task 3.6).

For each CME in cme_events, compute 6-hour pre-start window averages from
solar_wind_hourly and store in sw_ambient_context.

Rules enforced:
- RULE-004: Upsert idempotent on activity_id PK
- RULE-030: sqlite dialect insert
- RULE-033: Session context manager
- RULE-070: bz_gsm used (sw_bz_ambient)
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from ..database.schema import CmeEvent, SwAmbientContext, make_engine

logger = logging.getLogger(__name__)

_WINDOW_HOURS = 6


def _floor_hour(dt: datetime) -> str:
    """Truncate datetime to hour → 'YYYY-MM-DD HH:00'."""
    return dt.strftime("%Y-%m-%d %H:00")


def _parse_cme_time_safe(ts: str) -> datetime | None:
    """Robust CME timestamp parser."""
    if not ts:
        return None
    clean = str(ts).replace("T", " ").rstrip("Z").strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(clean[:16 if "%M" in fmt else 10], fmt)
        except ValueError:
            continue
    return None


def compute_ambient_context(db_path: str, force: bool = False) -> int:
    """Compute 6h pre-arrival ambient SW context for all CMEs.

    Returns number of rows upserted.
    """
    engine = make_engine(db_path)
    fetch_ts = datetime.now(timezone.utc).isoformat()

    with Session(engine) as s:
        # Load all CME activity_ids and start_times
        cme_rows = s.execute(
            sa.select(CmeEvent.activity_id, CmeEvent.start_time)
        ).fetchall()

        # Load existing context IDs to skip if not force
        existing_ids: set[str] = set()
        if not force:
            try:
                existing_ids = {
                    row[0]
                    for row in s.execute(
                        sa.select(SwAmbientContext.activity_id)
                    ).fetchall()
                }
            except Exception:
                pass  # table may not exist yet

    logger.info("Computing ambient SW context for %d CMEs", len(cme_rows))

    rows: list[dict[str, Any]] = []
    skipped = 0

    with Session(engine) as s:
        for activity_id, start_time in cme_rows:
            if not force and activity_id in existing_ids:
                skipped += 1
                continue

            cme_dt = _parse_cme_time_safe(start_time or "")
            if cme_dt is None:
                continue

            window_end = cme_dt
            window_start = cme_dt - timedelta(hours=_WINDOW_HOURS)

            ws = _floor_hour(window_start)
            we = _floor_hour(window_end)

            # Query solar_wind_hourly for the window
            sw_rows = s.execute(
                sa.text("""
                    SELECT flow_speed, proton_density, b_scalar_avg, bz_gsm
                    FROM solar_wind_hourly
                    WHERE datetime >= :ws AND datetime < :we
                      AND (flow_speed IS NOT NULL
                           OR proton_density IS NOT NULL
                           OR b_scalar_avg IS NOT NULL
                           OR bz_gsm IS NOT NULL)
                """),
                {"ws": ws, "we": we},
            ).fetchall()

            if not sw_rows:
                continue

            speeds = [r[0] for r in sw_rows if r[0] is not None]
            densities = [r[1] for r in sw_rows if r[1] is not None]
            bts = [r[2] for r in sw_rows if r[2] is not None]
            bzs = [r[3] for r in sw_rows if r[3] is not None]

            rows.append({
                "activity_id": activity_id,
                "window_start": ws,
                "window_end": we,
                "n_hours": len(sw_rows),
                "sw_speed_ambient": sum(speeds) / len(speeds) if speeds else None,
                "sw_density_ambient": sum(densities) / len(densities) if densities else None,
                "sw_bt_ambient": sum(bts) / len(bts) if bts else None,
                "sw_bz_ambient": sum(bzs) / len(bzs) if bzs else None,
                "source_catalog": "OMNI",
                "fetch_timestamp": fetch_ts,
                "data_version": "1.0",
            })

    logger.info("Computed %d ambient context rows (skipped %d existing)", len(rows), skipped)

    if not rows:
        return 0

    # Batch upsert (RULE-036: commit every 5K rows for large sets)
    batch_size = 5000
    total_upserted = 0

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        with Session(engine) as s, s.begin():
            stmt = insert(SwAmbientContext)
            update_cols = {
                c: stmt.excluded[c]
                for c in [
                    "window_start", "window_end", "n_hours",
                    "sw_speed_ambient", "sw_density_ambient",
                    "sw_bt_ambient", "sw_bz_ambient",
                    "source_catalog", "fetch_timestamp", "data_version",
                ]
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=["activity_id"], set_=update_cols
            )
            s.execute(stmt, batch)
        total_upserted += len(batch)

    logger.info("Upserted %d ambient SW context rows.", total_upserted)
    return total_upserted
