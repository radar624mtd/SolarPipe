"""Task 5.4 — Geomagnetic storm matching.

Given an ICME arrival time (from cme_icme_matcher), find the Dst minimum
and Kp maximum within a 24–48 hour post-arrival window.

L1 lag adjustment: shift L1 timestamps by +45 min before Dst correlation
(ACE/DSCOVR is ~45 min upstream of Earth).

Storm threshold: Dst < -30 nT.
Data source: dst_hourly (Kyoto WDC) and kp_3hr (GFZ).

Returns per-CME: dst_min_nt, dst_min_time, kp_max, storm_threshold_met.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from ..database.schema import DstHourly, Kp3hr, make_engine

logger = logging.getLogger(__name__)

# L1 upstream lag
_L1_LAG = timedelta(minutes=45)
# Post-arrival search window
_WINDOW_START = timedelta(hours=0)
_WINDOW_END = timedelta(hours=48)
# Storm threshold
_DST_STORM_THRESHOLD = -30.0


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


def find_storm_response(
    icme_arrival_time: str | None,
    dst_rows: list[dict[str, Any]],
    kp_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Find Dst min and Kp max in the 0–48 h window after ICME arrival.

    dst_rows / kp_rows: pre-loaded dicts with 'datetime', 'dst_nt', 'kp' keys.
    L1 lag: arrival_time is L1 passage → add 45 min to get Earth arrival.
    """
    result: dict[str, Any] = {
        "dst_min_nt": None,
        "dst_min_time": None,
        "kp_max": None,
        "storm_threshold_met": None,
    }

    arrival_dt = _parse_dt(icme_arrival_time)
    if arrival_dt is None:
        return result

    # Shift L1 timestamp to Earth
    earth_arrival = arrival_dt + _L1_LAG
    window_start = earth_arrival + _WINDOW_START
    window_end = earth_arrival + _WINDOW_END

    # --- Dst minimum ---
    dst_in_window = [
        r for r in dst_rows
        if r.get("dst_nt") is not None
        and _parse_dt(r.get("datetime")) is not None
        and window_start <= _parse_dt(r["datetime"]) <= window_end  # type: ignore[arg-type]
    ]

    if dst_in_window:
        best = min(dst_in_window, key=lambda r: r["dst_nt"])
        result["dst_min_nt"] = best["dst_nt"]
        result["dst_min_time"] = best["datetime"]
        result["storm_threshold_met"] = best["dst_nt"] < _DST_STORM_THRESHOLD

    # --- Kp maximum ---
    kp_in_window = [
        r for r in kp_rows
        if r.get("kp") is not None
        and _parse_dt(r.get("datetime")) is not None
        and window_start <= _parse_dt(r["datetime"]) <= window_end  # type: ignore[arg-type]
    ]

    if kp_in_window:
        result["kp_max"] = max(r["kp"] for r in kp_in_window)

    return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_storm_matching(
    db_path: str,
    icme_matches: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """For each CME with an ICME match, find the Dst/Kp storm response.

    icme_matches: {activity_id: icme_match_dict} from run_cme_icme_matching().
    Returns {activity_id: storm_dict}.
    """
    engine = make_engine(db_path)

    with Session(engine) as s:
        dst_rows = [dict(r._mapping) for r in s.execute(
            sa.select(DstHourly.__table__)
        ).fetchall()]
        kp_rows = [dict(r._mapping) for r in s.execute(
            sa.select(Kp3hr.__table__)
        ).fetchall()]

    logger.info(
        "Storm matching: %d Dst rows, %d Kp rows loaded",
        len(dst_rows), len(kp_rows),
    )

    results: dict[str, dict[str, Any]] = {}
    storm_count = 0

    for activity_id, icme_m in icme_matches.items():
        arrival = icme_m.get("icme_arrival_time")
        storm = find_storm_response(arrival, dst_rows, kp_rows)
        results[activity_id] = storm
        if storm.get("storm_threshold_met"):
            storm_count += 1

    logger.info(
        "Storm matching complete: %d/%d CMEs triggered storm (Dst < %d nT)",
        storm_count, len(icme_matches), int(_DST_STORM_THRESHOLD),
    )
    return results
