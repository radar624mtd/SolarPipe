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


def _build_indexed_series(
    rows: list[dict[str, Any]],
    dt_key: str,
    val_key: str,
) -> list[tuple[datetime, float]]:
    """Pre-parse datetimes and drop None values → sorted list of (dt, value).

    Parsing once here avoids O(n*m) repeated _parse_dt() calls in the inner loop.
    """
    out: list[tuple[datetime, float]] = []
    for r in rows:
        val = r.get(val_key)
        if val is None:
            continue
        dt = _parse_dt(r.get(dt_key))
        if dt is None:
            continue
        out.append((dt, float(val)))
    out.sort(key=lambda t: t[0])
    return out


def _window_slice(
    series: list[tuple[datetime, float]],
    window_start: datetime,
    window_end: datetime,
) -> list[tuple[datetime, float]]:
    """Binary-search slice of a sorted (dt, value) list within [start, end]."""
    import bisect
    lo = bisect.bisect_left(series, (window_start, float("-inf")))
    hi = bisect.bisect_right(series, (window_end, float("inf")))
    return series[lo:hi]


def find_storm_response_indexed(
    icme_arrival_time: str | None,
    dst_series: list[tuple[datetime, float]],
    kp_series: list[tuple[datetime, float]],
) -> dict[str, Any]:
    """Find Dst min / Kp max in 0–48 h window. Uses pre-parsed, sorted series."""
    result: dict[str, Any] = {
        "dst_min_nt": None,
        "dst_min_time": None,
        "kp_max": None,
        "storm_threshold_met": None,
    }

    arrival_dt = _parse_dt(icme_arrival_time)
    if arrival_dt is None:
        return result

    earth_arrival = arrival_dt + _L1_LAG
    window_start = earth_arrival + _WINDOW_START
    window_end = earth_arrival + _WINDOW_END

    dst_window = _window_slice(dst_series, window_start, window_end)
    if dst_window:
        best_dt, best_val = min(dst_window, key=lambda t: t[1])
        result["dst_min_nt"] = best_val
        result["dst_min_time"] = best_dt.isoformat()
        result["storm_threshold_met"] = best_val < _DST_STORM_THRESHOLD

    kp_window = _window_slice(kp_series, window_start, window_end)
    if kp_window:
        result["kp_max"] = max(v for _, v in kp_window)

    return result


# Keep original for backwards compatibility with tests
def find_storm_response(
    icme_arrival_time: str | None,
    dst_rows: list[dict[str, Any]],
    kp_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compatibility wrapper — builds indexed series then delegates."""
    dst_series = _build_indexed_series(dst_rows, "datetime", "dst_nt")
    kp_series = _build_indexed_series(kp_rows, "datetime", "kp")
    return find_storm_response_indexed(icme_arrival_time, dst_series, kp_series)


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

    # Pre-parse once — avoids O(n*m) datetime parsing in the inner loop
    dst_series = _build_indexed_series(dst_rows, "datetime", "dst_nt")
    kp_series = _build_indexed_series(kp_rows, "datetime", "kp")

    results: dict[str, dict[str, Any]] = {}
    storm_count = 0

    for activity_id, icme_m in icme_matches.items():
        arrival = icme_m.get("icme_arrival_time")
        storm = find_storm_response_indexed(arrival, dst_series, kp_series)
        results[activity_id] = storm
        if storm.get("storm_threshold_met"):
            storm_count += 1

    logger.info(
        "Storm matching complete: %d/%d CMEs triggered storm (Dst < %d nT)",
        storm_count, len(icme_matches), int(_DST_STORM_THRESHOLD),
    )
    return results
