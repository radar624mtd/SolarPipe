"""Task 5.6 — Quality flag scoring for feature_vectors.

Quality flag (1–5):
    5 — All 10 key features present; definitive ICME match
    4 — 1–2 null key features
    3 — 3–4 null key features (default; still usable for ML)
    2 — ≥5 null key features OR CDAW "Poor Event" (quality_flag=2)
    1 — CDAW "Very Poor Event" (quality_flag=1)

Key features (10):
    cme_speed_kms, cme_latitude, cme_longitude,
    flare_class_letter, sharp_harpnum, usflux,
    sw_speed_ambient, icme_arrival_time, dst_min_nt, f10_7
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from ..database.schema import CdawCmeEvent, make_engine

logger = logging.getLogger(__name__)

# Key features used for null-gap scoring. Must total exactly 10.
KEY_FEATURES = [
    "cme_speed_kms",
    "cme_latitude",
    "cme_longitude",
    "flare_class_letter",
    "sharp_harpnum",
    "usflux",
    "sw_speed_ambient",
    "icme_arrival_time",
    "dst_min_nt",
    "f10_7",
]


def compute_quality_flag(
    row: dict[str, Any],
    cdaw_quality: int | None = None,
) -> int:
    """Compute a 1–5 quality flag for an assembled feature_vectors row.

    cdaw_quality: quality_flag from nearest CDAW event (1 or 2 = degraded).
    Returns an integer in [1, 5].
    """
    # Flag 1: CDAW very poor event
    if cdaw_quality == 1:
        return 1

    # Flag 2: CDAW poor event
    if cdaw_quality == 2:
        return 2

    null_count = sum(1 for f in KEY_FEATURES if row.get(f) is None)

    if null_count >= 5:
        return 2
    if null_count >= 3:
        return 3
    if null_count >= 1:
        return 4

    # All 10 key features present — require definitive ICME data for flag 5
    icme_method = row.get("icme_match_method", "none")
    icme_conf = row.get("icme_match_confidence") or 0.0
    definitive = icme_method == "linked" or (
        icme_method == "transit_estimate" and icme_conf >= 0.8
    )
    return 5 if definitive else 4


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


def build_cdaw_quality_index(engine: sa.Engine) -> dict[str, int]:
    """Return {YYYY-MM-DD HH:MM: quality_flag} for CDAW poor events only.

    Keyed by first 16 chars of datetime for fast nearest-time lookup.
    Only includes rows with quality_flag < 3 (i.e., poor or very poor).
    """
    with Session(engine) as s:
        rows = s.execute(
            sa.select(
                CdawCmeEvent.__table__.c.datetime,
                CdawCmeEvent.__table__.c.quality_flag,
            ).where(CdawCmeEvent.__table__.c.quality_flag < 3)
        ).fetchall()
    return {(r.datetime or "")[:16]: r.quality_flag for r in rows if r.datetime}


def lookup_cdaw_quality(
    launch_time: str | None,
    cdaw_quality_index: dict[str, int],
    window_minutes: int = 30,
) -> int | None:
    """Return CDAW quality_flag for the nearest event within ±window_minutes.

    Returns None if no poor CDAW event is found nearby (quality assumed 3).
    """
    if not launch_time or not cdaw_quality_index:
        return None
    launch_dt = _parse_dt(launch_time)
    if launch_dt is None:
        return None
    best_flag: int | None = None
    best_gap = timedelta(minutes=window_minutes + 1)
    for ts_str, flag in cdaw_quality_index.items():
        try:
            cdaw_dt = _parse_dt(ts_str + ":00")
            if cdaw_dt is None:
                continue
            gap = abs(launch_dt - cdaw_dt)
            if gap < best_gap:
                best_gap = gap
                best_flag = flag
        except Exception:  # noqa: BLE001
            continue
    return best_flag
