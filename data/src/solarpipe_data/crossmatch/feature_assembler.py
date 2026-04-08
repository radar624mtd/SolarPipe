"""Task 5.5 — Feature vector assembly.

Combines outputs from all four matchers (5.1–5.4) plus CME kinematics,
ambient solar wind, and activity context (F10.7, sunspot number) into a
single row per CME for the feature_vectors table.

Quality flag is computed in Task 5.6; this module writes the default
value (3) unless a quality-degrading condition is detected here.

Deferred (null-filled):
    dimming_area, dimming_asymmetry — AIA processing (Phase 6)
    hcs_tilt_angle, hcs_distance   — PFSS/GONG maps (Phase 6)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from ..database.schema import (
    CmeAnalysis,
    CmeEvent,
    F107Daily,
    FeatureVector,
    SilsoDailySSN,
    SwAmbientContext,
    make_engine,
)
from ..database.queries import upsert
from .cme_flare_matcher import run_cme_flare_matching
from .cme_icme_matcher import run_cme_icme_matching
from .cme_sharp_matcher import run_cme_sharp_matching
from .storm_matcher import run_storm_matching

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_from_ts(ts: str | None) -> str | None:
    """Extract YYYY-MM-DD from an ISO timestamp string."""
    if not ts:
        return None
    return ts[:10]


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


def _lookup_f107(
    launch_date: str | None,
    f107_index: dict[str, float],
) -> float | None:
    """Return F10.7 observed value for a launch date (YYYY-MM-DD key)."""
    if not launch_date:
        return None
    return f107_index.get(launch_date)


def _lookup_ssn(
    launch_date: str | None,
    ssn_index: dict[str, float],
) -> float | None:
    """Return SILSO sunspot number for a launch date."""
    if not launch_date:
        return None
    return ssn_index.get(launch_date)


def _best_cme_speed(
    cme: dict[str, Any],
    analyses: list[dict[str, Any]],
) -> float | None:
    """Return speed from the most accurate CME analysis; fall back to cme_events."""
    # Prefer level_of_data=2 (definitive) → 1 (NRT) → 0 (realtime) → cme_events
    for level in (2, 1, 0):
        for a in analyses:
            if a.get("cme_activity_id") == cme["activity_id"] and a.get("level_of_data") == level:
                if a.get("speed_kms") is not None:
                    return a["speed_kms"]
    return cme.get("speed_kms")


def _best_analysis(
    cme: dict[str, Any],
    analyses: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the most accurate CmeAnalysis row for this CME."""
    for level in (2, 1, 0):
        for a in analyses:
            if a.get("cme_activity_id") == cme["activity_id"] and a.get("level_of_data") == level:
                return a
    return None


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def assemble_feature_vector(
    cme: dict[str, Any],
    analyses: list[dict[str, Any]],
    flare_match: dict[str, Any],
    icme_match: dict[str, Any],
    sharp_match: dict[str, Any],
    storm_match: dict[str, Any],
    ambient: dict[str, Any] | None,
    f107_index: dict[str, float],
    ssn_index: dict[str, float],
) -> dict[str, Any]:
    """Assemble one feature_vectors row from all matcher outputs.

    Returns a dict that maps directly to FeatureVector columns.
    All deferred features (dimming_*, hcs_*) are null.
    Quality flag defaults to 3; Task 5.6 will refine it.
    """
    activity_id = cme["activity_id"]
    launch_date = _date_from_ts(cme.get("start_time"))
    best = _best_analysis(cme, analyses)

    return {
        # Identity
        "activity_id": activity_id,
        "launch_time": cme.get("start_time"),

        # Kinematic — prefer analysis-level fields over cme_events
        "cme_speed_kms": _best_cme_speed(cme, analyses),
        "cme_half_angle_deg": (best or {}).get("half_angle_deg") or cme.get("half_angle_deg"),
        "cme_latitude": (best or {}).get("latitude") or cme.get("latitude"),
        "cme_longitude": (best or {}).get("longitude") or cme.get("longitude"),
        "cme_mass_grams": cme.get("cme_mass_grams"),          # CDAW only; ~40% null
        "cme_angular_width_deg": cme.get("cme_angular_width_deg"),  # CDAW only

        # Flare
        "linked_flare_id": flare_match.get("linked_flare_id"),
        "flare_class_letter": flare_match.get("flare_class_letter"),
        "flare_class_numeric": flare_match.get("flare_class_numeric"),
        "flare_peak_time": flare_match.get("flare_peak_time"),
        "flare_active_region": flare_match.get("flare_active_region"),
        "flare_match_method": flare_match.get("flare_match_method", "none"),

        # SHARP
        "sharp_harpnum": sharp_match.get("sharp_harpnum"),
        "sharp_noaa_ar": sharp_match.get("sharp_noaa_ar"),
        "sharp_snapshot_context": sharp_match.get("sharp_snapshot_context"),
        "usflux": sharp_match.get("usflux"),
        "meangam": sharp_match.get("meangam"),
        "meangbt": sharp_match.get("meangbt"),
        "meangbz": sharp_match.get("meangbz"),
        "meangbh": sharp_match.get("meangbh"),
        "meanjzd": sharp_match.get("meanjzd"),
        "totusjz": sharp_match.get("totusjz"),
        "meanalp": sharp_match.get("meanalp"),
        "meanjzh": sharp_match.get("meanjzh"),
        "totusjh": sharp_match.get("totusjh"),
        "absnjzh": sharp_match.get("absnjzh"),
        "savncpp": sharp_match.get("savncpp"),
        "meanpot": sharp_match.get("meanpot"),
        "totpot": sharp_match.get("totpot"),
        "meanshr": sharp_match.get("meanshr"),
        "shrgt45": sharp_match.get("shrgt45"),
        "r_value": sharp_match.get("r_value"),
        "area_acr": sharp_match.get("area_acr"),
        "sharp_match_method": sharp_match.get("sharp_match_method", "none"),

        # Ambient solar wind
        "sw_speed_ambient": (ambient or {}).get("sw_speed_ambient"),
        "sw_density_ambient": (ambient or {}).get("sw_density_ambient"),
        "sw_bt_ambient": (ambient or {}).get("sw_bt_ambient"),
        "sw_bz_ambient": (ambient or {}).get("sw_bz_ambient"),

        # ICME
        "linked_ips_id": icme_match.get("linked_ips_id"),
        "icme_arrival_time": icme_match.get("icme_arrival_time"),
        "transit_time_hours": icme_match.get("transit_time_hours"),
        "icme_match_method": icme_match.get("icme_match_method", "none"),
        "icme_match_confidence": icme_match.get("icme_match_confidence"),

        # Geomagnetic response
        "dst_min_nt": storm_match.get("dst_min_nt"),
        "dst_min_time": storm_match.get("dst_min_time"),
        "kp_max": storm_match.get("kp_max"),
        "storm_threshold_met": storm_match.get("storm_threshold_met"),

        # Deferred (null until Phase 6)
        "dimming_area": None,
        "dimming_asymmetry": None,
        "hcs_tilt_angle": None,
        "hcs_distance": None,

        # Activity context
        "f10_7": _lookup_f107(launch_date, f107_index),
        "sunspot_number": _lookup_ssn(launch_date, ssn_index),

        # Quality — default 3; refined by Task 5.6
        "quality_flag": 3,

        # Provenance
        "source_catalog": "crossmatch",
    }


# ---------------------------------------------------------------------------
# Index builders (in-memory lookups)
# ---------------------------------------------------------------------------

def _build_f107_index(engine: sa.Engine) -> dict[str, float]:
    """Return {YYYY-MM-DD: f10_7_obs} from f107_daily."""
    with Session(engine) as s:
        rows = s.execute(
            sa.select(F107Daily.__table__.c.date, F107Daily.__table__.c.f10_7_obs)
        ).fetchall()
    return {r.date: r.f10_7_obs for r in rows if r.f10_7_obs is not None}


def _build_ssn_index(engine: sa.Engine) -> dict[str, float]:
    """Return {YYYY-MM-DD: sunspot_number} from silso_daily_ssn."""
    with Session(engine) as s:
        rows = s.execute(
            sa.select(
                SilsoDailySSN.__table__.c.date,
                SilsoDailySSN.__table__.c.sunspot_number,
            )
        ).fetchall()
    return {r.date: r.sunspot_number for r in rows if r.sunspot_number is not None}


def _build_ambient_index(engine: sa.Engine) -> dict[str, dict[str, Any]]:
    """Return {activity_id: ambient_row_dict} from sw_ambient_context."""
    with Session(engine) as s:
        rows = [
            dict(r._mapping)
            for r in s.execute(sa.select(SwAmbientContext.__table__)).fetchall()
        ]
    return {r["activity_id"]: r for r in rows}


# ---------------------------------------------------------------------------
# Batch assembler
# ---------------------------------------------------------------------------

def run_feature_assembly(db_path: str) -> int:
    """Run all matchers and assemble feature_vectors for every CME.

    Returns the number of rows written.
    """
    engine = make_engine(db_path)

    # --- Load shared lookup tables ---
    logger.info("Loading lookup tables...")
    f107_index = _build_f107_index(engine)
    ssn_index = _build_ssn_index(engine)
    ambient_index = _build_ambient_index(engine)

    with Session(engine) as s:
        cmes = [dict(r._mapping) for r in s.execute(
            sa.select(CmeEvent.__table__)
        ).fetchall()]
        analyses = [dict(r._mapping) for r in s.execute(
            sa.select(CmeAnalysis.__table__)
        ).fetchall()]

    logger.info(
        "Assembling feature vectors for %d CMEs (f107=%d, ssn=%d, ambient=%d)",
        len(cmes), len(f107_index), len(ssn_index), len(ambient_index),
    )

    # --- Run matchers ---
    flare_matches = run_cme_flare_matching(db_path)
    ambient_speeds = {aid: a["sw_speed_ambient"] for aid, a in ambient_index.items()
                      if a.get("sw_speed_ambient") is not None}
    icme_matches = run_cme_icme_matching(db_path, ambient_speeds=ambient_speeds)
    sharp_matches = run_cme_sharp_matching(db_path)
    storm_matches = run_storm_matching(db_path, icme_matches)

    # --- Assemble rows ---
    rows: list[dict[str, Any]] = []
    for cme in cmes:
        aid = cme["activity_id"]
        row = assemble_feature_vector(
            cme=cme,
            analyses=analyses,
            flare_match=flare_matches.get(aid, {}),
            icme_match=icme_matches.get(aid, {}),
            sharp_match=sharp_matches.get(aid, {}),
            storm_match=storm_matches.get(aid, {}),
            ambient=ambient_index.get(aid),
            f107_index=f107_index,
            ssn_index=ssn_index,
        )
        rows.append(row)

    # --- Upsert ---
    logger.info("Upserting %d feature vectors...", len(rows))
    upsert(engine, FeatureVector, rows)

    logger.info("Feature assembly complete: %d rows written", len(rows))
    return len(rows)
