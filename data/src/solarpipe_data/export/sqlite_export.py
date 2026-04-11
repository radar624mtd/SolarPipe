"""Task 6.6 — CME Catalog SQLite Export.

Assembles `data/output/cme_catalog.db` from `feature_vectors` in staging.db.

Output tables (column names match configs/flux_rope_propagation_v1.yaml exactly):
  cme_events     — CME identity, kinematics, source region, environment, quality
  flux_rope_fits — Flux-rope fit parameters + Bz proxy (PRIMARY prediction target)
  l1_arrivals    — L1 arrival timing and geomagnetic response

Mapping from staging.db:
  feature_vectors → all three output tables
  cme_events (staging) → launch_time, source_location, noaa_ar
  interplanetary_shocks → shock_arrival_time
  dst_hourly → dst_min_nT post-arrival

Bz proxy: observed_rotation_angle is null-filled until real flux-rope fitting
is implemented. observed_bz_min is populated from dst_hourly storm window.
has_in_situ_fit = 1 only when icme_arrival_time is not null AND dst_min_nT is not null.

Rules:
  RULE-030: from sqlalchemy.dialects.sqlite import insert
  RULE-031: Path(p).as_posix() in all connection strings
  RULE-032: WAL mode via event listener
  RULE-008: schema version tracked in schema_version table
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy import Column, Float, Integer, String, Text, event
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import DeclarativeBase, Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output DB schema — separate Base from staging DB
# ---------------------------------------------------------------------------


class OutputBase(DeclarativeBase):
    pass


class OutputCmeEvent(OutputBase):
    __tablename__ = "cme_events"

    event_id: str = Column(String, primary_key=True)
    launch_time: str = Column(String, nullable=True)
    source_location: str = Column(String, nullable=True)
    noaa_ar: int = Column(Integer, nullable=True)
    # Kinematic
    cme_speed: float = Column(Float, nullable=True)
    cme_mass: float = Column(Float, nullable=True)
    cme_angular_width: float = Column(Float, nullable=True)
    flare_class_numeric: float = Column(Float, nullable=True)
    # Source region
    chirality: str = Column(String, nullable=True)
    initial_axis_angle: float = Column(Float, nullable=True)
    usflux: float = Column(Float, nullable=True)
    totpot: float = Column(Float, nullable=True)
    r_value: float = Column(Float, nullable=True)
    meanshr: float = Column(Float, nullable=True)
    totusjz: float = Column(Float, nullable=True)
    # Environmental
    coronal_hole_proximity: float = Column(Float, nullable=True)
    coronal_hole_polarity: int = Column(Integer, nullable=True)
    hcs_tilt_angle: float = Column(Float, nullable=True)
    hcs_distance: float = Column(Float, nullable=True)
    # Ambient L1
    sw_speed_ambient: float = Column(Float, nullable=True)
    sw_density_ambient: float = Column(Float, nullable=True)
    sw_bt_ambient: float = Column(Float, nullable=True)
    f10_7: float = Column(Float, nullable=True)
    # Quality
    quality_flag: int = Column(Integer, nullable=False, default=3)


class OutputFluxRopeFit(OutputBase):
    __tablename__ = "flux_rope_fits"

    event_id: str = Column(String, primary_key=True)
    observed_rotation_angle: float = Column(Float, nullable=True)   # PRIMARY TARGET — null for now
    observed_bz_min: float = Column(Float, nullable=True)
    bz_polarity: int = Column(Integer, nullable=True)
    fit_method: str = Column(String, nullable=False, default="none")
    fit_quality: int = Column(Integer, nullable=True)
    has_in_situ_fit: int = Column(Integer, nullable=False, default=0)


class OutputL1Arrival(OutputBase):
    __tablename__ = "l1_arrivals"

    event_id: str = Column(String, primary_key=True)
    shock_arrival_time: str = Column(String, nullable=True)
    icme_start_time: str = Column(String, nullable=True)
    icme_end_time: str = Column(String, nullable=True)
    transit_time_hours: float = Column(Float, nullable=True)
    dst_min_nT: float = Column(Float, nullable=True)
    kp_max: float = Column(Float, nullable=True)
    has_in_situ_fit: int = Column(Integer, nullable=False, default=0)
    # Match provenance — carried through from staging so downstream consumers
    # can distinguish DONKI-linked progenitors from transit-window guesses
    # and drop ghost rows in merged-CME clusters.
    icme_match_method: str = Column(String, nullable=True)
    icme_match_confidence: float = Column(Float, nullable=True)


class OutputSchemaVersion(OutputBase):
    __tablename__ = "schema_version"

    version: int = Column(Integer, primary_key=True)
    applied_at: str = Column(String, nullable=False)
    description: str = Column(String, nullable=True)


_OUTPUT_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Engine factory for output DB (WAL mode, RULE-031/032)
# ---------------------------------------------------------------------------


def _make_output_engine(db_path: str):
    posix = Path(db_path).as_posix()
    url = f"sqlite:///{posix}"
    engine = sa.create_engine(url, echo=False)

    @event.listens_for(engine, "connect")
    def _set_wal(dbapi_conn, _conn_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    OutputBase.metadata.create_all(engine)
    return engine


# ---------------------------------------------------------------------------
# Row builders — map feature_vector row → output rows
# ---------------------------------------------------------------------------


def _build_cme_event(fv: dict) -> dict:
    return {
        "event_id":               fv["activity_id"],
        "launch_time":            fv.get("launch_time"),
        "source_location":        fv.get("source_location"),
        "noaa_ar":                fv.get("noaa_ar"),
        "cme_speed":              fv.get("cme_speed_kms"),
        "cme_mass":               fv.get("cme_mass_grams"),
        "cme_angular_width":      fv.get("cme_angular_width_deg"),
        "flare_class_numeric":    fv.get("flare_class_numeric"),
        "chirality":              None,                      # deferred
        "initial_axis_angle":     fv.get("meanshr"),         # proxy
        "usflux":                 fv.get("usflux"),
        "totpot":                 fv.get("totpot"),
        "r_value":                fv.get("r_value"),
        "meanshr":                fv.get("meanshr"),
        "totusjz":                fv.get("totusjz"),
        "coronal_hole_proximity": None,                      # deferred
        "coronal_hole_polarity":  None,                      # deferred
        "hcs_tilt_angle":         fv.get("hcs_tilt_angle"),
        "hcs_distance":           fv.get("hcs_distance"),
        "sw_speed_ambient":       fv.get("sw_speed_ambient"),
        "sw_density_ambient":     fv.get("sw_density_ambient"),
        "sw_bt_ambient":          fv.get("sw_bt_ambient"),
        "f10_7":                  fv.get("f10_7"),
        "quality_flag":           int(fv.get("quality_flag") or 3),
    }


def _build_flux_rope_fit(fv: dict) -> dict:
    dst = fv.get("dst_min_nt")
    bz_pol = None
    if dst is not None:
        bz_pol = -1 if dst < -30.0 else 1

    has_fit = 1 if (fv.get("icme_arrival_time") and dst is not None) else 0
    fit_q = int(fv.get("quality_flag") or 3) if has_fit else None

    return {
        "event_id":               fv["activity_id"],
        "observed_rotation_angle": None,     # deferred — requires flux-rope fitting
        "observed_bz_min":        dst,
        "bz_polarity":            bz_pol,
        "fit_method":             "proxy" if has_fit else "none",
        "fit_quality":            fit_q,
        "has_in_situ_fit":        has_fit,
    }


def _build_l1_arrival(fv: dict) -> dict:
    has_fit = 1 if (fv.get("icme_arrival_time") and fv.get("dst_min_nt") is not None) else 0
    return {
        "event_id":              fv["activity_id"],
        "shock_arrival_time":    fv.get("icme_arrival_time"),    # IPS event_time as shock proxy
        "icme_start_time":       fv.get("icme_arrival_time"),
        "icme_end_time":         None,                            # not tracked
        "transit_time_hours":    fv.get("transit_time_hours"),
        "dst_min_nT":            fv.get("dst_min_nt"),
        "kp_max":                fv.get("kp_max"),
        "has_in_situ_fit":       has_fit,
        "icme_match_method":     fv.get("icme_match_method"),
        "icme_match_confidence": fv.get("icme_match_confidence"),
    }


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------


def build_catalog(
    staging_db_path: str,
    output_db_path: str,
    min_quality: int = 1,
) -> int:
    """Assemble cme_catalog.db from staging feature_vectors.

    Args:
        staging_db_path: Path to staging.db.
        output_db_path: Destination path for cme_catalog.db.
        min_quality: Minimum quality_flag to export (default 1 = all events).

    Returns:
        Number of events exported.
    """
    from ..database.schema import make_engine as make_staging_engine

    # Create output directory if needed
    Path(output_db_path).parent.mkdir(parents=True, exist_ok=True)

    staging_engine = make_staging_engine(staging_db_path)
    output_engine = _make_output_engine(output_db_path)

    try:
        # Load feature vectors from staging
        q = sa.text("""
            SELECT
                fv.*,
                ce.source_location,
                ce.active_region_num AS noaa_ar
            FROM feature_vectors fv
            LEFT JOIN cme_events ce ON ce.activity_id = fv.activity_id
            WHERE fv.quality_flag >= :min_q
            ORDER BY fv.launch_time
        """)

        with staging_engine.connect() as conn:
            rows = conn.execute(q, {"min_q": min_quality}).mappings().fetchall()

        if not rows:
            logger.warning("build_catalog: no feature_vectors with quality≥%d", min_quality)
            return 0

        now_iso = datetime.now(timezone.utc).isoformat()
        n_written = 0

        with Session(output_engine) as session:
            # Write schema version
            session.execute(
                insert(OutputSchemaVersion).values(
                    version=_OUTPUT_SCHEMA_VERSION,
                    applied_at=now_iso,
                    description="Initial export from feature_vectors",
                ).on_conflict_do_nothing()
            )

            for row in rows:
                fv = dict(row)

                cme_row = _build_cme_event(fv)
                fr_row = _build_flux_rope_fit(fv)
                l1_row = _build_l1_arrival(fv)

                session.execute(
                    insert(OutputCmeEvent).values(**cme_row)
                    .on_conflict_do_update(
                        index_elements=["event_id"],
                        set_={k: v for k, v in cme_row.items() if k != "event_id"},
                    )
                )
                session.execute(
                    insert(OutputFluxRopeFit).values(**fr_row)
                    .on_conflict_do_update(
                        index_elements=["event_id"],
                        set_={k: v for k, v in fr_row.items() if k != "event_id"},
                    )
                )
                session.execute(
                    insert(OutputL1Arrival).values(**l1_row)
                    .on_conflict_do_update(
                        index_elements=["event_id"],
                        set_={k: v for k, v in l1_row.items() if k != "event_id"},
                    )
                )
                n_written += 1

            session.commit()

        # Create training_features view for the .NET pipeline — JOINs the 3 output
        # tables and aliases columns to match configs/flux_rope_propagation_v1.yaml
        with output_engine.connect() as conn:
            conn.execute(sa.text("DROP VIEW IF EXISTS training_features"))
            conn.execute(sa.text("""
                CREATE VIEW training_features AS
                SELECT
                    e.event_id,
                    e.launch_time,
                    e.cme_speed          AS cme_speed_kms,
                    e.sw_speed_ambient   AS sw_speed_ambient_kms,
                    e.sw_density_ambient AS sw_density_n_cc,
                    e.sw_bt_ambient      AS sw_bt_nt,
                    e.f10_7,
                    e.quality_flag,
                    f.observed_bz_min    AS bz_gsm_proxy_nt,
                    a.transit_time_hours,
                    a.dst_min_nT         AS dst_min_nT,
                    a.kp_max,
                    a.has_in_situ_fit
                FROM cme_events e
                JOIN l1_arrivals a ON e.event_id = a.event_id
                JOIN flux_rope_fits f ON e.event_id = f.event_id
                WHERE a.transit_time_hours IS NOT NULL
                  AND e.cme_speed IS NOT NULL
                  AND e.quality_flag >= 2
                  AND e.launch_time < '2026-01-01'
            """))
            conn.commit()

    finally:
        staging_engine.dispose()
        output_engine.dispose()

    logger.info("build_catalog: exported %d events → %s", n_written, output_db_path)
    return n_written


# ---------------------------------------------------------------------------
# Row count helper for validate_db.py
# ---------------------------------------------------------------------------


def count_catalog_rows(output_db_path: str) -> dict[str, int]:
    """Return row counts for all three output tables."""
    engine = _make_output_engine(output_db_path)
    counts = {}
    try:
        with engine.connect() as conn:
            for table in ("cme_events", "flux_rope_fits", "l1_arrivals"):
                try:
                    n = conn.execute(sa.text(f"SELECT COUNT(*) FROM {table}")).scalar()
                    counts[table] = int(n or 0)
                except Exception:
                    counts[table] = -1
    finally:
        engine.dispose()
    return counts
