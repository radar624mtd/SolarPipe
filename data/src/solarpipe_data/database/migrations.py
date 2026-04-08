"""Schema migration helpers for staging.db.

Migration functions follow the signature: migrate_vN(engine) -> None.
They are registered in MIGRATIONS dict keyed by target version integer.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import sqlalchemy as sa
from sqlalchemy.orm import Session

from .schema import SchemaVersion, make_engine


# Registry: version → migration function
MIGRATIONS: dict[int, tuple[str, Callable]] = {}


def _register(version: int, description: str):
    def decorator(fn: Callable) -> Callable:
        MIGRATIONS[version] = (description, fn)
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Migration functions — add new ones here as schema evolves
# ---------------------------------------------------------------------------

# Version 1 is the initial schema created by init_db(); no migration needed.

@_register(5, "Add feature_vectors table (Phase 5 cross-matching output)")
def migrate_v5(engine: sa.Engine) -> None:
    with engine.connect() as conn:
        conn.execute(sa.text("""
            CREATE TABLE IF NOT EXISTS feature_vectors (
                activity_id TEXT PRIMARY KEY,
                launch_time TEXT,
                cme_speed_kms REAL,
                cme_half_angle_deg REAL,
                cme_latitude REAL,
                cme_longitude REAL,
                cme_mass_grams REAL,
                cme_angular_width_deg REAL,
                linked_flare_id TEXT,
                flare_class_letter TEXT,
                flare_class_numeric REAL,
                flare_peak_time TEXT,
                flare_active_region INTEGER,
                flare_match_method TEXT,
                sharp_harpnum INTEGER,
                sharp_noaa_ar INTEGER,
                sharp_snapshot_context TEXT,
                usflux REAL, meangam REAL, meangbt REAL, meangbz REAL,
                meangbh REAL, meanjzd REAL, totusjz REAL, meanalp REAL,
                meanjzh REAL, totusjh REAL, absnjzh REAL, savncpp REAL,
                meanpot REAL, totpot REAL, meanshr REAL, shrgt45 REAL,
                r_value REAL, area_acr REAL,
                sharp_match_method TEXT,
                sw_speed_ambient REAL, sw_density_ambient REAL,
                sw_bt_ambient REAL, sw_bz_ambient REAL,
                linked_ips_id TEXT,
                icme_arrival_time TEXT,
                transit_time_hours REAL,
                icme_match_method TEXT,
                icme_match_confidence REAL,
                dst_min_nt REAL, dst_min_time TEXT,
                kp_max REAL, storm_threshold_met INTEGER,
                dimming_area REAL, dimming_asymmetry REAL,
                hcs_tilt_angle REAL, hcs_distance REAL,
                f10_7 REAL, sunspot_number REAL,
                quality_flag INTEGER NOT NULL DEFAULT 3,
                source_catalog TEXT NOT NULL DEFAULT 'crossmatch',
                fetch_timestamp TEXT
            )
        """))
        conn.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS ix_feature_vectors_quality "
            "ON feature_vectors (quality_flag)"
        ))
        conn.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS ix_feature_vectors_launch_time "
            "ON feature_vectors (launch_time)"
        ))
        conn.commit()


@_register(4, "Add activity_id to sharp_keywords for resume/dedup support")
def migrate_v4(engine: sa.Engine) -> None:
    with engine.connect() as conn:
        # SQLite allows adding a nullable column; safe on populated or empty table
        try:
            conn.execute(sa.text(
                "ALTER TABLE sharp_keywords ADD COLUMN activity_id TEXT"
            ))
        except Exception:
            pass  # Column already exists — idempotent
        conn.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS ix_sharp_keywords_activity_id "
            "ON sharp_keywords (activity_id, query_context)"
        ))
        conn.commit()


@_register(3, "Add harp_noaa_map table (Phase 4 HARP↔NOAA AR mapping)")
def migrate_v3(engine: sa.Engine) -> None:
    with engine.connect() as conn:
        conn.execute(sa.text("""
            CREATE TABLE IF NOT EXISTS harp_noaa_map (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                harpnum INTEGER NOT NULL,
                noaa_ar INTEGER,
                noaa_ars TEXT,
                t_rec TEXT,
                source_catalog TEXT NOT NULL DEFAULT 'JSOC',
                fetch_timestamp TEXT,
                data_version TEXT
            )
        """))
        conn.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS ix_harp_noaa_map_harpnum ON harp_noaa_map (harpnum)"
        ))
        conn.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS ix_harp_noaa_map_noaa_ar ON harp_noaa_map (noaa_ar)"
        ))
        conn.commit()


@_register(2, "Add sw_ambient_context table (Phase 3 ambient solar wind context)")
def migrate_v2(engine: sa.Engine) -> None:
    from .schema import Base
    with engine.connect() as conn:
        # Create table if not already present (idempotent)
        conn.execute(sa.text("""
            CREATE TABLE IF NOT EXISTS sw_ambient_context (
                activity_id TEXT PRIMARY KEY,
                window_start TEXT,
                window_end TEXT,
                n_hours INTEGER,
                sw_speed_ambient REAL,
                sw_density_ambient REAL,
                sw_bt_ambient REAL,
                sw_bz_ambient REAL,
                source_catalog TEXT NOT NULL DEFAULT 'OMNI',
                fetch_timestamp TEXT,
                data_version TEXT
            )
        """))
        conn.commit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def current_version(engine: sa.Engine) -> int:
    """Return the highest applied schema version, or 0 if table is empty."""
    with Session(engine) as s:
        row = s.execute(
            sa.select(sa.func.max(SchemaVersion.version))
        ).scalar_one_or_none()
        return row if row is not None else 0


def apply_pending(engine: sa.Engine) -> list[int]:
    """Run all migrations with version > current_version().

    Returns list of version numbers applied (empty if already up-to-date).
    """
    current = current_version(engine)
    applied: list[int] = []

    for version in sorted(MIGRATIONS):
        if version <= current:
            continue
        description, fn = MIGRATIONS[version]
        fn(engine)
        with Session(engine) as s, s.begin():
            s.add(SchemaVersion(
                version=version,
                applied_at=datetime.now(timezone.utc).isoformat(),
                description=description,
            ))
        applied.append(version)

    return applied


def migrate(db_path: str) -> list[int]:
    """Convenience: open engine by path, apply pending migrations, return applied versions."""
    engine = make_engine(db_path)
    return apply_pending(engine)
