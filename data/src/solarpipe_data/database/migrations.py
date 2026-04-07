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
# Add migrate_v2, migrate_v3, etc. here when columns/tables change.

# Example (commented out — do not activate until Phase 2):
# @_register(2, "Add quality_score column to cme_events")
# def migrate_v2(engine: sa.Engine) -> None:
#     with engine.connect() as conn:
#         conn.execute(sa.text(
#             "ALTER TABLE cme_events ADD COLUMN quality_score REAL"
#         ))
#         conn.commit()


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
