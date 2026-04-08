"""Common query patterns for ingestion and crossmatch modules.

Rules enforced:
- RULE-030: sqlite dialect insert for upserts
- RULE-031: callers pass posix paths; engine created by make_engine()
- RULE-033: Session context manager always
- RULE-034: explicit set_ keys for nullable column upserts
"""
from __future__ import annotations

from typing import Any, Type

import sqlalchemy as sa
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import DeclarativeBase, Session

from .schema import make_engine


def upsert(
    engine: sa.Engine,
    table_class: Type[DeclarativeBase],
    rows: list[dict[str, Any]],
    batch_size: int = 500,
) -> int:
    """Batch upsert rows into table_class, keyed on the table's primary key(s).

    Uses sqlite INSERT OR REPLACE semantics via sqlalchemy.dialects.sqlite.insert.
    Explicit set_ per RULE-034 — nullable columns are included explicitly.
    Returns total rows processed.
    """
    if not rows:
        return 0

    table = table_class.__table__
    pk_cols = {c.name for c in table.primary_key.columns}
    all_cols = [c.name for c in table.columns]
    update_cols = [c for c in all_cols if c not in pk_cols]

    total = 0
    with Session(engine) as s, s.begin():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            stmt = sqlite_insert(table).values(chunk)
            if update_cols:
                stmt = stmt.on_conflict_do_update(
                    index_elements=list(pk_cols),
                    set_={col: stmt.excluded[col] for col in update_cols},
                )
            s.execute(stmt)
            total += len(chunk)

    return total


def temporal_range(
    engine: sa.Engine,
    table_class: Type[DeclarativeBase],
    start: str,
    end: str,
    datetime_col: str = "datetime",
) -> list[Any]:
    """Return ORM objects where datetime_col is between start and end (inclusive).

    start / end are ISO 8601 strings (compared lexicographically — valid for ISO dates).
    """
    col = getattr(table_class, datetime_col)
    with Session(engine) as s:
        return s.execute(
            sa.select(table_class).where(col >= start, col <= end)
        ).scalars().all()


def max_timestamp(
    engine_or_table: "sa.Engine | str",
    table_class_or_col: "Type[DeclarativeBase] | str",
    col_or_engine: "str | sa.Engine" = "datetime",
) -> str | None:
    """Return the maximum value of col as a string, or None if the table is empty.

    Accepts two calling conventions:
      max_timestamp(engine, TableClass, col="datetime")   # ORM form
      max_timestamp("table_name", "col", engine)          # raw SQL form (used by ingest scripts)
    """
    # Detect raw SQL form: first arg is a string (table name)
    if isinstance(engine_or_table, str):
        table_name = engine_or_table
        col = table_class_or_col if isinstance(table_class_or_col, str) else "datetime"
        engine = col_or_engine  # type: ignore[assignment]
        with Session(engine) as s:  # type: ignore[arg-type]
            row = s.execute(
                sa.text(f"SELECT MAX({col}) FROM {table_name}")
            ).fetchone()
            return row[0] if row else None
    # ORM form: first arg is engine
    engine = engine_or_table
    table_class = table_class_or_col
    col = col_or_engine if isinstance(col_or_engine, str) else "datetime"
    column = getattr(table_class, col)
    with Session(engine) as s:  # type: ignore[arg-type]
        return s.execute(sa.select(sa.func.max(column))).scalar_one_or_none()


def row_count(engine: sa.Engine, table_class: Type[DeclarativeBase]) -> int:
    """Return total row count for a table."""
    with Session(engine) as s:
        return s.execute(
            sa.select(sa.func.count()).select_from(table_class)
        ).scalar_one()
