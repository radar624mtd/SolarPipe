"""Ingest CDAW LASCO UNIVERSAL_ver2 CME catalog into cdaw_cme_events.

Rules enforced:
- RULE-003: Sentinel conversion at ingest ("----", "---", empty string → None)
- RULE-004: Upsert idempotent on cdaw_id
- RULE-030: from sqlalchemy.dialects.sqlite import insert
- RULE-033: Session context manager
- RULE-034: Explicit set_ for nullable columns
- RULE-037: Skip pages with len < 100
- RULE-050: BeautifulSoup only
- RULE-051: UNIVERSAL_ver2 URL (enforced in CdawClient)
- RULE-052: Strip footnote markers before numeric casting
- RULE-053: speed_20rs_kms is canonical arrival-model speed

Column map (HTML table columns, left to right):
  0  Date           e.g. "1996/01/11"
  1  Time (UT)      e.g. "04:58:44"
  2  Central PA     degrees or "Halo"
  3  Width          degrees
  4  Linear Speed   km/s
  5  2nd-order Speed (init)
  6  2nd-order Speed (final)
  7  Speed at 20 Rs  ← speed_20rs_kms (RULE-053)
  8  Accel          km/s²
  9  Mass           grams
  10 Kinetic Energy ergs
  11 MPA            degrees
  12 Remarks
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from bs4 import BeautifulSoup
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from solarpipe_data.database.schema import CdawCmeEvent

logger = logging.getLogger(__name__)

_SENTINEL_STRINGS = frozenset({"----", "---", "--", "", "N/A", "nan"})

# Remarks text → quality_flag mapping
_QUALITY_MAP = {
    "very poor event": 1,
    "poor event": 2,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest_cdaw_month(
    engine: Engine,
    year: int,
    month: int,
    html: str,
) -> int:
    """Parse one CDAW monthly HTML page and upsert all rows.

    Returns number of rows upserted.
    """
    if len(html) < 100:
        logger.warning("cdaw %04d-%02d: page too short (%d bytes) — skipping", year, month, len(html))
        return 0

    rows = _parse_html(html, year, month)
    if not rows:
        return 0

    _upsert_rows(engine, rows)
    logger.info("cdaw %04d-%02d: upserted %d rows", year, month, len(rows))
    return len(rows)


# ---------------------------------------------------------------------------
# HTML parser
# ---------------------------------------------------------------------------

def _parse_html(html: str, year: int, month: int) -> list[dict[str, Any]]:
    """Parse CDAW monthly HTML table. Returns list of row dicts."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if table is None:
        logger.warning("cdaw %04d-%02d: no <table> found", year, month)
        return []

    rows: list[dict[str, Any]] = []
    fetch_ts = datetime.now(tz=timezone.utc).isoformat()

    for tr in table.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < 12:
            continue  # header or malformed row

        row = _parse_row(cells, fetch_ts)
        if row is not None:
            rows.append(row)

    return rows


def _parse_row(cells: list[str], fetch_ts: str) -> dict[str, Any] | None:
    """Parse one table row. Returns None if the row has no valid date."""
    date_str = cells[0].strip()
    time_str = cells[1].strip() if len(cells) > 1 else ""

    if not date_str or date_str in _SENTINEL_STRINGS:
        return None

    # Normalise date: "1996/01/11" → "1996-01-11"
    date_norm = date_str.replace("/", "-")

    # Build datetime string and cdaw_id
    try:
        datetime.strptime(date_norm, "%Y-%m-%d")
    except ValueError:
        return None

    time_norm = time_str if time_str and time_str not in _SENTINEL_STRINGS else "00:00:00"
    dt_iso = f"{date_norm}T{time_norm}Z"
    cdaw_id = f"{date_norm.replace('-', '')}.{time_norm.replace(':', '')}"

    # Parse central PA: "Halo" → None, angular_width → 360
    pa_raw = cells[2].strip() if len(cells) > 2 else ""
    is_halo = pa_raw.lower() == "halo"
    central_pa = None if is_halo else _to_float(pa_raw)
    angular_width_raw = cells[3].strip() if len(cells) > 3 else ""
    angular_width = 360.0 if is_halo else _to_float(angular_width_raw)

    # Remarks and quality flag
    remarks_raw = cells[12].strip() if len(cells) > 12 else ""
    quality_flag = _quality_from_remarks(remarks_raw)
    remarks = remarks_raw if remarks_raw and remarks_raw not in _SENTINEL_STRINGS else None

    return {
        "cdaw_id": cdaw_id,
        "date": date_norm,
        "time_ut": time_norm,
        "datetime": dt_iso,
        "central_pa_deg": central_pa,
        "angular_width_deg": angular_width,
        "linear_speed_kms": _to_float(cells[4]) if len(cells) > 4 else None,
        "second_order_speed_init": _to_float(cells[5]) if len(cells) > 5 else None,
        "second_order_speed_final": _to_float(cells[6]) if len(cells) > 6 else None,
        "speed_20rs_kms": _to_float(cells[7]) if len(cells) > 7 else None,
        "accel_kms2": _to_float(cells[8]) if len(cells) > 8 else None,
        "mass_grams": _to_float(cells[9]) if len(cells) > 9 else None,
        "kinetic_energy_ergs": _to_float(cells[10]) if len(cells) > 10 else None,
        "mpa_deg": _to_float(cells[11]) if len(cells) > 11 else None,
        "remarks": remarks,
        "quality_flag": quality_flag,
        "source_catalog": "CDAW",
        "fetch_timestamp": fetch_ts,
        "data_version": "UNIVERSAL_ver2",
    }


# ---------------------------------------------------------------------------
# DB upsert
# ---------------------------------------------------------------------------

def _upsert_rows(engine: Engine, rows: list[dict[str, Any]]) -> None:
    """Upsert a batch of rows into cdaw_cme_events (RULE-030, RULE-033, RULE-034)."""
    if not rows:
        return

    stmt = insert(CdawCmeEvent).values(rows)
    # RULE-034: explicit set_ for every nullable column
    stmt = stmt.on_conflict_do_update(
        index_elements=["cdaw_id"],
        set_={
            "date": stmt.excluded.date,
            "time_ut": stmt.excluded.time_ut,
            "datetime": stmt.excluded.datetime,
            "central_pa_deg": stmt.excluded.central_pa_deg,
            "angular_width_deg": stmt.excluded.angular_width_deg,
            "linear_speed_kms": stmt.excluded.linear_speed_kms,
            "second_order_speed_init": stmt.excluded.second_order_speed_init,
            "second_order_speed_final": stmt.excluded.second_order_speed_final,
            "speed_20rs_kms": stmt.excluded.speed_20rs_kms,
            "accel_kms2": stmt.excluded.accel_kms2,
            "mass_grams": stmt.excluded.mass_grams,
            "kinetic_energy_ergs": stmt.excluded.kinetic_energy_ergs,
            "mpa_deg": stmt.excluded.mpa_deg,
            "remarks": stmt.excluded.remarks,
            "quality_flag": stmt.excluded.quality_flag,
            "source_catalog": stmt.excluded.source_catalog,
            "fetch_timestamp": stmt.excluded.fetch_timestamp,
            "data_version": stmt.excluded.data_version,
        },
    )
    with Session(engine) as s, s.begin():
        s.execute(stmt)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_footnote(cell: str) -> str:
    """Strip footnote markers from numeric cells (RULE-052).

    e.g. "-54.7*1" → "-54.7", "360*2" → "360"
    Pattern: remove everything from first non-numeric/sign/dot character.
    """
    return re.sub(r"[^0-9.\-].*$", "", cell.strip())


def _to_float(cell: str) -> float | None:
    """Convert a cell string to float, returning None for sentinels."""
    s = _strip_footnote(cell)
    if not s or s in _SENTINEL_STRINGS:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _quality_from_remarks(remarks: str) -> int:
    """Derive quality flag from remarks text.

    Returns:
        1 — "Very Poor Event"
        2 — "Poor Event"
        3 — default (no quality flag in remarks)
    """
    lower = remarks.lower()
    for phrase, flag in _QUALITY_MAP.items():
        if phrase in lower:
            return flag
    return 3
