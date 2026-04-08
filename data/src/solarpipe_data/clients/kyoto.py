"""Kyoto WDC-C2 Dst index client.

Rules enforced:
- RULE-002: All HTTP via BaseClient
- RULE-037: Check len(body) > 100 before parsing (maintenance returns 200+empty)
- RULE-080: Try final → provisional → realtime for each month
- RULE-081: Post-2019-04-26: HTML table; pre-2019: WDC fixed-width (not supported here)
- RULE-082: Sentinel >500 or <-500 nT → None
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

from .base import BaseClient

logger = logging.getLogger(__name__)

# WDC uses a different URL structure per data_type
_BASE = "https://wdc.kugi.kyoto-u.ac.jp/dst_realtime"
_TYPES: list[str] = ["final", "provisional", "realtime"]

# Post-2019 HTML URL pattern; pre-2019 uses binary WDC format (RULE-081)
_HTML_CUTOFF = date(2019, 4, 26)

_URL_TEMPLATES: dict[str, str] = {
    "final":       "https://wdc.kugi.kyoto-u.ac.jp/dst_final/{year}{month:02d}/index.html",
    "provisional": "https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/{year}{month:02d}/index.html",
    "realtime":    "https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/{year}{month:02d}/index.html",
}


class KyotoClient(BaseClient):
    """Client for Kyoto WDC-C2 Dst hourly index."""

    source_name = "kyoto"

    async def fetch_month(
        self, year: int, month: int
    ) -> tuple[list[dict[str, Any]], str]:
        """Fetch Dst for one month using cascade: final > provisional > realtime.

        Returns (records, data_type_used). Records may be empty if body too short
        (maintenance window) or pre-2019 (WDC binary format not supported).

        RULE-081: only HTML (post-2019) dates are supported here.
        RULE-037: empty body guard.
        """
        target = date(year, month, 1)
        if target < _HTML_CUTOFF:
            # RULE-081: pre-2019 WDC binary format — not supported; caller
            # should rely on ported data from solar_data.db for that range.
            logger.debug(
                "kyoto: %d-%02d is pre-2019 WDC format — skip (ported data covers this)",
                year, month,
            )
            return [], "unsupported"

        for data_type in _TYPES:
            url = _URL_TEMPLATES[data_type].format(year=year, month=month)
            cache_key = f"kyoto_{data_type}_{year}{month:02d}"
            try:
                html = await self.get_html(url, cache_key=cache_key)
            except Exception as exc:
                logger.debug("kyoto %s %d-%02d fetch failed: %s", data_type, year, month, exc)
                continue

            # RULE-037: empty body guard
            if len(html) < 100:
                logger.debug(
                    "kyoto %s %d-%02d: body too short (%d bytes), skipping",
                    data_type, year, month, len(html),
                )
                continue

            records = _parse_dst_html(html, year, month, data_type)
            if records:
                logger.info(
                    "kyoto: %d-%02d fetched %d records (%s)",
                    year, month, len(records), data_type,
                )
                return records, data_type

        logger.warning("kyoto: no usable Dst data for %d-%02d", year, month)
        return [], "none"

    async def fetch_range(
        self, start: date, end: date
    ) -> list[dict[str, Any]]:
        """Fetch Dst across a date range, month by month."""
        all_records: list[dict[str, Any]] = []
        year, month = start.year, start.month

        while date(year, month, 1) <= date(end.year, end.month, 1):
            records, _ = await self.fetch_month(year, month)
            all_records.extend(records)
            month += 1
            if month > 12:
                month = 1
                year += 1

        return all_records


# ---------------------------------------------------------------------------
# HTML parser — post-2019 Kyoto format
# ---------------------------------------------------------------------------

def _parse_dst_html(
    html: str, year: int, month: int, data_type: str
) -> list[dict[str, Any]]:
    """Parse Kyoto Dst HTML table: day rows × 24 hourly columns.

    Returns list of dicts: {datetime, dst_nt, data_type, ...}
    RULE-082: values > 500 or < -500 → None (sentinel).
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    fetch_ts = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []

    # Find the main Dst data table — Kyoto has a specific <pre> or <table> layout
    # Post-2019 uses a <pre>-based layout with fixed columns
    pre_tags = soup.find_all("pre")
    for pre in pre_tags:
        text = pre.get_text()
        if "DST" in text.upper() or "Dst" in text:
            parsed = _parse_dst_pre_block(text, year, month, data_type, fetch_ts)
            if parsed:
                return parsed

    # Fallback: try HTML table parse
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        parsed = _parse_dst_table(rows, year, month, data_type, fetch_ts)
        if parsed:
            return parsed

    return records


def _parse_dst_pre_block(
    text: str, year: int, month: int, data_type: str, fetch_ts: str
) -> list[dict[str, Any]]:
    """Parse Kyoto Dst <pre> fixed-width block.

    Format (post-2019):
      Line starts with day number, then 24 hourly values in columns.
    """
    import re
    records: list[dict[str, Any]] = []

    # Kyoto pre-block: lines like "  1   -12   -8  ..." or "DST  1  ..."
    # Each data line: day (1-31), then 24 values
    line_re = re.compile(r"^\s*(\d{1,2})\s+([\s\-\d]+)$")

    for line in text.splitlines():
        m = line_re.match(line)
        if not m:
            continue
        day = int(m.group(1))
        parts = m.group(2).split()
        if len(parts) < 24:
            continue

        for hour in range(24):
            val_str = parts[hour]
            try:
                val = int(val_str)
            except ValueError:
                continue

            # RULE-082: sentinel
            dst_nt: float | None = float(val)
            if dst_nt > 500 or dst_nt < -500:
                dst_nt = None

            try:
                dt = datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)
            except ValueError:
                continue

            records.append({
                "datetime": dt.strftime("%Y-%m-%d %H:%M"),
                "dst_nt": dst_nt,
                "data_type": data_type,
                "source_catalog": "Kyoto",
                "fetch_timestamp": fetch_ts,
                "data_version": "1.0",
            })

    return records


def _parse_dst_table(
    rows: Any, year: int, month: int, data_type: str, fetch_ts: str
) -> list[dict[str, Any]]:
    """Parse an HTML <table> with day rows × 24 hourly columns."""
    import re
    records: list[dict[str, Any]] = []

    for row in rows:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        # First cell should be a day number
        day_text = cells[0].get_text(strip=True)
        try:
            day = int(day_text)
        except ValueError:
            continue
        if day < 1 or day > 31:
            continue
        if len(cells) < 25:
            continue

        for hour in range(24):
            cell_text = cells[hour + 1].get_text(strip=True)
            # Strip non-numeric suffixes
            cell_text = re.sub(r"[^0-9\-]", "", cell_text)
            if not cell_text:
                continue
            try:
                val = int(cell_text)
            except ValueError:
                continue

            dst_nt: float | None = float(val)
            if dst_nt > 500 or dst_nt < -500:
                dst_nt = None

            try:
                dt = datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)
            except ValueError:
                continue

            records.append({
                "datetime": dt.strftime("%Y-%m-%d %H:%M"),
                "dst_nt": dst_nt,
                "data_type": data_type,
                "source_catalog": "Kyoto",
                "fetch_timestamp": fetch_ts,
                "data_version": "1.0",
            })

    return records
