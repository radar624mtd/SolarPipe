"""Kyoto WDC-C2 Dst index client.

Rules enforced:
- RULE-002: All HTTP via BaseClient
- RULE-037: Check len(body) > 100 before parsing (maintenance returns 200+empty)
- RULE-080: Try final → provisional → realtime for each month
- RULE-081: WDC fixed-width format (.for.request) used for all months; HTML fallback
  for months where the raw file is unavailable (post-2019 realtime only)
- RULE-082: Sentinel >500 or <-500 nT → None

WDC format (120 chars/line):
  cols  0- 2: "DST"
  cols  3- 4: YY (2-digit year)
  cols  5- 6: MM
  col   7   : '*'
  cols  8- 9: DD
  cols 10-11: spaces or "RR"
  col  12   : 'X'
  col  13   : version code (0=QL,1=prov,2=final)
  cols 14-15: century (19/20)
  cols 16-19: base value (×100 nT, always 0 for Dst)
  cols 20-115: 24×4-char hourly values (nT); 9999 = missing
  cols 116-119: daily mean (nT)
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

from .base import BaseClient

logger = logging.getLogger(__name__)

_TYPES: list[str] = ["final", "provisional", "realtime"]

# URL templates for the WDC fixed-width raw data file
_WDC_URL_TEMPLATES: dict[str, str] = {
    "final":       "https://wdc.kugi.kyoto-u.ac.jp/dst_final/{year}{month:02d}/dst{yy}{month:02d}.for.request",
    "provisional": "https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/{year}{month:02d}/dst{yy}{month:02d}.for.request",
    "realtime":    "https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/{year}{month:02d}/dst{yy}{month:02d}.for.request",
}

# HTML index page URLs (fallback for realtime months with no .for.request)
_HTML_URL_TEMPLATES: dict[str, str] = {
    "final":       "https://wdc.kugi.kyoto-u.ac.jp/dst_final/{year}{month:02d}/index.html",
    "provisional": "https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/{year}{month:02d}/index.html",
    "realtime":    "https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/{year}{month:02d}/index.html",
}


class KyotoClient(BaseClient):
    """Client for Kyoto WDC-C2 Dst hourly index.

    Fetches all months via the WDC fixed-width .for.request file (pre- and
    post-2019). Falls back to HTML parsing when the raw file is unavailable
    (typically the current partial month in realtime).
    """

    source_name = "kyoto"

    async def fetch_month(
        self, year: int, month: int
    ) -> tuple[list[dict[str, Any]], str]:
        """Fetch Dst for one month using cascade: final > provisional > realtime.

        Returns (records, data_type_used). Records may be empty if the month
        is unavailable at all quality levels.

        RULE-080: preference cascade final > provisional > realtime.
        RULE-037: empty body guard.
        RULE-081: WDC format tried first for all months; HTML fallback for
                  months where .for.request is not available.
        """
        yy = year % 100  # 2-digit year for filename

        for data_type in _TYPES:
            # --- Try WDC fixed-width format first ---
            wdc_url = _WDC_URL_TEMPLATES[data_type].format(
                year=year, month=month, yy=yy
            )
            wdc_cache_key = f"kyoto_wdc_{data_type}_{year}{month:02d}"
            try:
                raw = await self.get_html(wdc_url, cache_key=wdc_cache_key)
                if raw and len(raw) > 50:
                    records = _parse_wdc_format(raw, year, month, data_type)
                    if records:
                        logger.info(
                            "kyoto: %d-%02d fetched %d records (%s, wdc)",
                            year, month, len(records), data_type,
                        )
                        return records, data_type
            except Exception:
                pass  # WDC file not available; try HTML fallback

            # --- HTML fallback (index.html with <pre> or <table> block) ---
            html_url = _HTML_URL_TEMPLATES[data_type].format(year=year, month=month)
            html_cache_key = f"kyoto_{data_type}_{year}{month:02d}"
            try:
                html = await self.get_html(html_url, cache_key=html_cache_key)
            except Exception as exc:
                logger.debug("kyoto %s %d-%02d HTML fetch failed: %s", data_type, year, month, exc)
                continue

            if len(html) < 100:
                logger.debug(
                    "kyoto %s %d-%02d: body too short (%d bytes), skipping",
                    data_type, year, month, len(html),
                )
                continue

            records = _parse_dst_html(html, year, month, data_type)
            if records:
                logger.info(
                    "kyoto: %d-%02d fetched %d records (%s, html)",
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
# WDC fixed-width format parser
# ---------------------------------------------------------------------------

def _parse_wdc_format(
    text: str, year: int, month: int, data_type: str
) -> list[dict[str, Any]]:
    """Parse WDC Dst fixed-width format (120 chars/line).

    One line per day; 24 hourly values at cols 20-115 (4 chars each).
    Missing data: 9999. RULE-082: |val| > 500 → None.
    """
    fetch_ts = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []

    for line in text.splitlines():
        if len(line) < 116:
            continue
        if not line.startswith("DST"):
            continue

        # Parse day from cols 8-9
        try:
            day = int(line[8:10])
        except ValueError:
            continue
        if day < 1 or day > 31:
            continue

        # Parse century + YY for cross-check (cols 14-15 + 3-4)
        # We trust the caller's year/month args since we fetched from that URL

        for hour in range(24):
            start_col = 20 + hour * 4
            val_str = line[start_col:start_col + 4].strip()
            if not val_str:
                continue
            try:
                val = int(val_str)
            except ValueError:
                continue

            # Missing data sentinel
            if val == 9999:
                dst_nt: float | None = None
            else:
                dst_nt = float(val)
                # RULE-082
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


# ---------------------------------------------------------------------------
# HTML parser — fallback for months without .for.request
# ---------------------------------------------------------------------------

def _parse_dst_html(
    html: str, year: int, month: int, data_type: str
) -> list[dict[str, Any]]:
    """Parse Kyoto Dst HTML page: <pre> block or <table>."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    fetch_ts = datetime.now(timezone.utc).isoformat()

    pre_tags = soup.find_all("pre")
    for pre in pre_tags:
        text = pre.get_text()
        if "DST" in text.upper() or "Dst" in text:
            parsed = _parse_dst_pre_block(text, year, month, data_type, fetch_ts)
            if parsed:
                return parsed

    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        parsed = _parse_dst_table(rows, year, month, data_type, fetch_ts)
        if parsed:
            return parsed

    return []


def _parse_dst_pre_block(
    text: str, year: int, month: int, data_type: str, fetch_ts: str
) -> list[dict[str, Any]]:
    """Parse Kyoto Dst <pre> fixed-width block (post-2019 HTML variant)."""
    import re
    records: list[dict[str, Any]] = []
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
            try:
                val = int(parts[hour])
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
