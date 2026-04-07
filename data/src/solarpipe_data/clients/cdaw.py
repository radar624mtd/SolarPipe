"""CDAW LASCO CME catalog client.

Rules enforced:
- RULE-002: All HTTP via BaseClient — no direct httpx calls
- RULE-021: File cache checked before rate-limit quota
- RULE-037: Check len(html) > 100 before returning (maintenance guard)
- RULE-050: BeautifulSoup only — never pandas.read_html()
- RULE-051: UNIVERSAL_ver2/ URL path (not legacy UNIVERSAL/)
- RULE-052: Strip footnote markers before numeric casting

URL pattern:
  https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL_ver2/{YYYY}_{MM}/univ{YYYY}_{MM}.html
Rate: 0.5 req/s (1 req per 2 s).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from solarpipe_data.config import Settings

from .base import BaseClient


class CdawClient(BaseClient):
    """Client for CDAW LASCO UNIVERSAL_ver2 CME catalog."""

    source_name = "cdaw"

    def __init__(self, settings: Settings) -> None:
        super().__init__(
            rate_limit=settings.cdaw_rate_limit,
            cache_dir=Path(settings.data_dir) / "raw" / "cdaw",
            cache_ttl_hours=settings.cache_ttl_hours,
            http_timeout_s=settings.http_timeout_s,
            http_max_retries=settings.http_max_retries,
            http_backoff_base_s=settings.http_backoff_base_s,
            cache_enabled=settings.cache_enabled,
        )
        self._base_url = settings.cdaw_base_url.rstrip("/")

    async def fetch_month(
        self, year: int, month: int, force: bool = False
    ) -> str:
        """Fetch one monthly HTML page from UNIVERSAL_ver2.

        Returns raw HTML string. Caller must check len > 100 (RULE-037).
        Cache key is per-month so re-runs avoid re-fetching (RULE-021).
        """
        url = (
            f"{self._base_url}/UNIVERSAL_ver2/"
            f"{year:04d}_{month:02d}/univ{year:04d}_{month:02d}.html"
        )
        cache_key = f"cdaw_{year:04d}_{month:02d}"
        return await self.get_html(url, cache_key=cache_key, force=force)

    async def fetch_range(
        self,
        start: date,
        end: date,
        force: bool = False,
    ) -> list[tuple[int, int, str]]:
        """Fetch all monthly pages covering [start, end] (inclusive).

        Returns list of (year, month, html) tuples — one per calendar month
        intersecting the requested range. Pages with body < 100 bytes are
        silently skipped (maintenance guard — RULE-037).
        """
        results: list[tuple[int, int, str]] = []
        y, m = start.year, start.month
        end_y, end_m = end.year, end.month

        while (y, m) <= (end_y, end_m):
            html = await self.fetch_month(y, m, force=force)
            if len(html) >= 100:
                results.append((y, m, html))
            else:
                pass  # maintenance / empty month — skip quietly
            m += 1
            if m > 12:
                m = 1
                y += 1

        return results
