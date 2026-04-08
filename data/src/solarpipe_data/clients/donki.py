"""NASA DONKI API client.

Rules enforced:
- RULE-040: Only /notifications needs 30-day chunking; all other endpoints accept any range
- RULE-041: CMEAnalysis always uses mostAccurateOnly=true
- RULE-042: time21_5 is not event start — do not alias it
- RULE-043: DONKI timestamps use "2016-09-06T14:18Z" (no seconds)
- RULE-044: level_of_data preference 2 > 1 > 0
- RULE-045: Broken linkedEvents references handled with null FKs

Primary base URL: https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get (no api_key needed)
Fallback base URL: https://api.nasa.gov/DONKI (requires api_key, historically unreliable)
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from solarpipe_data.config import Settings

from .base import BaseClient

logger = logging.getLogger(__name__)

_DATE_FMT = "%Y-%m-%d"
_CHUNK_DAYS = 30  # only for /notifications


class DonkiClient(BaseClient):
    """Client for NASA CCMC DONKI API."""

    source_name = "donki"

    def __init__(self, settings: Settings) -> None:
        super().__init__(
            rate_limit=settings.donki_rate_limit,
            cache_dir=Path(settings.data_dir) / "raw" / "donki",
            cache_ttl_hours=settings.cache_ttl_hours,
            http_timeout_s=settings.http_timeout_s,
            http_max_retries=settings.http_max_retries,
            http_backoff_base_s=settings.http_backoff_base_s,
            cache_enabled=settings.cache_enabled,
        )
        self._base_url = settings.donki_base_url.rstrip("/")
        self._api_key = settings.nasa_api_key

    def _params(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        # kauai.ccmc.gsfc.nasa.gov does not use api_key; api.nasa.gov does
        p: dict[str, Any] = {}
        if "api.nasa.gov" in self._base_url:
            p["api_key"] = self._api_key
        if extra:
            p.update(extra)
        return p

    def _cache_key(self, endpoint: str, start: str, end: str, extra: str = "") -> str:
        return f"{endpoint}_{start}_{end}{extra}".replace("/", "_")

    # ------------------------------------------------------------------
    # Public fetch methods — one per endpoint
    # ------------------------------------------------------------------

    async def fetch_cme(self, start: str, end: str, force: bool = False) -> list[dict]:
        """Fetch CME events. Accepts any date range (RULE-040)."""
        url = f"{self._base_url}/CME"
        return await self.get(
            url,
            params=self._params({"startDate": start, "endDate": end}),
            cache_key=self._cache_key("cme", start, end),
            force=force,
        ) or []

    async def fetch_cme_analysis(
        self, start: str, end: str, force: bool = False
    ) -> list[dict]:
        """Fetch CMEAnalysis records with mostAccurateOnly=true (RULE-041)."""
        url = f"{self._base_url}/CMEAnalysis"
        return await self.get(
            url,
            params=self._params({
                "startDate": start,
                "endDate": end,
                "mostAccurateOnly": "true",
            }),
            cache_key=self._cache_key("cme_analysis", start, end),
            force=force,
        ) or []

    async def fetch_flares(self, start: str, end: str, force: bool = False) -> list[dict]:
        """Fetch solar flares (FLR endpoint)."""
        url = f"{self._base_url}/FLR"
        return await self.get(
            url,
            params=self._params({"startDate": start, "endDate": end}),
            cache_key=self._cache_key("flr", start, end),
            force=force,
        ) or []

    async def fetch_gst(self, start: str, end: str, force: bool = False) -> list[dict]:
        """Fetch geomagnetic storm events (GST endpoint)."""
        url = f"{self._base_url}/GST"
        return await self.get(
            url,
            params=self._params({"startDate": start, "endDate": end}),
            cache_key=self._cache_key("gst", start, end),
            force=force,
        ) or []

    async def fetch_ips(self, start: str, end: str, force: bool = False) -> list[dict]:
        """Fetch interplanetary shocks (IPS endpoint)."""
        url = f"{self._base_url}/IPS"
        return await self.get(
            url,
            params=self._params({
                "startDate": start,
                "endDate": end,
                "location": "Earth",
                "catalog": "ALL",
            }),
            cache_key=self._cache_key("ips", start, end),
            force=force,
        ) or []

    async def fetch_enlil(self, start: str, end: str, force: bool = False) -> list[dict]:
        """Fetch WSA-ENLIL simulation records."""
        url = f"{self._base_url}/WSAEnlilSimulations"
        return await self.get(
            url,
            params=self._params({"startDate": start, "endDate": end}),
            cache_key=self._cache_key("enlil", start, end),
            force=force,
        ) or []

    async def fetch_notifications(
        self, start: str, end: str, force: bool = False
    ) -> list[dict]:
        """Fetch notifications — chunked to ≤30 days per request (RULE-040)."""
        chunks = list(_date_chunks(start, end, _CHUNK_DAYS))
        results: list[dict] = []
        url = f"{self._base_url}/notifications"
        for chunk_start, chunk_end in chunks:
            chunk = await self.get(
                url,
                params=self._params({"startDate": chunk_start, "endDate": chunk_end}),
                cache_key=self._cache_key("notif", chunk_start, chunk_end),
                force=force,
            ) or []
            results.extend(chunk)
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_chunks(
    start: str, end: str, chunk_days: int
) -> list[tuple[str, str]]:
    """Split a date range into chunks of at most chunk_days days."""
    d_start = date.fromisoformat(start)
    d_end = date.fromisoformat(end)
    chunks = []
    current = d_start
    while current <= d_end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), d_end)
        chunks.append((current.strftime(_DATE_FMT), chunk_end.strftime(_DATE_FMT)))
        current = chunk_end + timedelta(days=1)
    return chunks
