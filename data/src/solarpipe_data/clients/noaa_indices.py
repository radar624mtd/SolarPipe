"""NOAA SWPC indices client — GOES X-ray flares.

Rules enforced:
- RULE-002: All HTTP via BaseClient
- RULE-021: Cache before rate limiting
- RULE-022: Structured error messages
- RULE-024: httpx.Timeout, not asyncio.wait_for

Endpoints used:
  7-day recent:  /json/goes/primary/xray-flares-7-day.json
  Archive:       /json/goes/{satellite}/xray-flares-{YYYY}.json
                 e.g. /json/goes/16/xray-flares-2023.json

Timestamps include trailing "Z" — strip before fromisoformat() (Phase 3 rule
documented here for reference; applied in ingest_flares.py).
"""
from __future__ import annotations

import logging
from pathlib import Path

from solarpipe_data.config import Settings

from .base import BaseClient

logger = logging.getLogger(__name__)

# Satellites with publicly available flare archives
_ARCHIVE_SATELLITES = [16, 17, 18]
_LEGACY_SATELLITES = [13, 14, 15]  # older GOES


class NoaaIndicesClient(BaseClient):
    """Client for NOAA SWPC JSON endpoints."""

    source_name = "noaa_indices"

    def __init__(self, settings: Settings) -> None:
        super().__init__(
            rate_limit=settings.swpc_rate_limit,
            cache_dir=Path(settings.data_dir) / "raw" / "noaa_indices",
            cache_ttl_hours=settings.cache_ttl_hours,
            http_timeout_s=settings.http_timeout_s,
            http_max_retries=settings.http_max_retries,
            http_backoff_base_s=settings.http_backoff_base_s,
            cache_enabled=settings.cache_enabled,
        )
        self._base_url = settings.swpc_base_url.rstrip("/")

    async def fetch_flares_recent(self, force: bool = False) -> list[dict]:
        """Fetch 7-day rolling GOES flare list (primary satellite)."""
        url = f"{self._base_url}/json/goes/primary/xray-flares-7-day.json"
        return await self.get(url, cache_key="goes_flares_7day", force=force) or []

    async def fetch_flares_year(
        self, satellite: int, year: int, force: bool = False
    ) -> list[dict]:
        """Fetch one year of GOES flares from SWPC archive.

        Args:
            satellite: GOES satellite number (e.g. 16, 17, 18)
            year: calendar year (e.g. 2023)
        """
        url = f"{self._base_url}/json/goes/{satellite}/xray-flares-{year}.json"
        cache_key = f"goes_flares_{satellite}_{year}"
        result = await self.get(url, cache_key=cache_key, force=force)
        return result if isinstance(result, list) else []
