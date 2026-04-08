"""SWPC solar wind client — NOAA Space Weather Prediction Center.

Rules enforced:
- RULE-002: All HTTP via BaseClient
- RULE-070: bz_gsm only, never bz_gse
- RULE-071: ts.rstrip("Z") before fromisoformat() on all SWPC timestamps
- RULE-072: mag-5-minute.json is a single record; use mag-7-day.json for time series
- RULE-073: L1 propagation delay is physical — store as measured, lag at crossmatch
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .base import BaseClient

_BASE = "https://services.swpc.noaa.gov"


class SwpcClient(BaseClient):
    """Client for NOAA SWPC real-time solar wind products."""

    source_name = "swpc"

    # ------------------------------------------------------------------
    # Magnetic field
    # ------------------------------------------------------------------

    async def fetch_mag_7day(self) -> list[dict[str, Any]]:
        """7-day 1-minute magnetic field time series (bz_gsm canonical).

        Returns list of dicts with keys:
          time_tag, bx_gse, by_gse, bz_gse, lon_gse, lat_gse,
          bt, bz_gsm (RULE-070)
        """
        url = f"{_BASE}/products/solar-wind/mag-7-day.json"
        raw = await self.get(url, cache_key="swpc_mag_7day")
        return _parse_swpc_table(raw)

    async def fetch_plasma_7day(self) -> list[dict[str, Any]]:
        """7-day 1-minute plasma time series.

        Returns list of dicts with keys:
          time_tag, density, speed, temperature
        """
        url = f"{_BASE}/products/solar-wind/plasma-7-day.json"
        raw = await self.get(url, cache_key="swpc_plasma_7day")
        return _parse_swpc_table(raw)

    # ------------------------------------------------------------------
    # NCEI archival CSV — fills gaps pre-7-day window
    # ------------------------------------------------------------------

    async def fetch_omni_hourly_csv(self, year: int) -> str:
        """Fetch OMNI2 hourly CSV for a given year from NCEI.

        Returns raw CSV text. Caller handles parsing and sentinel conversion.
        """
        url = (
            f"https://www.ngdc.noaa.gov/stp/space-weather/solar-data/"
            f"solar-indices/omni/omni2_hourly.csv?year={year}"
        )
        cache_key = f"omni_hourly_{year}"
        html = await self.get_html(url, cache_key=cache_key)
        return html

    # ------------------------------------------------------------------
    # Auxiliary: recent geomagnetic indices from SWPC JSON feeds
    # ------------------------------------------------------------------

    async def fetch_kp_1hour(self) -> list[dict[str, Any]]:
        """Planetary K index (1-hour cadence) from SWPC."""
        url = f"{_BASE}/products/noaa-planetary-k-index.json"
        raw = await self.get(url, cache_key="swpc_kp_1hr")
        return _parse_swpc_table(raw)

    async def fetch_geomag_storm_summary(self) -> list[dict[str, Any]]:
        """Recent geomagnetic storm summary from SWPC 3-day forecast."""
        url = f"{_BASE}/products/noaa-geomagnetic-forecast.json"
        raw = await self.get(url, cache_key="swpc_geomag_forecast")
        if isinstance(raw, list) and len(raw) > 1:
            headers = raw[0]
            return [dict(zip(headers, row)) for row in raw[1:]]
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_swpc_table(raw: Any) -> list[dict[str, Any]]:
    """SWPC JSON is [header_row, data_row, data_row, ...].

    Normalises timestamps: strips trailing Z before fromisoformat (RULE-071).
    Converts bz_gse key alias so callers always see bz_gsm (RULE-070).
    """
    if not isinstance(raw, list) or len(raw) < 2:
        return []

    headers: list[str] = raw[0]
    records: list[dict[str, Any]] = []

    for row in raw[1:]:
        if len(row) != len(headers):
            continue
        d: dict[str, Any] = dict(zip(headers, row))

        # RULE-071: normalise timestamp
        ts = d.get("time_tag") or d.get("time")
        if ts:
            clean_ts = str(ts).rstrip("Z")
            try:
                dt = datetime.fromisoformat(clean_ts).replace(tzinfo=timezone.utc)
                d["time_tag"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                d["time_tag"] = clean_ts

        # Cast numeric fields from string
        for key in list(d.keys()):
            if key == "time_tag":
                continue
            d[key] = _safe_float(d[key])

        records.append(d)

    return records


def _safe_float(val: Any) -> float | None:
    """Convert to float; return None for missing/sentinel values."""
    if val is None or val == "" or val == "-999.9" or val == "999.9":
        return None
    try:
        f = float(val)
        # SWPC sentinels
        if abs(f) >= 99999.0:
            return None
        return f
    except (TypeError, ValueError):
        return None
