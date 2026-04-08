"""JSOC DRMS client for HMI SHARP space-weather keywords.

Rules enforced:
- RULE-060: Drop LON_FWT > 60° at ingest
- RULE-061: Wrap every drms.query() with ThreadPoolExecutor.result(timeout=60)
- RULE-062: CEA series only — hmi.sharp_cea_720s
- RULE-063: NOAA_AR = 0 → None; use NOAA_ARS tilde filter for multi-region HARPs
- RULE-064: DATE__OBS double underscore → t_rec column
- RULE-065: NRT series not used; 12-min cadence gaps are normal
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from ..config import get_settings

logger = logging.getLogger(__name__)

# RULE-062: CEA series for space-weather keywords
_SERIES_CEA = "hmi.sharp_cea_720s"
# RULE-063: CCD series for HARP↔NOAA mapping (NOAA_ARS available here)
_SERIES_CCD = "hmi.sharp_720s"

# 18 space-weather keywords + provenance fields (RULE-062)
_SW_KEYWORDS = [
    "USFLUX", "MEANGAM", "MEANGBT", "MEANGBZ", "MEANGBH",
    "MEANJZD", "TOTUSJZ", "MEANALP", "MEANJZH", "TOTUSJH",
    "ABSNJZH", "SAVNCPP", "MEANPOT", "TOTPOT", "MEANSHR",
    "SHRGT45", "R_VALUE", "AREA_ACR",
    # Position + provenance
    "LAT_FWT", "LON_FWT", "NOAA_AR", "HARPNUM", "T_REC", "DATE__OBS",
    "QUALITY",
]

# HARP↔NOAA mapping keywords (Task 4.3)
_MAPPING_KEYWORDS = ["HARPNUM", "NOAA_AR", "NOAA_ARS", "T_REC"]


class JsocClient:
    """JSOC DRMS query client.

    Not a BaseClient subclass — JSOC uses drms package (not httpx).
    All queries go through _query() which enforces the timeout (RULE-061).
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._email = cfg.jsoc_email
        self._timeout = cfg.jsoc_timeout_s

    def _get_drms_client(self):
        """Create drms.Client lazily (imports drms at runtime)."""
        import drms
        if not self._email:
            raise ValueError(
                "JSOC requires an email address. Set JSOC_EMAIL env var."
            )
        return drms.Client(email=self._email)

    def _query(self, ds: str, key: str) -> pd.DataFrame:
        """RULE-061: wrap drms.query() with ThreadPoolExecutor timeout."""
        c = self._get_drms_client()
        with ThreadPoolExecutor(max_workers=1) as pool:
            try:
                return pool.submit(c.query, ds, key=key).result(timeout=self._timeout)
            except FuturesTimeoutError:
                raise TimeoutError(
                    f"JSOC query timed out after {self._timeout}s: {ds!r}"
                )

    def fetch_sharp_at_time(
        self,
        noaa_ar: int,
        t_query: datetime,
        context: str = "at_eruption",
    ) -> list[dict[str, Any]]:
        """Fetch SHARP keywords for a given NOAA AR at a specific time.

        Uses a ±6-minute window (one 12-min cadence step either side).
        Returns list of row dicts; empty if no data or LON_FWT > 60°.

        RULE-060: drops LON_FWT > 60° here.
        RULE-063: NOAA_AR = 0 → None.
        RULE-064: DATE__OBS normalised to t_rec.
        """
        # JSOC time format: 2016.09.06_14:18:00_TAI  (uses _ not T)
        t_start = t_query - timedelta(minutes=6)
        t_end = t_query + timedelta(minutes=6)
        t_start_str = _to_jsoc_time(t_start)
        t_end_str = _to_jsoc_time(t_end)

        ds = f"{_SERIES_CEA}[? NOAA_AR = {noaa_ar} ?][{t_start_str}-{t_end_str}]"
        keys = ",".join(_SW_KEYWORDS)

        try:
            df = self._query(ds, keys)
        except Exception as exc:
            logger.warning("JSOC query failed for AR %d at %s: %s", noaa_ar, t_query, exc)
            return []

        if df is None or df.empty:
            return []

        return _parse_sharp_df(df, context)

    def fetch_sharp_by_harpnum(
        self,
        harpnum: int,
        t_query: datetime,
        context: str = "at_eruption",
    ) -> list[dict[str, Any]]:
        """Fetch SHARP keywords by HARPNUM (for multi-region HARP fallback).

        RULE-063: use HARPNUM when NOAA_AR is 0 or missing.
        """
        t_start = t_query - timedelta(minutes=6)
        t_end = t_query + timedelta(minutes=6)
        t_start_str = _to_jsoc_time(t_start)
        t_end_str = _to_jsoc_time(t_end)

        ds = f"{_SERIES_CEA}[{harpnum}][{t_start_str}-{t_end_str}]"
        keys = ",".join(_SW_KEYWORDS)

        try:
            df = self._query(ds, keys)
        except Exception as exc:
            logger.warning("JSOC HARP query failed for HARP %d at %s: %s", harpnum, t_query, exc)
            return []

        if df is None or df.empty:
            return []

        return _parse_sharp_df(df, context)

    def fetch_harp_noaa_mapping(
        self, t_start: datetime, t_end: datetime
    ) -> list[dict[str, Any]]:
        """Fetch HARP↔NOAA AR mapping for a time range (Task 4.3).

        Uses hmi.sharp_720s (CCD series — NOAA_ARS is available here).
        Returns list of {harpnum, noaa_ar, noaa_ars, t_rec}.
        """
        t_start_str = _to_jsoc_time(t_start)
        t_end_str = _to_jsoc_time(t_end)
        ds = f"{_SERIES_CCD}[][{t_start_str}-{t_end_str}]"
        keys = ",".join(_MAPPING_KEYWORDS)

        try:
            df = self._query(ds, keys)
        except Exception as exc:
            logger.warning("JSOC mapping query failed %s → %s: %s", t_start, t_end, exc)
            return []

        if df is None or df.empty:
            return []

        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            noaa_ar = _int_or_none(row.get("NOAA_AR"))
            # RULE-063: NOAA_AR = 0 → None
            if noaa_ar == 0:
                noaa_ar = None
            records.append({
                "harpnum": _int_or_none(row.get("HARPNUM")),
                "noaa_ar": noaa_ar,
                "noaa_ars": str(row.get("NOAA_ARS", "") or "").strip() or None,
                "t_rec": _normalise_t_rec(str(row.get("T_REC", "") or "")),
            })
        return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_jsoc_time(dt: datetime) -> str:
    """Convert datetime → JSOC time string 'YYYY.MM.DD_HH:MM:SS_TAI'."""
    return dt.strftime("%Y.%m.%d_%H:%M:%S_TAI")


def _normalise_t_rec(raw: str) -> str | None:
    """Convert JSOC T_REC '2016.09.06_14:12:00_TAI' → ISO 'YYYY-MM-DD HH:MM:SS'."""
    if not raw:
        return None
    # Remove _TAI suffix
    clean = raw.replace("_TAI", "").strip()
    # Replace dots and underscores
    clean = clean.replace(".", "-", 2).replace("_", " ", 1)
    try:
        datetime.strptime(clean[:16], "%Y-%m-%d %H:%M")
        return clean[:16]
    except ValueError:
        return clean


def _float_or_none(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and val != val):  # NaN check
        return None
    try:
        f = float(val)
        # JSOC missing value sentinel
        if abs(f) > 1e30:
            return None
        return f
    except (TypeError, ValueError):
        return None


def _int_or_none(val: Any) -> int | None:
    f = _float_or_none(val)
    return int(f) if f is not None else None


def _parse_sharp_df(
    df: pd.DataFrame, context: str
) -> list[dict[str, Any]]:
    """Convert SHARP DataFrame to list of row dicts.

    RULE-060: drop LON_FWT > 60°.
    RULE-063: NOAA_AR = 0 → None.
    RULE-064: DATE__OBS → t_rec if T_REC missing.
    """
    from datetime import datetime, timezone
    fetch_ts = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        lon_fwt = _float_or_none(row.get("LON_FWT"))
        # RULE-060: disk-passage filter — drop limb observations
        if lon_fwt is not None and abs(lon_fwt) > 60.0:
            continue

        # RULE-064: normalise T_REC (prefer T_REC, fallback DATE__OBS)
        t_rec_raw = str(row.get("T_REC") or row.get("DATE__OBS") or "")
        t_rec = _normalise_t_rec(t_rec_raw)

        noaa_ar = _int_or_none(row.get("NOAA_AR"))
        # RULE-063: 0 means no designation
        if noaa_ar == 0:
            noaa_ar = None

        records.append({
            "harpnum": _int_or_none(row.get("HARPNUM")),
            "noaa_ar": noaa_ar,
            "t_rec": t_rec,
            "usflux": _float_or_none(row.get("USFLUX")),
            "meangam": _float_or_none(row.get("MEANGAM")),
            "meangbt": _float_or_none(row.get("MEANGBT")),
            "meangbz": _float_or_none(row.get("MEANGBZ")),
            "meangbh": _float_or_none(row.get("MEANGBH")),
            "meanjzd": _float_or_none(row.get("MEANJZD")),
            "totusjz": _float_or_none(row.get("TOTUSJZ")),
            "meanalp": _float_or_none(row.get("MEANALP")),
            "meanjzh": _float_or_none(row.get("MEANJZH")),
            "totusjh": _float_or_none(row.get("TOTUSJH")),
            "absnjzh": _float_or_none(row.get("ABSNJZH")),
            "savncpp": _float_or_none(row.get("SAVNCPP")),
            "meanpot": _float_or_none(row.get("MEANPOT")),
            "totpot": _float_or_none(row.get("TOTPOT")),
            "meanshr": _float_or_none(row.get("MEANSHR")),
            "shrgt45": _float_or_none(row.get("SHRGT45")),
            "r_value": _float_or_none(row.get("R_VALUE")),
            "area_acr": _float_or_none(row.get("AREA_ACR")),
            "lat_fwt": _float_or_none(row.get("LAT_FWT")),
            "lon_fwt": lon_fwt,
            "query_context": context,
            "source_catalog": "JSOC",
            "fetch_timestamp": fetch_ts,
            "data_version": "hmi.sharp_cea_720s",
        })

    return records
