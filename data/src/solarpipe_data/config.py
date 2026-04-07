"""Pydantic settings — loaded from configs/default.yaml with env-var overrides."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_default_yaml() -> dict:
    candidate = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
    if candidate.exists():
        with open(candidate) as f:
            return yaml.safe_load(f) or {}
    return {}


_DEFAULTS = _load_default_yaml()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    # Paths
    data_dir: str = Field(default=_DEFAULTS.get("data_dir", "./data"))
    staging_db_path: str = Field(default=_DEFAULTS.get("staging_db_path", "./data/staging/staging.db"))
    output_db_path: str = Field(default=_DEFAULTS.get("output_db_path", "./data/output/cme_catalog.db"))
    log_level: str = Field(default=_DEFAULTS.get("log_level", "INFO"))

    # NASA DONKI
    nasa_api_key: str = Field(default=_DEFAULTS.get("nasa_api_key", "DEMO_KEY"))
    nasa_api_email: str = Field(default=_DEFAULTS.get("nasa_api_email", ""))
    donki_base_url: str = Field(default=_DEFAULTS.get("donki_base_url", "https://api.nasa.gov/DONKI"))
    donki_rate_limit: float = Field(default=_DEFAULTS.get("donki_rate_limit", 0.5))

    # CDAW
    cdaw_base_url: str = Field(default=_DEFAULTS.get("cdaw_base_url", "https://cdaw.gsfc.nasa.gov/CME_list"))
    cdaw_rate_limit: float = Field(default=_DEFAULTS.get("cdaw_rate_limit", 0.5))

    # SWPC
    swpc_base_url: str = Field(default=_DEFAULTS.get("swpc_base_url", "https://services.swpc.noaa.gov"))
    swpc_rate_limit: float = Field(default=_DEFAULTS.get("swpc_rate_limit", 1.0))

    # JSOC
    jsoc_email: str = Field(default=_DEFAULTS.get("jsoc_email", ""))
    jsoc_timeout_s: int = Field(default=_DEFAULTS.get("jsoc_timeout_s", 60))

    # Kyoto
    kyoto_base_url: str = Field(default=_DEFAULTS.get("kyoto_base_url", "https://wdc.kugi.kyoto-u.ac.jp"))
    kyoto_rate_limit: float = Field(default=_DEFAULTS.get("kyoto_rate_limit", 0.33))

    # GFZ
    gfz_kp_url: str = Field(default=_DEFAULTS.get("gfz_kp_url", "https://kp.gfz-potsdam.de/app/json/"))
    gfz_rate_limit: float = Field(default=_DEFAULTS.get("gfz_rate_limit", 0.5))

    # HTTP client
    http_timeout_s: int = Field(default=_DEFAULTS.get("http_timeout_s", 30))
    http_max_retries: int = Field(default=_DEFAULTS.get("http_max_retries", 3))
    http_backoff_base_s: float = Field(default=_DEFAULTS.get("http_backoff_base_s", 2.0))
    cache_enabled: bool = Field(default=_DEFAULTS.get("cache_enabled", True))
    cache_ttl_hours: int = Field(default=_DEFAULTS.get("cache_ttl_hours", 24))

    # Physics
    sharp_lon_fwt_max_deg: float = Field(default=_DEFAULTS.get("sharp_lon_fwt_max_deg", 60.0))
    donki_verified_start: str = Field(default=_DEFAULTS.get("donki_verified_start", "2012-01-01"))

    # Quality
    min_quality_flag: int = Field(default=_DEFAULTS.get("min_quality_flag", 3))

    @field_validator("staging_db_path", "output_db_path", mode="before")
    @classmethod
    def _posix_path(cls, v: str) -> str:
        return Path(v).as_posix()


def get_settings() -> Settings:
    return Settings()
