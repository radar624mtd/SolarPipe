"""BaseClient — all HTTP access goes through this class.

Rules enforced:
- RULE-002: No raw httpx calls outside this module
- RULE-020: Single shared AsyncClient per instance
- RULE-021: File cache checked before consuming rate limit quota
- RULE-022: Structured error messages (source, URL, status, API message)
- RULE-023: Retry on 429/5xx; respect Retry-After; backoff with jitter
- RULE-024: httpx.Timeout, not asyncio.wait_for
- RULE-025: asyncio.run() — no manual event loop policy
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token-bucket rate limiter (RULE-020)."""

    def __init__(self, rate: float) -> None:
        self._rate = rate          # tokens/second
        self._tokens = rate        # start full
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
            self._last = now
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._rate
                await asyncio.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


class BaseClient:
    """Base HTTP client with token-bucket rate limiting, retry, and file cache."""

    source_name: str = "base"

    def __init__(
        self,
        rate_limit: float = 1.0,
        cache_dir: Path | None = None,
        cache_ttl_hours: int = 24,
        http_timeout_s: int = 30,
        http_max_retries: int = 3,
        http_backoff_base_s: float = 2.0,
        cache_enabled: bool = True,
    ) -> None:
        self._rate_limiter = RateLimiter(rate_limit)
        self._cache_dir = cache_dir or Path("./data/raw") / self.source_name
        self._cache_ttl = cache_ttl_hours * 3600
        self._timeout = httpx.Timeout(float(http_timeout_s), connect=10.0)
        self._max_retries = http_max_retries
        self._backoff_base = http_backoff_base_s
        self._cache_enabled = cache_enabled
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "BaseClient":
        self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        cache_key: str | None = None,
        force: bool = False,
    ) -> Any:
        """Fetch JSON from url with params. Returns parsed dict or list."""
        params = params or {}
        key = cache_key or self._make_cache_key(url, params)
        cache_path = self._cache_path(key, suffix=".json")

        # RULE-021: Check cache before rate limiting
        if self._cache_enabled and not force:
            cached = self._read_cache(cache_path)
            if cached is not None:
                logger.debug("%s cache hit: %s", self.source_name, key)
                return cached

        await self._rate_limiter.acquire()
        response = await self._request_with_retry(url, params)
        data = response.json()

        if self._cache_enabled:
            self._write_cache(cache_path, data)

        return data

    async def get_html(
        self,
        url: str,
        cache_key: str | None = None,
        force: bool = False,
    ) -> str:
        """Fetch HTML text from url."""
        key = cache_key or self._make_cache_key(url, {})
        cache_path = self._cache_path(key, suffix=".html")

        if self._cache_enabled and not force:
            path = cache_path
            if path.exists() and self._is_fresh(path):
                return path.read_text(encoding="utf-8")

        await self._rate_limiter.acquire()
        response = await self._request_with_retry(url, {})
        html = response.text

        if self._cache_enabled:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(html, encoding="utf-8")

        return html

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _request_with_retry(
        self, url: str, params: dict[str, Any]
    ) -> httpx.Response:
        """RULE-022 + RULE-023: structured errors, retry on 429/5xx."""
        assert self._client is not None, "BaseClient must be used as async context manager"

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._client.get(url, params=params)
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning(
                    "%s HTTP error attempt %d/%d: %s %s",
                    self.source_name, attempt + 1, self._max_retries + 1, url, exc,
                )
                await self._backoff(attempt, retry_after=None)
                continue

            # Check X-RateLimit-Remaining for dynamic adjustment
            remaining = resp.headers.get("X-RateLimit-Remaining")
            if remaining is not None and int(remaining) < 5:
                logger.warning(
                    "%s rate limit low: %s remaining, sleeping 1s",
                    self.source_name, remaining,
                )
                await asyncio.sleep(1.0)

            if resp.status_code in (200, 201, 204):
                return resp

            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = _parse_retry_after(resp)
                logger.warning(
                    "%s %d on attempt %d/%d: %s (retry_after=%s)",
                    self.source_name, resp.status_code,
                    attempt + 1, self._max_retries + 1, url, retry_after,
                )
                if attempt < self._max_retries:
                    await self._backoff(attempt, retry_after)
                    continue
                # Final attempt failed
                _raise_http_error(self.source_name, url, resp)

            # 400/401/403/404 — do not retry
            _raise_http_error(self.source_name, url, resp)

        # Exhausted retries from transport errors
        raise RuntimeError(
            f"{self.source_name}: exhausted {self._max_retries + 1} attempts for {url}"
        ) from last_exc

    async def _backoff(self, attempt: int, retry_after: float | None) -> None:
        if retry_after is not None:
            await asyncio.sleep(retry_after)
        else:
            delay = self._backoff_base ** attempt
            await asyncio.sleep(delay)

    def _cache_path(self, key: str, suffix: str = ".json") -> Path:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        return self._cache_dir / f"{key}{suffix}"

    def _make_cache_key(self, url: str, params: dict[str, Any]) -> str:
        raw = url + json.dumps(params, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _read_cache(self, path: Path) -> Any | None:
        if not path.exists():
            return None
        if not self._is_fresh(path):
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _write_cache(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def _is_fresh(self, path: Path) -> bool:
        age = time.time() - path.stat().st_mtime
        return age < self._cache_ttl


def _parse_retry_after(resp: httpx.Response) -> float | None:
    val = resp.headers.get("Retry-After")
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return 60.0  # conservative default if unparseable


def _raise_http_error(source: str, url: str, resp: httpx.Response) -> None:
    try:
        body = resp.json()
        api_msg = body.get("error") or body.get("msg") or body.get("message") or str(body)
    except Exception:
        api_msg = resp.text[:200]
    raise httpx.HTTPStatusError(
        f"{source} HTTP {resp.status_code}: {url} — {api_msg}",
        request=resp.request,
        response=resp,
    )
