"""Unit tests for DonkiClient.

Rules:
- RULE-090: asyncio_mode=auto — no @pytest.mark.asyncio needed
- RULE-091: fixtures loaded from tests/fixtures/ — no network required
- RULE-092: @pytest.mark.unit
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def cme_sample():
    return _load("donki_cme_sample.json")


@pytest.fixture
def settings():
    from solarpipe_data.config import Settings
    return Settings(
        nasa_api_key="TEST_KEY",
        donki_base_url="https://api.nasa.gov/DONKI",
        donki_rate_limit=100.0,  # high rate so tests don't sleep
        cache_enabled=False,
        staging_db_path="/tmp/test_staging.db",
        output_db_path="/tmp/test_output.db",
        data_dir="/tmp/solarpipe_test",
    )


@pytest.mark.unit
class TestDateChunking:
    def test_single_chunk_within_30_days(self):
        from solarpipe_data.clients.donki import _date_chunks
        chunks = _date_chunks("2024-01-01", "2024-01-15", 30)
        assert chunks == [("2024-01-01", "2024-01-15")]

    def test_splits_across_30_day_boundary(self):
        from solarpipe_data.clients.donki import _date_chunks
        chunks = _date_chunks("2024-01-01", "2024-02-10", 30)
        assert len(chunks) == 2
        assert chunks[0] == ("2024-01-01", "2024-01-30")
        assert chunks[1] == ("2024-01-31", "2024-02-10")

    def test_exact_30_days_is_single_chunk(self):
        from solarpipe_data.clients.donki import _date_chunks
        chunks = _date_chunks("2024-01-01", "2024-01-30", 30)
        assert len(chunks) == 1

    def test_31_days_splits(self):
        from solarpipe_data.clients.donki import _date_chunks
        chunks = _date_chunks("2024-01-01", "2024-01-31", 30)
        assert len(chunks) == 2
        assert chunks[0][1] == "2024-01-30"
        assert chunks[1] == ("2024-01-31", "2024-01-31")


@pytest.mark.unit
class TestRateLimiter:
    async def test_acquire_does_not_block_when_full(self):
        import time
        from solarpipe_data.clients.base import RateLimiter
        rl = RateLimiter(rate=100.0)  # 100 req/s — won't block
        start = time.monotonic()
        await rl.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.5, f"Expected <0.5s, got {elapsed:.3f}s"

    async def test_acquire_consumes_token(self):
        from solarpipe_data.clients.base import RateLimiter
        rl = RateLimiter(rate=1.0)
        rl._tokens = 1.0
        await rl.acquire()
        assert rl._tokens == 0.0


@pytest.mark.unit
class TestBaseClientCacheHit:
    async def test_cache_hit_skips_request(self, tmp_path, settings):
        from solarpipe_data.clients.base import BaseClient

        class TestClient(BaseClient):
            source_name = "test"

        payload = [{"id": "abc"}]
        cache_file = tmp_path / "test" / "abc123.json"
        cache_file.parent.mkdir(parents=True)
        cache_file.write_text(json.dumps(payload))

        client = TestClient(
            rate_limit=100.0,
            cache_dir=tmp_path / "test",
            cache_ttl_hours=24,
            cache_enabled=True,
        )
        client._client = MagicMock()  # should never be called

        # Manually set cache file with matching key
        key = client._make_cache_key("http://example.com", {})
        cp = client._cache_path(key)
        cp.write_text(json.dumps(payload))

        result = await client.get("http://example.com", {})
        assert result == payload
        # _client.get should not be called — cache hit
        client._client.get.assert_not_called()

    async def test_cache_miss_calls_request(self, tmp_path):
        from solarpipe_data.clients.base import BaseClient

        class TestClient(BaseClient):
            source_name = "test"

        payload = {"result": "ok"}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = payload

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        client = TestClient(
            rate_limit=100.0,
            cache_dir=tmp_path / "test",
            cache_ttl_hours=24,
            cache_enabled=False,  # cache off → always call
        )
        client._client = mock_client

        result = await client.get("http://example.com/data", {})
        assert result == payload
        mock_client.get.assert_called_once()


@pytest.mark.unit
class TestDonkiClientFetch:
    async def test_fetch_cme_constructs_correct_url_and_params(self, settings, cme_sample):
        from solarpipe_data.clients.donki import DonkiClient

        async with DonkiClient(settings) as client:
            with patch.object(client, "get", new=AsyncMock(return_value=cme_sample)) as mock_get:
                result = await client.fetch_cme("2024-01-01", "2024-03-31")

        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert "2024-01-01" in str(call_kwargs)
        assert "2024-03-31" in str(call_kwargs)
        assert result == cme_sample

    async def test_fetch_notifications_chunks_range(self, settings):
        from solarpipe_data.clients.donki import DonkiClient

        call_log = []

        async def fake_get(url, params=None, cache_key=None, force=False):
            call_log.append(params)
            return []

        async with DonkiClient(settings) as client:
            with patch.object(client, "get", side_effect=fake_get):
                await client.fetch_notifications("2024-01-01", "2024-03-15")

        # 74 days / 30 = 3 chunks
        assert len(call_log) == 3

    async def test_fetch_cme_analysis_sets_most_accurate_only(self, settings):
        from solarpipe_data.clients.donki import DonkiClient

        async with DonkiClient(settings) as client:
            with patch.object(client, "get", new=AsyncMock(return_value=[])) as mock_get:
                await client.fetch_cme_analysis("2024-01-01", "2024-01-31")

        _, kwargs = mock_get.call_args
        params = kwargs.get("params") or mock_get.call_args[0][1]
        assert params.get("mostAccurateOnly") == "true"

    async def test_fetch_returns_empty_list_on_none_response(self, settings):
        from solarpipe_data.clients.donki import DonkiClient

        async with DonkiClient(settings) as client:
            with patch.object(client, "get", new=AsyncMock(return_value=None)):
                result = await client.fetch_cme("2024-01-01", "2024-01-31")

        assert result == []
