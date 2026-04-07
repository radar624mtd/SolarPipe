"""Unit tests for ingest_donki_cme.

Rules:
- RULE-090: asyncio_mode=auto — no @pytest.mark.asyncio
- RULE-091: fixtures from tests/fixtures/
- RULE-092: @pytest.mark.unit
- Uses in-memory SQLite (no staging.db required)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def cme_sample():
    return _load("donki_cme_sample.json")


@pytest.fixture
def in_memory_engine():
    from solarpipe_data.database.schema import Base, init_db
    engine = init_db(":memory:")
    yield engine
    engine.dispose()


@pytest.mark.unit
class TestParseCmeRecord:
    def test_full_record_parses_correctly(self, cme_sample):
        from solarpipe_data.ingestion.ingest_donki_cme import _parse_cme_record
        rec = _parse_cme_record(cme_sample[0])

        assert rec["activity_id"] == "2016-09-06T14:18Z-CME-001"
        assert rec["start_time"] == "2016-09-06T14:18:00+00:00"
        assert rec["source_location"] == "S09W11"
        assert rec["active_region_num"] == 12673
        assert rec["speed_kms"] == 1571.0
        assert rec["half_angle_deg"] == 34.0
        assert rec["is_earth_directed"] is True
        assert rec["is_most_accurate"] is True
        assert rec["source_catalog"] == "DONKI"

    def test_active_region_zero_converted_to_none(self, cme_sample):
        from solarpipe_data.ingestion.ingest_donki_cme import _parse_cme_record
        rec = _parse_cme_record(cme_sample[1])
        assert rec["active_region_num"] is None

    def test_null_source_location_is_none(self, cme_sample):
        from solarpipe_data.ingestion.ingest_donki_cme import _parse_cme_record
        rec = _parse_cme_record(cme_sample[1])
        assert rec["source_location"] is None

    def test_linked_flare_extracted(self, cme_sample):
        from solarpipe_data.ingestion.ingest_donki_cme import _parse_cme_record
        rec = _parse_cme_record(cme_sample[0])
        assert rec["linked_flare_id"] == "2016-09-06T14:00Z-FLR-001"

    def test_linked_gst_extracted(self, cme_sample):
        from solarpipe_data.ingestion.ingest_donki_cme import _parse_cme_record
        rec = _parse_cme_record(cme_sample[0])
        gst_ids = json.loads(rec["linked_gst_ids"])
        assert "2016-09-09T23:10Z-GST-001" in gst_ids

    def test_linked_ips_extracted(self, cme_sample):
        from solarpipe_data.ingestion.ingest_donki_cme import _parse_cme_record
        rec = _parse_cme_record(cme_sample[2])
        ips_ids = json.loads(rec["linked_ips_ids"])
        assert "2023-12-17T00:00Z-IPS-001" in ips_ids

    def test_missing_activity_id_returns_none(self):
        from solarpipe_data.ingestion.ingest_donki_cme import _parse_cme_record
        result = _parse_cme_record({"startTime": "2024-01-01T00:00Z"})
        assert result is None

    def test_best_analysis_prefers_most_accurate_higher_level(self, cme_sample):
        from solarpipe_data.ingestion.ingest_donki_cme import _parse_cme_record
        # cme_sample[2] has two analyses — isMostAccurate=true at level 2 wins
        rec = _parse_cme_record(cme_sample[2])
        assert rec["speed_kms"] == 1488.0  # level 2 most-accurate

    def test_no_analyses_produces_none_speed(self, cme_sample):
        from solarpipe_data.ingestion.ingest_donki_cme import _parse_cme_record
        rec = _parse_cme_record(cme_sample[1])
        assert rec["speed_kms"] is None

    def test_instruments_json_array(self, cme_sample):
        from solarpipe_data.ingestion.ingest_donki_cme import _parse_cme_record
        rec = _parse_cme_record(cme_sample[0])
        instruments = json.loads(rec["instruments"])
        assert "SOHO: LASCO/C2" in instruments

    def test_n_linked_events_count(self, cme_sample):
        from solarpipe_data.ingestion.ingest_donki_cme import _parse_cme_record
        rec = _parse_cme_record(cme_sample[0])
        assert rec["n_linked_events"] == 2


@pytest.mark.unit
class TestIngestCmeBatch:
    def test_upsert_idempotent_insert_twice_count_once(self, cme_sample, in_memory_engine):
        from solarpipe_data.database.queries import row_count
        from solarpipe_data.database.schema import CmeEvent
        from solarpipe_data.ingestion.ingest_donki_cme import ingest_cme_batch

        n1 = ingest_cme_batch(in_memory_engine, cme_sample)
        n2 = ingest_cme_batch(in_memory_engine, cme_sample)

        assert n1 == 3
        assert n2 == 3
        # Count in DB must still be 3 after double insert
        assert row_count(in_memory_engine, CmeEvent) == 3

    def test_empty_batch_returns_zero(self, in_memory_engine):
        from solarpipe_data.ingestion.ingest_donki_cme import ingest_cme_batch
        assert ingest_cme_batch(in_memory_engine, []) == 0

    def test_records_skip_missing_activity_id(self, in_memory_engine):
        from solarpipe_data.database.queries import row_count
        from solarpipe_data.database.schema import CmeEvent
        from solarpipe_data.ingestion.ingest_donki_cme import ingest_cme_batch

        bad = [{"startTime": "2024-01-01T00:00Z", "cmeAnalyses": [], "linkedEvents": []}]
        n = ingest_cme_batch(in_memory_engine, bad)
        assert n == 0
        assert row_count(in_memory_engine, CmeEvent) == 0

    def test_upsert_overwrites_on_second_insert(self, in_memory_engine):
        from sqlalchemy.orm import Session
        from solarpipe_data.database.schema import CmeEvent
        from solarpipe_data.ingestion.ingest_donki_cme import ingest_cme_batch

        first = [{
            "activityID": "2024-01-01T00:00Z-CME-001",
            "startTime": "2024-01-01T00:00Z",
            "sourceLocation": "N00E00",
            "activeRegionNum": 12345,
            "catalog": "M2M_CATALOG",
            "note": "original",
            "instruments": [],
            "link": None,
            "cmeAnalyses": [],
            "linkedEvents": [],
        }]
        second = [{**first[0], "note": "updated"}]

        ingest_cme_batch(in_memory_engine, first)
        ingest_cme_batch(in_memory_engine, second)

        with Session(in_memory_engine) as s:
            row = s.get(CmeEvent, "2024-01-01T00:00Z-CME-001")
        assert row.note == "updated"


@pytest.mark.unit
class TestNormaliseTimestamp:
    def test_donki_format_no_seconds(self):
        from solarpipe_data.ingestion.ingest_donki_cme import _normalise_ts
        result = _normalise_ts("2016-09-06T14:18Z")
        assert result == "2016-09-06T14:18:00+00:00"

    def test_none_returns_none(self):
        from solarpipe_data.ingestion.ingest_donki_cme import _normalise_ts
        assert _normalise_ts(None) is None

    def test_empty_string_returns_none(self):
        from solarpipe_data.ingestion.ingest_donki_cme import _normalise_ts
        assert _normalise_ts("") is None
