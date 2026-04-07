"""Unit tests for ingest_flares and dedup_flares.

Rules:
- RULE-090: asyncio_mode=auto
- RULE-091: fixtures from tests/fixtures/
- RULE-092: @pytest.mark.unit
- In-memory SQLite — no staging.db required
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def goes_sample():
    return _load("goes_flares_sample.json")


@pytest.fixture
def donki_sample():
    return _load("donki_flares_sample.json")


@pytest.fixture
def in_memory_engine():
    from solarpipe_data.database.schema import init_db
    engine = init_db(":memory:")
    yield engine
    engine.dispose()


# ---------------------------------------------------------------------------
# GOES parsing
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParseGoesFlares:
    def test_flare_id_constructed_from_begin_time_and_satellite(self, goes_sample):
        from solarpipe_data.ingestion.ingest_flares import _parse_goes_record
        row = _parse_goes_record(goes_sample[0], "G16", "2024-01-01T00:00:00Z")
        assert row is not None
        assert "G16" in row["flare_id"]

    def test_class_letter_and_magnitude_parsed(self, goes_sample):
        from solarpipe_data.ingestion.ingest_flares import _parse_goes_record
        row = _parse_goes_record(goes_sample[0], "G16", "2024-01-01T00:00:00Z")
        assert row["class_letter"] == "X"
        assert row["class_magnitude"] == 1.5

    def test_m_class_parsed(self, goes_sample):
        from solarpipe_data.ingestion.ingest_flares import _parse_goes_record
        row = _parse_goes_record(goes_sample[1], "G16", "ts")
        assert row["class_letter"] == "M"
        assert row["class_magnitude"] == 2.3

    def test_null_ar_is_none(self, goes_sample):
        from solarpipe_data.ingestion.ingest_flares import _parse_goes_record
        row = _parse_goes_record(goes_sample[2], "G16", "ts")
        assert row["active_region_num"] is None

    def test_null_peak_time_is_none(self, goes_sample):
        from solarpipe_data.ingestion.ingest_flares import _parse_goes_record
        row = _parse_goes_record(goes_sample[2], "G16", "ts")
        assert row["peak_time"] is None

    def test_source_catalog_is_goes(self, goes_sample):
        from solarpipe_data.ingestion.ingest_flares import _parse_goes_record
        row = _parse_goes_record(goes_sample[0], "G16", "ts")
        assert row["source_catalog"] == "GOES"

    def test_no_begin_time_returns_none(self):
        from solarpipe_data.ingestion.ingest_flares import _parse_goes_record
        row = _parse_goes_record({}, "G16", "ts")
        assert row is None

    def test_upsert_idempotent(self, goes_sample, in_memory_engine):
        from solarpipe_data.database.queries import row_count
        from solarpipe_data.database.schema import Flare
        from solarpipe_data.ingestion.ingest_flares import ingest_goes_flares

        n1 = ingest_goes_flares(in_memory_engine, goes_sample, "G16")
        n2 = ingest_goes_flares(in_memory_engine, goes_sample, "G16")
        assert n1 == 3
        assert n2 == 3
        assert row_count(in_memory_engine, Flare) == 3


# ---------------------------------------------------------------------------
# DONKI FLR parsing
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParseDonkiFlares:
    def test_flare_id_is_flr_id(self, donki_sample):
        from solarpipe_data.ingestion.ingest_flares import _parse_donki_flare
        row = _parse_donki_flare(donki_sample[0], "ts")
        assert row["flare_id"] == "2024-01-03T04:15Z-FLR-001"

    def test_ar_zero_converted_to_none(self, donki_sample):
        from solarpipe_data.ingestion.ingest_flares import _parse_donki_flare
        # donki_sample[1] has activeRegionNum=0
        row = _parse_donki_flare(donki_sample[1], "ts")
        assert row["active_region_num"] is None

    def test_linked_events_json_array(self, donki_sample):
        from solarpipe_data.ingestion.ingest_flares import _parse_donki_flare
        row = _parse_donki_flare(donki_sample[0], "ts")
        ids = json.loads(row["linked_event_ids"])
        assert "2024-01-03T04:15Z-CME-001" in ids

    def test_instruments_json_array(self, donki_sample):
        from solarpipe_data.ingestion.ingest_flares import _parse_donki_flare
        row = _parse_donki_flare(donki_sample[0], "ts")
        inst = json.loads(row["instruments"])
        assert "GOES-16: EXIS 1-8A" in inst

    def test_source_catalog_is_donki(self, donki_sample):
        from solarpipe_data.ingestion.ingest_flares import _parse_donki_flare
        row = _parse_donki_flare(donki_sample[0], "ts")
        assert row["source_catalog"] == "DONKI"

    def test_missing_flr_id_returns_none(self):
        from solarpipe_data.ingestion.ingest_flares import _parse_donki_flare
        row = _parse_donki_flare({"beginTime": "2024-01-01T00:00Z"}, "ts")
        assert row is None

    def test_upsert_idempotent(self, donki_sample, in_memory_engine):
        from solarpipe_data.database.queries import row_count
        from solarpipe_data.database.schema import Flare
        from solarpipe_data.ingestion.ingest_flares import ingest_donki_flares

        n1 = ingest_donki_flares(in_memory_engine, donki_sample)
        n2 = ingest_donki_flares(in_memory_engine, donki_sample)
        assert n1 == 2
        assert n2 == 2
        assert row_count(in_memory_engine, Flare) == 2


# ---------------------------------------------------------------------------
# _parse_class helper
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParseClass:
    def test_x_class(self):
        from solarpipe_data.ingestion.ingest_flares import _parse_class
        assert _parse_class("X1.5") == ("X", 1.5)

    def test_m_class(self):
        from solarpipe_data.ingestion.ingest_flares import _parse_class
        assert _parse_class("M2.3") == ("M", 2.3)

    def test_c_class(self):
        from solarpipe_data.ingestion.ingest_flares import _parse_class
        assert _parse_class("C3.0") == ("C", 3.0)

    def test_none_returns_none_none(self):
        from solarpipe_data.ingestion.ingest_flares import _parse_class
        assert _parse_class(None) == (None, None)

    def test_empty_string(self):
        from solarpipe_data.ingestion.ingest_flares import _parse_class
        assert _parse_class("") == (None, None)


# ---------------------------------------------------------------------------
# Flare dedup
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDedupFlares:
    def test_goes_donki_duplicate_merged(self, in_memory_engine):
        from solarpipe_data.database.queries import row_count
        from solarpipe_data.database.schema import Flare
        from solarpipe_data.ingestion.dedup_flares import dedup_flares
        from solarpipe_data.ingestion.ingest_flares import (
            ingest_donki_flares,
            ingest_goes_flares,
        )

        # GOES record — same AR and begin_time as DONKI record
        goes = [{
            "begin_time": "2024-01-03T04:15Z",
            "peak_time": "2024-01-03T04:22Z",
            "end_time": None,
            "max_class": "X1.5",
            "source_location": "S09W11",
            "noaa_ar": 13536,
        }]
        donki = [{
            "flrID": "2024-01-03T04:15Z-FLR-001",
            "beginTime": "2024-01-03T04:15Z",
            "peakTime": "2024-01-03T04:22Z",
            "endTime": None,
            "classType": "X1.5",
            "sourceLocation": "S09W11",
            "activeRegionNum": 13536,
            "catalog": "M2M_CATALOG",
            "note": "X-class flare",
            "instruments": [],
            "linkedEvents": [],
            "link": None,
        }]

        ingest_goes_flares(in_memory_engine, goes, "G16")
        ingest_donki_flares(in_memory_engine, donki)
        assert row_count(in_memory_engine, Flare) == 2

        removed = dedup_flares(in_memory_engine)
        assert removed == 1
        assert row_count(in_memory_engine, Flare) == 1

    def test_no_duplicates_no_removal(self, in_memory_engine):
        from solarpipe_data.ingestion.dedup_flares import dedup_flares
        from solarpipe_data.ingestion.ingest_flares import ingest_goes_flares

        goes = [{
            "begin_time": "2024-01-03T04:15Z",
            "max_class": "X1.5",
            "noaa_ar": 13536,
        }]
        ingest_goes_flares(in_memory_engine, goes, "G16")
        removed = dedup_flares(in_memory_engine)
        assert removed == 0

    def test_different_ar_not_merged(self, in_memory_engine):
        from solarpipe_data.database.queries import row_count
        from solarpipe_data.database.schema import Flare
        from solarpipe_data.ingestion.dedup_flares import dedup_flares
        from solarpipe_data.ingestion.ingest_flares import (
            ingest_donki_flares,
            ingest_goes_flares,
        )

        goes = [{
            "begin_time": "2024-01-03T04:15Z",
            "max_class": "X1.5",
            "noaa_ar": 11111,
        }]
        donki = [{
            "flrID": "2024-01-03T04:15Z-FLR-001",
            "beginTime": "2024-01-03T04:15Z",
            "classType": "X1.5",
            "activeRegionNum": 99999,
            "instruments": [],
            "linkedEvents": [],
            "catalog": "M2M_CATALOG",
            "note": None,
            "link": None,
        }]
        ingest_goes_flares(in_memory_engine, goes, "G16")
        ingest_donki_flares(in_memory_engine, donki)
        removed = dedup_flares(in_memory_engine)
        assert removed == 0
        assert row_count(in_memory_engine, Flare) == 2
