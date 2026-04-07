"""Unit tests for DONKI ancillary ingestors: ENLIL, GST, IPS.

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
def gst_sample():
    return _load("donki_gst_sample.json")


@pytest.fixture
def ips_sample():
    return _load("donki_ips_sample.json")


@pytest.fixture
def enlil_sample():
    return _load("donki_enlil_sample.json")


@pytest.fixture
def in_memory_engine():
    from solarpipe_data.database.schema import init_db
    engine = init_db(":memory:")
    yield engine
    engine.dispose()


# ---------------------------------------------------------------------------
# GST ingestor
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIngestGst:
    def test_kp_max_computed(self, gst_sample, in_memory_engine):
        from sqlalchemy.orm import Session
        from solarpipe_data.database.schema import GeomagneticStorm
        from solarpipe_data.ingestion.ingest_donki_gst import ingest_gst

        ingest_gst(in_memory_engine, gst_sample)
        with Session(in_memory_engine) as s:
            row = s.get(GeomagneticStorm, "2024-01-03T12:00Z-GST-001")
        assert row is not None
        assert row.kp_index_max == 7.0

    def test_empty_kp_list_gives_none_max(self, gst_sample, in_memory_engine):
        from sqlalchemy.orm import Session
        from solarpipe_data.database.schema import GeomagneticStorm
        from solarpipe_data.ingestion.ingest_donki_gst import ingest_gst

        ingest_gst(in_memory_engine, gst_sample)
        with Session(in_memory_engine) as s:
            row = s.get(GeomagneticStorm, "2024-01-07T06:00Z-GST-001")
        assert row.kp_index_max is None

    def test_linked_event_ids_json(self, gst_sample, in_memory_engine):
        from sqlalchemy.orm import Session
        from solarpipe_data.database.schema import GeomagneticStorm
        from solarpipe_data.ingestion.ingest_donki_gst import ingest_gst

        ingest_gst(in_memory_engine, gst_sample)
        with Session(in_memory_engine) as s:
            row = s.get(GeomagneticStorm, "2024-01-03T12:00Z-GST-001")
        ids = json.loads(row.linked_event_ids)
        assert "2024-01-03T04:15Z-FLR-001" in ids

    def test_upsert_idempotent(self, gst_sample, in_memory_engine):
        from solarpipe_data.database.queries import row_count
        from solarpipe_data.database.schema import GeomagneticStorm
        from solarpipe_data.ingestion.ingest_donki_gst import ingest_gst

        ingest_gst(in_memory_engine, gst_sample)
        ingest_gst(in_memory_engine, gst_sample)
        assert row_count(in_memory_engine, GeomagneticStorm) == 2

    def test_missing_gst_id_skipped(self, in_memory_engine):
        from solarpipe_data.ingestion.ingest_donki_gst import ingest_gst
        bad = [{"startTime": "2024-01-01T00:00Z", "allKpIndex": [], "linkedEvents": []}]
        n = ingest_gst(in_memory_engine, bad)
        assert n == 0

    def test_empty_records_returns_zero(self, in_memory_engine):
        from solarpipe_data.ingestion.ingest_donki_gst import ingest_gst
        assert ingest_gst(in_memory_engine, []) == 0


# ---------------------------------------------------------------------------
# IPS ingestor
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIngestIps:
    def test_ips_id_is_activity_id(self, ips_sample, in_memory_engine):
        from sqlalchemy.orm import Session
        from solarpipe_data.database.schema import InterplanetaryShock
        from solarpipe_data.ingestion.ingest_donki_ips import ingest_ips

        ingest_ips(in_memory_engine, ips_sample)
        with Session(in_memory_engine) as s:
            row = s.get(InterplanetaryShock, "2024-01-07T00:00Z-IPS-001")
        assert row is not None
        assert row.location == "Earth"

    def test_instruments_json_array(self, ips_sample, in_memory_engine):
        from sqlalchemy.orm import Session
        from solarpipe_data.database.schema import InterplanetaryShock
        from solarpipe_data.ingestion.ingest_donki_ips import ingest_ips

        ingest_ips(in_memory_engine, ips_sample)
        with Session(in_memory_engine) as s:
            row = s.get(InterplanetaryShock, "2024-01-07T00:00Z-IPS-001")
        inst = json.loads(row.instruments)
        assert "ACE: SWEPAM" in inst
        assert "DSCOVR: PLASMAG" in inst

    def test_linked_cme_in_linked_event_ids(self, ips_sample, in_memory_engine):
        from sqlalchemy.orm import Session
        from solarpipe_data.database.schema import InterplanetaryShock
        from solarpipe_data.ingestion.ingest_donki_ips import ingest_ips

        ingest_ips(in_memory_engine, ips_sample)
        with Session(in_memory_engine) as s:
            row = s.get(InterplanetaryShock, "2024-01-07T00:00Z-IPS-001")
        ids = json.loads(row.linked_event_ids)
        assert "2024-01-03T04:15Z-CME-001" in ids

    def test_upsert_idempotent(self, ips_sample, in_memory_engine):
        from solarpipe_data.database.queries import row_count
        from solarpipe_data.database.schema import InterplanetaryShock
        from solarpipe_data.ingestion.ingest_donki_ips import ingest_ips

        ingest_ips(in_memory_engine, ips_sample)
        ingest_ips(in_memory_engine, ips_sample)
        assert row_count(in_memory_engine, InterplanetaryShock) == 2

    def test_missing_activity_id_skipped(self, in_memory_engine):
        from solarpipe_data.ingestion.ingest_donki_ips import ingest_ips
        bad = [{"eventTime": "2024-01-01T00:00Z", "location": "Earth"}]
        assert ingest_ips(in_memory_engine, bad) == 0


# ---------------------------------------------------------------------------
# ENLIL ingestor
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIngestEnlil:
    def test_simulation_id_is_model_completion_time(self, enlil_sample, in_memory_engine):
        from sqlalchemy.orm import Session
        from solarpipe_data.database.schema import EnlilSimulation
        from solarpipe_data.ingestion.ingest_donki_enlil import ingest_enlil

        ingest_enlil(in_memory_engine, enlil_sample)
        with Session(in_memory_engine) as s:
            row = s.get(EnlilSimulation, "2024-01-03T06:00Z")
        assert row is not None
        assert row.au == 1.0

    def test_linked_cme_ids_extracted(self, enlil_sample, in_memory_engine):
        from sqlalchemy.orm import Session
        from solarpipe_data.database.schema import EnlilSimulation
        from solarpipe_data.ingestion.ingest_donki_enlil import ingest_enlil

        ingest_enlil(in_memory_engine, enlil_sample)
        with Session(in_memory_engine) as s:
            row = s.get(EnlilSimulation, "2024-01-03T06:00Z")
        ids = json.loads(row.linked_cme_ids)
        assert "2024-01-03T04:15Z-CME-001" in ids

    def test_empty_cme_inputs_gives_none_ids(self, enlil_sample, in_memory_engine):
        from sqlalchemy.orm import Session
        from solarpipe_data.database.schema import EnlilSimulation
        from solarpipe_data.ingestion.ingest_donki_enlil import ingest_enlil

        ingest_enlil(in_memory_engine, enlil_sample)
        with Session(in_memory_engine) as s:
            row = s.get(EnlilSimulation, "2024-01-07T10:00Z")
        assert row.linked_cme_ids is None

    def test_upsert_idempotent(self, enlil_sample, in_memory_engine):
        from solarpipe_data.database.queries import row_count
        from solarpipe_data.database.schema import EnlilSimulation
        from solarpipe_data.ingestion.ingest_donki_enlil import ingest_enlil

        ingest_enlil(in_memory_engine, enlil_sample)
        ingest_enlil(in_memory_engine, enlil_sample)
        assert row_count(in_memory_engine, EnlilSimulation) == 2

    def test_missing_model_completion_time_skipped(self, in_memory_engine):
        from solarpipe_data.ingestion.ingest_donki_enlil import ingest_enlil
        bad = [{"au": 1.0, "cmeInputs": []}]
        assert ingest_enlil(in_memory_engine, bad) == 0
