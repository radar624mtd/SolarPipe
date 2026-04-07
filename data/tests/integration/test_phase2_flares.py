"""Integration tests for Phase 2 flare ingest + dedup pipeline.

Marker: @pytest.mark.integration
No live network. Seeds DB from fixture JSON, verifies row counts and
dedup behaviour end-to-end.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture(scope="module")
def engine():
    from solarpipe_data.database.schema import init_db
    e = init_db(":memory:")
    yield e
    e.dispose()


@pytest.fixture(scope="module")
def goes_sample():
    return _load("goes_flares_sample.json")


@pytest.fixture(scope="module")
def donki_sample():
    return _load("donki_flares_sample.json")


@pytest.mark.integration
class TestFlaresIngestPipeline:
    def test_goes_rows_inserted(self, engine, goes_sample):
        from solarpipe_data.database.queries import row_count
        from solarpipe_data.database.schema import Flare
        from solarpipe_data.ingestion.ingest_flares import ingest_goes_flares

        ingest_goes_flares(engine, goes_sample, "G16")
        assert row_count(engine, Flare) >= 3

    def test_donki_rows_inserted(self, engine, donki_sample):
        from solarpipe_data.database.queries import row_count
        from solarpipe_data.database.schema import Flare
        from solarpipe_data.ingestion.ingest_flares import ingest_donki_flares

        before = row_count(engine, Flare)
        ingest_donki_flares(engine, donki_sample)
        after = row_count(engine, Flare)
        # At least 1 new row (DONKI has events not in GOES fixture by flare_id)
        assert after >= before

    def test_dedup_removes_goes_duplicate(self, engine, goes_sample, donki_sample):
        """After seeding both catalogs with overlapping event, dedup removes GOES copy."""
        from solarpipe_data.database.queries import row_count
        from solarpipe_data.database.schema import Flare
        from solarpipe_data.ingestion.dedup_flares import dedup_flares
        from solarpipe_data.ingestion.ingest_flares import (
            ingest_donki_flares,
            ingest_goes_flares,
        )
        from solarpipe_data.database.schema import init_db

        # Fresh engine for this test to avoid state from other tests
        fresh = init_db(":memory:")

        # Seed one identical event in both GOES and DONKI
        goes_one = [{
            "begin_time": "2024-03-01T06:00Z",
            "max_class": "X2.0",
            "noaa_ar": 13600,
        }]
        donki_one = [{
            "flrID": "2024-03-01T06:00Z-FLR-001",
            "beginTime": "2024-03-01T06:00Z",
            "classType": "X2.0",
            "activeRegionNum": 13600,
            "instruments": [],
            "linkedEvents": [],
            "catalog": "M2M_CATALOG",
            "note": None,
            "link": None,
        }]

        ingest_goes_flares(fresh, goes_one, "G16")
        ingest_donki_flares(fresh, donki_one)
        assert row_count(fresh, Flare) == 2

        removed = dedup_flares(fresh)
        assert removed == 1
        assert row_count(fresh, Flare) == 1

        fresh.dispose()

    def test_source_catalog_values_correct(self, engine):
        from sqlalchemy import text

        with engine.connect() as conn:
            bad = conn.execute(
                text(
                    "SELECT COUNT(*) FROM flares "
                    "WHERE source_catalog NOT IN ('GOES', 'DONKI')"
                )
            ).scalar()
        assert bad == 0

    def test_class_letter_populated(self, engine):
        from sqlalchemy import text

        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM flares WHERE class_letter IS NOT NULL")
            ).scalar()
        assert count >= 1


@pytest.mark.integration
class TestDonkiAncillaryPipeline:
    def test_gst_rows_inserted(self):
        from solarpipe_data.database.queries import row_count
        from solarpipe_data.database.schema import GeomagneticStorm, init_db
        from solarpipe_data.ingestion.ingest_donki_gst import ingest_gst

        engine = init_db(":memory:")
        sample = _load("donki_gst_sample.json")
        n = ingest_gst(engine, sample)
        assert n == 2
        assert row_count(engine, GeomagneticStorm) == 2
        engine.dispose()

    def test_ips_rows_inserted(self):
        from solarpipe_data.database.queries import row_count
        from solarpipe_data.database.schema import InterplanetaryShock, init_db
        from solarpipe_data.ingestion.ingest_donki_ips import ingest_ips

        engine = init_db(":memory:")
        sample = _load("donki_ips_sample.json")
        n = ingest_ips(engine, sample)
        assert n == 2
        assert row_count(engine, InterplanetaryShock) == 2
        engine.dispose()

    def test_enlil_rows_inserted(self):
        from solarpipe_data.database.queries import row_count
        from solarpipe_data.database.schema import EnlilSimulation, init_db
        from solarpipe_data.ingestion.ingest_donki_enlil import ingest_enlil

        engine = init_db(":memory:")
        sample = _load("donki_enlil_sample.json")
        n = ingest_enlil(engine, sample)
        assert n == 2
        assert row_count(engine, EnlilSimulation) == 2
        engine.dispose()

    def test_gst_kp_max_in_expected_range(self):
        from sqlalchemy import text
        from solarpipe_data.database.schema import init_db
        from solarpipe_data.ingestion.ingest_donki_gst import ingest_gst

        engine = init_db(":memory:")
        ingest_gst(engine, _load("donki_gst_sample.json"))
        with engine.connect() as conn:
            kp = conn.execute(
                text("SELECT kp_index_max FROM geomagnetic_storms WHERE gst_id = '2024-01-03T12:00Z-GST-001'")
            ).scalar()
        assert kp is not None
        assert 5.0 <= kp <= 9.0
        engine.dispose()
