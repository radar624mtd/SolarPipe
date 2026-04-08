"""Unit tests for ingest_solar_wind.py — hourly averaging and upsert.

RULE-090: asyncio_mode=auto
RULE-092: @pytest.mark.unit
"""
from __future__ import annotations

import pytest


@pytest.fixture
def in_memory_engine():
    from solarpipe_data.database.schema import init_db
    engine = init_db(":memory:")
    yield engine
    engine.dispose()


@pytest.mark.unit
class TestBuildRow:
    def test_build_row_fields(self):
        from solarpipe_data.ingestion.ingest_solar_wind import _build_row
        mag_avg = {"bt": 5.2, "bz_gsm": -4.2, "bz_gse": -4.0, "bx_gse": -2.3, "by_gse": 1.1}
        plasma_avg = {"speed": 420.0, "density": 5.2, "temperature": 85000.0}
        row = _build_row("2026-04-06 00:00", mag_avg, plasma_avg, "2026-04-06T00:00:00Z")

        assert row["datetime"] == "2026-04-06 00:00"
        assert row["bz_gsm"] == pytest.approx(-4.2)
        assert row["flow_speed"] == pytest.approx(420.0)
        assert row["proton_density"] == pytest.approx(5.2)
        assert row["spacecraft"] == "DSCOVR"
        assert row["source_catalog"] == "SWPC"

    def test_build_row_ace_before_transition(self):
        from solarpipe_data.ingestion.ingest_solar_wind import _build_row
        row = _build_row("2015-01-01 00:00", {}, {}, "ts")
        assert row["spacecraft"] == "ACE"

    def test_build_row_sentinel_propagated(self):
        from solarpipe_data.ingestion.ingest_solar_wind import _build_row
        mag_avg = {"bt": 99999.9, "bz_gsm": -4.2}
        plasma_avg = {}
        row = _build_row("2026-04-06 01:00", mag_avg, plasma_avg, "ts")
        assert row["b_scalar_avg"] is None
        assert row["bz_gsm"] == pytest.approx(-4.2)


@pytest.mark.unit
class TestIngestSolarWindUpsert:
    async def test_upsert_is_idempotent(self, in_memory_engine):
        """Running upsert twice with same rows should not change row count."""
        import sqlalchemy as sa
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        from sqlalchemy.orm import Session
        from solarpipe_data.database.schema import SolarWindHourly

        row = {
            "datetime": "2026-04-06 02:00",
            "date": "2026-04-06",
            "year": 2026, "doy": 96, "hour": 2,
            "bz_gsm": -4.5,
            "flow_speed": 425.0,
            "spacecraft": "DSCOVR",
            "source_catalog": "SWPC",
            "fetch_timestamp": "ts",
            "data_version": "7day-rt",
        }
        for _ in range(2):
            with Session(in_memory_engine) as s, s.begin():
                stmt = sqlite_insert(SolarWindHourly)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["datetime"],
                    set_={"bz_gsm": stmt.excluded.bz_gsm},
                )
                s.execute(stmt, [row])

        with Session(in_memory_engine) as s:
            count = s.execute(
                sa.select(sa.func.count()).select_from(SolarWindHourly)
            ).scalar()
        assert count == 1
