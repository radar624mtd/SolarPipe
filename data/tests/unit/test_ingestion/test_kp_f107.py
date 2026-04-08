"""Unit tests for ingest_kp.py and ingest_f107.py.

RULE-090: asyncio_mode=auto
RULE-092: @pytest.mark.unit
"""
from __future__ import annotations

from datetime import date

import pytest


@pytest.fixture
def in_memory_engine():
    from solarpipe_data.database.schema import init_db
    engine = init_db(":memory:")
    yield engine
    engine.dispose()


@pytest.mark.unit
class TestKpParsing:
    def test_parse_record_standard(self):
        from solarpipe_data.ingestion.ingest_kp import _parse_record
        rec = {
            "datetime": "2026-04-06 00:00:00",
            "Kp": "3.3",
            "ap": "15",
            "definitive": True,
            "ap_daily": "12.5",
        }
        parsed = _parse_record(rec, "ts")
        assert parsed is not None
        assert parsed["kp"] == pytest.approx(3.3)
        assert parsed["ap"] == 15
        assert parsed["datetime"] == "2026-04-06 00:00"
        assert parsed["source_catalog"] == "GFZ"

    def test_parse_record_iso_with_t(self):
        from solarpipe_data.ingestion.ingest_kp import _parse_record
        rec = {"datetime": "2026-04-06T03:00:00Z", "Kp": "2.7"}
        parsed = _parse_record(rec, "ts")
        assert parsed is not None
        assert parsed["datetime"] == "2026-04-06 03:00"

    def test_parse_record_sentinel_kp_becomes_none(self):
        from solarpipe_data.ingestion.ingest_kp import _parse_record
        rec = {"datetime": "2026-04-06 00:00:00", "Kp": "999.0"}
        parsed = _parse_record(rec, "ts")
        assert parsed["kp"] is None

    def test_parse_record_missing_datetime_returns_none(self):
        from solarpipe_data.ingestion.ingest_kp import _parse_record
        assert _parse_record({}, "ts") is None

    def test_sentinel_function(self):
        from solarpipe_data.ingestion.ingest_kp import _sentinel
        assert _sentinel(-1.0) is None
        assert _sentinel(999.0) is None
        assert _sentinel(3.3) == pytest.approx(3.3)
        assert _sentinel(None) is None
        assert _sentinel("bad") is None


@pytest.mark.unit
class TestF107Parsing:
    def test_parse_record_ym_format(self):
        from solarpipe_data.ingestion.ingest_f107 import _parse_record
        rec = {
            "time-tag": "2026-03",
            "observed_swpc_solar_flux": "152.3",
            "smoothed_swpc_solar_flux": "148.0",
            "ssn": "85.0",
        }
        parsed = _parse_record(rec, "ts")
        assert parsed is not None
        assert parsed["date"] == "2026-03-01"
        assert parsed["f10_7_obs"] == pytest.approx(152.3)
        assert parsed["f10_7_adj"] == pytest.approx(148.0)
        assert parsed["sunspot_number"] == pytest.approx(85.0)

    def test_parse_record_sentinel_flux(self):
        from solarpipe_data.ingestion.ingest_f107 import _parse_record
        rec = {"time-tag": "2026-03", "observed_swpc_solar_flux": "999.9"}
        parsed = _parse_record(rec, "ts")
        assert parsed["f10_7_obs"] is None

    def test_parse_record_missing_time_tag(self):
        from solarpipe_data.ingestion.ingest_f107 import _parse_record
        assert _parse_record({}, "ts") is None

    def test_parse_record_sentinel_negative_ssn(self):
        from solarpipe_data.ingestion.ingest_f107 import _sentinel
        assert _sentinel(-1.0) is None
        assert _sentinel(-999.9) is None


@pytest.mark.unit
class TestAmbientContext:
    def test_parse_cme_time_safe_standard(self):
        from solarpipe_data.ingestion.ingest_sw_ambient import _parse_cme_time_safe
        dt = _parse_cme_time_safe("2016-09-06T14:18Z")
        assert dt is not None
        assert dt.year == 2016
        assert dt.month == 9
        assert dt.day == 6
        assert dt.hour == 14

    def test_parse_cme_time_safe_space_format(self):
        from solarpipe_data.ingestion.ingest_sw_ambient import _parse_cme_time_safe
        dt = _parse_cme_time_safe("2020-01-15 08:00")
        assert dt is not None
        assert dt.hour == 8

    def test_parse_cme_time_safe_invalid(self):
        from solarpipe_data.ingestion.ingest_sw_ambient import _parse_cme_time_safe
        assert _parse_cme_time_safe("") is None
        assert _parse_cme_time_safe(None) is None

    def test_floor_hour(self):
        from solarpipe_data.ingestion.ingest_sw_ambient import _floor_hour
        from datetime import datetime
        dt = datetime(2026, 4, 6, 14, 37, 22)
        assert _floor_hour(dt) == "2026-04-06 14:00"
