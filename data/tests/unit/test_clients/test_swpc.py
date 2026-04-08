"""Unit tests for clients/swpc.py.

RULE-090: asyncio_mode=auto — no @pytest.mark.asyncio
RULE-092: @pytest.mark.unit
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text())


@pytest.mark.unit
class TestParseSwpcTable:
    def test_mag_parses_all_fields(self):
        from solarpipe_data.clients.swpc import _parse_swpc_table
        raw = _load("swpc_mag_sample.json")
        records = _parse_swpc_table(raw)
        # 6 data rows
        assert len(records) == 6

    def test_timestamp_stripped_of_z(self):
        from solarpipe_data.clients.swpc import _parse_swpc_table
        raw = [
            ["time_tag", "bt", "bz_gsm"],
            ["2026-04-06T00:01:00Z", "5.2", "-4.2"],
        ]
        records = _parse_swpc_table(raw)
        assert len(records) == 1
        # Z should be stripped and time normalised
        assert "Z" not in records[0]["time_tag"]

    def test_sentinel_99999_becomes_none(self):
        from solarpipe_data.clients.swpc import _parse_swpc_table
        raw = _load("swpc_mag_sample.json")
        records = _parse_swpc_table(raw)
        # Last record has all 99999.9 — should be None
        last = records[-1]
        assert last["bt"] is None
        assert last["bz_gsm"] is None

    def test_bz_gsm_key_present(self):
        from solarpipe_data.clients.swpc import _parse_swpc_table
        raw = _load("swpc_mag_sample.json")
        records = _parse_swpc_table(raw)
        # RULE-070: bz_gsm must be in records
        assert "bz_gsm" in records[0]

    def test_plasma_parses(self):
        from solarpipe_data.clients.swpc import _parse_swpc_table
        raw = _load("swpc_plasma_sample.json")
        records = _parse_swpc_table(raw)
        assert len(records) == 6
        assert records[0]["speed"] == pytest.approx(420.3)
        assert records[0]["density"] == pytest.approx(5.2)

    def test_empty_table_returns_empty_list(self):
        from solarpipe_data.clients.swpc import _parse_swpc_table
        assert _parse_swpc_table([]) == []
        assert _parse_swpc_table([["time_tag"]]) == []

    def test_safe_float_converts_string(self):
        from solarpipe_data.clients.swpc import _safe_float
        assert _safe_float("3.14") == pytest.approx(3.14)
        assert _safe_float("-999.9") is None
        assert _safe_float("999.9") is None
        assert _safe_float("") is None
        assert _safe_float(None) is None
        assert _safe_float("99999.0") is None


@pytest.mark.unit
class TestHourlyAveraging:
    def test_merge_by_hour_groups_correctly(self):
        from solarpipe_data.ingestion.ingest_solar_wind import _merge_by_hour, _hourly_key

        mag = [
            {"time_tag": "2026-04-06 00:01:00", "bt": 5.2, "bz_gsm": -4.2},
            {"time_tag": "2026-04-06 00:30:00", "bt": 4.9, "bz_gsm": -3.9},
            {"time_tag": "2026-04-06 01:00:00", "bt": 6.1, "bz_gsm": -4.8},
        ]
        plasma = [
            {"time_tag": "2026-04-06 00:15:00", "speed": 420.0, "density": 5.2},
        ]
        grouped = _merge_by_hour(mag, plasma)
        assert "2026-04-06 00:00" in grouped
        assert "2026-04-06 01:00" in grouped
        assert len(grouped["2026-04-06 00:00"]["mag"]) == 2
        assert len(grouped["2026-04-06 00:00"]["plasma"]) == 1

    def test_average_records_ignores_none(self):
        from solarpipe_data.ingestion.ingest_solar_wind import _average_records
        records = [
            {"bz_gsm": -4.2},
            {"bz_gsm": None},
            {"bz_gsm": -3.8},
        ]
        avg = _average_records(records, ["bz_gsm"])
        assert avg["bz_gsm"] == pytest.approx(-4.0)

    def test_average_records_all_none_returns_none(self):
        from solarpipe_data.ingestion.ingest_solar_wind import _average_records
        records = [{"bz_gsm": None}, {"bz_gsm": None}]
        avg = _average_records(records, ["bz_gsm"])
        assert avg["bz_gsm"] is None

    def test_spacecraft_ace_before_transition(self):
        from solarpipe_data.ingestion.ingest_solar_wind import _spacecraft_for
        from datetime import datetime, timezone
        dt = datetime(2016, 7, 26, tzinfo=timezone.utc)
        assert _spacecraft_for(dt) == "ACE"

    def test_spacecraft_dscovr_after_transition(self):
        from solarpipe_data.ingestion.ingest_solar_wind import _spacecraft_for
        from datetime import datetime, timezone
        dt = datetime(2016, 7, 28, tzinfo=timezone.utc)
        assert _spacecraft_for(dt) == "DSCOVR"

    def test_sentinel_conversion(self):
        from solarpipe_data.ingestion.ingest_solar_wind import _sentinel
        assert _sentinel(99999.9) is None
        assert _sentinel(-99999.9) is None
        assert _sentinel(420.0) == pytest.approx(420.0)
        assert _sentinel(None) is None
