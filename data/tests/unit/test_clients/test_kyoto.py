"""Unit tests for clients/kyoto.py — Dst HTML parser, cascade logic.

RULE-090: asyncio_mode=auto
RULE-092: @pytest.mark.unit
"""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"


@pytest.mark.unit
class TestKyotoDstParser:
    def test_parse_dst_pre_block_day1(self):
        from solarpipe_data.clients.kyoto import _parse_dst_pre_block
        # Simple pre-block with one day
        text = (
            "DST INDEX\n"
            "  1   -5   -4   -3   -2   -1    0    1    2    3    2    1    0"
            "   -1   -2   -3   -4   -5   -4   -3   -2   -1    0    1    2\n"
        )
        records = _parse_dst_pre_block(text, 2025, 4, "final", "ts")
        assert len(records) == 24
        # Hour 0 = -5
        assert records[0]["dst_nt"] == pytest.approx(-5.0)
        assert records[0]["datetime"] == "2025-04-01 00:00"
        assert records[0]["data_type"] == "final"

    def test_parse_dst_sentinel_to_none(self):
        from solarpipe_data.clients.kyoto import _parse_dst_pre_block
        # Build a line where one value is 9999 (sentinel)
        vals = [-5] * 23 + [9999]
        vals_str = "   ".join(str(v) for v in vals)
        text = f"  1  {vals_str}\n"
        records = _parse_dst_pre_block(text, 2025, 4, "final", "ts")
        # Last hour should be None (RULE-082: >500 → None)
        last_record = [r for r in records if r["datetime"].endswith("23:00")]
        assert last_record[0]["dst_nt"] is None

    def test_parse_dst_html_fixture(self):
        from solarpipe_data.clients.kyoto import _parse_dst_html
        html = (FIXTURES / "kyoto_dst_sample.html").read_text()
        records = _parse_dst_html(html, 2025, 4, "provisional")
        # Should parse 3 days × 24 hours = 72 records
        assert len(records) == 72
        # Day 3 hour 2 = -30
        day3_h2 = next(r for r in records if r["datetime"] == "2025-04-03 02:00")
        assert day3_h2["dst_nt"] == pytest.approx(-30.0)

    def test_empty_body_guard(self):
        from solarpipe_data.clients.kyoto import _parse_dst_html
        # Body < 100 chars should return empty (RULE-037 enforced in client.fetch_month)
        records = _parse_dst_html("<html></html>", 2025, 4, "realtime")
        assert records == []

    def test_wdc_format_9999_sentinel(self):
        """WDC format: 9999 value → None (missing data)."""
        from solarpipe_data.clients.kyoto import _parse_wdc_format
        # Build a line with 9999 in hour 5 (col 40-43)
        base = "DST1001*01  X220   0" + "   5" * 5 + "9999" + "   5" * 18 + "   5"
        records = _parse_wdc_format(base + "\n", 2010, 1, "final")
        assert records[5]["dst_nt"] is None
        assert records[4]["dst_nt"] == 5.0

    def test_wdc_format_parses_pre_2019(self):
        """WDC format parser supports pre-2019 data (no longer blocked)."""
        from solarpipe_data.clients.kyoto import _parse_wdc_format
        # Real Jan 2010 WDC format line
        line = "DST1001*01  X220   0   5   4   4   2   0   0   2   2   2   0  -1   1   2   4   5   7   3   6  12  13  14  14  13  11   5"
        records = _parse_wdc_format(line + "\n", 2010, 1, "final")
        assert len(records) == 24
        assert records[0]["dst_nt"] == 5.0   # hour 00: col 20-23 = '   5'
        assert records[10]["dst_nt"] == -1.0  # hour 10: col 60-63 = '  -1'
        assert records[0]["data_type"] == "final"
        assert records[0]["datetime"] == "2010-01-01 00:00"


@pytest.mark.unit
class TestDstCascade:
    def test_cascade_final_beats_provisional(self):
        from solarpipe_data.ingestion.ingest_dst import _should_update
        # Existing = provisional, incoming = final → should update
        assert _should_update("provisional", "final") is True

    def test_cascade_provisional_beats_realtime(self):
        from solarpipe_data.ingestion.ingest_dst import _should_update
        assert _should_update("realtime", "provisional") is True

    def test_cascade_no_downgrade_final(self):
        from solarpipe_data.ingestion.ingest_dst import _should_update
        # Existing = final, incoming = realtime → must NOT update (RULE-080)
        assert _should_update("final", "realtime") is False

    def test_cascade_no_downgrade_provisional(self):
        from solarpipe_data.ingestion.ingest_dst import _should_update
        assert _should_update("provisional", "realtime") is False

    def test_cascade_same_type_updates(self):
        from solarpipe_data.ingestion.ingest_dst import _should_update
        # Same type should update (refresh data)
        assert _should_update("final", "final") is True
        assert _should_update("provisional", "provisional") is True

    def test_cascade_none_existing_always_updates(self):
        from solarpipe_data.ingestion.ingest_dst import _should_update
        assert _should_update(None, "realtime") is True
        assert _should_update(None, "final") is True
