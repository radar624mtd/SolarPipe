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

    def test_pre_2019_date_is_unsupported(self):
        """pre-2019 dates should return unsupported — logged, not parsed."""
        # We can't fully test async client here, but we can test the date comparison
        from datetime import date
        from solarpipe_data.clients.kyoto import _HTML_CUTOFF
        assert date(2018, 12, 1) < _HTML_CUTOFF
        assert date(2019, 5, 1) >= _HTML_CUTOFF


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
