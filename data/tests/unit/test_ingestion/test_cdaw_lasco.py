"""Unit tests for ingest_cdaw_lasco.

Rules:
- RULE-090: asyncio_mode=auto
- RULE-091: fixture from tests/fixtures/cdaw_month_sample.html
- RULE-092: @pytest.mark.unit
- In-memory SQLite — no staging.db required
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"


@pytest.fixture
def html_sample():
    return (FIXTURES / "cdaw_month_sample.html").read_text(encoding="utf-8")


@pytest.fixture
def in_memory_engine():
    from solarpipe_data.database.schema import init_db
    engine = init_db(":memory:")
    yield engine
    engine.dispose()


# ---------------------------------------------------------------------------
# _parse_html
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParseHtml:
    def test_returns_three_rows(self, html_sample):
        from solarpipe_data.ingestion.ingest_cdaw_lasco import _parse_html
        rows = _parse_html(html_sample, 2024, 1)
        assert len(rows) == 3

    def test_halo_cpa_is_none_width_is_360(self, html_sample):
        from solarpipe_data.ingestion.ingest_cdaw_lasco import _parse_html
        rows = _parse_html(html_sample, 2024, 1)
        halo = rows[0]
        assert halo["central_pa_deg"] is None
        assert halo["angular_width_deg"] == 360.0

    def test_non_halo_cpa_parsed(self, html_sample):
        from solarpipe_data.ingestion.ingest_cdaw_lasco import _parse_html
        rows = _parse_html(html_sample, 2024, 1)
        assert rows[1]["central_pa_deg"] == 250.0
        assert rows[1]["angular_width_deg"] == 45.0

    def test_footnote_stripped_from_speed(self, html_sample):
        from solarpipe_data.ingestion.ingest_cdaw_lasco import _parse_html
        rows = _parse_html(html_sample, 2024, 1)
        # "450*1" → 450.0
        assert rows[1]["linear_speed_kms"] == 450.0

    def test_mass_sentinel_is_none(self, html_sample):
        from solarpipe_data.ingestion.ingest_cdaw_lasco import _parse_html
        rows = _parse_html(html_sample, 2024, 1)
        # "----" → None
        assert rows[1]["mass_grams"] is None

    def test_poor_event_quality_flag_2(self, html_sample):
        from solarpipe_data.ingestion.ingest_cdaw_lasco import _parse_html
        rows = _parse_html(html_sample, 2024, 1)
        assert rows[1]["quality_flag"] == 2

    def test_very_poor_event_quality_flag_1(self, html_sample):
        from solarpipe_data.ingestion.ingest_cdaw_lasco import _parse_html
        rows = _parse_html(html_sample, 2024, 1)
        assert rows[2]["quality_flag"] == 1

    def test_default_quality_flag_3(self, html_sample):
        from solarpipe_data.ingestion.ingest_cdaw_lasco import _parse_html
        rows = _parse_html(html_sample, 2024, 1)
        assert rows[0]["quality_flag"] == 3

    def test_cdaw_id_format(self, html_sample):
        from solarpipe_data.ingestion.ingest_cdaw_lasco import _parse_html
        rows = _parse_html(html_sample, 2024, 1)
        # "2024/01/03" + "04:00:05" → "20240103.040005"
        assert rows[0]["cdaw_id"] == "20240103.040005"

    def test_speed_20rs_is_canonical_speed(self, html_sample):
        from solarpipe_data.ingestion.ingest_cdaw_lasco import _parse_html
        rows = _parse_html(html_sample, 2024, 1)
        assert rows[0]["speed_20rs_kms"] == 1620.0

    def test_source_catalog_is_cdaw(self, html_sample):
        from solarpipe_data.ingestion.ingest_cdaw_lasco import _parse_html
        rows = _parse_html(html_sample, 2024, 1)
        assert all(r["source_catalog"] == "CDAW" for r in rows)

    def test_data_version_is_universal_ver2(self, html_sample):
        from solarpipe_data.ingestion.ingest_cdaw_lasco import _parse_html
        rows = _parse_html(html_sample, 2024, 1)
        assert all(r["data_version"] == "UNIVERSAL_ver2" for r in rows)

    def test_short_page_returns_empty(self):
        from solarpipe_data.ingestion.ingest_cdaw_lasco import _parse_html
        rows = _parse_html("<html></html>", 2024, 1)
        assert rows == []

    def test_page_under_100_bytes_skipped(self):
        from solarpipe_data.ingestion.ingest_cdaw_lasco import ingest_cdaw_month
        from solarpipe_data.database.schema import init_db
        engine = init_db(":memory:")
        n = ingest_cdaw_month(engine, 2024, 1, "<html></html>")
        assert n == 0
        engine.dispose()


# ---------------------------------------------------------------------------
# Upsert idempotency
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIngestCdawMonth:
    def test_upsert_idempotent(self, html_sample, in_memory_engine):
        from solarpipe_data.database.queries import row_count
        from solarpipe_data.database.schema import CdawCmeEvent
        from solarpipe_data.ingestion.ingest_cdaw_lasco import ingest_cdaw_month

        n1 = ingest_cdaw_month(in_memory_engine, 2024, 1, html_sample)
        n2 = ingest_cdaw_month(in_memory_engine, 2024, 1, html_sample)

        assert n1 == 3
        assert n2 == 3
        assert row_count(in_memory_engine, CdawCmeEvent) == 3

    def test_empty_batch_returns_zero(self, in_memory_engine):
        from solarpipe_data.ingestion.ingest_cdaw_lasco import ingest_cdaw_month
        n = ingest_cdaw_month(in_memory_engine, 2024, 1, "")
        assert n == 0


# ---------------------------------------------------------------------------
# _strip_footnote
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStripFootnote:
    def test_asterisk_marker(self):
        from solarpipe_data.ingestion.ingest_cdaw_lasco import _strip_footnote
        assert _strip_footnote("-54.7*1") == "-54.7"

    def test_no_marker(self):
        from solarpipe_data.ingestion.ingest_cdaw_lasco import _strip_footnote
        assert _strip_footnote("1600") == "1600"

    def test_positive_with_marker(self):
        from solarpipe_data.ingestion.ingest_cdaw_lasco import _strip_footnote
        assert _strip_footnote("360*2") == "360"

    def test_empty_string(self):
        from solarpipe_data.ingestion.ingest_cdaw_lasco import _strip_footnote
        assert _strip_footnote("") == ""
