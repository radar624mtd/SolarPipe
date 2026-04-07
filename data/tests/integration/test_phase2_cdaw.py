"""Integration tests for Phase 2 CDAW ingest pipeline.

Marker: @pytest.mark.integration
No live network. Uses fixture HTML — verifies end-to-end parse → DB flow
and expected row-count ranges (RULE-092).
"""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def engine():
    from solarpipe_data.database.schema import init_db
    e = init_db(":memory:")
    yield e
    e.dispose()


@pytest.fixture(scope="module")
def html_sample():
    return (FIXTURES / "cdaw_month_sample.html").read_text(encoding="utf-8")


@pytest.mark.integration
class TestCdawIngestPipeline:
    def test_three_rows_ingested(self, engine, html_sample):
        from solarpipe_data.database.queries import row_count
        from solarpipe_data.database.schema import CdawCmeEvent
        from solarpipe_data.ingestion.ingest_cdaw_lasco import ingest_cdaw_month

        n = ingest_cdaw_month(engine, 2024, 1, html_sample)
        assert n == 3
        assert row_count(engine, CdawCmeEvent) >= 3

    def test_halo_event_present(self, engine, html_sample):
        from sqlalchemy import text
        ingest = __import__(
            "solarpipe_data.ingestion.ingest_cdaw_lasco",
            fromlist=["ingest_cdaw_month"],
        )
        ingest.ingest_cdaw_month(engine, 2024, 1, html_sample)
        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM cdaw_cme_events WHERE central_pa_deg IS NULL AND angular_width_deg = 360")
            ).scalar()
        assert count >= 1

    def test_poor_event_quality_flags_set(self, engine, html_sample):
        from sqlalchemy import text
        from solarpipe_data.ingestion.ingest_cdaw_lasco import ingest_cdaw_month
        ingest_cdaw_month(engine, 2024, 1, html_sample)
        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM cdaw_cme_events WHERE quality_flag < 3")
            ).scalar()
        assert count >= 2  # one "Poor Event" (flag=2) + one "Very Poor Event" (flag=1)

    def test_speed_20rs_populated(self, engine, html_sample):
        from sqlalchemy import text
        from solarpipe_data.ingestion.ingest_cdaw_lasco import ingest_cdaw_month
        ingest_cdaw_month(engine, 2024, 1, html_sample)
        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM cdaw_cme_events WHERE speed_20rs_kms IS NOT NULL")
            ).scalar()
        assert count >= 3

    def test_source_catalog_all_cdaw(self, engine, html_sample):
        from sqlalchemy import text
        from solarpipe_data.ingestion.ingest_cdaw_lasco import ingest_cdaw_month
        ingest_cdaw_month(engine, 2024, 1, html_sample)
        with engine.connect() as conn:
            bad = conn.execute(
                text("SELECT COUNT(*) FROM cdaw_cme_events WHERE source_catalog != 'CDAW'")
            ).scalar()
        assert bad == 0

    def test_re_ingest_idempotent_row_count(self, engine, html_sample):
        from solarpipe_data.database.queries import row_count
        from solarpipe_data.database.schema import CdawCmeEvent
        from solarpipe_data.ingestion.ingest_cdaw_lasco import ingest_cdaw_month

        before = row_count(engine, CdawCmeEvent)
        ingest_cdaw_month(engine, 2024, 1, html_sample)
        after = row_count(engine, CdawCmeEvent)
        assert after == before  # idempotent — no new rows
