"""Integration tests for Phase 4 SHARP ingestion pipeline.

Marker: @pytest.mark.integration
No live network — mocks JsocClient to return fixture data.
Verifies: DB schema, parse→upsert flow, LON_FWT filter, coverage metric.

RULE-090: asyncio_mode=auto
"""
from __future__ import annotations

import pytest
import pandas as pd
import sqlalchemy as sa


@pytest.fixture(scope="module")
def engine():
    from solarpipe_data.database.schema import init_db
    from solarpipe_data.database.migrations import apply_pending
    e = init_db(":memory:")
    apply_pending(e)
    yield e
    e.dispose()


def _make_sharp_df(noaa_ar: int, lon_fwt: float = 15.0) -> pd.DataFrame:
    """Build a minimal SHARP DataFrame for testing."""
    return pd.DataFrame([{
        "HARPNUM": 4000 + noaa_ar,
        "NOAA_AR": noaa_ar,
        "T_REC": "2016.09.06_14:12:00_TAI",
        "DATE__OBS": "2016.09.06_14:12:00_TAI",
        "LON_FWT": lon_fwt,
        "LAT_FWT": -8.0,
        "USFLUX": 1.2e22,
        "MEANGAM": 5.3,
        "MEANGBT": 100.0,
        "MEANGBZ": -50.0,
        "MEANGBH": 80.0,
        "MEANJZD": 0.01,
        "TOTUSJZ": 1.5e12,
        "MEANALP": 0.05,
        "MEANJZH": 0.02,
        "TOTUSJH": 3.0e12,
        "ABSNJZH": 2.5e12,
        "SAVNCPP": 120.0,
        "MEANPOT": 1000.0,
        "TOTPOT": 5.0e24,
        "MEANSHR": 20.0,
        "SHRGT45": 0.15,
        "R_VALUE": 3.5,
        "AREA_ACR": 800.0,
        "QUALITY": 0,
    }])


@pytest.mark.integration
class TestSharpSchemaAndParsing:
    def test_sharp_keywords_table_exists(self, engine):
        from solarpipe_data.database.schema import SharpKeyword
        with engine.connect() as conn:
            count = conn.execute(
                sa.select(sa.func.count()).select_from(SharpKeyword.__table__)
            ).scalar()
        assert count == 0  # empty in fresh DB

    def test_harp_noaa_map_table_exists(self, engine):
        from solarpipe_data.database.schema import HarpNoaaMap
        with engine.connect() as conn:
            count = conn.execute(
                sa.select(sa.func.count()).select_from(HarpNoaaMap.__table__)
            ).scalar()
        assert count == 0

    def test_parse_sharp_df_and_insert(self, engine):
        from solarpipe_data.clients.jsoc import _parse_sharp_df
        from solarpipe_data.database.schema import SharpKeyword
        from sqlalchemy.orm import Session

        df = _make_sharp_df(noaa_ar=12673)
        records = _parse_sharp_df(df, "at_eruption")
        assert len(records) == 1

        with Session(engine) as s, s.begin():
            s.execute(SharpKeyword.__table__.insert(), records)

        with engine.connect() as conn:
            count = conn.execute(
                sa.select(sa.func.count()).select_from(SharpKeyword.__table__)
            ).scalar()
        assert count == 1

    def test_lon_fwt_filter_at_boundary(self, engine):
        from solarpipe_data.clients.jsoc import _parse_sharp_df
        # Exactly 60° should pass; >60° should be dropped (RULE-060)
        df_pass = _make_sharp_df(noaa_ar=11111, lon_fwt=60.0)
        df_fail = _make_sharp_df(noaa_ar=22222, lon_fwt=60.1)

        records_pass = _parse_sharp_df(df_pass, "at_eruption")
        records_fail = _parse_sharp_df(df_fail, "at_eruption")

        assert len(records_pass) == 1
        assert len(records_fail) == 0

    def test_noaa_ar_zero_stored_as_none(self, engine):
        from solarpipe_data.clients.jsoc import _parse_sharp_df
        df = pd.DataFrame([{
            "HARPNUM": 9999, "NOAA_AR": 0,
            "T_REC": "2016.09.07_00:00:00_TAI",
            "LON_FWT": 10.0, "LAT_FWT": 5.0,
        }])
        records = _parse_sharp_df(df, "at_eruption")
        assert len(records) == 1
        assert records[0]["noaa_ar"] is None  # RULE-063


@pytest.mark.integration
class TestSharpCoverageMetric:
    def test_coverage_metric_runs_on_empty_db(self):
        from solarpipe_data.database.schema import init_db
        from solarpipe_data.database.migrations import apply_pending
        from solarpipe_data.ingestion.select_sharp_features import compute_sharp_coverage

        import tempfile, os, gc
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            e = init_db(db_path)
            apply_pending(e)
            e.dispose()
            del e
            result = compute_sharp_coverage(db_path)
            assert result["total_cmes_with_ar"] == 0
            assert result["coverage_pct"] == 0.0
        finally:
            gc.collect()
            try:
                os.unlink(db_path)
            except PermissionError:
                pass  # Windows: SQLite WAL files may linger briefly

    def test_optimal_snapshot_prefers_at_eruption(self, engine):
        from solarpipe_data.clients.jsoc import _parse_sharp_df
        from solarpipe_data.ingestion.select_sharp_features import get_best_sharp_snapshot
        from solarpipe_data.database.schema import SharpKeyword
        from sqlalchemy.orm import Session

        # Insert at_eruption and minus_6h snapshots for same AR
        for context in ["minus_6h", "at_eruption"]:
            df = _make_sharp_df(noaa_ar=55555)
            records = _parse_sharp_df(df, context)
            with Session(engine) as s, s.begin():
                s.execute(SharpKeyword.__table__.insert(), records)

        result = get_best_sharp_snapshot(engine, noaa_ar=55555, harpnum=None, t_eruption="")
        assert result is not None
        assert result["query_context"] == "at_eruption"
