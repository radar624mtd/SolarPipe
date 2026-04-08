"""Task 6.8 — End-to-end integration test for Phase 6 pipeline.

Covers: seed staging DB → populate feature_vectors → emulate ensemble →
        write Parquet → build cme_catalog.db → validate.

Marker: @pytest.mark.integration
No live network. All data is seeded inline. Run via:
    pytest -m "integration and not live" -k e2e
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers — seed a minimal staging DB
# ---------------------------------------------------------------------------

def _seed_staging_db(db_path: str) -> None:
    """Seed staging.db with enough rows to drive the full pipeline.

    Only inserts into tables that the Phase 6 export/emulate pipeline reads:
      - cme_events   (for source_location / noaa_ar join in build_catalog)
      - feature_vectors  (primary input to emulate_from_db and build_catalog)
    """
    from sqlalchemy.dialects.sqlite import insert

    from solarpipe_data.database.schema import (
        CmeEvent,
        FeatureVector,
        init_db,
    )

    engine = init_db(db_path)

    # ---- cme_events --------------------------------------------------------
    # Minimal required columns (all others nullable)
    cme_rows = [
        {
            "activity_id": f"2017-09-{10 + i:02d}T12:00:00-CME-001",
            "start_time": f"2017-09-{10 + i:02d}T12:00:00Z",
            "source_location": "N14W10",
            "active_region_num": 12673,
        }
        for i in range(5)
    ]
    with engine.begin() as conn:
        conn.execute(insert(CmeEvent.__table__).values(cme_rows).on_conflict_do_nothing())

    # ---- feature_vectors (cross-match output, Phase 5 deliverable) ---------
    fv_rows = [
        {
            "activity_id": f"2017-09-{10 + i:02d}T12:00:00-CME-001",
            "launch_time": f"2017-09-{10 + i:02d}T12:00:00Z",
            "cme_speed_kms": 500.0 + i * 100,
            "cme_latitude": 14.0,
            "cme_longitude": -10.0,
            "cme_angular_width_deg": 120.0,
            "cme_mass_grams": 5e14,
            "linked_flare_id": None,
            "flare_class_numeric": 1.2e-4 + i * 1e-5,
            "icme_arrival_time": f"2017-09-{12 + i:02d}T06:00:00Z",
            "icme_match_confidence": 0.8,
            "transit_time_hours": 42.0 + i * 2,
            "sharp_harpnum": 7115,
            "sharp_noaa_ar": 12673,
            "sharp_snapshot_context": "at_eruption",
            "usflux": 4.5e22,
            "meanshr": 25.0,
            "totpot": 3.8e23,
            "r_value": 1.8e21,
            "totusjz": 2.1e12,
            "sw_speed_ambient": 450.0,
            "sw_density_ambient": 5.0,
            "sw_bt_ambient": 6.0,
            "f10_7": 165.0,
            "hcs_tilt_angle": None,
            "hcs_distance": None,
            "dst_min_nt": -90.0 - i * 10,
            "kp_max": 7.0,
            "storm_threshold_met": 1,
            "quality_flag": 4,
        }
        for i in range(5)
    ]
    with engine.begin() as conn:
        conn.execute(insert(FeatureVector.__table__).values(fv_rows).on_conflict_do_nothing())

    engine.dispose()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tmp_dirs():
    """Return (staging_db_path, catalog_db_path, parquet_path) in a temp dir.

    Uses ignore_cleanup_errors=True to handle Windows file-lock on SQLite WAL
    files that remain briefly open after engine.dispose().
    """
    td = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    d = td.name
    staging = str(Path(d) / "staging.db")
    catalog = str(Path(d) / "cme_catalog.db")
    parquet = str(Path(d) / "enlil_runs" / "enlil_ensemble_v1.parquet")
    _seed_staging_db(staging)
    yield staging, catalog, parquet
    td.cleanup()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestE2EPipeline:
    """Full pipeline: feature_vectors → emulate → export → validate."""

    def test_staging_db_seeded(self, tmp_dirs):
        """Staging DB has 5 feature_vectors rows after seeding."""
        import sqlalchemy as sa
        from solarpipe_data.database.schema import make_engine, FeatureVector

        staging, _, _ = tmp_dirs
        engine = make_engine(staging)
        with engine.connect() as conn:
            n = conn.execute(sa.text("SELECT COUNT(*) FROM feature_vectors")).scalar()
        engine.dispose()
        assert n == 5

    def test_parquet_export(self, tmp_dirs):
        """build_parquet_from_db writes a valid Parquet file with ≥5*n_members rows."""
        from solarpipe_data.export.parquet_export import build_parquet_from_db
        from solarpipe_data.synthetic.enlil_emulator import EmulatorConfig

        staging, _, parquet = tmp_dirs
        cfg = EmulatorConfig(n_members=10, seed=7)  # small ensemble for speed
        n_rows = build_parquet_from_db(staging, parquet, config=cfg, min_quality=1)
        assert n_rows == 5 * 10  # 5 events × 10 members
        assert Path(parquet).exists()

    def test_parquet_schema(self, tmp_dirs):
        """Parquet file has all required columns from validate_db expected set."""
        import pyarrow.parquet as pq
        from scripts.validate_db import _EXPECTED_PARQUET_COLS

        _, _, parquet = tmp_dirs
        schema = pq.read_schema(parquet)
        actual = set(schema.names)
        missing = _EXPECTED_PARQUET_COLS - actual
        assert not missing, f"Missing Parquet columns: {missing}"

    def test_parquet_metadata(self, tmp_dirs):
        """Parquet file has generation_at and source metadata keys."""
        import pyarrow.parquet as pq

        _, _, parquet = tmp_dirs
        meta = pq.read_schema(parquet).metadata or {}
        decoded = {k.decode(): v.decode() for k, v in meta.items()}
        assert "generated_at" in decoded
        assert "source" in decoded
        assert "n_members" in decoded
        assert decoded["n_members"] == "10"

    def test_parquet_row_group_size(self, tmp_dirs):
        """Parquet row groups respect the ≤64 MB target."""
        import pyarrow.parquet as pq
        from solarpipe_data.export.parquet_export import _ROW_GROUP_TARGET_BYTES

        _, _, parquet = tmp_dirs
        file_meta = pq.read_metadata(parquet)
        for rg_idx in range(file_meta.num_row_groups):
            rg = file_meta.row_group(rg_idx)
            # Rough estimate: total_byte_size is compressed; uncompressed ok
            assert rg.total_byte_size <= _ROW_GROUP_TARGET_BYTES * 2  # 2× compressed margin

    def test_build_catalog(self, tmp_dirs):
        """build_catalog writes cme_catalog.db with 5 rows in all three tables."""
        from solarpipe_data.export.sqlite_export import build_catalog, count_catalog_rows

        staging, catalog, _ = tmp_dirs
        n = build_catalog(staging, catalog, min_quality=1)
        assert n == 5

        counts = count_catalog_rows(catalog)
        assert counts["cme_events"] == 5
        assert counts["flux_rope_fits"] == 5
        assert counts["l1_arrivals"] == 5

    def test_catalog_three_table_join(self, tmp_dirs):
        """Three-table join executes and returns rows for quality≥3 events."""
        import sqlite3
        from scripts.validate_db import _THREE_TABLE_JOIN

        _, catalog, _ = tmp_dirs
        conn = sqlite3.connect(catalog)
        rows = conn.execute(_THREE_TABLE_JOIN).fetchall()
        conn.close()
        # All 5 seeded events have quality_flag=4 and has_in_situ_fit=1
        assert len(rows) == 5

    def test_catalog_speed_range(self, tmp_dirs):
        """No cme_speed values outside 100–3000 km/s (no sentinels)."""
        import sqlite3

        _, catalog, _ = tmp_dirs
        conn = sqlite3.connect(catalog)
        n_bad = conn.execute("""
            SELECT COUNT(*) FROM cme_events
            WHERE cme_speed IS NOT NULL AND (cme_speed < 100 OR cme_speed > 3000)
        """).fetchone()[0]
        conn.close()
        assert n_bad == 0

    def test_catalog_dst_non_null(self, tmp_dirs):
        """dst_min_nT is non-null for all quality≥3 events (seeded with -90 to -130)."""
        import sqlite3

        _, catalog, _ = tmp_dirs
        conn = sqlite3.connect(catalog)
        n = conn.execute("""
            SELECT COUNT(*) FROM l1_arrivals a
            JOIN cme_events e ON e.event_id = a.event_id
            WHERE e.quality_flag >= 3 AND a.dst_min_nT IS NOT NULL
        """).fetchone()[0]
        conn.close()
        assert n == 5

    def test_catalog_schema_columns(self, tmp_dirs):
        """cme_catalog.db has exactly the expected columns in all three tables."""
        import sqlite3
        from scripts.validate_db import (
            _EXPECTED_CME_EVENTS_COLS,
            _EXPECTED_FLUX_ROPE_COLS,
            _EXPECTED_L1_ARRIVALS_COLS,
            _get_columns,
        )

        _, catalog, _ = tmp_dirs
        conn = sqlite3.connect(catalog)

        for table, expected in [
            ("cme_events",     _EXPECTED_CME_EVENTS_COLS),
            ("flux_rope_fits", _EXPECTED_FLUX_ROPE_COLS),
            ("l1_arrivals",    _EXPECTED_L1_ARRIVALS_COLS),
        ]:
            actual = _get_columns(conn, table)
            missing = expected - actual
            assert not missing, f"{table}: missing columns {missing}"

        conn.close()

    def test_validate_db_script_exits_zero(self, tmp_dirs):
        """validate_db.py check_sqlite returns all-pass for the seeded catalog."""
        from scripts.validate_db import check_sqlite

        _, catalog, _ = tmp_dirs
        results = check_sqlite(catalog, min_rows=5)
        failed = [r for r in results if not r.passed]
        assert not failed, f"validate_db failures: {[str(r) for r in failed]}"

    def test_validate_db_parquet_checks(self, tmp_dirs):
        """check_parquet returns all-pass for the generated Parquet file."""
        from scripts.validate_db import check_parquet

        _, _, parquet = tmp_dirs
        results = check_parquet(parquet)
        failed = [r for r in results if not r.passed]
        assert not failed, f"check_parquet failures: {[str(r) for r in failed]}"

    def test_has_in_situ_fit_populated(self, tmp_dirs):
        """has_in_situ_fit=1 for events that have both icme_arrival_time and dst_min_nt."""
        import sqlite3

        _, catalog, _ = tmp_dirs
        conn = sqlite3.connect(catalog)
        n_fit = conn.execute(
            "SELECT COUNT(*) FROM flux_rope_fits WHERE has_in_situ_fit = 1"
        ).fetchone()[0]
        conn.close()
        # All 5 seeded events have icme_arrival_time + dst_min_nt → all should be 1
        assert n_fit == 5

    def test_parquet_arrival_speeds_positive(self, tmp_dirs):
        """All speed_arrival values in Parquet are ≥ 0."""
        import pyarrow.parquet as pq
        import numpy as np

        _, _, parquet = tmp_dirs
        tbl = pq.read_table(parquet, columns=["speed_arrival"])
        speeds = tbl["speed_arrival"].to_pylist()
        assert all(s >= 0.0 for s in speeds), "Negative arrival speed found"

    def test_parquet_transit_hours_positive(self, tmp_dirs):
        """All transit_hours values in Parquet are > 0."""
        import pyarrow.parquet as pq

        _, _, parquet = tmp_dirs
        tbl = pq.read_table(parquet, columns=["transit_hours"])
        transits = tbl["transit_hours"].to_pylist()
        assert all(t > 0.0 for t in transits), "Non-positive transit time found"

    def test_catalog_idempotent(self, tmp_dirs):
        """Running build_catalog twice does not duplicate rows."""
        from solarpipe_data.export.sqlite_export import build_catalog, count_catalog_rows

        staging, catalog, _ = tmp_dirs
        build_catalog(staging, catalog, min_quality=1)
        counts = count_catalog_rows(catalog)
        assert counts["cme_events"] == 5
        assert counts["flux_rope_fits"] == 5
        assert counts["l1_arrivals"] == 5
