"""Unit tests for python/feature_schema.py (G1 gate).

Tests validate:
  1. Total column count == 133
  2. Phantom (drop) count == 6
  3. Flat column count == 105
  4. No duplicate column names in the schema
  5. Every flat column has a valid null_policy (dense or sparse)
  6. SEQUENCE_CHANNELS length == 22
  7. LABEL_COL is present and has role "label"
  8. assert_schema_matches_db() passes against live staging.db
  9. assert_schema_matches_db() raises on a synthetic drift

Run with:
    cd python && python -m pytest tests/test_feature_schema.py -v
"""
from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Insert the python/ directory on path so imports resolve without install
# ---------------------------------------------------------------------------
import sys
import pathlib

_HERE = pathlib.Path(__file__).parent
_PYTHON_DIR = _HERE.parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from feature_schema import (  # noqa: E402
    FEATURE_SCHEMA,
    FLAT_COLS,
    FLAT_DENSE_COLS,
    FLAT_SPARSE_COLS,
    DROP_COLS,
    KEY_COLS,
    LABEL_COL,
    BOOKKEEP_COLS,
    SEQUENCE_CHANNELS,
    N_SEQ_CHANNELS,
    TOTAL_COLS,
    FLAT_COUNT,
    PHANTOM_COUNT,
    assert_schema_matches_db,
)

# ---------------------------------------------------------------------------
# Path to staging.db -- resolved relative to repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = _PYTHON_DIR.parent
_DB_PATH = _REPO_ROOT / "data" / "data" / "staging" / "staging.db"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSchemaCounts:
    """Hard count assertions -- fail immediately if schema was edited."""

    def test_total_col_count(self) -> None:
        assert len(FEATURE_SCHEMA) == 133, (
            f"Expected 133 total columns, got {len(FEATURE_SCHEMA)}"
        )

    def test_total_cols_constant_matches(self) -> None:
        assert TOTAL_COLS == 133

    def test_phantom_count(self) -> None:
        assert len(DROP_COLS) == 6, (
            f"Expected 6 phantom columns, got {len(DROP_COLS)}: {DROP_COLS}"
        )

    def test_phantom_count_constant_matches(self) -> None:
        assert PHANTOM_COUNT == 6

    def test_flat_count(self) -> None:
        assert FLAT_COUNT == 105, (
            f"Expected 105 flat columns, got {FLAT_COUNT}"
        )

    def test_key_count(self) -> None:
        assert len(KEY_COLS) == 6, (
            f"Expected 6 key columns, got {len(KEY_COLS)}: {KEY_COLS}"
        )

    def test_label_is_single(self) -> None:
        labels = [c for c in FEATURE_SCHEMA if c.role == "label"]
        assert len(labels) == 1
        assert labels[0].name == LABEL_COL

    def test_bookkeep_count(self) -> None:
        assert len(BOOKKEEP_COLS) == 15, (
            f"Expected 15 bookkeeping columns, got {len(BOOKKEEP_COLS)}"
        )

    def test_role_partition_sums_to_total(self) -> None:
        n = (
            len(KEY_COLS)
            + 1  # label
            + FLAT_COUNT
            + PHANTOM_COUNT
            + len(BOOKKEEP_COLS)
        )
        assert n == 133, f"Role partition sums to {n}, expected 133"

    def test_dense_plus_sparse_equals_flat(self) -> None:
        assert len(FLAT_DENSE_COLS) + len(FLAT_SPARSE_COLS) == FLAT_COUNT

    def test_sequence_channel_count(self) -> None:
        assert N_SEQ_CHANNELS == 22
        assert len(SEQUENCE_CHANNELS) == 22


class TestSchemaIntegrity:
    """Structural integrity of the schema definition."""

    def test_no_duplicate_names(self) -> None:
        names = [c.name for c in FEATURE_SCHEMA]
        seen: set[str] = set()
        dupes = [n for n in names if n in seen or seen.add(n)]  # type: ignore[func-returns-value]
        assert dupes == [], f"Duplicate column names: {dupes}"

    def test_flat_columns_have_valid_null_policy(self) -> None:
        bad = [
            c.name for c in FEATURE_SCHEMA
            if c.role == "flat" and c.null_policy not in ("dense", "sparse")
        ]
        assert bad == [], f"Flat cols with invalid null_policy: {bad}"

    def test_phantom_cols_have_phantom_null_policy(self) -> None:
        bad = [
            c.name for c in FEATURE_SCHEMA
            if c.role == "drop" and c.null_policy != "phantom"
        ]
        assert bad == [], f"Drop cols not marked phantom: {bad}"

    def test_known_phantoms_present(self) -> None:
        expected = {
            "dimming_area",
            "dimming_asymmetry",
            "eit_wave_speed_kms",
            "rc_bz_min",
            "phase8_pred_transit_hours",
            "pinn_v1_pred_transit_hours",
        }
        actual = set(DROP_COLS)
        assert actual == expected, (
            f"Phantom mismatch.\n  Expected: {sorted(expected)}"
            f"\n  Got:      {sorted(actual)}"
        )

    def test_label_col_is_float32(self) -> None:
        label_specs = [c for c in FEATURE_SCHEMA if c.role == "label"]
        assert label_specs[0].dtype == "float32"

    def test_flat_cols_are_float32_or_int32(self) -> None:
        bad = [
            c.name for c in FEATURE_SCHEMA
            if c.role == "flat" and c.dtype not in ("float32", "int32")
        ]
        assert bad == [], f"Flat cols with non-numeric dtype: {bad}"

    def test_tier_values_valid(self) -> None:
        bad = [c.name for c in FEATURE_SCHEMA if c.tier not in (0, 1, 2)]
        assert bad == [], f"Invalid tier values: {bad}"

    def test_tier2_flat_cols_count(self) -> None:
        t2 = [c for c in FEATURE_SCHEMA if c.role == "flat" and c.tier == 2]
        assert len(t2) == 6, (
            f"Expected 6 Tier-2 flat cols (has_mpc, mpc_delay, "
            f"stereo x4), got {len(t2)}"
        )

    def test_sequence_channels_no_duplicates(self) -> None:
        assert len(SEQUENCE_CHANNELS) == len(set(SEQUENCE_CHANNELS))

    def test_omni_channels_not_in_flat(self) -> None:
        """OMNI sequence channels should not appear as flat column names."""
        flat_set = set(FLAT_COLS)
        # Only the aggregate scalars (omni_24h_*, omni_48h_*, omni_150h_*)
        # belong in flat; the raw channel names (bz_gsm, speed, ...) must not.
        raw_omni = {"bz_gsm", "speed", "density", "pressure", "temperature"}
        overlap = raw_omni & flat_set
        assert overlap == set(), (
            f"Raw OMNI channel names found in flat cols: {overlap}"
        )


class TestDbSchemaAssertion:
    """assert_schema_matches_db() tests against live and synthetic DBs."""

    @pytest.mark.skipif(
        not _DB_PATH.exists(),
        reason=f"staging.db not found at {_DB_PATH}",
    )
    def test_passes_against_live_db(self) -> None:
        """Should not raise against the real staging.db."""
        assert_schema_matches_db(str(_DB_PATH))

    def test_raises_on_missing_column(self) -> None:
        """Drop a column from the VIEW -- should trigger RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            conn = sqlite3.connect(db_path)
            # Build a minimal view with one column missing (activity_id absent)
            cols_without_first = [
                c.name for c in FEATURE_SCHEMA if c.name != "activity_id"
            ]
            col_defs = ", ".join(f"{n} TEXT" for n in cols_without_first)
            conn.execute(f"CREATE TABLE training_features ({col_defs})")
            conn.close()
            with pytest.raises(RuntimeError, match="schema drift"):
                assert_schema_matches_db(db_path)

    def test_raises_on_extra_column(self) -> None:
        """Add a spurious column to the DB -- should trigger RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            conn = sqlite3.connect(db_path)
            all_names = [c.name for c in FEATURE_SCHEMA]
            col_defs = ", ".join(f"{n} TEXT" for n in all_names)
            # Add one extra column
            col_defs += ", spurious_extra_col TEXT"
            conn.execute(f"CREATE TABLE training_features ({col_defs})")
            conn.close()
            with pytest.raises(RuntimeError, match="schema drift"):
                assert_schema_matches_db(db_path)

    def test_passes_on_exact_match(self) -> None:
        """A DB with exactly the right columns should pass silently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            conn = sqlite3.connect(db_path)
            all_names = [c.name for c in FEATURE_SCHEMA]
            col_defs = ", ".join(f"{n} TEXT" for n in all_names)
            conn.execute(f"CREATE TABLE training_features ({col_defs})")
            conn.close()
            # Must not raise
            assert_schema_matches_db(db_path)
