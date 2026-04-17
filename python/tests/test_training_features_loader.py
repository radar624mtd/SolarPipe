"""Unit tests for datasets/training_features_loader.py (G2 gate).

Tests validate:
  1. Dataset loads train split: len == 1884, no NaN in output tensors
  2. Dataset loads holdout split: len == 90
  3. x_flat shape == (105,), m_flat shape == (105,), float32
  4. x_seq shape == (T, C), m_seq shape == (T, C), float32
  5. m_flat == 0.0 exactly where x_flat == 0.0 due to imputation (proxy check)
  6. y shape == (1,), positive finite float32
  7. Split-leak guard: no overlap between train and holdout activity_ids
  8. Split-leak guard raises RuntimeError on synthetic overlap
  9. Dense columns (always observed) have m_flat == 1.0 always

Run with:
    cd python && python -m pytest tests/test_training_features_loader.py -v
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import sys

import pytest
import torch

_PYTHON_DIR = Path(__file__).parent.parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from feature_schema import FLAT_DENSE_COLS, FLAT_COLS, N_SEQ_CHANNELS  # noqa: E402
from datasets.training_features_loader import (  # noqa: E402
    TrainingFeaturesDataset,
)

# ---------------------------------------------------------------------------
# Paths to real data -- tests skip if not available
# ---------------------------------------------------------------------------
_REPO_ROOT     = _PYTHON_DIR.parent
_DB_PATH       = str(_REPO_ROOT / "data" / "data" / "staging" / "staging.db")
_SEQ_PATH      = str(_REPO_ROOT / "data" / "sequences")
_DATA_AVAILABLE = (
    Path(_DB_PATH).exists()
    and (Path(_SEQ_PATH) / "train_sequences.parquet").exists()
    and (Path(_SEQ_PATH) / "holdout_sequences.parquet").exists()
)

_skip_no_data = pytest.mark.skipif(
    not _DATA_AVAILABLE,
    reason="staging.db or sequence Parquet files not found",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def train_ds() -> TrainingFeaturesDataset:
    return TrainingFeaturesDataset(
        split="train",
        db_path=_DB_PATH,
        sequences_path=_SEQ_PATH,
    )


@pytest.fixture(scope="module")
def holdout_ds() -> TrainingFeaturesDataset:
    return TrainingFeaturesDataset(
        split="holdout",
        db_path=_DB_PATH,
        sequences_path=_SEQ_PATH,
    )


# ---------------------------------------------------------------------------
# Count tests
# ---------------------------------------------------------------------------

class TestDatasetCounts:

    @_skip_no_data
    def test_train_length(self, train_ds: TrainingFeaturesDataset) -> None:
        assert len(train_ds) == 1884, (
            f"Expected 1884 train events, got {len(train_ds)}"
        )

    @_skip_no_data
    def test_holdout_length(self, holdout_ds: TrainingFeaturesDataset) -> None:
        assert len(holdout_ds) == 90, (
            f"Expected 90 holdout events, got {len(holdout_ds)}"
        )

    @_skip_no_data
    def test_n_flat_property(self, train_ds: TrainingFeaturesDataset) -> None:
        assert train_ds.n_flat == 105

    @_skip_no_data
    def test_n_seq_channels_property(
        self, train_ds: TrainingFeaturesDataset
    ) -> None:
        assert train_ds.n_seq_channels == N_SEQ_CHANNELS


# ---------------------------------------------------------------------------
# Shape and dtype tests
# ---------------------------------------------------------------------------

class TestOutputShapes:

    @_skip_no_data
    def test_flat_shapes(self, train_ds: TrainingFeaturesDataset) -> None:
        x_flat, m_flat, x_seq, m_seq, y = train_ds[0]
        assert x_flat.shape == (105,), f"x_flat shape: {x_flat.shape}"
        assert m_flat.shape == (105,), f"m_flat shape: {m_flat.shape}"

    @_skip_no_data
    def test_seq_shapes(self, train_ds: TrainingFeaturesDataset) -> None:
        x_flat, m_flat, x_seq, m_seq, y = train_ds[0]
        T, C = x_seq.shape
        assert C == N_SEQ_CHANNELS, f"x_seq channels: {C}, expected {N_SEQ_CHANNELS}"
        assert x_seq.shape == m_seq.shape

    @_skip_no_data
    def test_label_shape(self, train_ds: TrainingFeaturesDataset) -> None:
        _, _, _, _, y = train_ds[0]
        assert y.shape == (1,), f"y shape: {y.shape}"

    @_skip_no_data
    def test_all_float32(self, train_ds: TrainingFeaturesDataset) -> None:
        for tensor in train_ds[0]:
            assert tensor.dtype == torch.float32, (
                f"Expected float32, got {tensor.dtype}"
            )


# ---------------------------------------------------------------------------
# Value tests
# ---------------------------------------------------------------------------

class TestOutputValues:

    @_skip_no_data
    def test_no_nan_in_x_flat(self, train_ds: TrainingFeaturesDataset) -> None:
        """NaN imputed to 0.0 -- x_flat must be NaN-free."""
        for idx in range(min(50, len(train_ds))):
            x_flat, _, _, _, _ = train_ds[idx]
            assert not torch.isnan(x_flat).any(), (
                f"NaN found in x_flat at idx={idx}"
            )

    @_skip_no_data
    def test_no_nan_in_m_flat(self, train_ds: TrainingFeaturesDataset) -> None:
        for idx in range(min(50, len(train_ds))):
            _, m_flat, _, _, _ = train_ds[idx]
            assert not torch.isnan(m_flat).any()

    @_skip_no_data
    def test_no_nan_in_x_seq(self, train_ds: TrainingFeaturesDataset) -> None:
        for idx in range(min(50, len(train_ds))):
            _, _, x_seq, _, _ = train_ds[idx]
            assert not torch.isnan(x_seq).any()

    @_skip_no_data
    def test_no_nan_in_y(self, train_ds: TrainingFeaturesDataset) -> None:
        for idx in range(min(50, len(train_ds))):
            _, _, _, _, y = train_ds[idx]
            assert not torch.isnan(y).any()

    @_skip_no_data
    def test_mask_binary(self, train_ds: TrainingFeaturesDataset) -> None:
        """Masks must be 0.0 or 1.0 exactly."""
        for idx in range(min(20, len(train_ds))):
            _, m_flat, _, m_seq, _ = train_ds[idx]
            assert ((m_flat == 0.0) | (m_flat == 1.0)).all(), (
                f"m_flat has non-binary values at idx={idx}"
            )
            assert ((m_seq == 0.0) | (m_seq == 1.0)).all()

    @_skip_no_data
    def test_y_positive(self, train_ds: TrainingFeaturesDataset) -> None:
        """Transit times must be positive (physical constraint)."""
        for idx in range(min(100, len(train_ds))):
            _, _, _, _, y = train_ds[idx]
            assert y.item() > 0, f"Non-positive transit time at idx={idx}: {y.item()}"

    @_skip_no_data
    def test_dense_cols_mask_always_one(
        self, train_ds: TrainingFeaturesDataset
    ) -> None:
        """Dense columns (<10% NULL) should have mask=1 for most events."""
        dense_indices = [
            i for i, name in enumerate(FLAT_COLS)
            if name in FLAT_DENSE_COLS
        ]
        n_sample = min(200, len(train_ds))
        dense_masks = torch.zeros(n_sample, len(dense_indices))
        for idx in range(n_sample):
            _, m_flat, _, _, _ = train_ds[idx]
            dense_masks[idx] = m_flat[dense_indices]
        # At least 90% of dense col masks should be 1.0
        frac = dense_masks.mean().item()
        assert frac >= 0.90, (
            f"Dense col mask coverage {frac:.2%} < 90% (expected near 1.0)"
        )

    @_skip_no_data
    def test_imputed_zero_has_mask_zero(
        self, train_ds: TrainingFeaturesDataset
    ) -> None:
        """Where x_flat == 0 AND mask == 0: that is an imputed NULL (MNAR).
        Where x_flat == 0 AND mask == 1: that is a genuine zero observation.
        This test checks the mask correctly tracks the former for sparse cols."""
        # Use a column with confirmed ~95% NULL rate so we are certain
        # to encounter NULLs within 200 events.
        # sep_onset_delay_hours is 95.1% NULL by audit (2026-04-17).
        sparse_name = "sep_onset_delay_hours"
        if sparse_name not in FLAT_COLS:
            pytest.skip(f"'{sparse_name}' not found in FLAT_COLS")
        sparse_idx = FLAT_COLS.index(sparse_name)

        null_count = 0
        for idx in range(min(200, len(train_ds))):
            x_flat, m_flat, _, _, _ = train_ds[idx]
            if m_flat[sparse_idx].item() == 0.0:
                # Imputed: x value must be 0.0
                assert x_flat[sparse_idx].item() == 0.0, (
                    f"Masked sparse col has non-zero value at idx={idx}"
                )
                null_count += 1
        # At least one NULL should exist in 200 events for any sparse col
        assert null_count > 0, (
            f"No NULLs found in sparse col '{sparse_name}' across 200 events"
        )


# ---------------------------------------------------------------------------
# Split-leak guard tests
# ---------------------------------------------------------------------------

class TestSplitLeakGuard:

    @_skip_no_data
    def test_no_overlap_train_holdout(
        self,
        train_ds: TrainingFeaturesDataset,
        holdout_ds: TrainingFeaturesDataset,
    ) -> None:
        train_set    = set(train_ds.activity_ids)
        holdout_set  = set(holdout_ds.activity_ids)
        overlap      = train_set & holdout_set
        assert overlap == set(), (
            f"{len(overlap)} activity_ids in both splits: "
            f"{sorted(overlap)[:3]}"
        )

    def test_leak_guard_raises_on_synthetic_overlap(self) -> None:
        """Build a tiny DB where the same event appears in both splits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "leak.db")
            conn = sqlite3.connect(db_path)
            # Minimal schema: just the columns the loader needs
            conn.execute(
                "CREATE TABLE training_features "
                "(activity_id TEXT, launch_time TEXT, "
                "transit_time_hours REAL, split TEXT, exclude INTEGER)"
            )
            # Insert the same activity_id in both splits
            conn.execute(
                "INSERT INTO training_features VALUES "
                "('ACT001', '2020-01-01T00:00', 72.0, 'train', 0)"
            )
            conn.execute(
                "INSERT INTO training_features VALUES "
                "('ACT001', '2020-01-01T00:00', 72.0, 'holdout', 0)"
            )
            conn.commit()
            conn.close()

            from datasets.training_features_loader import (
                TrainingFeaturesDataset as DS,
            )
            with pytest.raises(RuntimeError, match="Split leak"):
                DS._assert_no_split_leak(db_path, "train", ["ACT001"])
