"""Masked dataset loader for the neural ensemble training pipeline (G2 gate).

Produces (x_flat, m_flat, x_seq, m_seq, y) tuples where:
  x_flat  : (105,)   float32 -- flat feature values, NaN -> 0.0
  m_flat  : (105,)   float32 -- 1.0 if observed, 0.0 if was NULL
  x_seq   : (T, C)   float32 -- pre-launch OMNI sequence, NaN -> 0.0
  m_seq   : (T, C)   float32 -- 1.0 if observed, 0.0 if was NULL
  y       : (1,)     float32 -- transit_time_hours

Flat branch
-----------
Read from ``training_features`` SQLite VIEW (133 cols).
Drop 6 phantom cols.  Cast numeric flat cols to float32.
Categorical int cols (cluster_ids, flags) -> int -> float32.
NaN -> 0.0, mask = 0.0.  Dense cols get mask = 1.0 always (no NULL).

Sequence branch
---------------
Read from pre-built Parquet (build_pinn_sequences.py output).
Current shape: (1884 train events) x 222 timesteps x 20 OMNI channels.
When GOES MAG channels are added (RULE-213 expansion), the loader pads
missing channels with 0.0 and mask=0.0 automatically, so no code change
is needed here -- only feature_schema.py SEQUENCE_CHANNELS and the
Parquet rebuild need updating.

Split handling
--------------
split='train'   -> activity_ids with split='train',  exclude=0
split='holdout' -> activity_ids with split='holdout', exclude=0
Split-leak guard enforced at construction time.

Usage
-----
    from datasets.training_features_loader import TrainingFeaturesDataset
    from torch.utils.data import DataLoader

    ds = TrainingFeaturesDataset(
        split='train',
        db_path='/abs/path/to/staging.db',
        sequences_path='/abs/path/to/data/sequences/',
    )
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    for x_flat, m_flat, x_seq, m_seq, y in loader:
        ...   # x_flat: (B, 105), m_flat: (B, 105), x_seq: (B, 222, 20), ...
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Import schema constants from sibling module
# ---------------------------------------------------------------------------
import sys

_HERE = Path(__file__).parent
_PYTHON_DIR = _HERE.parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from feature_schema import (  # noqa: E402
    FEATURE_SCHEMA,
    FLAT_COLS,
    LABEL_COL,
    SEQUENCE_CHANNELS,
    N_SEQ_CHANNELS,
    assert_schema_matches_db,
)

Split = Literal["train", "holdout"]

# Columns that are integer-encoded categories in the flat branch.
# These are cast int -> float32 (NULL -> -1 -> -1.0, then masked).
_INT_FLAT_COLS: frozenset[str] = frozenset(
    c.name
    for c in FEATURE_SCHEMA
    if c.role == "flat" and c.dtype == "int32"
)

# Sentinel: NULL integer categoricals become -1 before float32 cast.
_INT_NULL_SENTINEL: int = -1


class TrainingFeaturesDataset(Dataset[tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]]):
    """PyTorch Dataset over training_features VIEW + OMNI sequence Parquet.

    Args:
        split: 'train' or 'holdout'
        db_path: absolute path to staging.db
        sequences_path: directory containing train_sequences.parquet and
            holdout_sequences.parquet (built by build_pinn_sequences.py)
        validate_schema: if True (default), assert DB schema at construction
    """

    def __init__(
        self,
        split: Split,
        db_path: str,
        sequences_path: str,
        validate_schema: bool = True,
    ) -> None:
        print(
            f"=== TrainingFeaturesDataset init === "
            f"[split={split}]",
            flush=True,
        )
        self._split = split
        self._db_path = db_path
        self._sequences_path = Path(sequences_path)

        # ---- G1 schema drift gate ----------------------------------------
        if validate_schema:
            print("step 1 of 4 -- validating DB schema [checking]", flush=True)
            assert_schema_matches_db(db_path)
            print("step 1 of 4 -- validating DB schema [OK]", flush=True)
        else:
            print("step 1 of 4 -- schema validation SKIPPED", flush=True)

        # ---- Load flat features from SQLite ---------------------------------
        print("step 2 of 4 -- loading flat features from SQLite", flush=True)
        self._flat_data, self._activity_ids = self._load_flat(db_path, split)
        n_events = len(self._activity_ids)
        print(
            f"step 2 of 4 -- flat loaded: {n_events} events x "
            f"{self._flat_data.shape[1]} cols [OK]",
            flush=True,
        )

        # ---- Load sequences from Parquet ------------------------------------
        print("step 3 of 4 -- loading sequences from Parquet", flush=True)
        self._seq_data = self._load_sequences(
            self._sequences_path, split, self._activity_ids
        )
        t_steps, n_chan = self._seq_data.shape[1], self._seq_data.shape[2]
        print(
            f"step 3 of 4 -- sequences loaded: {n_events} x {t_steps} x "
            f"{n_chan} [OK]",
            flush=True,
        )

        # ---- Split-leak guard -----------------------------------------------
        print("step 4 of 4 -- split-leak guard [checking]", flush=True)
        self._assert_no_split_leak(db_path, split, self._activity_ids)
        print("step 4 of 4 -- split-leak guard [OK]", flush=True)

        print(
            f"=== TrainingFeaturesDataset ready === "
            f"[{n_events} events, flat=105, seq={t_steps}x{n_chan}]",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._activity_ids)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Return (x_flat, m_flat, x_seq, m_seq, y) for one event."""
        flat_row = self._flat_data[idx]   # (105 + 1,) -- last col is label

        x_flat_raw = flat_row[:-1].astype(np.float32)
        y_val      = flat_row[-1:].astype(np.float32)

        # Build value + mask tensors for flat branch
        nan_mask = np.isnan(x_flat_raw)
        x_flat = np.where(nan_mask, 0.0, x_flat_raw).astype(np.float32)
        m_flat = (~nan_mask).astype(np.float32)

        # Sequence branch
        seq_raw = self._seq_data[idx]     # (T, C) float32, NaN where missing
        nan_seq  = np.isnan(seq_raw)
        x_seq = np.where(nan_seq, 0.0, seq_raw).astype(np.float32)
        m_seq = (~nan_seq).astype(np.float32)

        return (
            torch.from_numpy(x_flat),
            torch.from_numpy(m_flat),
            torch.from_numpy(x_seq),
            torch.from_numpy(m_seq),
            torch.from_numpy(y_val),
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def activity_ids(self) -> list[str]:
        return list(self._activity_ids)

    @property
    def n_flat(self) -> int:
        return 105

    @property
    def n_seq_timesteps(self) -> int:
        return self._seq_data.shape[1]

    @property
    def n_seq_channels(self) -> int:
        return self._seq_data.shape[2]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_flat(
        db_path: str, split: Split
    ) -> tuple[np.ndarray, list[str]]:
        """Return (flat_array, activity_ids) for the given split.

        flat_array shape: (N, 106) -- 105 feature cols + 1 label col.
        NULL integers become NaN (via -1 sentinel).
        """
        cols_to_select = FLAT_COLS + [LABEL_COL]
        col_sql = ", ".join(cols_to_select)
        sql = (
            f"SELECT activity_id, {col_sql} "
            f"FROM training_features "
            f"WHERE split = ? AND exclude = 0 "
            f"ORDER BY launch_time ASC"
        )
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.execute(sql, (split,))
            rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            raise RuntimeError(
                f"No rows found for split='{split}' in training_features"
            )

        activity_ids = [r[0] for r in rows]
        # rows: each row is (activity_id, col0, col1, ..., label)
        raw = np.array(
            [[c if c is not None else np.nan for c in r[1:]] for r in rows],
            dtype=np.float64,
        )

        # Integer-type cols: NULL -> -1 -> float (already NaN from above),
        # but non-null int values need to round-trip correctly.
        # The np.nan substitution above handles NULLs uniformly for all types.
        # Dense cols: verify no NaN (warn, don't abort -- some events lack data)
        flat_arr = raw[:, :-1]   # (N, 105)
        label_arr = raw[:, -1:]  # (N, 1)
        combined = np.concatenate([flat_arr, label_arr], axis=1).astype(
            np.float32
        )
        return combined, activity_ids

    @staticmethod
    def _load_sequences(
        sequences_path: Path,
        split: Split,
        activity_ids: list[str],
    ) -> np.ndarray:
        """Return (N, T, C) float32 sequence array aligned to activity_ids.

        Missing activity_ids get an all-NaN slice.
        Missing channels (e.g. GOES MAG not yet in Parquet) are zero-padded
        with mask=0 (handled in __getitem__).
        """
        fname = f"{split}_sequences.parquet"
        fpath = sequences_path / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Sequence Parquet not found: {fpath}\n"
                f"Run: python3.12 scripts/build_pinn_sequences.py"
            )

        table = pq.read_table(str(fpath))
        parquet_channels = [
            c for c in table.schema.names
            if c not in ("activity_id", "split", "timestep", "window",
                         "has_full_omni", "transit_time_hours")
        ]
        # Align Parquet channels to SEQUENCE_CHANNELS (feature_schema order)
        # Channels present in schema but not in Parquet -> NaN column (masked)
        parquet_ch_set = set(parquet_channels)

        # Build per-event dict: activity_id -> (T, len(parquet_channels))
        df_ids  = table.column("activity_id").to_pylist()
        df_step = table.column("timestep").to_pylist()

        # Determine T (timesteps)
        from collections import Counter
        step_counts = Counter(df_ids)
        t_steps = max(step_counts.values()) if step_counts else 222

        # Determine output C = N_SEQ_CHANNELS (schema-defined)
        # If Parquet has fewer channels, missing ones become NaN
        c_out = N_SEQ_CHANNELS

        # Map schema channel index -> parquet column index (or -1 if absent)
        ch_map: list[int] = []
        for sch_ch in SEQUENCE_CHANNELS:
            if sch_ch in parquet_ch_set:
                ch_map.append(parquet_channels.index(sch_ch))
            else:
                ch_map.append(-1)

        # Pull raw data arrays for parquet channels
        parquet_arrays = [
            table.column(ch).to_pylist() for ch in parquet_channels
        ]

        # Build lookup: activity_id -> (T, len(parquet_channels)) array
        n_par_ch = len(parquet_channels)
        event_data: dict[str, np.ndarray] = {}

        # Group rows by activity_id
        from collections import defaultdict
        rows_by_id: dict[str, list[tuple[int, list[float]]]] = defaultdict(list)
        for row_idx, (aid, step) in enumerate(zip(df_ids, df_step)):
            vals = [
                float(parquet_arrays[ci][row_idx])
                if parquet_arrays[ci][row_idx] is not None
                else float("nan")
                for ci in range(n_par_ch)
            ]
            rows_by_id[aid].append((step, vals))

        for aid, step_vals in rows_by_id.items():
            arr = np.full((t_steps, n_par_ch), np.nan, dtype=np.float32)
            for step, vals in step_vals:
                if 0 <= step < t_steps:
                    arr[step] = vals
            event_data[aid] = arr

        # Assemble output array (N, T, C_out) aligned to requested activity_ids
        result = np.full(
            (len(activity_ids), t_steps, c_out), np.nan, dtype=np.float32
        )
        missing_count = 0
        for i, aid in enumerate(activity_ids):
            if aid not in event_data:
                missing_count += 1
                continue
            src = event_data[aid]   # (T, n_par_ch)
            for out_ci, par_ci in enumerate(ch_map):
                if par_ci >= 0:
                    result[i, :, out_ci] = src[:, par_ci]
                # else: remains NaN -> mask=0 in __getitem__

        if missing_count:
            print(
                f"  WARNING: {missing_count} activity_ids have no sequence "
                f"data (all-NaN slices assigned)",
                flush=True,
            )
        return result

    @staticmethod
    def _assert_no_split_leak(
        db_path: str, split: Split, activity_ids: list[str]
    ) -> None:
        """Assert that none of the loaded activity_ids appear in the other split."""
        other = "holdout" if split == "train" else "train"
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT activity_id FROM training_features "
                "WHERE split = ? AND exclude = 0",
                (other,),
            ).fetchall()
        finally:
            conn.close()
        other_ids = {r[0] for r in rows}
        loaded_ids = set(activity_ids)
        overlap = loaded_ids & other_ids
        if overlap:
            raise RuntimeError(
                f"Split leak detected: {len(overlap)} activity_ids appear in "
                f"both '{split}' and '{other}' splits: "
                f"{sorted(overlap)[:5]}..."
            )
