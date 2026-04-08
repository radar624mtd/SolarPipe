"""Task 6.5 — ENLIL Ensemble Parquet Export.

Writes the synthetic ENLIL ensemble to:
    data/output/enlil_runs/enlil_ensemble_v1.parquet

Requirements (ADR-D002):
  - PyArrow writer
  - Row groups ≤ 64 MB (ParquetSharp optimal read granularity)
  - File-level metadata: generation timestamp, seed, parameter distributions
  - Column names must match C# ParquetProvider expectations

Row group size is computed dynamically based on estimated bytes-per-row.
The target is 64 MB per row group (67,108,864 bytes).

No other pipeline data uses Parquet — only this file.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..synthetic.enlil_emulator import EmulatorConfig, EmulatorResult, generation_metadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROW_GROUP_TARGET_BYTES: int = 64 * 1024 * 1024  # 64 MB
_BYTES_PER_ROW_ESTIMATE: int = 19 * 8 + 50       # 19 float/int cols + event_id overhead

# ---------------------------------------------------------------------------
# Arrow schema
# ---------------------------------------------------------------------------

_PARQUET_SCHEMA = pa.schema([
    pa.field("event_id",            pa.string()),
    pa.field("member_id",           pa.int64()),
    pa.field("seed",                pa.int64()),
    pa.field("speed_initial",       pa.float64()),
    pa.field("speed_arrival",       pa.float64()),
    pa.field("ambient_wind",        pa.float64()),
    pa.field("transit_hours",       pa.float64()),
    pa.field("gamma",               pa.float64()),
    pa.field("latitude_deg",        pa.float64()),
    pa.field("longitude_deg",       pa.float64()),
    pa.field("axis_angle_deg",      pa.float64()),
    pa.field("angular_width",       pa.float64()),
    pa.field("hcs_deflection",      pa.float64()),
    pa.field("ch_deflection",       pa.float64()),
    pa.field("bias_correction",     pa.float64()),
    pa.field("noise_sigma",         pa.float64()),
    pa.field("quality_flag",        pa.int64()),
    pa.field("flare_class_numeric", pa.float64()),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _results_to_table(results: list[EmulatorResult]) -> pa.Table:
    """Concatenate all EmulatorResult dicts into a single Arrow table."""
    if not results:
        return pa.table({f.name: pa.array([], type=f.type) for f in _PARQUET_SCHEMA}, schema=_PARQUET_SCHEMA)

    columns: dict[str, list] = {f.name: [] for f in _PARQUET_SCHEMA}

    for result in results:
        d = result.as_dict()
        for col in columns:
            arr = d[col]
            if isinstance(arr, np.ndarray):
                columns[col].append(arr)
            else:
                columns[col].append(np.array(arr))

    arrays = {}
    for col, chunks in columns.items():
        flat = np.concatenate(chunks) if chunks else np.array([])
        field_type = _PARQUET_SCHEMA.field(col).type
        if field_type == pa.string():
            arrays[col] = pa.array(flat.tolist(), type=pa.string())
        elif field_type == pa.int64():
            arrays[col] = pa.array(flat.astype(np.int64), type=pa.int64())
        else:
            arrays[col] = pa.array(flat.astype(np.float64), type=pa.float64())

    return pa.table(arrays, schema=_PARQUET_SCHEMA)


def _compute_row_group_size(n_total_rows: int) -> int:
    """Compute row group size to stay at or below 64 MB."""
    rows_per_group = _ROW_GROUP_TARGET_BYTES // _BYTES_PER_ROW_ESTIMATE
    # At least 1,000 rows per group; at most n_total_rows
    return int(max(1000, min(rows_per_group, n_total_rows)))


# ---------------------------------------------------------------------------
# Public write function
# ---------------------------------------------------------------------------

def write_parquet(
    results: list[EmulatorResult],
    output_path: str,
    config: EmulatorConfig | None = None,
) -> int:
    """Write ensemble results to Parquet.

    Args:
        results: List of EmulatorResult from the emulator.
        output_path: Destination path for the Parquet file.
        config: EmulatorConfig for metadata. Optional.

    Returns:
        Total number of rows written.
    """
    if not results:
        logger.warning("write_parquet: no results to write — skipping")
        return 0

    cfg = config or EmulatorConfig()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    table = _results_to_table(results)
    n_rows = len(table)

    row_group_size = _compute_row_group_size(n_rows)

    metadata = generation_metadata(cfg, n_events=len(results))
    # Metadata must be attached to the schema (PyArrow 23+)
    metadata_bytes = {k.encode(): v.encode() for k, v in metadata.items()}
    schema_with_meta = table.schema.with_metadata(metadata_bytes)
    table = table.cast(schema_with_meta)

    # Atomic write: write to temp then rename
    tmp_path = output_path + ".tmp"
    try:
        pq.write_table(
            table,
            tmp_path,
            row_group_size=row_group_size,
            compression="snappy",
            write_statistics=True,
        )
        # Atomic rename (works on same filesystem)
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(tmp_path, output_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    n_row_groups = -(-n_rows // row_group_size)  # ceiling division
    logger.info(
        "parquet_export: wrote %d rows, %d row groups → %s",
        n_rows, n_row_groups, output_path,
    )
    return n_rows


# ---------------------------------------------------------------------------
# Full pipeline: DB → emulate → write
# ---------------------------------------------------------------------------

def build_parquet_from_db(
    staging_db_path: str,
    output_path: str,
    config: EmulatorConfig | None = None,
    min_quality: int = 3,
) -> int:
    """End-to-end: load feature_vectors, emulate, write Parquet.

    Args:
        staging_db_path: Path to staging.db.
        output_path: Path for output Parquet file.
        config: EmulatorConfig.
        min_quality: Minimum quality_flag.

    Returns:
        Number of rows written (0 if no events).
    """
    from ..synthetic.enlil_emulator import emulate_from_db

    cfg = config or EmulatorConfig()
    results = emulate_from_db(staging_db_path, config=cfg, min_quality=min_quality)

    if not results:
        logger.warning("build_parquet_from_db: no events emulated — output not written")
        return 0

    return write_parquet(results, output_path, config=cfg)
