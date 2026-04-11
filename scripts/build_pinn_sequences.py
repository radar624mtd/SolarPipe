"""Extract per-event OMNI time series sequences for TFT/sequence model training.

Outputs two Parquet files under SOLARPIPE_SEQUENCES_PATH (default: data/sequences/):
  train_sequences.parquet   — 1,884 events × 150 pre-launch timesteps × 20 channels
  holdout_sequences.parquet — 90 events × 150 pre-launch timesteps × 20 channels

Each row in the Parquet represents one (event, timestep) pair:
  activity_id     TEXT   — CME event ID
  split           TEXT   — 'train' or 'holdout'
  timestep        INT16  — hours before launch: -150 to -1 (pre-launch)
                           or hours after launch: 0 to +71 (in-transit)
  window          TEXT   — 'pre_launch' or 'in_transit'
  has_full_omni   BOOL   — True if proton_density fill rate >= 80% for this event/window
  transit_time_hours FLOAT32 — label (same for all rows of same event)
  [20 OMNI channels as FLOAT32]

OMNI channels extracted (20):
  Bz_GSM, flow_speed, proton_density, flow_pressure, AE_nT, Dst_nT, Kp_x10,
  B_scalar_avg, By_GSM, electric_field, plasma_beta, alfven_mach,
  sigma_Bz, sigma_N, sigma_V, flow_longitude, flow_latitude,
  alpha_proton_ratio, F10_7_index, Bx_GSE

Sentinel handling: values >= 9990 or <= -1e29 → NaN (float32 NaN, not dropped).
Gap interpolation: linear fill for runs of <= 6 consecutive NaN within a window.
Fill rate < 80% sets has_full_omni=False for that event/window combination.

Usage:
    python scripts/build_pinn_sequences.py [--out-dir PATH] [--split train|holdout|both]
                                           [--validate-only] [--max-events N]

Prerequisites:
    pinn_training_flat must exist in staging.db.
    SOLARPIPE_SEQUENCES_PATH env var overrides --out-dir.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SOLAR_DB = Path("C:/Users/radar/SolarPipe/solar_data.db")
STAGING_DB = Path("C:/Users/radar/SolarPipe/data/data/staging/staging.db")
DEFAULT_OUT_DIR = Path("C:/Users/radar/SolarPipe/data/sequences")

# Window lengths
PRE_LAUNCH_HOURS = 150
IN_TRANSIT_HOURS = 72

# Minimum proton_density fill rate to mark has_full_omni=True
FILL_RATE_THRESHOLD = 0.80

# Sentinel threshold (OMNI uses 9999.9 and -1e31 for missing)
_SENTINEL_LARGE = 9990.0

# 20 OMNI channels (all float32)
OMNI_CHANNELS = [
    "Bz_GSM", "flow_speed", "proton_density", "flow_pressure",
    "AE_nT", "Dst_nT", "Kp_x10",
    "B_scalar_avg", "By_GSM", "electric_field", "plasma_beta", "alfven_mach",
    "sigma_Bz", "sigma_N", "sigma_V",
    "flow_longitude", "flow_latitude", "alpha_proton_ratio",
    "F10_7_index", "Bx_GSE",
]

_TZ_RE = re.compile(r"[+-]\d{2}:\d{2}$|Z$")


def _parse_dt(s: str) -> datetime | None:
    if not s:
        return None
    s2 = _TZ_RE.sub("", s.strip()).replace("T", " ").replace("Z", "")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s2.strip(), fmt)
        except ValueError:
            continue
    return None


def _dt_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


def _clean_sentinel(v: float | None) -> float:
    """Convert sentinel or None to float NaN."""
    if v is None:
        return float("nan")
    if abs(v) >= _SENTINEL_LARGE or v <= -1e29:
        return float("nan")
    return float(v)


def _interpolate_nans(arr: np.ndarray, max_gap: int = 6) -> np.ndarray:
    """Linear interpolation for runs of NaN up to max_gap in length."""
    arr = arr.copy()
    n = len(arr)
    i = 0
    while i < n:
        if math.isnan(arr[i]):
            run_start = i
            while i < n and math.isnan(arr[i]):
                i += 1
            run_len = i - run_start
            if run_len <= max_gap:
                left = arr[run_start - 1] if run_start > 0 and not math.isnan(arr[run_start - 1]) else None
                right = arr[i] if i < n and not math.isnan(arr[i]) else None
                if left is not None and right is not None:
                    for j in range(run_len):
                        frac = (j + 1) / (run_len + 1)
                        arr[run_start + j] = left + frac * (right - left)
        else:
            i += 1
    return arr


def extract_window(
    conn_solar: sqlite3.Connection,
    launch: datetime,
    hours_before: int,
    hours_after: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract OMNI sequence for a single event.

    Returns:
        pre_arr:  float32 array (hours_before, len(OMNI_CHANNELS))
        transit_arr: float32 array (hours_after, len(OMNI_CHANNELS))
        density_fill_rate: fraction of pre-launch proton_density rows that are non-NaN
    """
    # Floor launch to the nearest hour so lookup keys align to omni_hourly HH:00 rows.
    # e.g. launch=09:54 → launch_h=09:00; pre_start keys are then 03:00, 04:00, ...
    launch_h = launch.replace(minute=0, second=0, microsecond=0)
    t_pre_start = launch_h - timedelta(hours=hours_before)
    t_transit_end = launch_h + timedelta(hours=hours_after)

    col_sql = ", ".join(f'"{c}"' for c in OMNI_CHANNELS)
    rows = conn_solar.execute(
        f'SELECT datetime, {col_sql} FROM omni_hourly '
        f'WHERE datetime >= ? AND datetime < ? ORDER BY datetime',
        (_dt_str(t_pre_start), _dt_str(t_transit_end)),
    ).fetchall()

    # Build hourly lookup {floor_hour_str → values}
    lookup: dict[str, list[float]] = {}
    for row in rows:
        dt_str = row[0][:16]  # "YYYY-MM-DD HH:MM"
        lookup[dt_str] = [_clean_sentinel(row[i + 1]) for i in range(len(OMNI_CHANNELS))]

    nan_row = [float("nan")] * len(OMNI_CHANNELS)

    def _build_array(start: datetime, n_hours: int) -> np.ndarray:
        arr = np.empty((n_hours, len(OMNI_CHANNELS)), dtype=np.float32)
        for h in range(n_hours):
            t = start + timedelta(hours=h)
            key = _dt_str(t)
            vals = lookup.get(key, nan_row)
            arr[h] = vals
        # Interpolate NaN per channel
        for c in range(len(OMNI_CHANNELS)):
            arr[:, c] = _interpolate_nans(arr[:, c].astype(float)).astype(np.float32)
        return arr

    pre_arr = _build_array(t_pre_start, hours_before)
    transit_arr = _build_array(launch_h, hours_after)

    # Density fill rate on pre-launch window
    density_col_idx = OMNI_CHANNELS.index("proton_density")
    n_density = np.sum(~np.isnan(pre_arr[:, density_col_idx]))
    fill_rate = float(n_density) / hours_before

    return pre_arr, transit_arr, fill_rate


def _build_arrow_schema(window: str) -> pa.Schema:
    """Build Arrow schema for sequence rows."""
    fields = [
        pa.field("activity_id", pa.string()),
        pa.field("split", pa.string()),
        pa.field("timestep", pa.int16()),
        pa.field("window", pa.string()),
        pa.field("has_full_omni", pa.bool_()),
        pa.field("transit_time_hours", pa.float32()),
    ]
    for ch in OMNI_CHANNELS:
        fields.append(pa.field(ch, pa.float32()))
    return pa.schema(fields)


def process_events(
    conn_solar: sqlite3.Connection,
    conn_staging: sqlite3.Connection,
    split_filter: str,
    out_dir: Path,
    max_events: int | None = None,
) -> None:
    """Extract sequences for events in the given split and write Parquet."""
    if split_filter == "both":
        splits = ("train", "holdout")
    else:
        splits = (split_filter,)

    for split in splits:
        print(f"\n  Processing split='{split}'...")
        rows = conn_staging.execute(
            "SELECT activity_id, launch_time, transit_time_hours "
            "FROM pinn_training_flat WHERE split=? AND exclude=0 ORDER BY launch_time",
            (split,),
        ).fetchall()

        if max_events:
            rows = rows[:max_events]

        print(f"  Events to process: {len(rows)}")

        schema = _build_arrow_schema(split)

        # Accumulate batches; flush every 100 events
        batch_records: list[dict] = []
        writer: pq.ParquetWriter | None = None
        out_path = out_dir / f"{split}_sequences.parquet"

        n_full_omni = 0
        n_processed = 0

        for idx, (aid, launch_str, tt) in enumerate(rows):
            launch = _parse_dt(launch_str)
            if launch is None:
                continue

            try:
                pre_arr, transit_arr, fill_rate = extract_window(
                    conn_solar, launch, PRE_LAUNCH_HOURS, IN_TRANSIT_HOURS
                )
            except Exception as exc:
                print(f"  WARNING: {aid} extraction failed: {exc}")
                continue

            has_full = fill_rate >= FILL_RATE_THRESHOLD
            if has_full:
                n_full_omni += 1

            # Pre-launch rows: timestep = -150 to -1
            for h in range(PRE_LAUNCH_HOURS):
                rec = {
                    "activity_id": aid,
                    "split": split,
                    "timestep": np.int16(h - PRE_LAUNCH_HOURS),  # -150...-1
                    "window": "pre_launch",
                    "has_full_omni": has_full,
                    "transit_time_hours": np.float32(tt) if tt is not None else np.float32("nan"),
                }
                for c_idx, ch in enumerate(OMNI_CHANNELS):
                    rec[ch] = np.float32(pre_arr[h, c_idx])
                batch_records.append(rec)

            # In-transit rows: timestep = 0 to +71
            for h in range(IN_TRANSIT_HOURS):
                rec = {
                    "activity_id": aid,
                    "split": split,
                    "timestep": np.int16(h),  # 0...71
                    "window": "in_transit",
                    "has_full_omni": has_full,
                    "transit_time_hours": np.float32(tt) if tt is not None else np.float32("nan"),
                }
                for c_idx, ch in enumerate(OMNI_CHANNELS):
                    rec[ch] = np.float32(transit_arr[h, c_idx])
                batch_records.append(rec)

            n_processed += 1

            # Flush every 100 events
            if len(batch_records) >= 100 * (PRE_LAUNCH_HOURS + IN_TRANSIT_HOURS):
                table = _records_to_table(batch_records, schema)
                if writer is None:
                    writer = pq.ParquetWriter(str(out_path), schema, compression="snappy")
                writer.write_table(table)
                batch_records = []

            if (idx + 1) % 100 == 0:
                print(f"    {idx + 1}/{len(rows)} events processed...")

        # Final flush
        if batch_records:
            table = _records_to_table(batch_records, schema)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), schema, compression="snappy")
            writer.write_table(table)

        if writer is not None:
            writer.close()

        print(f"  Written: {out_path}")
        print(f"  Events processed: {n_processed}")
        print(f"  has_full_omni=True: {n_full_omni}/{n_processed} "
              f"({100*n_full_omni/max(n_processed,1):.1f}%)")
        rows_written = n_processed * (PRE_LAUNCH_HOURS + IN_TRANSIT_HOURS)
        print(f"  Total rows written: {rows_written:,} "
              f"({n_processed} events × {PRE_LAUNCH_HOURS + IN_TRANSIT_HOURS} timesteps)")


def _records_to_table(records: list[dict], schema: pa.Schema) -> pa.Table:
    """Convert list of row dicts to a pyarrow Table matching schema."""
    cols: dict[str, list] = {f.name: [] for f in schema}
    for rec in records:
        for name in cols:
            cols[name].append(rec.get(name))

    arrays = []
    for field in schema:
        col = cols[field.name]
        if field.type == pa.string():
            arrays.append(pa.array(col, type=pa.string()))
        elif field.type == pa.bool_():
            arrays.append(pa.array(col, type=pa.bool_()))
        elif field.type == pa.int16():
            arrays.append(pa.array(col, type=pa.int16()))
        elif field.type == pa.float32():
            arrays.append(pa.array(col, type=pa.float32()))
        else:
            arrays.append(pa.array(col))

    return pa.table(dict(zip([f.name for f in schema], arrays)), schema=schema)


def validate_output(out_dir: Path, split: str) -> None:
    path = out_dir / f"{split}_sequences.parquet"
    if not path.exists():
        print(f"  {path} does not exist")
        return
    tbl = pq.read_table(str(path))
    n_rows = len(tbl)
    n_events = tbl["activity_id"].unique().to_pylist()
    import pyarrow.compute as pc
    pre_mask = pc.and_(
        pc.equal(tbl["window"], "pre_launch"),
        pc.equal(tbl["timestep"], pa.scalar(-1, pa.int16())),
    )
    import pyarrow.compute as pc2  # noqa: PLC0415
    n_full = int(pc2.sum(tbl.filter(pre_mask).column("has_full_omni").cast(pa.int32())).as_py())
    print(f"\n  {split}_sequences.parquet:")
    print(f"    rows: {n_rows:,}")
    print(f"    events: {len(n_events)}")
    print(f"    schema: {[f.name for f in tbl.schema][:8]}...")
    print(f"    has_full_omni events: {n_full}/{len(n_events)}")
    print(f"    Bz_GSM dtype: {tbl.schema.field('Bz_GSM').type}")
    # Spot-check no wrong dtypes
    for ch in OMNI_CHANNELS[:3]:
        assert tbl.schema.field(ch).type == pa.float32(), f"{ch} is not float32"
    print(f"    float32 spot-check: OK")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None,
                    help="Output directory (default: $SOLARPIPE_SEQUENCES_PATH or data/sequences/)")
    ap.add_argument("--split", default="both", choices=["train", "holdout", "both"])
    ap.add_argument("--validate-only", action="store_true",
                    help="Only validate existing output files")
    ap.add_argument("--max-events", type=int, default=None,
                    help="Limit events processed (for testing)")
    args = ap.parse_args()

    out_dir = Path(
        args.out_dir
        or os.environ.get("SOLARPIPE_SEQUENCES_PATH", "")
        or str(DEFAULT_OUT_DIR)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        splits = ["train", "holdout"] if args.split == "both" else [args.split]
        for s in splits:
            validate_output(out_dir, s)
        return 0

    print("=== build_pinn_sequences ===")
    print(f"solar_db:   {SOLAR_DB}")
    print(f"staging_db: {STAGING_DB}")
    print(f"out_dir:    {out_dir}")
    print(f"channels:   {len(OMNI_CHANNELS)} OMNI channels")
    print(f"pre-launch: {PRE_LAUNCH_HOURS}h window")
    print(f"in-transit: {IN_TRANSIT_HOURS}h window")

    # Verify prereqs
    if not SOLAR_DB.exists():
        print(f"ERROR: solar_data.db not found at {SOLAR_DB}")
        return 1
    if not STAGING_DB.exists():
        print(f"ERROR: staging.db not found at {STAGING_DB}")
        return 1

    conn_solar = sqlite3.connect(str(SOLAR_DB))
    conn_staging = sqlite3.connect(str(STAGING_DB))

    try:
        n = conn_staging.execute("SELECT COUNT(*) FROM pinn_training_flat").fetchone()[0]
    except sqlite3.OperationalError:
        print("ERROR: pinn_training_flat not found — run build_pinn_feature_matrix.py first")
        return 1
    print(f"\n  pinn_training_flat: {n} events")

    process_events(conn_solar, conn_staging, args.split, out_dir, args.max_events)

    # Validate output
    splits = ["train", "holdout"] if args.split == "both" else [args.split]
    for s in splits:
        validate_output(out_dir, s)

    conn_solar.close()
    conn_staging.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
