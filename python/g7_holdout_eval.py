"""G7 holdout evaluation: run tft_pinn_aa409d9d on 90-event holdout, compute MAE.

Usage:
    python g7_holdout_eval.py --model-id tft_pinn_aa409d9d
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.ipc as pa_ipc


def main(model_id: str, model_base: str, seq_base: str, db_path: str, out_dir: str) -> None:
    print(f"=== g7_holdout_eval start === model_id={model_id}", flush=True)

    onnx_path = str(Path(model_base) / model_id / "model_matmul_reduce.onnx")
    meta_path = Path(model_base) / model_id / "meta.json"

    print("step 1 of 4 — loading model meta + ONNX", flush=True)
    meta = json.loads(meta_path.read_text())
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    ep = sess.get_providers()[0]
    print(f"  ORT provider: {ep}", flush=True)

    print("step 2 of 4 — loading holdout flat features from staging.db", flush=True)
    x_flat_np, m_flat_np, y_true, aids = _load_holdout_flat(db_path, meta)
    N = len(y_true)
    print(f"  holdout events: {N}", flush=True)

    print("step 3 of 4 — loading holdout sequences from Parquet", flush=True)
    x_seq_np, m_seq_np = _load_holdout_sequences(seq_base, aids)
    print(f"  sequences shape: {x_seq_np.shape}", flush=True)

    print("step 4 of 4 — running inference + computing MAE", flush=True)
    p10_all, p50_all, p90_all = [], [], []
    batch = 16
    for i in range(0, N, batch):
        sl = slice(i, i + batch)
        out = sess.run(
            ["p10", "p50", "p90"],
            {
                "x_flat": x_flat_np[sl],
                "m_flat": m_flat_np[sl],
                "x_seq":  x_seq_np[sl],
                "m_seq":  m_seq_np[sl],
            },
        )
        p10_all.append(out[0].flatten())
        p50_all.append(out[1].flatten())
        p90_all.append(out[2].flatten())
        print(f"step 4 of 4 — inference batch {i//batch+1} [OK]", flush=True)

    p10 = np.concatenate(p10_all)
    p50 = np.concatenate(p50_all)
    p90 = np.concatenate(p90_all)
    y = np.array(y_true)

    mae  = float(np.abs(p50 - y).mean())
    rmse = float(np.sqrt(np.mean((p50 - y) ** 2)))
    bias = float((p50 - y).mean())
    coverage = float(np.mean((y >= p10) & (y <= p90)))

    result = {
        "model_id": model_id,
        "split": "holdout",
        "n_events": N,
        "mae_hours": mae,
        "rmse_hours": rmse,
        "bias_hours": bias,
        "p10_p90_coverage": coverage,
        "ort_provider": ep,
        "baseline_g6_mae": 30.018800735473633,
        "g7_gate_pass": mae <= 6.0,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = Path(out_dir) / "g7_holdout_eval.json"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\n{'='*60}", flush=True)
    print(f"  G7 HOLDOUT EVAL RESULT", flush=True)
    print(f"  model_id : {model_id}", flush=True)
    print(f"  MAE      : {mae:.3f} h  (gate: ≤ 6.0 h)", flush=True)
    print(f"  RMSE     : {rmse:.3f} h", flush=True)
    print(f"  Bias     : {bias:.3f} h", flush=True)
    print(f"  P10-P90 coverage: {coverage:.1%}  (ideal: 80%)", flush=True)
    print(f"  G7 gate  : {'✅ PASS' if result['g7_gate_pass'] else '❌ FAIL'}", flush=True)
    print(f"  vs G6 baseline: {30.018800735473633:.2f} h → {mae:.2f} h  ({(30.018800735473633-mae)/30.018800735473633*100:.1f}% improvement)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"\n=== g7_holdout_eval done → {out_path} ===", flush=True)


def _load_holdout_flat(db_path: str, meta: dict) -> tuple:
    import sqlite3
    from feature_schema import FLAT_COLS, FLAT_SPARSE_COLS

    conn = sqlite3.connect(db_path)
    cols_sql = ", ".join(f'"{c}"' for c in FLAT_COLS + ["transit_time_hours", "activity_id"])
    rows = conn.execute(
        f"SELECT {cols_sql} FROM training_features WHERE split = 'holdout' AND exclude = 0"
    ).fetchall()
    conn.close()

    col_names = FLAT_COLS + ["transit_time_hours", "activity_id"]
    n_flat = len(FLAT_COLS)
    x_list, m_list, y_list, aid_list = [], [], [], []

    feat_mean = np.array(meta.get("feat_mean", [0.0] * n_flat), dtype=np.float32)
    feat_std  = np.array(meta.get("feat_std",  [1.0] * n_flat), dtype=np.float32)

    for row in rows:
        vals = row[:n_flat]
        transit = row[n_flat]
        aid = str(row[n_flat + 1])
        if transit is None or math.isnan(float(transit)):
            continue

        x = np.zeros(n_flat, dtype=np.float32)
        m = np.zeros(n_flat, dtype=np.float32)
        for j, v in enumerate(vals):
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                fv = float(v)
                if abs(fv) < 9990:
                    x[j] = fv
                    m[j] = 1.0

        # Standardize using training stats from meta
        x = (x - feat_mean) / np.where(feat_std > 1e-8, feat_std, 1.0)
        x_list.append(x)
        m_list.append(m)
        y_list.append(float(transit))
        aid_list.append(aid)

    return (
        np.stack(x_list).astype(np.float32),
        np.stack(m_list).astype(np.float32),
        y_list,
        aid_list,
    )


def _load_holdout_sequences(seq_base: str, aids: list[str]) -> tuple:
    from feature_schema import SEQUENCE_CHANNELS as SEQ_CHANNELS, N_SEQ_CHANNELS

    seq_file = str(Path(seq_base) / "holdout_sequences.parquet")
    tbl = pq.read_table(seq_file)

    aid_col      = tbl.column("activity_id").to_pylist()
    window_col   = tbl.column("window").to_pylist()
    timestep_col = tbl.column("timestep").to_pylist()

    # Determine T from pre_launch timestep range
    pre_ts = [int(timestep_col[i]) for i in range(tbl.num_rows) if window_col[i] == "pre_launch"]
    t_min, t_max = min(pre_ts), max(pre_ts)
    T = t_max - t_min + 1
    C = N_SEQ_CHANNELS

    aid_index = {aid: i for i, aid in enumerate(aids)}
    N = len(aids)
    x_np = np.zeros((N, T, C), dtype=np.float32)
    m_np = np.zeros((N, T, C), dtype=np.float32)

    schema_names = set(tbl.schema.names)
    ch_arrays: dict = {}
    for ch in SEQ_CHANNELS:
        ch_arrays[ch] = tbl.column(ch).to_pylist() if ch in schema_names else None

    for row_i in range(tbl.num_rows):
        if window_col[row_i] != "pre_launch":
            continue
        aid = aid_col[row_i]
        if aid not in aid_index:
            continue
        n_idx = aid_index[aid]
        t_idx = int(timestep_col[row_i]) - t_min
        if t_idx < 0 or t_idx >= T:
            continue
        for c_idx, ch in enumerate(SEQ_CHANNELS):
            arr = ch_arrays[ch]
            if arr is None:
                continue
            v = arr[row_i]
            if v is None:
                continue
            fv = float(v)
            if fv != fv:
                continue
            x_np[n_idx, t_idx, c_idx] = fv
            m_np[n_idx, t_idx, c_idx] = 1.0

    print(f"  sequences loaded: shape={x_np.shape}, obs_rate={m_np.mean():.2%}", flush=True)
    return x_np, m_np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="tft_pinn_aa409d9d")
    parser.add_argument("--model-base", default="C:/Users/radar/SolarPipe/models")
    parser.add_argument("--seq-base", default="C:/Users/radar/SolarPipe/data/sequences")
    parser.add_argument("--db", default="C:/Users/radar/SolarPipe/data/data/staging/staging.db")
    parser.add_argument("--out-dir", default="C:/Users/radar/SolarPipe/output/neural_ensemble_v1")
    args = parser.parse_args()
    main(args.model_id, args.model_base, args.seq_base, args.db, args.out_dir)
