"""Train the TFT neural ensemble for CME transit time prediction.

Reads:
  staging.db:pinn_expanded_flat     — 75-column scalar feature matrix
  data/sequences/train_sequences.parquet   — 150h pre-launch + 72h in-transit
  data/sequences/holdout_sequences.parquet — same for holdout

Existing model predictions injected as ensemble head inputs:
  output/phase8_domain_results.json    → phase8_pred_transit_hours
  output/pinn_v1/pinn_v1_results.json  → pinn_v1_pred_transit_hours

Outputs:
  output/tft_v1/tft_v1_results.json   — per-event holdout predictions + metrics
  models/registry/tft_v1_fold{N}.pt   — per-fold model weights
  models/registry/tft_v1_meta.json    — architecture + feature config

Usage:
    python scripts/train_tft_model.py [--epochs 50] [--device cpu|cuda]
                                      [--folds 5] [--validate-only]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

# Add python/ to path for tft_model import
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from tft_model import TransitTimeTFT, TFTTrainer  # noqa: E402

STAGING_DB = Path("C:/Users/radar/SolarPipe/data/data/staging/staging.db")
SEQ_DIR = Path(
    os.environ.get("SOLARPIPE_SEQUENCES_PATH", "")
    or "C:/Users/radar/SolarPipe/data/sequences"
)
OUTPUT_DIR = Path("C:/Users/radar/SolarPipe/output/tft_v1")
MODEL_DIR = Path("C:/Users/radar/SolarPipe/models/registry")

PHASE8_RESULTS = Path("C:/Users/radar/SolarPipe/output/phase8_domain_results.json")
PINN_V1_RESULTS = Path("C:/Users/radar/SolarPipe/output/pinn_v1/pinn_v1_results.json")

# ── Feature definitions ───────────────────────────────────────────────────────

# All scalar features from pinn_expanded_flat (excluding metadata and label)
STATIC_FEATURES = [
    # Original 45-col features (numeric only)
    "omni_24h_bz_mean", "omni_24h_bz_std", "omni_24h_bz_min",
    "omni_24h_speed_mean", "omni_24h_density_mean", "omni_24h_pressure_mean",
    "omni_24h_ae_max", "omni_24h_dst_min", "omni_24h_kp_max",
    "f10_7", "cluster_id_k5", "cluster_assigned",
    "preceding_cme_count_48h", "preceding_cme_speed_max", "preceding_cme_speed_mean",
    "preceding_cme_angular_sep_min", "is_multi_cme",
    "omni_48h_density_spike_max", "omni_48h_speed_gradient",
    "cme_speed_kms", "cme_half_angle_deg", "cme_latitude", "cme_longitude",
    "cme_angular_width_deg",
    "cdaw_linear_speed_kms", "cdaw_angular_width_deg",
    "cdaw_mass_log10", "cdaw_ke_log10", "cdaw_matched",
    "flare_class_numeric", "has_flare", "flare_source_longitude",
    "omni_150h_density_median", "omni_150h_speed_median",
    "sw_bz_ambient", "delta_v_kms", "usflux", "sharp_available",
    # New CDAW kinematics
    "second_order_speed_init", "second_order_speed_final",
    "second_order_speed_20Rs", "accel_kms2", "mpa_deg",
    # New SHARP magnetic
    "meangam", "meangbt", "meangbz", "meangbh",
    "meanjzd", "totusjz", "meanjzh", "totusjh", "absnjzh",
    "meanalp", "savncpp", "meanpot", "totpot",
    "meanshr", "shrgt45", "r_value", "area_acr",
    # New cluster labels
    "cluster_id_k8", "cluster_id_k12", "cluster_id_dbscan",
    # ENLIL
    "enlil_predicted_arrival_hours", "enlil_au", "enlil_matched",
]

# 20 OMNI sequence channels (must match build_pinn_sequences.py OMNI_CHANNELS)
SEQ_CHANNELS = [
    "Bz_GSM", "flow_speed", "proton_density", "flow_pressure",
    "AE_nT", "Dst_nT", "Kp_x10",
    "B_scalar_avg", "By_GSM", "electric_field", "plasma_beta", "alfven_mach",
    "sigma_Bz", "sigma_N", "sigma_V",
    "flow_longitude", "flow_latitude", "alpha_proton_ratio",
    "F10_7_index", "Bx_GSE",
]

# Existing model predictions (ensemble head inputs)
EXISTING_PRED_COLS = ["phase8_pred_transit_hours", "pinn_v1_pred_transit_hours"]

N_STATIC = len(STATIC_FEATURES)
N_SEQ_CH = len(SEQ_CHANNELS)
N_EXISTING = len(EXISTING_PRED_COLS)
PRE_LEN = 150
TRANSIT_LEN = 72


# ── Data loading ─────────────────────────────────────────────────────────────

def load_existing_predictions() -> dict[str, dict[str, float]]:
    """Load Phase 8 and PINN V1 per-event predictions. Returns {activity_id: {col: val}}."""
    preds: dict[str, dict[str, float]] = {}

    if PHASE8_RESULTS.exists():
        data = json.loads(PHASE8_RESULTS.read_text())
        for ev in data.get("events", []):
            aid = ev.get("activity_id") or ev.get("cme_id")
            val = ev.get("predicted_transit_hours") or ev.get("pred_transit_hours")
            if aid and val is not None:
                preds.setdefault(aid, {})["phase8_pred_transit_hours"] = float(val)

    if PINN_V1_RESULTS.exists():
        data = json.loads(PINN_V1_RESULTS.read_text())
        # Load OOF (training) + holdout predictions; both use "predicted" key
        for section in ("oof_predictions", "holdout_predictions"):
            for ev in data.get(section, []):
                aid = ev.get("activity_id") or ev.get("cme_id")
                val = ev.get("predicted") or ev.get("pred_transit_hours")
                if aid and val is not None:
                    preds.setdefault(aid, {})["pinn_v1_pred_transit_hours"] = float(val)

    return preds


def load_scalar_features(split: str, existing_preds: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load static features and labels from pinn_expanded_flat.

    Returns: activity_ids, features (N, S) float32, labels (N,) float32
    """
    conn = sqlite3.connect(str(STAGING_DB))
    cols_sql = ", ".join(STATIC_FEATURES + EXISTING_PRED_COLS + ["activity_id", "transit_time_hours"])
    rows = conn.execute(
        f"SELECT {cols_sql} FROM pinn_expanded_flat WHERE split=? AND exclude=0 "
        "ORDER BY launch_time",
        (split,),
    ).fetchall()
    conn.close()

    col_names = STATIC_FEATURES + EXISTING_PRED_COLS + ["activity_id", "transit_time_hours"]
    col_idx = {c: i for i, c in enumerate(col_names)}

    activity_ids = [r[col_idx["activity_id"]] for r in rows]
    labels = np.array([
        float("nan") if r[col_idx["transit_time_hours"]] is None
        else float(r[col_idx["transit_time_hours"]])
        for r in rows
    ], dtype=np.float32)

    feat_arr = np.full((len(rows), N_STATIC + N_EXISTING), float("nan"), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, col in enumerate(STATIC_FEATURES):
            v = r[col_idx[col]]
            if v is not None:
                feat_arr[i, j] = float(v)
        # Merge existing predictions from JSON if DB column is NULL
        for k, col in enumerate(EXISTING_PRED_COLS):
            v = r[col_idx[col]]
            if v is not None:
                feat_arr[i, N_STATIC + k] = float(v)
            elif activity_ids[i] in existing_preds:
                vj = existing_preds[activity_ids[i]].get(col)
                if vj is not None:
                    feat_arr[i, N_STATIC + k] = float(vj)

    static_arr = feat_arr[:, :N_STATIC]
    existing_arr = feat_arr[:, N_STATIC:]
    return activity_ids, static_arr, existing_arr, labels


def load_sequences(split: str, activity_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-launch and in-transit sequences from Parquet.

    Returns:
        pre_arr:     (N, 150, C) float32
        transit_arr: (N, 72, C) float32
        transit_mask:(N, 72) bool
    """
    path = SEQ_DIR / f"{split}_sequences.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Sequence file not found: {path}\n"
            "Run: python scripts/build_pinn_sequences.py"
        )

    tbl = pq.read_table(str(path))
    # Build event index
    aid_col = tbl["activity_id"].to_pylist()
    ts_col = tbl["timestep"].to_pylist()
    window_col = tbl["window"].to_pylist()
    ch_data = {ch: tbl[ch].to_pylist() for ch in SEQ_CHANNELS}

    # Index rows by (activity_id, window, timestep)
    row_lookup: dict[tuple, int] = {}
    for i, (aid, window, ts) in enumerate(zip(aid_col, window_col, ts_col)):
        row_lookup[(aid, window, ts)] = i

    N = len(activity_ids)
    pre_arr = np.full((N, PRE_LEN, N_SEQ_CH), float("nan"), dtype=np.float32)
    transit_arr = np.full((N, TRANSIT_LEN, N_SEQ_CH), float("nan"), dtype=np.float32)
    transit_mask = np.zeros((N, TRANSIT_LEN), dtype=bool)

    for i, aid in enumerate(activity_ids):
        for h in range(PRE_LEN):
            ts = h - PRE_LEN  # -150...-1
            row_i = row_lookup.get((aid, "pre_launch", ts))
            if row_i is not None:
                for c_idx, ch in enumerate(SEQ_CHANNELS):
                    v = ch_data[ch][row_i]
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        pre_arr[i, h, c_idx] = float(v)

        for h in range(TRANSIT_LEN):
            row_i = row_lookup.get((aid, "in_transit", h))
            if row_i is not None:
                transit_mask[i, h] = True
                for c_idx, ch in enumerate(SEQ_CHANNELS):
                    v = ch_data[ch][row_i]
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        transit_arr[i, h, c_idx] = float(v)

    return pre_arr, transit_arr, transit_mask


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr)


def build_dataset(split: str, existing_preds: dict) -> dict:
    """Load all tensors for a split. Returns dict ready for TFTTrainer."""
    print(f"  Loading scalar features [{split}]...")
    activity_ids, static_arr, existing_arr, labels = load_scalar_features(split, existing_preds)
    print(f"    {len(activity_ids)} events, {N_STATIC} static + {N_EXISTING} pred features")

    print(f"  Loading sequences [{split}]...")
    pre_arr, transit_arr, transit_mask = load_sequences(split, activity_ids)
    print(f"    pre_seq: {pre_arr.shape}, transit_seq: {transit_arr.shape}")

    # Per-feature mean imputation (fit on train, apply consistently)
    # NaN → 0 after z-score norm (z-score computed outside, here just pass as-is;
    # the TFT's _replace_nan(0.0) handles NaN at forward time)

    return {
        "activity_ids": activity_ids,
        "static": _to_tensor(static_arr),
        "pre_seq": _to_tensor(pre_arr),
        "transit_seq": _to_tensor(transit_arr),
        "transit_mask": torch.from_numpy(transit_mask),
        "existing_preds": _to_tensor(existing_arr),
        "target": _to_tensor(labels).unsqueeze(1),
    }


# ── Temporal CV fold splitting ────────────────────────────────────────────────

def get_fold_indices(
    activity_ids: list[str],
    n_folds: int,
    gap_days: int,
    conn: sqlite3.Connection,
) -> list[tuple[list[int], list[int]]]:
    """Expanding-window temporal CV. Last fold is calibration-only (RULE-164)."""
    rows = conn.execute(
        "SELECT activity_id, launch_time FROM pinn_expanded_flat "
        "WHERE split='train' AND exclude=0 ORDER BY launch_time"
    ).fetchall()
    launch_map = {r[0]: r[1] for r in rows}

    from datetime import timedelta
    times = []
    for aid in activity_ids:
        lt = launch_map.get(aid, "")
        try:
            t = datetime.strptime(lt[:16], "%Y-%m-%d %H:%M")
        except Exception:
            t = datetime(2010, 1, 1)
        times.append(t)

    N = len(activity_ids)
    sorted_idx = sorted(range(N), key=lambda i: times[i])
    t_sorted = [times[i] for i in sorted_idx]

    t_min, t_max = t_sorted[0], t_sorted[-1]
    span = (t_max - t_min).days
    fold_size = span // n_folds
    gap = timedelta(days=gap_days)

    folds = []
    for f in range(n_folds):
        val_start = t_min + (f + 1) * timedelta(days=fold_size) - timedelta(days=fold_size // 2)
        val_end = t_min + (f + 2) * timedelta(days=fold_size)
        if f == n_folds - 1:
            val_end = t_max + timedelta(days=1)

        train_idx = [i for i in range(N) if times[i] < val_start - gap]
        val_idx = [i for i in range(N) if val_start <= times[i] < val_end]

        if train_idx and val_idx:
            folds.append((train_idx, val_idx))

    return folds


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(preds_p50: np.ndarray, labels: np.ndarray) -> dict:
    valid = ~np.isnan(labels) & ~np.isnan(preds_p50)
    if not valid.any():
        return {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan"), "n": 0}
    y, p = labels[valid], preds_p50[valid]
    err = p - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "n": int(valid.sum()),
    }


# ── Main training loop ────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        results_path = OUTPUT_DIR / "tft_v1_results.json"
        if results_path.exists():
            r = json.loads(results_path.read_text())
            print(f"Holdout MAE: {r['holdout_mae']:.3f}h")
            print(f"Holdout RMSE: {r['holdout_rmse']:.3f}h")
            print(f"Holdout N: {r['holdout_n']}")
        else:
            print("No results found — run without --validate-only first")
        return 0

    print("=== train_tft_model ===")
    print(f"device: {args.device}")
    print(f"d_model: {args.d_model}, n_heads: {args.n_heads}, epochs: {args.epochs}")
    print(f"static features: {N_STATIC}, seq channels: {N_SEQ_CH}, existing preds: {N_EXISTING}")

    print("\n[1/4] Loading existing model predictions...")
    existing_preds = load_existing_predictions()
    print(f"  Phase 8 preds loaded: {sum(1 for v in existing_preds.values() if 'phase8_pred_transit_hours' in v)}")
    print(f"  PINN V1 preds loaded: {sum(1 for v in existing_preds.values() if 'pinn_v1_pred_transit_hours' in v)}")

    print("\n[2/4] Building datasets...")
    train_ds = build_dataset("train", existing_preds)
    holdout_ds = build_dataset("holdout", existing_preds)

    model_kwargs = {
        "n_static": N_STATIC,
        "n_seq_channels": N_SEQ_CH,
        "n_existing_preds": N_EXISTING,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_lstm_layers": 2,
        "dropout": 0.1,
        "max_pre_len": PRE_LEN,
        "max_transit_len": TRANSIT_LEN,
    }

    trainer = TFTTrainer(
        model_kwargs=model_kwargs,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        n_folds=args.folds,
        gap_days=14,
        device=args.device,
    )

    print("\n[3/4] Running temporal CV...")
    conn = sqlite3.connect(str(STAGING_DB))
    folds = get_fold_indices(train_ds["activity_ids"], args.folds, gap_days=14, conn=conn)
    conn.close()
    print(f"  {len(folds)} folds (last fold = calibration-only)")

    fold_metrics = []
    best_model = None
    best_val_mae = float("inf")

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        is_calibration = (fold_i == len(folds) - 1)
        print(f"\n  Fold {fold_i + 1}/{len(folds)}"
              f" {'[CALIBRATION]' if is_calibration else ''}"
              f" train={len(train_idx)} val={len(val_idx)}")

        fold_train = {k: v[train_idx] if torch.is_tensor(v) else v
                      for k, v in train_ds.items() if k != "activity_ids"}
        fold_val = {k: v[val_idx] if torch.is_tensor(v) else v
                    for k, v in train_ds.items() if k != "activity_ids"}
        fold_val_ids = [train_ds["activity_ids"][i] for i in val_idx]

        def progress_cb(epoch, tl, vl, f=fold_i):
            if epoch % 10 == 0:
                print(f"    epoch {epoch:3d}: train={tl:.4f} val={vl:.4f}")

        # RULE-164: last fold is calibration-only — skip training, evaluate only
        if is_calibration:
            if best_model is None:
                print("  WARNING: no best model found — skipping calibration fold")
                continue
            model = best_model
        else:
            import threading
            ct_event = threading.Event()
            model = trainer.train_fold(fold_train, fold_val, progress_cb, ct_event)

        val_preds = trainer.predict(model, fold_val).numpy()
        metrics = compute_metrics(val_preds[:, 1], fold_val["target"].squeeze().numpy())
        fold_metrics.append({**metrics, "fold": fold_i + 1, "calibration": is_calibration,
                              "n_train": len(train_idx), "n_val": len(val_idx)})
        print(f"    val MAE={metrics['mae']:.3f}h RMSE={metrics['rmse']:.3f}h")

        if not is_calibration and metrics["mae"] < best_val_mae:
            best_val_mae = metrics["mae"]
            best_model = model
            torch.save(model.state_dict(), str(MODEL_DIR / f"tft_v1_fold{fold_i+1}.pt"))
            print(f"    new best model saved (fold {fold_i+1})")

    print("\n[4/4] Evaluating on holdout...")
    if best_model is None:
        print("ERROR: no model trained")
        return 1

    holdout_preds = trainer.predict(best_model, holdout_ds).numpy()  # (90, 3)
    holdout_labels = holdout_ds["target"].squeeze().numpy()
    holdout_ids = holdout_ds["activity_ids"]

    metrics = compute_metrics(holdout_preds[:, 1], holdout_labels)
    print(f"\n  Holdout MAE:  {metrics['mae']:.3f}h")
    print(f"  Holdout RMSE: {metrics['rmse']:.3f}h")
    print(f"  Holdout bias: {metrics['bias']:.3f}h")
    print(f"  Holdout N:    {metrics['n']}")

    # Build per-event output
    events_out = []
    for i, aid in enumerate(holdout_ids):
        if math.isnan(holdout_labels[i]):
            continue
        events_out.append({
            "activity_id": aid,
            "truth_transit_hours": float(holdout_labels[i]),
            "pred_p10": float(holdout_preds[i, 0]),
            "pred_p50": float(holdout_preds[i, 1]),
            "pred_p90": float(holdout_preds[i, 2]),
            "error_hours": float(holdout_preds[i, 1] - holdout_labels[i]),
        })

    results = {
        "model": "tft_v1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "architecture": model_kwargs,
        "training": {"epochs": args.epochs, "lr": args.lr, "device": args.device},
        "n_static_features": N_STATIC,
        "n_seq_channels": N_SEQ_CH,
        "n_existing_preds": N_EXISTING,
        "cv_fold_metrics": fold_metrics,
        "cv_mae_mean": float(np.mean([f["mae"] for f in fold_metrics if not f["calibration"]])),
        "holdout_mae": metrics["mae"],
        "holdout_rmse": metrics["rmse"],
        "holdout_bias": metrics["bias"],
        "holdout_n": metrics["n"],
        "static_feature_names": STATIC_FEATURES,
        "seq_channel_names": SEQ_CHANNELS,
        "existing_pred_names": EXISTING_PRED_COLS,
        "holdout_predictions": events_out,
    }

    results_path = OUTPUT_DIR / "tft_v1_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved: {results_path}")

    # Save meta for sidecar inference
    meta_path = MODEL_DIR / "tft_v1_meta.json"
    meta_path.write_text(json.dumps({
        "model_type": "TFT_v1",
        "model_kwargs": model_kwargs,
        "static_features": STATIC_FEATURES,
        "seq_channels": SEQ_CHANNELS,
        "existing_pred_cols": EXISTING_PRED_COLS,
    }, indent=2))
    print(f"  Meta saved: {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
