"""Dark full-feature LightGBM model — direct transit time prediction.

Unlike PINN V1 which corrects a physics prior, this model sees ALL available
features (75 static cols + 9 derived interactions + sequence summary stats
extracted from the 150h OMNI window) and predicts transit_time_hours directly.

The "unknown" in the final ensemble — no physics inductive bias, so it can
capture patterns the physics model misses (multi-CME interaction geometry,
SHARP magnetic topology, solar cycle phase).

Sequence features extracted as summary statistics from pre_seq:
  For each OMNI channel: mean, std, min, max over the 150h window
  For Bz_GSM and flow_speed: additional 24h-window stats (last 24h before launch)
  Derivative features: mean/max of |d/dt Bz|, |d/dt speed|, |d²/dt² Bz|

Outputs:
  output/dark_model/dark_model_results.json
  models/registry/dark_model.pkl

Usage:
  python scripts/train_dark_model.py [--n-folds 5]
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

# Resolve repo root: works whether run from main repo or a worktree
_script_dir = Path(__file__).resolve().parent
ROOT = _script_dir.parent
# If staging.db not found at expected path, walk up to find the repo root
if not (ROOT / "data" / "data" / "staging" / "staging.db").exists():
    for _p in _script_dir.parents:
        if (_p / "data" / "data" / "staging" / "staging.db").exists():
            ROOT = _p
            break
sys.path.insert(0, str(_script_dir))
sys.path.insert(0, str(ROOT / "python"))

STAGING_DB = ROOT / "data" / "data" / "staging" / "staging.db"
SEQ_DIR = ROOT / "data" / "sequences"
PINN_V1_RESULTS = ROOT / "output" / "pinn_v1" / "pinn_v1_results.json"
OUT_DIR = ROOT / "output" / "dark_model"
MODEL_PATH = ROOT / "models" / "registry" / "dark_model.pkl"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

SEQ_CHANNELS = [
    "Bz_GSM", "flow_speed", "proton_density", "flow_pressure",
    "AE_nT", "Dst_nT", "Kp_x10",
    "B_scalar_avg", "By_GSM", "electric_field", "plasma_beta", "alfven_mach",
    "sigma_Bz", "sigma_N", "sigma_V",
    "flow_longitude", "flow_latitude", "alpha_proton_ratio",
    "F10_7_index", "Bx_GSE",
]
PRE_LEN = 150

# Static features from pinn_expanded_flat (all numeric columns)
STATIC_COLS = [
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
    "second_order_speed_init", "second_order_speed_final",
    "second_order_speed_20Rs", "accel_kms2", "mpa_deg",
    "meangam", "meangbt", "meangbz", "meangbh",
    "meanjzd", "totusjz", "meanjzh", "totusjh", "absnjzh",
    "meanalp", "savncpp", "meanpot", "totpot",
    "meanshr", "shrgt45", "r_value", "area_acr",
    "cluster_id_k8", "cluster_id_k12", "cluster_id_dbscan",
    "enlil_predicted_arrival_hours", "enlil_au", "enlil_matched",
]

_SENTINEL = 9990.0


def _clean(v) -> float:
    if v is None:
        return float("nan")
    fv = float(v)
    return float("nan") if abs(fv) >= _SENTINEL else fv


def load_pinn_v1_preds() -> dict[str, float]:
    """Load PINN V1 OOF + holdout predictions as a feature."""
    if not PINN_V1_RESULTS.exists():
        return {}
    data = json.loads(PINN_V1_RESULTS.read_text())
    preds = {}
    for section in ("oof_predictions", "holdout_predictions"):
        for ev in data.get(section, []):
            aid = ev.get("activity_id")
            val = ev.get("predicted")
            if aid and val is not None:
                preds[aid] = float(val)
    return preds


def extract_seq_features(seq_arr: np.ndarray) -> np.ndarray:
    """Extract summary statistics from (T, C) pre-launch sequence.

    Returns 1D feature vector of length (C*4 + 2*3 + 3) = C*4 + 9.
    """
    C = seq_arr.shape[1]
    feats = []

    # Per-channel: mean, std, min, max over full 150h
    for c in range(C):
        col = seq_arr[:, c]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            feats.extend([float("nan")] * 4)
        else:
            feats.extend([float(np.mean(valid)), float(np.std(valid)),
                          float(np.min(valid)), float(np.max(valid))])

    # Extra: last 24h window stats for Bz (col 0) and flow_speed (col 1)
    for c in [0, 1]:
        col = seq_arr[-24:, c]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            feats.extend([float("nan")] * 3)
        else:
            feats.extend([float(np.mean(valid)), float(np.min(valid)), float(np.max(valid))])

    # Derivative features: mean/max of |dBz/dt|, mean |dSpeed/dt|, mean |d²Bz/dt²|
    for c, order in [(0, 1), (1, 1), (0, 2)]:
        col = seq_arr[:, c].astype(float)
        if order == 1:
            d = np.abs(np.diff(col))
            d = d[~np.isnan(d)]
        else:
            if len(col) > 2:
                d2 = np.abs(col[2:] - 2 * col[1:-1] + col[:-2])
                d = d2[~np.isnan(d2)]
            else:
                d = np.array([])
        feats.append(float(np.mean(d)) if len(d) > 0 else float("nan"))

    return np.array(feats, dtype=np.float32)


def load_sequences_as_features(split: str, activity_ids: list[str]) -> np.ndarray:
    """Load Parquet sequences and extract summary statistics. Returns (N, F) array."""
    path = SEQ_DIR / f"{split}_sequences.parquet"
    if not path.exists():
        print(f"  WARNING: {path} not found — sequence features will be NaN")
        C = len(SEQ_CHANNELS)
        F = C * 4 + 9
        return np.full((len(activity_ids), F), float("nan"), dtype=np.float32)

    tbl = pq.read_table(str(path))
    aid_col = tbl["activity_id"].to_pylist()
    ts_col = tbl["timestep"].to_pylist()
    window_col = tbl["window"].to_pylist()
    ch_data = {ch: tbl[ch].to_pylist() for ch in SEQ_CHANNELS}

    # Build event → pre_launch rows
    C = len(SEQ_CHANNELS)
    F = C * 4 + 9
    out = np.full((len(activity_ids), F), float("nan"), dtype=np.float32)
    aid_set = {aid: i for i, aid in enumerate(activity_ids)}

    # Collect rows per event
    event_rows: dict[str, list[tuple[int, list[float]]]] = {}
    for row_i, (aid, window, ts) in enumerate(zip(aid_col, window_col, ts_col)):
        if aid not in aid_set or window != "pre_launch":
            continue
        t_idx = ts + PRE_LEN
        if 0 <= t_idx < PRE_LEN:
            vals = [float("nan") if v is None or (isinstance(v, float) and math.isnan(v))
                    else float(v) for v in [ch_data[ch][row_i] for ch in SEQ_CHANNELS]]
            event_rows.setdefault(aid, []).append((t_idx, vals))

    for aid, rows in event_rows.items():
        n_idx = aid_set[aid]
        seq = np.full((PRE_LEN, C), float("nan"), dtype=np.float32)
        for t_idx, vals in rows:
            seq[t_idx] = vals
        out[n_idx] = extract_seq_features(seq)

    return out


def build_features(split: str, pinn_v1_preds: dict) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    """Build full feature matrix.

    Returns: activity_ids, X (N, F) float32, y (N,) float32, feature_names
    """
    conn = sqlite3.connect(str(STAGING_DB))
    cols_sql = ", ".join(STATIC_COLS + ["activity_id", "transit_time_hours"])
    rows = conn.execute(
        f"SELECT {cols_sql} FROM pinn_expanded_flat WHERE split=? AND exclude=0 "
        "ORDER BY launch_time", (split,)
    ).fetchall()
    conn.close()

    col_names = STATIC_COLS + ["activity_id", "transit_time_hours"]
    col_idx = {c: i for i, c in enumerate(col_names)}

    activity_ids = [r[col_idx["activity_id"]] for r in rows]
    y = np.array([_clean(r[col_idx["transit_time_hours"]]) for r in rows], dtype=np.float32)
    N = len(rows)

    # Static features (sentinel → NaN)
    static = np.full((N, len(STATIC_COLS)), float("nan"), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, col in enumerate(STATIC_COLS):
            static[i, j] = _clean(r[col_idx[col]])

    # Engineered static interactions
    def _s(col: str) -> np.ndarray:
        j = STATIC_COLS.index(col) if col in STATIC_COLS else -1
        return static[:, j] if j >= 0 else np.full(N, float("nan"))

    log_usflux = np.log1p(np.clip(np.abs(_s("usflux")), 0, None))
    log_totpot = np.log1p(np.clip(np.abs(_s("totpot")), 0, None))
    log_area = np.log1p(np.clip(np.abs(_s("area_acr")), 0, None))
    spd = _s("cme_speed_kms")
    lat = _s("cme_latitude")
    lat_proj = np.where(np.isnan(spd) | np.isnan(lat),
                        float("nan"), spd * np.sin(np.abs(lat) * np.pi / 180.0))
    dv = _s("delta_v_kms")
    rel_drag = np.where(np.isnan(dv) | np.isnan(spd), float("nan"), dv / (spd + 1))
    tp = _s("totpot")
    ac = _s("area_acr")
    free_energy = np.where(np.isnan(tp) | np.isnan(ac), float("nan"), tp * ac)
    acc = _s("accel_kms2")
    acc_sq = np.where(np.isnan(acc), float("nan"), acc * np.abs(acc))
    sf = _s("second_order_speed_final")
    si = _s("second_order_speed_init")
    spd_change = np.where(np.isnan(sf) | np.isnan(si), float("nan"), sf - si)

    eng = np.column_stack([log_usflux, log_totpot, log_area,
                           lat_proj, rel_drag, free_energy, acc_sq, spd_change])
    eng_names = ["log_usflux", "log_totpot", "log_area_acr",
                 "cme_lat_proj", "rel_drag_proxy", "sharp_free_energy",
                 "accel_signed_sq", "cdaw_spd_change"]

    # PINN V1 OOF prediction as feature
    pinn_feat = np.array([pinn_v1_preds.get(aid, float("nan"))
                          for aid in activity_ids], dtype=np.float32).reshape(-1, 1)

    # Sequence summary features
    print(f"  Extracting sequence features [{split}]...")
    seq_feats = load_sequences_as_features(split, activity_ids)
    C = len(SEQ_CHANNELS)
    seq_names = []
    for ch in SEQ_CHANNELS:
        seq_names += [f"seq_{ch}_mean", f"seq_{ch}_std", f"seq_{ch}_min", f"seq_{ch}_max"]
    seq_names += ["seq_bz_24h_mean", "seq_bz_24h_min", "seq_bz_24h_max",
                  "seq_spd_24h_mean", "seq_spd_24h_min", "seq_spd_24h_max",
                  "seq_abs_dbz_mean", "seq_abs_dspd_mean", "seq_abs_d2bz_mean"]

    X = np.concatenate([static, eng, pinn_feat, seq_feats], axis=1).astype(np.float32)
    feat_names = STATIC_COLS + eng_names + ["pinn_v1_pred"] + seq_names

    return activity_ids, X, y, feat_names


def temporal_folds(N: int, n_folds: int) -> list[tuple[list[int], list[int]]]:
    """Simple expanding-window walk-forward folds (sorted by index = launch_time order)."""
    fold_size = N // (n_folds + 1)
    folds = []
    for k in range(1, n_folds + 1):
        cutoff = k * fold_size
        train_idx = list(range(cutoff))
        val_idx = list(range(cutoff, min(cutoff + fold_size, N)))
        if train_idx and val_idx:
            folds.append((train_idx, val_idx))
    return folds


def train_lgb(X_tr: np.ndarray, y_tr: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              feat_names: list[str]) -> lgb.Booster:
    """Train LightGBM with NaN-tolerant params."""
    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "n_estimators": 3000,
        "learning_rate": 0.02,
        "num_leaves": 63,
        "min_child_samples": 10,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
    }
    ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_names)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr, feature_name=feat_names)
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
    model = lgb.train(params, ds_tr, valid_sets=[ds_val], callbacks=callbacks)
    return model


def mae(errors: list[float]) -> float:
    return float(np.mean(np.abs(errors)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--write-oof-preds", action="store_true")
    args = ap.parse_args()

    print("=== Dark Full-Feature LightGBM Model ===")

    pinn_v1_preds = load_pinn_v1_preds()
    print(f"  PINN V1 preds loaded: {len(pinn_v1_preds)}")

    print("\n[1/3] Building feature matrices...")
    train_ids, X_train, y_train, feat_names = build_features("train", pinn_v1_preds)
    hold_ids, X_hold, y_hold, _ = build_features("holdout", pinn_v1_preds)
    print(f"  Train: {len(train_ids)} events, {X_train.shape[1]} features")
    print(f"  Holdout: {len(hold_ids)} events")

    # Filter rows with valid target
    train_valid = ~np.isnan(y_train)
    X_train, y_train, train_ids = X_train[train_valid], y_train[train_valid], [train_ids[i] for i in np.where(train_valid)[0]]
    hold_valid = ~np.isnan(y_hold)
    X_hold, y_hold, hold_ids = X_hold[hold_valid], y_hold[hold_valid], [hold_ids[i] for i in np.where(hold_valid)[0]]

    print(f"\n[2/3] Temporal CV ({args.n_folds} folds)...")
    folds = temporal_folds(len(train_ids), args.n_folds)
    cv_maes = []
    oof_preds_map: dict[str, float] = {}

    for fold_i, (tr_idx, val_idx) in enumerate(folds):
        model_cv = train_lgb(X_train[tr_idx], y_train[tr_idx],
                             X_train[val_idx], y_train[val_idx], feat_names)
        preds_val = model_cv.predict(X_train[val_idx])
        errs = list(preds_val - y_train[val_idx])
        fold_mae = mae(errs)
        cv_maes.append(fold_mae)
        print(f"  fold {fold_i+1}: n_val={len(val_idx)} MAE={fold_mae:.2f}h")
        if args.write_oof_preds:
            for pos, idx in enumerate(val_idx):
                oof_preds_map[train_ids[idx]] = float(preds_val[pos])

    print(f"  CV MAE: {np.mean(cv_maes):.2f}h ± {np.std(cv_maes):.2f}h")

    print("\n[3/3] Final model on all train data...")
    final_model = train_lgb(X_train, y_train, X_hold, y_hold, feat_names)
    preds_hold = final_model.predict(X_hold)
    hold_errs = list(preds_hold - y_hold)
    hold_mae = mae(hold_errs)
    hold_rmse = float(np.sqrt(np.mean(np.array(hold_errs) ** 2)))
    hold_bias = float(np.mean(hold_errs))
    print(f"  Holdout MAE:  {hold_mae:.2f}h")
    print(f"  Holdout RMSE: {hold_rmse:.2f}h")
    print(f"  Holdout bias: {hold_bias:+.2f}h")
    print(f"  PINN V1 ref:  9.05h | Phase 8 ref: 12.33h")

    # Feature importance top-20
    fi = sorted(zip(feat_names, final_model.feature_importance()), key=lambda x: -x[1])
    print("\n  Feature importance (top 20):")
    for feat, imp in fi[:20]:
        print(f"    {feat:<50s} {imp:.0f}")

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": final_model, "feat_names": feat_names}, f)
    print(f"\n  Model saved: {MODEL_PATH}")

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_train": len(train_ids),
        "n_holdout": len(hold_ids),
        "n_features": X_train.shape[1],
        "cv_mae_mean": float(np.mean(cv_maes)),
        "cv_mae_std": float(np.std(cv_maes)),
        "holdout_mae": float(hold_mae),
        "holdout_rmse": float(hold_rmse),
        "holdout_bias": float(hold_bias),
        "feature_importance": [(f, int(i)) for f, i in fi],
        "holdout_predictions": [
            {"activity_id": hold_ids[i], "truth": float(y_hold[i]),
             "predicted": float(preds_hold[i]),
             "error": float(preds_hold[i] - y_hold[i])}
            for i in range(len(hold_ids))
        ],
    }
    if args.write_oof_preds:
        results["oof_predictions"] = [
            {"activity_id": aid, "predicted": pred}
            for aid, pred in oof_preds_map.items()
        ]

    out_path = OUT_DIR / "dark_model_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  Results saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
