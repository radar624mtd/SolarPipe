"""PINN transit-time model: physics-prior ODE residual + LightGBM on pinn_training_flat.

Physics layer:
  transit_physics = ODE drag model (closed-form approximation)
  label = transit_time_hours (observed)
  residual = transit_time_hours - transit_physics

ML layer:
  LightGBM regressor trained on residual using all pinn_training_flat features.
  Temporal CV: walk-forward folds by launch_time, no random k-fold.

Outputs:
  output/pinn_v1/pinn_v1_results.json
  output/pinn_v1/pinn_v1_comparison.md
  models/registry/pinn_v1_model.pkl

Usage:
  python scripts/train_pinn_model.py [--no-residual] [--n-folds 5]
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

ROOT       = Path(__file__).resolve().parents[1]
STAGING_DB = ROOT / "data" / "data" / "staging" / "staging.db"
SOLAR_DB   = ROOT / "solar_data.db"
OUT_DIR    = ROOT / "output" / "pinn_v1"
MODEL_PATH = ROOT / "models" / "registry" / "pinn_v1_model.pkl"

# Physics constants
R_SUN_KM  = 695700.0
R_START   = 21.5    # solar radii (injection point, ~0.1 AU)
R_TARGET  = 215.0   # solar radii (1 AU)

# M8 calibrated drag parameter (global prior; per-event modulation is the ML residual's job)
GAMMA0_DEFAULT = 2.2974e-8   # km⁻¹, from M8 refined calibration

# Feature columns to pass to LightGBM (NaN-tolerant; categorical handled separately)
NUMERIC_FEATURES = [
    # Stage 1 — Regime
    "omni_24h_bz_mean", "omni_24h_bz_std", "omni_24h_bz_min",
    "omni_24h_speed_mean", "omni_24h_density_mean", "omni_24h_pressure_mean",
    "omni_24h_ae_max", "omni_24h_dst_min", "omni_24h_kp_max",
    "f10_7",
    # Stage 2 — Interaction
    "preceding_cme_count_48h", "preceding_cme_speed_max", "preceding_cme_speed_mean",
    "preceding_cme_angular_sep_min", "is_multi_cme",
    "omni_48h_density_spike_max", "omni_48h_speed_gradient",
    # Stage 3 — Physics
    "cme_speed_kms", "cme_half_angle_deg", "cme_latitude", "cme_longitude",
    "cme_angular_width_deg", "cdaw_linear_speed_kms",
    "cdaw_mass_log10", "cdaw_ke_log10", "cdaw_matched",
    "has_flare", "flare_class_numeric", "flare_source_longitude",
    "omni_150h_density_median", "omni_150h_speed_median",
    "sw_bz_ambient", "delta_v_kms",
    "sharp_available", "usflux",
    # Cluster label (ordinal, LightGBM treats as numeric)
    "cluster_id_k5",
    # Engineered
    "transit_physics",   # added below
]

CATEGORICAL_FEATURES: list[str] = []   # cluster_id handled as numeric


# ── Physics ODE (closed-form approximation) ──────────────────────────────────

def drag_transit_hours(v0: float, w: float, gamma: float,
                       r_start: float = R_START, r_target: float = R_TARGET,
                       dt_h: float = 0.25) -> float:
    """RK4 drag integration — returns transit hours or NaN if timeout."""
    r, v, t = r_start, v0, 0.0
    max_t = 200.0
    h_sec = dt_h * 3600.0
    scale = 3600.0 / R_SUN_KM

    def rhs(r_, v_):
        dr = v_ * scale
        dv = -gamma * (v_ - w) * abs(v_ - w) * 3600.0
        return dr, dv

    while r < r_target and t < max_t:
        k1r, k1v = rhs(r, v)
        k2r, k2v = rhs(r + 0.5*dt_h*k1r, v + 0.5*dt_h*k1v)
        k3r, k3v = rhs(r + 0.5*dt_h*k2r, v + 0.5*dt_h*k2v)
        k4r, k4v = rhs(r + dt_h*k3r, v + dt_h*k3v)
        r_new = r + (dt_h/6.0)*(k1r + 2*k2r + 2*k3r + k4r)
        v_new = v + (dt_h/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
        # Linear interpolation to crossing
        if r < r_target <= r_new:
            frac = (r_target - r) / (r_new - r)
            return t + frac * dt_h
        r, v, t = r_new, v_new, t + dt_h

    return float("nan") if r < r_target else t


def compute_physics_column(rows: list[dict]) -> np.ndarray:
    """Compute per-event physics ODE transit (hours) using pre-launch OMNI median speed."""
    out = []
    for row in rows:
        v0 = row.get("cme_speed_kms") or float("nan")
        w  = row.get("omni_150h_speed_median") or row.get("omni_24h_speed_mean") or 400.0
        if math.isnan(v0) or v0 < 100:
            out.append(float("nan"))
            continue
        w = max(200.0, min(800.0, w))
        # Clip v0 to physical range
        v0 = max(200.0, min(3500.0, v0))
        t = drag_transit_hours(v0, w, GAMMA0_DEFAULT)
        out.append(t)
    return np.array(out)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[list[dict], list[dict]]:
    conn = sqlite3.connect(str(STAGING_DB))
    conn.row_factory = sqlite3.Row
    all_rows = conn.execute(
        "SELECT * FROM pinn_training_flat WHERE exclude=0"
    ).fetchall()
    conn.close()

    train = [dict(r) for r in all_rows if r["split"] == "train"]
    holdout = [dict(r) for r in all_rows if r["split"] == "holdout"]
    return train, holdout


def to_matrix(rows: list[dict], features: list[str]) -> np.ndarray:
    X = np.full((len(rows), len(features)), float("nan"))
    for i, row in enumerate(rows):
        for j, f in enumerate(features):
            v = row.get(f)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                X[i, j] = float(v)
    # Return as pandas DataFrame for proper LightGBM feature-name handling
    import pandas as pd
    return pd.DataFrame(X, columns=features)


# ── Temporal CV ───────────────────────────────────────────────────────────────

def temporal_folds(rows: list[dict], n_folds: int) -> list[tuple[list[int], list[int]]]:
    """Walk-forward temporal folds. Sort by launch_time, split into n_folds+1 bands."""
    sorted_idx = sorted(range(len(rows)), key=lambda i: rows[i]["launch_time"])
    n = len(sorted_idx)
    fold_size = n // (n_folds + 1)
    folds = []
    for k in range(1, n_folds + 1):
        cutoff = k * fold_size
        train_idx = sorted_idx[:cutoff]
        val_idx   = sorted_idx[cutoff:cutoff + fold_size]
        if len(val_idx) == 0:
            continue
        folds.append((train_idx, val_idx))
    return folds


# ── LightGBM training ─────────────────────────────────────────────────────────

LGB_PARAMS = {
    "objective":       "regression_l1",   # MAE loss — matches our eval metric
    "metric":          "mae",
    "n_estimators":    800,
    "learning_rate":   0.03,
    "num_leaves":      63,
    "min_child_samples": 20,
    "feature_fraction":  0.7,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "reg_alpha":       0.1,
    "reg_lambda":      0.1,
    "verbose":         -1,
    "n_jobs":          -1,
    "random_state":    42,
}


def train_model(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
                feature_names: list[str] | None = None) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    eval_set = [(X_val, y_val)] if X_val is not None else None
    cb = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)] if eval_set else []
    model.fit(X_train, y_train,
              eval_set=eval_set,
              callbacks=cb,
              feature_name=feature_names or "auto")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def mae(errors: list[float]) -> float:
    return sum(abs(e) for e in errors) / len(errors) if errors else float("nan")

def rmse(errors: list[float]) -> float:
    return math.sqrt(sum(e**2 for e in errors) / len(errors)) if errors else float("nan")

def bias(errors: list[float]) -> float:
    return sum(errors) / len(errors) if errors else float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-residual", action="store_true",
                    help="Train on raw transit time, not physics residual")
    ap.add_argument("--n-folds", type=int, default=5,
                    help="Number of temporal CV folds (default 5)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=== PINN Model Training (pinn_v1) ===")
    print(f"  staging.db: {STAGING_DB}")
    print(f"  mode: {'raw transit' if args.no_residual else 'physics residual'}")

    # Load data
    train_rows, holdout_rows = load_data()
    print(f"\n  Train: {len(train_rows)} rows | Holdout: {len(holdout_rows)} rows")

    # Physics column
    print("  Computing physics ODE column...")
    phys_train   = compute_physics_column(train_rows)
    phys_holdout = compute_physics_column(holdout_rows)
    for i, row in enumerate(train_rows):
        row["transit_physics"] = phys_train[i]
    for i, row in enumerate(holdout_rows):
        row["transit_physics"] = phys_holdout[i]

    # Labels
    y_train_raw = np.array([r["transit_time_hours"] for r in train_rows])
    y_hold_raw  = np.array([r["transit_time_hours"] for r in holdout_rows])

    if args.no_residual:
        y_train = y_train_raw
        y_hold  = y_hold_raw
    else:
        # Residual = observed - physics; train ML to correct physics bias
        y_train = y_train_raw - phys_train
        y_hold  = y_hold_raw - phys_holdout

    print(f"  Physics MAE (train):   {mae(list(y_train_raw - phys_train)):.2f}h")
    print(f"  Physics MAE (holdout): {mae(list(y_hold_raw - phys_holdout)):.2f}h")

    # Feature matrix
    features = NUMERIC_FEATURES[:]
    X_train_full = to_matrix(train_rows, features)
    X_hold_full  = to_matrix(holdout_rows, features)

    # ── Temporal CV ──────────────────────────────────────────────────────────
    print(f"\n  Temporal CV ({args.n_folds} walk-forward folds)...")
    folds = temporal_folds(train_rows, args.n_folds)
    cv_maes = []
    cv_physics_maes = []

    for fold_i, (tr_idx, val_idx) in enumerate(folds):
        X_tr  = X_train_full.iloc[tr_idx]
        y_tr  = y_train[tr_idx]
        X_val = X_train_full.iloc[val_idx]
        y_val = y_train[val_idx]
        y_val_raw = y_train_raw[val_idx]
        phys_val  = phys_train[val_idx]

        model_cv = train_model(X_tr, y_tr, X_val, y_val, feature_names=features)
        pred_resid = model_cv.predict(X_val)

        if args.no_residual:
            pred_transit = pred_resid
        else:
            pred_transit = phys_val + pred_resid

        fold_errors = list(pred_transit - y_val_raw)
        fold_mae = mae(fold_errors)
        phys_only_mae = mae(list(phys_val - y_val_raw))
        cv_maes.append(fold_mae)
        cv_physics_maes.append(phys_only_mae)
        print(f"    fold {fold_i+1}: n_val={len(val_idx)} | physics MAE={phys_only_mae:.2f}h | model MAE={fold_mae:.2f}h | Δ={fold_mae-phys_only_mae:+.2f}h")

    print(f"  CV MAE mean: {np.mean(cv_maes):.2f}h ± {np.std(cv_maes):.2f}h")
    print(f"  CV physics MAE mean: {np.mean(cv_physics_maes):.2f}h")

    # ── Final model (all train data) ─────────────────────────────────────────
    print("\n  Training final model on all train data...")
    final_model = train_model(X_train_full, y_train, feature_names=features)

    # Holdout predictions
    pred_hold_resid = final_model.predict(X_hold_full)
    if args.no_residual:
        pred_hold = pred_hold_resid
    else:
        pred_hold = phys_holdout + pred_hold_resid

    hold_errors = list(pred_hold - y_hold_raw)
    hold_mae  = mae(hold_errors)
    hold_rmse = rmse(hold_errors)
    hold_bias = bias(hold_errors)
    phys_hold_mae = mae(list(phys_holdout - y_hold_raw))

    print(f"\n  Holdout (2026, n={len(holdout_rows)})")
    print(f"    Physics-only MAE: {phys_hold_mae:.2f}h")
    print(f"    PINN v1 MAE:      {hold_mae:.2f}h  RMSE={hold_rmse:.2f}h  bias={hold_bias:+.2f}h")
    print(f"    Phase 8 baseline: 12.33h")
    print(f"    Δ vs baseline:    {hold_mae - 12.33:+.2f}h")

    # Feature importance top-20
    fi = sorted(zip(features, final_model.feature_importances_), key=lambda x: -x[1])
    print("\n  Feature importance (top 20):")
    for feat, imp in fi[:20]:
        print(f"    {feat:<40s} {imp:.0f}")

    # ── CCMC 4-event comparison ───────────────────────────────────────────────
    ccmc_events = [
        ("Jan-18 X1.9", "2026-01-18T18:09:00-CME-001", 24.77, 43.52),
        ("Mar-18 SIDC", "2026-03-18T09:23:00-CME-001", 58.90, 58.78),
        ("Mar-30",      "2026-03-30T03:24:00-CME-001", 56.08, 48.62),
        ("Apr-01",      "2026-04-01T23:45:00-CME-001", 39.28, 47.77),
    ]

    print("\n  CCMC 4-event comparison:")
    ccmc_errs = []
    ccmc_rows_out = []
    hold_by_id = {r["activity_id"]: (i, r) for i, r in enumerate(holdout_rows)}

    for tag, aid, truth, ph8_pred in ccmc_events:
        if aid not in hold_by_id:
            print(f"    {tag}: NOT FOUND in holdout (activity_id={aid})")
            continue
        idx, row = hold_by_id[aid]
        pred = pred_hold[idx]
        phys = phys_holdout[idx]
        err  = pred - truth
        ccmc_errs.append(abs(err))
        print(f"    {tag:<14s}  truth={truth:.2f}h  phys={phys:.2f}h  pinn={pred:.2f}h  err={err:+.2f}h  (ph8={ph8_pred:.2f}h)")
        ccmc_rows_out.append({
            "tag": tag, "activity_id": aid, "truth": truth,
            "physics": float(phys), "pinn_v1": float(pred),
            "error": float(err), "ph8_pred": ph8_pred,
        })

    if ccmc_errs:
        print(f"    CCMC MAE: {sum(ccmc_errs)/len(ccmc_errs):.2f}h  (Ph8: 8.57h)")

    # ── Save model + results ──────────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": final_model, "features": features,
                     "mode": "residual" if not args.no_residual else "direct",
                     "gamma0": GAMMA0_DEFAULT}, f)
    print(f"\n  Model saved: {MODEL_PATH}")

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "residual" if not args.no_residual else "direct",
        "n_train": len(train_rows),
        "n_holdout": len(holdout_rows),
        "cv_mae_mean": float(np.mean(cv_maes)),
        "cv_mae_std":  float(np.std(cv_maes)),
        "holdout_mae":  float(hold_mae),
        "holdout_rmse": float(hold_rmse),
        "holdout_bias": float(hold_bias),
        "physics_holdout_mae": float(phys_hold_mae),
        "phase8_baseline_mae": 12.33,
        "ccmc_4event": ccmc_rows_out,
        "feature_importance": [(f, int(i)) for f, i in fi],
    }
    with open(OUT_DIR / "pinn_v1_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Markdown report
    md = f"""# PINN v1 Training Results

Generated: {results['generated_at']}
Mode: {results['mode']}
Train: {results['n_train']} events | Holdout: {results['n_holdout']} events (2026)

## Holdout Performance

| Model | MAE | RMSE | Bias |
|---|---:|---:|---:|
| Physics-only (ODE) | {phys_hold_mae:.2f}h | — | — |
| Phase 8 baseline | 12.33h | — | — |
| PINN v1 (this) | {hold_mae:.2f}h | {hold_rmse:.2f}h | {hold_bias:+.2f}h |

CV MAE: {np.mean(cv_maes):.2f}h ± {np.std(cv_maes):.2f}h (walk-forward, {args.n_folds} folds)

## CCMC 4-Event Comparison

| Event | Truth | Physics | PINN v1 | Error | Ph8 |
|---|---:|---:|---:|---:|---:|
"""
    for r in ccmc_rows_out:
        md += f"| {r['tag']} | {r['truth']:.2f}h | {r['physics']:.2f}h | {r['pinn_v1']:.2f}h | {r['error']:+.2f}h | {r['ph8_pred']:.2f}h |\n"
    if ccmc_errs:
        md += f"\nCCMC MAE: **{sum(ccmc_errs)/len(ccmc_errs):.2f}h** (Ph8: 8.57h)\n"

    md += "\n## Top 20 Feature Importances\n\n| Feature | Importance |\n|---|---:|\n"
    for feat, imp in fi[:20]:
        md += f"| {feat} | {imp:.0f} |\n"

    with open(OUT_DIR / "pinn_v1_comparison.md", "w") as f:
        f.write(md)

    print(f"  Results saved: {OUT_DIR}/pinn_v1_results.json")
    print(f"  Report saved:  {OUT_DIR}/pinn_v1_comparison.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
