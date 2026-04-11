"""NNLS ensemble: optimally blend all model predictions.

Models blended:
  1. TFT v1 final          (output/tft_v1/tft_v1_results.json)
  2. TFT sweep winner      (output/tft_v1/sweep_results.json → re-train best config)
  3. PINN V1               (output/pinn_v1/pinn_v1_results.json)
  4. Dark full-feature LGB (output/dark_model/dark_model_results.json)
  5. Phase 9 progressive   (output/phase9_*/phase9_results.json — best available)
  6. Physics ODE baseline  (computed from pinn_training_flat physics column)

NNLS (Non-Negative Least Squares) optimization:
  - Fit weights on calibration set (last 20% of train, chronological)
  - Apply to holdout
  - Quantile output: individual model P50s blended, TFT P10/P90 used for uncertainty

Outputs:
  output/ensemble/ensemble_results.json   — per-event predictions + weights
  output/ensemble/ensemble_report.md      — performance summary

Usage:
  python scripts/build_nnls_ensemble.py
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
STAGING_DB = ROOT / "data" / "data" / "staging" / "staging.db"
OUT_DIR = ROOT / "output" / "ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Model result loaders ──────────────────────────────────────────────────────

def load_tft_v1() -> dict[str, dict]:
    """Load TFT v1 per-event predictions → {activity_id: {p10,p50,p90}}"""
    path = ROOT / "output" / "tft_v1" / "tft_v1_results.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out = {}
    for ev in data.get("holdout_events", []):
        aid = ev.get("activity_id")
        if aid:
            out[aid] = {
                "p10": ev.get("pred_p10", float("nan")),
                "p50": ev.get("pred_p50", float("nan")),
                "p90": ev.get("pred_p90", float("nan")),
            }
    return out


def load_pinn_v1() -> dict[str, float]:
    """Load PINN V1 holdout P50 predictions → {activity_id: predicted}"""
    path = ROOT / "output" / "pinn_v1" / "pinn_v1_results.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {ev["activity_id"]: ev["predicted"]
            for ev in data.get("holdout_predictions", [])
            if ev.get("activity_id") and ev.get("predicted") is not None}


def load_dark_model() -> dict[str, float]:
    """Load dark LGB holdout predictions → {activity_id: predicted}"""
    path = ROOT / "output" / "dark_model" / "dark_model_results.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {ev["activity_id"]: ev["predicted"]
            for ev in data.get("holdout_predictions", [])
            if ev.get("activity_id") and ev.get("predicted") is not None}


def load_phase9_best() -> dict[str, float]:
    """Load best Phase 9 progressive drag predictions → {activity_id: predicted}"""
    # Try phase9 output directories in order of quality
    for subdir in ["phase9_m8", "phase9_m7_static", "phase9_m7", "phase9_m6_6",
                   "phase9_m6_5", "phase9_m6"]:
        path = ROOT / "output" / subdir
        results_path = path / f"{subdir}_results.json"
        if results_path.exists():
            data = json.loads(results_path.read_text())
            preds = {}
            for ev in data.get("predictions", data.get("events", [])):
                aid = ev.get("activity_id") or ev.get("cme_id")
                val = ev.get("predicted_transit_hours") or ev.get("pred_transit_hours") or ev.get("predicted")
                if aid and val is not None:
                    preds[aid] = float(val)
            if preds:
                print(f"  Phase 9: loaded {len(preds)} preds from {subdir}")
                return preds
    return {}


def load_physics_ode(hold_ids: list[str]) -> dict[str, float]:
    """Load physics ODE predictions from pinn_training_flat via drag model."""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from train_pinn_model import compute_physics_column, load_data, _parse_dt
        _, holdout_rows = load_data()
        phys = compute_physics_column(holdout_rows)
        return {r["activity_id"]: float(p) for r, p in zip(holdout_rows, phys)}
    except Exception as e:
        print(f"  Physics ODE: failed ({e})")
        return {}


def load_ground_truth() -> dict[str, float]:
    """Load holdout ground truth from pinn_training_flat."""
    conn = sqlite3.connect(str(STAGING_DB))
    rows = conn.execute(
        "SELECT activity_id, transit_time_hours FROM pinn_expanded_flat "
        "WHERE split='holdout' AND exclude=0 ORDER BY launch_time"
    ).fetchall()
    conn.close()
    return {r[0]: float(r[1]) for r in rows if r[1] is not None}


# ── NNLS optimization ─────────────────────────────────────────────────────────

def build_prediction_matrix(
    activity_ids: list[str],
    model_preds: dict[str, dict[str, float]],
    model_names: list[str],
) -> np.ndarray:
    """Build (N, M) prediction matrix. NaN where model has no prediction."""
    N = len(activity_ids)
    M = len(model_names)
    mat = np.full((N, M), float("nan"), dtype=np.float64)
    for j, name in enumerate(model_names):
        preds = model_preds[name]
        for i, aid in enumerate(activity_ids):
            v = preds.get(aid, float("nan"))
            mat[i, j] = float(v) if v is not None and not math.isnan(v) else float("nan")
    return mat


def nnls_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit NNLS weights on rows where all models have predictions.

    X: (N, M), y: (N,)
    Returns: weights (M,) — non-negative, sum to 1.
    """
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    if valid.sum() < 10:
        print(f"  WARNING: only {valid.sum()} complete rows for NNLS — falling back to uniform")
        M = X.shape[1]
        return np.ones(M) / M

    X_v, y_v = X[valid], y[valid]
    # Solve: min ||X_v @ w - y_v||² s.t. w >= 0
    w, _ = nnls(X_v, y_v)
    if w.sum() < 1e-10:
        w = np.ones(X_v.shape[1])
    w = w / w.sum()
    return w


def blend(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Apply weights to prediction matrix; NaN rows get mean of non-NaN."""
    out = np.full(X.shape[0], float("nan"), dtype=np.float64)
    for i in range(X.shape[0]):
        row = X[i]
        avail = ~np.isnan(row)
        if avail.any():
            w_avail = w[avail]
            w_avail = w_avail / w_avail.sum()
            out[i] = (row[avail] * w_avail).sum()
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=== NNLS Ensemble ===")

    print("\n[1/4] Loading model predictions...")
    tft_v1_preds = load_tft_v1()
    pinn_v1_preds = load_pinn_v1()
    dark_preds = load_dark_model()
    phase9_preds = load_phase9_best()
    gt = load_ground_truth()

    print(f"  TFT v1:      {len(tft_v1_preds)} events")
    print(f"  PINN V1:     {len(pinn_v1_preds)} events")
    print(f"  Dark LGB:    {len(dark_preds)} events")
    print(f"  Phase 9:     {len(phase9_preds)} events")
    print(f"  Ground truth:{len(gt)} events")

    # Physics ODE
    hold_ids_sorted = sorted(gt.keys())
    ode_preds = load_physics_ode(hold_ids_sorted)
    print(f"  Physics ODE: {len(ode_preds)} events")

    # Use TFT p50 as primary prediction
    tft_p50 = {aid: v["p50"] for aid, v in tft_v1_preds.items()}
    tft_p10 = {aid: v["p10"] for aid, v in tft_v1_preds.items()}
    tft_p90 = {aid: v["p90"] for aid, v in tft_v1_preds.items()}

    # Model dict
    all_model_preds = {
        "tft_v1_p50": tft_p50,
        "pinn_v1": pinn_v1_preds,
        "dark_lgb": dark_preds,
        "phase9": phase9_preds,
        "physics_ode": ode_preds,
    }
    model_names = [m for m, p in all_model_preds.items() if p]
    available_preds = {m: all_model_preds[m] for m in model_names}
    print(f"\n  Models available for ensemble: {model_names}")

    print("\n[2/4] Building prediction matrices...")
    # All holdout events
    hold_ids = sorted(gt.keys())
    y_hold = np.array([gt[aid] for aid in hold_ids], dtype=np.float64)
    X_hold = build_prediction_matrix(hold_ids, available_preds, model_names)

    # Calibration set: last 20% of train events chronologically
    conn = sqlite3.connect(str(STAGING_DB))
    train_rows = conn.execute(
        "SELECT activity_id, transit_time_hours FROM pinn_expanded_flat "
        "WHERE split='train' AND exclude=0 ORDER BY launch_time"
    ).fetchall()
    conn.close()

    N_cal = max(20, int(0.20 * len(train_rows)))
    cal_rows = train_rows[-N_cal:]
    cal_ids = [r[0] for r in cal_rows]
    y_cal = np.array([float(r[1]) for r in cal_rows], dtype=np.float64)

    # Load calibration predictions from OOF results
    def load_oof(path_key: str, section: str = "oof_predictions") -> dict[str, float]:
        path = ROOT / "output" / path_key
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        return {ev["activity_id"]: float(ev["predicted"])
                for ev in data.get(section, [])
                if ev.get("activity_id") and ev.get("predicted") is not None}

    pinn_oof = load_oof("pinn_v1/pinn_v1_results.json", "oof_predictions")
    dark_oof = load_oof("dark_model/dark_model_results.json", "oof_predictions")
    tft_oof = load_oof("tft_v1/tft_v1_results.json", "oof_predictions")

    # Physics ODE for cal events
    sys.path.insert(0, str(ROOT / "scripts"))
    cal_ode: dict[str, float] = {}
    try:
        from train_pinn_model import compute_physics_column, load_data
        train_events, _ = load_data()
        phys_train = compute_physics_column(train_events)
        cal_ode = {r["activity_id"]: float(p) for r, p in zip(train_events, phys_train)}
    except Exception:
        pass

    cal_model_preds = {
        "tft_v1_p50": tft_oof,
        "pinn_v1": pinn_oof,
        "dark_lgb": dark_oof,
        "phase9": {},  # Phase 9 OOF not available
        "physics_ode": cal_ode,
    }
    cal_available = {m: cal_model_preds[m] for m in model_names}
    X_cal = build_prediction_matrix(cal_ids, cal_available, model_names)

    print(f"  Calibration rows: {len(cal_ids)} (last 20% of train)")
    cal_complete = (~np.isnan(X_cal).any(axis=1) & ~np.isnan(y_cal)).sum()
    print(f"  Complete calibration rows: {cal_complete}")

    print("\n[3/4] Fitting NNLS weights...")
    w = nnls_fit(X_cal, y_cal)
    print(f"  Weights:")
    for name, weight in zip(model_names, w):
        print(f"    {name:<20s}: {weight:.4f}")

    print("\n[4/4] Evaluating holdout...")
    blend_hold = blend(X_hold, w)

    valid = ~np.isnan(blend_hold) & ~np.isnan(y_hold)
    errs = blend_hold[valid] - y_hold[valid]
    ens_mae = float(np.mean(np.abs(errs)))
    ens_rmse = float(np.sqrt(np.mean(errs ** 2)))
    ens_bias = float(np.mean(errs))

    print(f"\n  Ensemble MAE:  {ens_mae:.3f}h (N={valid.sum()})")
    print(f"  Ensemble RMSE: {ens_rmse:.3f}h")
    print(f"  Ensemble bias: {ens_bias:+.3f}h")

    # Individual model MAEs for comparison
    print(f"\n  Individual model holdout MAEs:")
    individual_maes = {}
    for j, name in enumerate(model_names):
        col = X_hold[:, j]
        v = ~np.isnan(col) & ~np.isnan(y_hold)
        if v.any():
            m = float(np.mean(np.abs(col[v] - y_hold[v])))
            individual_maes[name] = m
            print(f"    {name:<20s}: {m:.3f}h (N={v.sum()})")

    # Uncertainty bounds from TFT
    events_out = []
    for i, aid in enumerate(hold_ids):
        p50 = float(blend_hold[i]) if not math.isnan(blend_hold[i]) else float("nan")
        p10 = tft_p10.get(aid, float("nan"))
        p90 = tft_p90.get(aid, float("nan"))
        events_out.append({
            "activity_id": aid,
            "truth_transit_hours": float(y_hold[i]),
            "ensemble_p50": p50,
            "ensemble_p10": p10,
            "ensemble_p90": p90,
            "error_hours": p50 - float(y_hold[i]) if not math.isnan(p50) else float("nan"),
            "model_preds": {name: float(X_hold[i, j]) for j, name in enumerate(model_names)},
        })

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": model_names,
        "nnls_weights": {name: float(w[j]) for j, name in enumerate(model_names)},
        "ensemble_mae": ens_mae,
        "ensemble_rmse": ens_rmse,
        "ensemble_bias": ens_bias,
        "n_holdout": int(valid.sum()),
        "individual_maes": individual_maes,
        "events": events_out,
    }

    out_path = OUT_DIR / "ensemble_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved: {out_path}")

    # Markdown report
    md = f"""# NNLS Ensemble Results

Generated: {results['generated_at']}
Models: {', '.join(model_names)}

## Holdout Performance (N={valid.sum()})

| Model | MAE | Weight |
|---|---:|---:|
"""
    for j, name in enumerate(model_names):
        m = individual_maes.get(name, float("nan"))
        md += f"| {name} | {m:.2f}h | {w[j]:.4f} |\n"
    md += f"| **NNLS Ensemble** | **{ens_mae:.2f}h** | — |\n"
    md += f"\nRMSE: {ens_rmse:.2f}h | Bias: {ens_bias:+.2f}h\n"
    md += f"\nReference: PINN V1 = 9.05h | Phase 8 = 12.33h | Target = 3h ± 2h\n"

    report_path = OUT_DIR / "ensemble_report.md"
    report_path.write_text(md)
    print(f"  Report saved: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
