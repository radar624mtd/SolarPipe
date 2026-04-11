#!/usr/bin/env python
"""Phase 9 M8 — scipy refinement of (γ₀, n_ref_scale) from grid warm-start.

Warm-starts from grid-search best (γ₀=2.297e-8, n_ref_scale=2.75) and uses
scipy.optimize.minimize (Nelder-Mead) to find the continuous optimum.

Outputs:
  output/phase9_m8/refined_progressive_results.csv
  output/phase9_m8/phase9_m8_refined_comparison.md
  output/phase9_m8/phase9_m8_refined_comparison.json
"""
from __future__ import annotations

import csv
import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize  # type: ignore

ROOT      = Path(__file__).resolve().parents[1]
DB_PATH   = ROOT / "solar_data.db"
M66_CSV   = ROOT / "output" / "phase9_m6_6" / "density" / "progressive_results.csv"
STATIC_CSV = ROOT / "output" / "phase9_m7_static" / "progressive_results.csv"
OUT_DIR   = ROOT / "output" / "phase9_m8"

R_SUN_KM  = 695700.0
R_START   = 21.5
R_TARGET  = 215.0
W_DEFAULT = 400.0
GAMMA_MIN = 1e-9
GAMMA_MAX = 1e-6
ODE_CAP   = 72.0

# Warm-start from grid best
GAMMA0_INIT     = 2.2974e-8
N_REF_SCALE_INIT = 2.75


# ---------------------------------------------------------------------------
# ODE (identical to calibrate script)
# ---------------------------------------------------------------------------

def drag_rhs(r, v, gamma, w):
    dr = v * 3600.0 / R_SUN_KM
    dv = -gamma * (v - w) * abs(v - w) * 3600.0
    return dr, dv


def rk4_step(r, v, gamma, w, h):
    k1r, k1v = drag_rhs(r, v, gamma, w)
    k2r, k2v = drag_rhs(r + 0.5*h*k1r, v + 0.5*h*k1v, gamma, w)
    k3r, k3v = drag_rhs(r + 0.5*h*k2r, v + 0.5*h*k2v, gamma, w)
    k4r, k4v = drag_rhs(r + h*k3r,     v + h*k3v,     gamma, w)
    return (r + (h/6.0)*(k1r + 2*k2r + 2*k3r + k4r),
            v + (h/6.0)*(k1v + 2*k2v + 2*k3v + k4v))


# ---------------------------------------------------------------------------
# OMNI helpers
# ---------------------------------------------------------------------------

def load_omni_window(conn, launch_dt, hours_before=24, hours_after=96):
    s = (launch_dt - timedelta(hours=hours_before)).strftime("%Y-%m-%d %H:%M")
    e = (launch_dt + timedelta(hours=hours_after)).strftime("%Y-%m-%d %H:%M")
    rows = conn.execute(
        "SELECT datetime, flow_speed, proton_density FROM omni_hourly "
        "WHERE datetime >= ? AND datetime <= ? ORDER BY datetime", (s, e)
    ).fetchall()
    result = []
    for r in rows:
        try:
            dt = datetime.strptime(r[0], "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        result.append((dt, r[1], r[2]))
    return result


def compute_pre_launch_median(conn, launch_dt, lookback_hours=150.0):
    n_slots = int(round(lookback_hours))
    window_start = launch_dt - timedelta(hours=lookback_hours)
    rows = conn.execute(
        "SELECT datetime, proton_density FROM omni_hourly "
        "WHERE datetime >= ? AND datetime < ? ORDER BY datetime",
        (window_start.strftime("%Y-%m-%d %H:%M"), launch_dt.strftime("%Y-%m-%d %H:%M"))
    ).fetchall()
    readings = {}
    for dtstr, n in rows:
        if n is None:
            continue
        try:
            dt = datetime.strptime(dtstr, "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        if not (0 < n < 9999.0):
            continue
        slot = int(round((dt - window_start).total_seconds() / 3600))
        if 0 <= slot < n_slots:
            readings[slot] = n
    if not readings:
        return float("nan")
    sorted_slots = sorted(readings)
    filled = []
    for i in range(n_slots):
        if i in readings:
            filled.append(readings[i])
            continue
        lefts  = [k for k in sorted_slots if k < i]
        rights = [k for k in sorted_slots if k > i]
        if lefts and rights:
            l2, r2 = lefts[-1], rights[0]
            t = (i - l2) / (r2 - l2)
            filled.append(readings[l2] + t * (readings[r2] - readings[l2]))
        elif lefts:
            filled.append(readings[lefts[-1]])
        else:
            filled.append(readings[rights[0]])
    filled.sort()
    mid = len(filled) // 2
    return filled[mid] if len(filled) % 2 else (filled[mid - 1] + filled[mid]) / 2.0


def compute_pre_launch_w(omni_rows, launch_dt):
    pre_vs = [row[1] for row in omni_rows if row[0] <= launch_dt and row[1] is not None]
    if pre_vs:
        return sum(pre_vs[-6:]) / max(len(pre_vs[-6:]), 1)
    return W_DEFAULT


# ---------------------------------------------------------------------------
# Core integrator
# ---------------------------------------------------------------------------

def integrate(v0, gamma0, n_ref_abs, omni_rows, launch_dt, h=1.0):
    """Returns (arrival_hours, shock_detected_t, n_missing, gamma_vals)."""
    w = compute_pre_launch_w(omni_rows, launch_dt)
    post = {}
    for row in omni_rows:
        if row[0] > launch_dt:
            hr_idx = int((row[0] - launch_dt).total_seconds() // 3600)
            post[hr_idx] = row

    r, v = R_START, v0
    t = 0.0
    n_missing = 0
    gamma_vals = []
    shock_t = None
    prev_n, prev_v_l1 = None, None

    while t < ODE_CAP:
        if r >= R_TARGET:
            return t, shock_t, n_missing, gamma_vals

        hr_idx = int(t) + 1
        step = post.get(hr_idx)
        if step is None or step[2] is None or step[2] <= 0:
            gamma_eff = gamma0
            n_missing += 1
            n_obs = None
            v_obs_l1 = None
        else:
            n_obs    = step[2]
            v_obs_l1 = step[1]
            gamma_eff = gamma0 * (n_obs / n_ref_abs)
            gamma_eff = max(GAMMA_MIN, min(GAMMA_MAX, gamma_eff))

            if (prev_n is not None and prev_v_l1 is not None
                    and v_obs_l1 is not None and prev_n > 0):
                if (v_obs_l1 - prev_v_l1) > 200.0 and (n_obs / prev_n) >= 3.0:
                    shock_t = t + 1.0
                    gamma_vals.append(gamma_eff)
                    return shock_t, shock_t, n_missing, gamma_vals

            prev_n = n_obs
            prev_v_l1 = v_obs_l1

        gamma_vals.append(gamma_eff)
        r, v = rk4_step(r, v, gamma_eff, w, h)
        t += h

    return None, shock_t, n_missing, gamma_vals


# ---------------------------------------------------------------------------
# Pre-load per-event data so the objective function doesn't hit the DB
# ---------------------------------------------------------------------------

def preload_event_data(conn, events):
    """Returns list of dicts with omni and n_med pre-computed."""
    data = []
    for ev in events:
        launch_dt = parse_launch(ev["launch_time"])
        omni  = load_omni_window(conn, launch_dt)
        n_med = compute_pre_launch_median(conn, launch_dt)
        total_slots = int(ODE_CAP)
        n_present   = sum(1 for row in omni
                          if row[0] > launch_dt
                          and row[2] is not None and row[2] > 0
                          and (row[0] - launch_dt).total_seconds() / 3600 <= total_slots)
        data.append({
            **ev,
            "launch_dt":       launch_dt,
            "omni":            omni,
            "n_med":           n_med,
            "density_coverage": n_present / total_slots,
        })
    return data


def parse_launch(iso: str) -> datetime:
    s = iso.replace("Z", "+00:00")
    return datetime.fromisoformat(s).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Objective: MAE on training arrivals, parameterised in log-space
# ---------------------------------------------------------------------------

_eval_count = 0

def objective(params, preloaded):
    global _eval_count
    _eval_count += 1

    log_g0, log_ns = params
    gamma0     = math.exp(log_g0)
    n_ref_scale = math.exp(log_ns)

    # Bounds guard (Nelder-Mead doesn't enforce bounds natively)
    if gamma0 < 1e-9 or gamma0 > 1e-6:
        return 1e6
    if n_ref_scale < 0.1 or n_ref_scale > 10.0:
        return 1e6

    errors = []
    for ev in preloaded:
        if ev["observed_h"] is None:
            continue
        n_med = ev["n_med"]
        n_ref_abs = (n_med if not math.isnan(n_med) else 5.0) * n_ref_scale
        arr, _, _, _ = integrate(ev["initial_speed"], gamma0, n_ref_abs,
                                  ev["omni"], ev["launch_dt"])
        if arr is not None:
            errors.append(arr - ev["observed_h"])

    if not errors:
        return 1e6
    return sum(abs(e) for e in errors) / len(errors)


# ---------------------------------------------------------------------------
# Full run with given params
# ---------------------------------------------------------------------------

def run_all(preloaded, gamma0, n_ref_scale):
    results = []
    for ev in preloaded:
        n_med = ev["n_med"]
        n_ref_abs = (n_med if not math.isnan(n_med) else 5.0) * n_ref_scale

        arr, shock_t, n_missing, gamma_vals = integrate(
            ev["initial_speed"], gamma0, n_ref_abs, ev["omni"], ev["launch_dt"])

        gamma_eff_mean  = sum(gamma_vals) / len(gamma_vals) if gamma_vals else gamma0
        gamma_eff_final = gamma_vals[-1] if gamma_vals else gamma0

        error_h = (arr - ev["observed_h"]) if (arr is not None and ev["observed_h"] is not None) else None
        term    = "shock_detected" if shock_t else ("target_reached" if arr else "timeout")

        results.append({
            "activity_id":           ev["activity_id"],
            "launch_time":           ev["launch_time"],
            "initial_speed_kms":     ev["initial_speed"],
            "observed_transit_hours": ev["observed_h"],
            "arrival_time_hours":    arr,
            "error_hours":           error_h,
            "termination_reason":    term,
            "shock_arrived":         1 if shock_t else 0,
            "n_missing_hours":       n_missing,
            "density_coverage":      ev["density_coverage"],
            "gamma_eff_mean":        gamma_eff_mean,
            "gamma_eff_final":       gamma_eff_final,
            "n_med":                 ev["n_med"],
            "n_ref_abs":             n_ref_abs,
        })
    return results


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "activity_id", "launch_time", "initial_speed_kms", "observed_transit_hours",
    "arrival_time_hours", "error_hours", "termination_reason", "shock_arrived",
    "n_missing_hours", "density_coverage", "gamma_eff_mean", "gamma_eff_final",
]


def write_csv(results, path):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in CSV_FIELDS}
            for k in ("arrival_time_hours", "error_hours", "observed_transit_hours"):
                if row[k] is None:
                    row[k] = ""
                elif isinstance(row[k], float):
                    row[k] = f"{row[k]:.3f}"
            for k in ("gamma_eff_mean", "gamma_eff_final"):
                if isinstance(row[k], float):
                    row[k] = f"{row[k]:.6E}"
            if isinstance(row["density_coverage"], float):
                row["density_coverage"] = f"{row['density_coverage']:.3f}"
            w.writerow(row)


def load_ref_csv(path):
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {r["activity_id"]: r for r in csv.DictReader(f)}


def _mae(vs): return sum(abs(v) for v in vs)/len(vs) if vs else None
def _rmse(vs): return math.sqrt(sum(v*v for v in vs)/len(vs)) if vs else None
def _bias(vs): return sum(vs)/len(vs) if vs else None
def fv(v): return f"{v:.2f}" if v is not None else "—"


def write_markdown(results, m66, static, gamma0, n_ref_scale, path):
    arrived = [r for r in results if r["error_hours"] is not None]
    errs    = [r["error_hours"] for r in arrived]

    m66_errs = [float(v["error_hours"]) for v in m66.values()
                if v.get("error_hours") and v["error_hours"] != ""]
    st_errs  = [float(v["error_hours"]) for v in static.values()
                if v.get("error_hours") and v["error_hours"] != ""]

    shock_rows = [r for r in results if r["shock_arrived"] == 1]

    lines = ["# Phase 9 M8 — Refined Joint Calibration Report\n"]
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
    lines.append(f"Optimiser: scipy Nelder-Mead (warm-start γ₀=2.297e-8, n_ref_scale=2.75)\n")
    lines.append(f"Refined params: γ₀ = {gamma0:.4E} km⁻¹  |  n_ref_scale = {n_ref_scale:.4f}\n")

    lines.append("\n## Headline numbers (arrived-event MAE)\n")
    lines.append("| Run | n_arrived / 71 | MAE | RMSE | Bias |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, n, e_list in [
        ("Phase 8 Static (baseline)", len(st_errs), st_errs),
        ("M6.6 All-density (n_ref=10)", len(m66_errs), m66_errs),
        (f"M8 Refined (γ₀={gamma0:.2E}, scale={n_ref_scale:.3f})", len(arrived), errs),
    ]:
        lines.append(f"| {label} | {n}/71 | {fv(_mae(e_list))} "
                     f"| {fv(_rmse(e_list))} | {fv(_bias(e_list))} |")

    lines.append("\n\n## Shock-detection events\n")
    lines.append("| Launch | v₀ | obs | M8r pred | M8r err | M6.6 err | Static err |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in shock_rows:
        act = r["activity_id"]
        m66e = float(m66[act]["error_hours"]) if m66.get(act, {}).get("error_hours") else None
        ste  = float(static[act]["error_hours"]) if static.get(act, {}).get("error_hours") else None
        lines.append(f"| {r['launch_time'][:19]} | {r['initial_speed_kms']:.0f} "
                     f"| {fv(r['observed_transit_hours'])} "
                     f"| {fv(r['arrival_time_hours'])} "
                     f"| {fv(r['error_hours'])} "
                     f"| {fv(m66e)} | {fv(ste)} |")

    lines.append("\n\n## Per-event table (arrived only)\n")
    lines.append("| Launch | v₀ | obs | M8r pred | M8r err | M6.6 err | Static err | shock |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in sorted(results, key=lambda x: x["launch_time"]):
        if r["error_hours"] is None:
            continue
        act = r["activity_id"]
        m66e = float(m66[act]["error_hours"]) if m66.get(act, {}).get("error_hours") else None
        ste  = float(static[act]["error_hours"]) if static.get(act, {}).get("error_hours") else None
        lines.append(
            f"| {r['launch_time'][:19]} | {r['initial_speed_kms']:.0f} "
            f"| {fv(r['observed_transit_hours'])} "
            f"| {fv(r['arrival_time_hours'])} "
            f"| {fv(r['error_hours'])} "
            f"| {fv(m66e)} | {fv(ste)} "
            f"| {'Y' if r['shock_arrived'] else '-'} |"
        )

    path.write_text("\n".join(lines))


def write_json(results, m66, static, gamma0, n_ref_scale, path):
    arrived = [r for r in results if r["error_hours"] is not None]
    errs    = [r["error_hours"] for r in arrived]
    m66_errs = [float(v["error_hours"]) for v in m66.values()
                if v.get("error_hours") and v["error_hours"] != ""]
    st_errs  = [float(v["error_hours"]) for v in static.values()
                if v.get("error_hours") and v["error_hours"] != ""]

    summary = {
        "generated_at":           datetime.now(timezone.utc).isoformat(),
        "gamma0_refined":         gamma0,
        "n_ref_scale_refined":    n_ref_scale,
        "phase8_static":          {"n_arrived": len(st_errs), "mae": _mae(st_errs),
                                   "rmse": _rmse(st_errs), "bias": _bias(st_errs)},
        "m6_6_density_nref10":    {"n_arrived": len(m66_errs), "mae": _mae(m66_errs),
                                   "rmse": _rmse(m66_errs), "bias": _bias(m66_errs)},
        "m8_refined":             {"n_arrived": len(arrived), "mae": _mae(errs),
                                   "rmse": _rmse(errs), "bias": _bias(errs)},
        "per_event": results,
    }
    path.write_text(json.dumps(summary, indent=2, default=str))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_event_list():
    events = []
    with M66_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            obs = float(row["observed_transit_hours"]) if row["observed_transit_hours"] else None
            events.append({
                "activity_id":   row["activity_id"],
                "launch_time":   row["launch_time"],
                "initial_speed": float(row["initial_speed_kms"]),
                "observed_h":    obs,
            })
    return events


def main():
    global _eval_count

    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found")
        return

    conn = sqlite3.connect(str(DB_PATH))
    events = load_event_list()
    print(f"Loaded {len(events)} events. Pre-loading OMNI windows...")

    preloaded = preload_event_data(conn, events)
    conn.close()
    print("Pre-load complete.")

    train = [e for e in preloaded if e["observed_h"] is not None]
    print(f"Training set: {len(train)} events with ground truth")

    # Nelder-Mead in log-space for scale-invariant search
    x0 = [math.log(GAMMA0_INIT), math.log(N_REF_SCALE_INIT)]
    print(f"\nStarting Nelder-Mead from γ₀={GAMMA0_INIT:.4E}, n_ref_scale={N_REF_SCALE_INIT:.3f}")
    print(f"  Warm-start MAE = {objective(x0, train):.4f}h")

    _eval_count = 0
    result = minimize(
        objective,
        x0,
        args=(train,),
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-4, "maxiter": 2000, "disp": True},
    )

    gamma0_opt     = math.exp(result.x[0])
    n_ref_scale_opt = math.exp(result.x[1])
    final_mae       = result.fun

    print(f"\nOptimisation complete ({_eval_count} evaluations)")
    print(f"  γ₀        = {gamma0_opt:.6E}  km⁻¹")
    print(f"  n_ref_scale = {n_ref_scale_opt:.6f}")
    print(f"  Training MAE = {final_mae:.4f}h")

    # Full 71-event run
    print("\nRunning all 71 events with refined params...")
    results = run_all(preloaded, gamma0_opt, n_ref_scale_opt)

    m66    = load_ref_csv(M66_CSV)
    static = load_ref_csv(STATIC_CSV)

    write_csv(results, OUT_DIR / "refined_progressive_results.csv")
    write_markdown(results, m66, static, gamma0_opt, n_ref_scale_opt,
                   OUT_DIR / "phase9_m8_refined_comparison.md")
    write_json(results, m66, static, gamma0_opt, n_ref_scale_opt,
               OUT_DIR / "phase9_m8_refined_comparison.json")

    arrived = [r for r in results if r["error_hours"] is not None]
    errs    = [r["error_hours"] for r in arrived]
    m66_errs = [float(v["error_hours"]) for v in m66.values()
                if v.get("error_hours") and v["error_hours"] != ""]
    st_errs  = [float(v["error_hours"]) for v in static.values()
                if v.get("error_hours") and v["error_hours"] != ""]

    print(f"\n{'='*60}")
    print(f"Phase 8 static:          n={len(st_errs):2}/71  MAE={_mae(st_errs):.2f}h  "
          f"RMSE={_rmse(st_errs):.2f}h  bias={_bias(st_errs):+.2f}h")
    print(f"M6.6 density n_ref=10:   n={len(m66_errs):2}/71  MAE={_mae(m66_errs):.2f}h  "
          f"RMSE={_rmse(m66_errs):.2f}h  bias={_bias(m66_errs):+.2f}h")
    print(f"M8 refined:              n={len(arrived):2}/71  MAE={_mae(errs):.2f}h  "
          f"RMSE={_rmse(errs):.2f}h  bias={_bias(errs):+.2f}h")
    print(f"  γ₀={gamma0_opt:.4E}  n_ref_scale={n_ref_scale_opt:.4f}")
    print(f"\nWrote output/phase9_m8/refined_*")


if __name__ == "__main__":
    main()
