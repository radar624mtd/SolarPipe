#!/usr/bin/env python
"""Phase 9 M8 — joint (γ₀, n_ref_scale) re-calibration.

The M7 post-mortem showed that γ₀ and n_ref are not independent: Phase 8
calibrated γ₀=2e-8 without density modulation, so n_ref cannot be treated as
a physics constant when we switch on γ_eff = γ₀·(n_obs/n_ref).

M8 approach:
  γ_eff(t) = γ₀ · (n_obs(t) / (n_ref_scale · n_med))

where:
  - n_med  = per-event 150h pre-launch OMNI density median (same derivation as M7)
  - n_ref_scale is a dimensionless free parameter (floats jointly with γ₀)

Training set  = all 71-event events that have observed_transit_hours (i.e. arrived
                in at least one reference run).  We optimise on the arrived subset.
Hold-out note = this is a throwaway script; the full 71-event run IS the
                hold-out (no separate test set — same as M6/M7 policy).

Grid search:
  γ₀         ∈ logspace(0.5e-8, 4e-8, 16 points)
  n_ref_scale ∈ linspace(0.5, 3.0, 11 points)   → 176 grid cells

For each (γ₀, n_ref_scale) pair we compute MAE over the training arrivals.
Best pair → re-run all 71 events → write progressive_results.csv + comparison .md

Output:
  output/phase9_m8/calibration_grid.csv
  output/phase9_m8/progressive_results.csv
  output/phase9_m8/phase9_m8_comparison.md
  output/phase9_m8/phase9_m8_comparison.json
"""
from __future__ import annotations

import csv
import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parents[1]
DB_PATH    = ROOT / "solar_data.db"
M66_CSV    = ROOT / "output" / "phase9_m6_6" / "density" / "progressive_results.csv"
STATIC_CSV = ROOT / "output" / "phase9_m7_static" / "progressive_results.csv"
OUT_DIR    = ROOT / "output" / "phase9_m8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

R_SUN_KM   = 695700.0
R_START    = 21.5      # solar radii
R_TARGET   = 215.0     # 1 AU
W_DEFAULT  = 400.0     # km/s background wind fallback
GAMMA_MIN  = 1e-9
GAMMA_MAX  = 1e-6
ODE_CAP    = 72.0      # hours (spec-mandated)

# Grid dimensions
GAMMA0_POINTS     = 16
N_REF_SCALE_POINTS = 11

# ---------------------------------------------------------------------------
# ODE helpers (RK4, identical to throwaway scripts)
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
        "WHERE datetime >= ? AND datetime <= ? ORDER BY datetime",
        (s, e)
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
    """150h pre-launch OMNI proton_density median (gap-filled, sentinels stripped)."""
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
    """Pre-launch 6h mean ambient wind speed."""
    pre_vs = [row[1] for row in omni_rows if row[0] <= launch_dt and row[1] is not None]
    if pre_vs:
        return sum(pre_vs[-6:]) / max(len(pre_vs[-6:]), 1)
    return W_DEFAULT

# ---------------------------------------------------------------------------
# Core integrator — density-modulated with joint (γ₀, n_ref_abs) where
#   n_ref_abs = n_ref_scale * n_med  (effective n_ref, cm⁻³)
# ---------------------------------------------------------------------------

def integrate_density_modulated(v0, gamma0, omni_rows, launch_dt,
                                 n_ref_abs, h=1.0):
    """
    Returns (arrival_hours, shock_detected, n_missing, gamma_eff_values).
    arrival_hours is None if 72h cap hit.
    """
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
    shock_detected_t = None
    prev_n, prev_v_l1 = None, None

    while t < ODE_CAP:
        if r >= R_TARGET:
            return t, shock_detected_t, n_missing, gamma_vals

        hr_idx = int(t) + 1
        step = post.get(hr_idx)
        if step is None or step[2] is None or step[2] <= 0:
            gamma_eff = gamma0
            n_missing += 1
            n_obs = None
            v_obs_l1 = None
        else:
            n_obs = step[2]
            v_obs_l1 = step[1]
            gamma_eff = gamma0 * (n_obs / n_ref_abs)
            gamma_eff = max(GAMMA_MIN, min(GAMMA_MAX, gamma_eff))

            # Shock detection
            if (prev_n is not None and prev_v_l1 is not None
                    and v_obs_l1 is not None and prev_n > 0):
                if (v_obs_l1 - prev_v_l1) > 200.0 and (n_obs / prev_n) >= 3.0:
                    shock_detected_t = t + 1.0
                    gamma_vals.append(gamma_eff)
                    return shock_detected_t, shock_detected_t, n_missing, gamma_vals

            prev_n = n_obs
            prev_v_l1 = v_obs_l1

        gamma_vals.append(gamma_eff)
        r, v = rk4_step(r, v, gamma_eff, w, h)
        t += h

    return None, shock_detected_t, n_missing, gamma_vals

# ---------------------------------------------------------------------------
# Load reference CSVs
# ---------------------------------------------------------------------------

def load_csv_by_id(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {r["activity_id"]: r for r in csv.DictReader(f)}


def load_m66() -> dict:
    return load_csv_by_id(M66_CSV)


def load_static() -> dict:
    return load_csv_by_id(STATIC_CSV)

# ---------------------------------------------------------------------------
# Build event list from M6.6 density CSV (authoritative 71-event set)
# ---------------------------------------------------------------------------

def load_event_list() -> list[dict]:
    events = []
    with M66_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            obs = float(row["observed_transit_hours"]) if row["observed_transit_hours"] else None
            events.append({
                "activity_id":    row["activity_id"],
                "launch_time":    row["launch_time"],
                "initial_speed":  float(row["initial_speed_kms"]),
                "observed_h":     obs,
            })
    return events


def parse_launch(iso: str) -> datetime:
    s = iso.replace("Z", "+00:00")
    return datetime.fromisoformat(s).replace(tzinfo=None)

# ---------------------------------------------------------------------------
# Single-event runner
# ---------------------------------------------------------------------------

def run_event(conn, event: dict, gamma0: float, n_ref_scale: float,
              n_med_cache: dict) -> dict:
    act_id   = event["activity_id"]
    launch_dt = parse_launch(event["launch_time"])
    v0       = event["initial_speed"]

    # Derive n_med (cached across grid points)
    if act_id not in n_med_cache:
        n_med_cache[act_id] = compute_pre_launch_median(conn, launch_dt)
    n_med = n_med_cache[act_id]

    if math.isnan(n_med):
        # Fallback: use n_ref_scale=1, n_med=5 as safe default
        n_ref_abs = 5.0 * n_ref_scale
    else:
        n_ref_abs = n_med * n_ref_scale

    omni = load_omni_window(conn, launch_dt)
    total_slots = int(ODE_CAP)
    n_present   = sum(1 for row in omni
                      if row[0] > launch_dt
                      and row[2] is not None and row[2] > 0
                      and (row[0] - launch_dt).total_seconds() / 3600 <= total_slots)
    density_coverage = n_present / total_slots

    arrival_h, shock_t, n_missing, gamma_vals = integrate_density_modulated(
        v0, gamma0, omni, launch_dt, n_ref_abs)

    gamma_eff_mean  = sum(gamma_vals) / len(gamma_vals) if gamma_vals else gamma0
    gamma_eff_final = gamma_vals[-1] if gamma_vals else gamma0

    if arrival_h is not None and event["observed_h"] is not None:
        error_h = arrival_h - event["observed_h"]
    else:
        error_h = None

    term = "shock_detected" if shock_t else ("target_reached" if arrival_h else "timeout")

    return {
        "activity_id":      act_id,
        "launch_time":      event["launch_time"],
        "initial_speed_kms": v0,
        "observed_transit_hours": event["observed_h"],
        "arrival_time_hours":     arrival_h,
        "error_hours":            error_h,
        "termination_reason":     term,
        "shock_arrived":          1 if shock_t else 0,
        "n_missing_hours":        n_missing,
        "density_coverage":       density_coverage,
        "gamma_eff_mean":         gamma_eff_mean,
        "gamma_eff_final":        gamma_eff_final,
        "n_med":                  n_med,
        "n_ref_abs":              n_ref_abs,
    }

# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def mae(vals):
    vs = [abs(v) for v in vals if v is not None and not math.isnan(v)]
    return sum(vs) / len(vs) if vs else float("inf")


def grid_search(conn, events: list[dict]) -> tuple[float, float, list]:
    """Return (best_gamma0, best_n_ref_scale, grid_rows)."""
    import numpy as np  # type: ignore

    gamma0_values     = np.logspace(math.log10(0.5e-8), math.log10(4e-8), GAMMA0_POINTS).tolist()
    n_ref_scale_values = [0.5 + i * (3.0 - 0.5) / (N_REF_SCALE_POINTS - 1)
                          for i in range(N_REF_SCALE_POINTS)]

    # Only calibrate on events that have ground truth
    train_events = [e for e in events if e["observed_h"] is not None]
    print(f"Grid search: {len(train_events)} training events with ground truth")
    print(f"  γ₀ grid: {GAMMA0_POINTS} pts in [{gamma0_values[0]:.2e}, {gamma0_values[-1]:.2e}]")
    print(f"  n_ref_scale grid: {N_REF_SCALE_POINTS} pts in [{n_ref_scale_values[0]:.2f}, {n_ref_scale_values[-1]:.2f}]")

    n_med_cache: dict[str, float] = {}
    grid_rows = []
    best_mae = float("inf")
    best_g0, best_ns = gamma0_values[8], n_ref_scale_values[4]

    total_cells = len(gamma0_values) * len(n_ref_scale_values)
    cell = 0

    for g0 in gamma0_values:
        for ns in n_ref_scale_values:
            cell += 1
            errors = []
            n_arrived = 0
            for ev in train_events:
                res = run_event(conn, ev, g0, ns, n_med_cache)
                if res["error_hours"] is not None:
                    errors.append(res["error_hours"])
                    n_arrived += 1

            cell_mae = mae(errors)
            grid_rows.append({
                "gamma0":          g0,
                "n_ref_scale":     ns,
                "n_arrived":       n_arrived,
                "mae_h":           cell_mae if cell_mae != float("inf") else None,
                "bias_h":          (sum(errors)/len(errors) if errors else None),
            })

            if cell_mae < best_mae:
                best_mae  = cell_mae
                best_g0   = g0
                best_ns   = ns

            if cell % 20 == 0 or cell == total_cells:
                print(f"  [{cell:3d}/{total_cells}] best so far: "
                      f"γ₀={best_g0:.3e}  n_ref_scale={best_ns:.2f}  MAE={best_mae:.2f}h")

    print(f"\nGrid search complete.")
    print(f"  Best: γ₀={best_g0:.3e}  n_ref_scale={best_ns:.2f}  MAE={best_mae:.2f}h "
          f"(on {len(train_events)} training events)")
    return best_g0, best_ns, grid_rows

# ---------------------------------------------------------------------------
# Full 71-event run with calibrated params
# ---------------------------------------------------------------------------

def run_all_events(conn, events, gamma0, n_ref_scale):
    n_med_cache: dict[str, float] = {}
    results = []
    for i, ev in enumerate(events):
        res = run_event(conn, ev, gamma0, n_ref_scale, n_med_cache)
        results.append(res)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(events)} events done")
    return results

# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "activity_id", "launch_time", "initial_speed_kms", "observed_transit_hours",
    "arrival_time_hours", "error_hours", "termination_reason", "shock_arrived",
    "n_missing_hours", "density_coverage", "gamma_eff_mean", "gamma_eff_final",
]

def write_progressive_csv(results: list[dict], path: Path) -> None:
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
            row["density_coverage"] = f"{row['density_coverage']:.3f}"
            w.writerow(row)


def write_calibration_csv(grid_rows: list[dict], path: Path) -> None:
    fields = ["gamma0", "n_ref_scale", "n_arrived", "mae_h", "bias_h"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in grid_rows:
            row = dict(r)
            row["gamma0"] = f"{row['gamma0']:.4E}"
            for k in ("mae_h", "bias_h"):
                row[k] = f"{row[k]:.3f}" if row[k] is not None else ""
            w.writerow(row)


def write_markdown(results: list[dict], m66: dict, static: dict,
                   gamma0: float, n_ref_scale: float, path: Path,
                   grid_best_mae: float) -> None:

    def fv(v): return f"{v:.2f}" if v is not None else "—"

    arrived_m8  = [r for r in results if r["error_hours"] is not None]
    m8_errs     = [r["error_hours"] for r in arrived_m8]

    m66_arrived = [k for k, v in m66.items()
                   if v.get("error_hours") and v["error_hours"] != ""]
    m66_errs    = [float(m66[k]["error_hours"]) for k in m66_arrived]

    st_arrived  = [k for k, v in static.items()
                   if v.get("error_hours") and v["error_hours"] != ""]
    st_errs     = [float(static[k]["error_hours"]) for k in st_arrived]

    def _mae(vs): return sum(abs(v) for v in vs)/len(vs) if vs else None
    def _rmse(vs): return math.sqrt(sum(v*v for v in vs)/len(vs)) if vs else None
    def _bias(vs): return sum(vs)/len(vs) if vs else None

    shock_rows  = [r for r in results if r["shock_arrived"] == 1]

    lines = ["# Phase 9 M8 — Joint (γ₀, n_ref_scale) Re-calibration Report\n"]
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
    lines.append(f"Calibrated params: γ₀ = {gamma0:.4E} km⁻¹  |  "
                 f"n_ref_scale = {n_ref_scale:.3f}  |  "
                 f"Training MAE (calibration) = {grid_best_mae:.2f}h\n")

    lines.append("\n## Headline numbers (arrived-event MAE)\n")
    lines.append("| Run | n_arrived / 71 | MAE | RMSE | Bias |")
    lines.append("|---|---:|---:|---:|---:|")

    def fmt(label, n, errs):
        lines.append(f"| {label} | {n}/71 | {fv(_mae(errs))} "
                     f"| {fv(_rmse(errs))} | {fv(_bias(errs))} |")

    fmt("Phase 8 Static (baseline)", len(st_arrived), st_errs)
    fmt("M6.6 All-density (n_ref=10)", len(m66_arrived), m66_errs)
    fmt("M8 Joint-calibrated", len(arrived_m8), m8_errs)

    lines.append("\n\n## Calibration summary\n")
    lines.append(f"Grid search: {GAMMA0_POINTS} × {N_REF_SCALE_POINTS} = "
                 f"{GAMMA0_POINTS*N_REF_SCALE_POINTS} cells  "
                 f"(γ₀ logspace 0.5e-8→4e-8, n_ref_scale linspace 0.5→3.0)\n")
    lines.append(f"Training set = events with ground truth in M6.6 CSV.  "
                 f"Hold-out = same 71-event set (no separate test split).\n")

    lines.append("\n## Shock-detection events\n")
    lines.append("| Launch | v₀ | obs | M8 pred | M8 err | M6.6 err | Static err |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in shock_rows:
        act = r["activity_id"]
        m66r = m66.get(act, {})
        str_ = static.get(act, {})
        m66e = float(m66r["error_hours"]) if m66r.get("error_hours") else None
        ste  = float(str_["error_hours"]) if str_.get("error_hours") else None
        lines.append(f"| {r['launch_time'][:19]} | {r['initial_speed_kms']:.0f} "
                     f"| {fv(r['observed_transit_hours'])} "
                     f"| {fv(r['arrival_time_hours'])} "
                     f"| {fv(r['error_hours'])} "
                     f"| {fv(m66e)} | {fv(ste)} |")

    lines.append("\n\n## Per-event table (arrived only)\n")
    lines.append("| Launch | v₀ | obs | M8 pred | M8 err | M6.6 err | Static err | shock |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in sorted(results, key=lambda x: x["launch_time"]):
        if r["error_hours"] is None:
            continue
        act = r["activity_id"]
        m66r = m66.get(act, {})
        str_ = static.get(act, {})
        m66e = float(m66r["error_hours"]) if m66r.get("error_hours") else None
        ste  = float(str_["error_hours"]) if str_.get("error_hours") else None
        lines.append(
            f"| {r['launch_time'][:19]} | {r['initial_speed_kms']:.0f} "
            f"| {fv(r['observed_transit_hours'])} "
            f"| {fv(r['arrival_time_hours'])} "
            f"| {fv(r['error_hours'])} "
            f"| {fv(m66e)} | {fv(ste)} "
            f"| {'Y' if r['shock_arrived'] else '-'} |"
        )

    path.write_text("\n".join(lines))


def write_json(results, m66, static, gamma0, n_ref_scale, grid_best_mae, path):
    def _mae(vs): return sum(abs(v) for v in vs)/len(vs) if vs else None
    def _rmse(vs): return math.sqrt(sum(v*v for v in vs)/len(vs)) if vs else None
    def _bias(vs): return sum(vs)/len(vs) if vs else None

    arrived  = [r for r in results if r["error_hours"] is not None]
    errs     = [r["error_hours"] for r in arrived]
    m66_errs = [float(v["error_hours"]) for v in m66.values()
                if v.get("error_hours") and v["error_hours"] != ""]
    st_errs  = [float(v["error_hours"]) for v in static.values()
                if v.get("error_hours") and v["error_hours"] != ""]

    summary = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "gamma0_calibrated": gamma0,
        "n_ref_scale_calibrated": n_ref_scale,
        "calibration_training_mae_h": grid_best_mae,
        "phase8_static": {
            "n_arrived": len(st_errs),
            "mae": _mae(st_errs), "rmse": _rmse(st_errs), "bias": _bias(st_errs),
        },
        "m6_6_density_nref10": {
            "n_arrived": len(m66_errs),
            "mae": _mae(m66_errs), "rmse": _rmse(m66_errs), "bias": _bias(m66_errs),
        },
        "m8_joint_calibrated": {
            "n_arrived": len(arrived),
            "mae": _mae(errs), "rmse": _rmse(errs), "bias": _bias(errs),
        },
        "per_event": results,
    }
    path.write_text(json.dumps(summary, indent=2, default=str))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not DB_PATH.exists():
        print(f"ERROR: database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH))

    events = load_event_list()
    print(f"Loaded {len(events)} events from M6.6 progressive CSV")

    m66    = load_m66()
    static = load_static()

    # --- Grid search ---
    print("\n=== Phase 1: Grid search ===")
    best_g0, best_ns, grid_rows = grid_search(conn, events)

    write_calibration_csv(grid_rows, OUT_DIR / "calibration_grid.csv")
    print(f"Wrote {OUT_DIR / 'calibration_grid.csv'}")

    # Find best MAE from grid_rows
    grid_best_mae = min((r["mae_h"] for r in grid_rows if r["mae_h"] is not None), default=float("inf"))

    # --- Full 71-event run ---
    print(f"\n=== Phase 2: Full 71-event run (γ₀={best_g0:.4E}, n_ref_scale={best_ns:.3f}) ===")
    results = run_all_events(conn, events, best_g0, best_ns)

    conn.close()

    # Write outputs
    write_progressive_csv(results, OUT_DIR / "progressive_results.csv")
    write_markdown(results, m66, static, best_g0, best_ns,
                   OUT_DIR / "phase9_m8_comparison.md", grid_best_mae)
    write_json(results, m66, static, best_g0, best_ns, grid_best_mae,
               OUT_DIR / "phase9_m8_comparison.json")

    # Console summary
    arrived = [r for r in results if r["error_hours"] is not None]
    errs    = [r["error_hours"] for r in arrived]
    def _mae(vs): return sum(abs(v) for v in vs)/len(vs) if vs else None
    def _bias(vs): return sum(vs)/len(vs) if vs else None
    def _rmse(vs): return math.sqrt(sum(v*v for v in vs)/len(vs)) if vs else None

    m66_errs = [float(v["error_hours"]) for v in m66.values()
                if v.get("error_hours") and v["error_hours"] != ""]
    st_errs  = [float(v["error_hours"]) for v in static.values()
                if v.get("error_hours") and v["error_hours"] != ""]

    print(f"\n{'='*60}")
    print(f"Phase 8 static:        n={len(st_errs):2}/71  MAE={_mae(st_errs):.2f}h  "
          f"RMSE={_rmse(st_errs):.2f}h  bias={_bias(st_errs):+.2f}h")
    print(f"M6.6 density n_ref=10: n={len(m66_errs):2}/71  MAE={_mae(m66_errs):.2f}h  "
          f"RMSE={_rmse(m66_errs):.2f}h  bias={_bias(m66_errs):+.2f}h")
    print(f"M8 joint-calibrated:   n={len(arrived):2}/71  MAE={_mae(errs):.2f}h  "
          f"RMSE={_rmse(errs):.2f}h  bias={_bias(errs):+.2f}h")
    print(f"  Calibrated: γ₀={best_g0:.4E}  n_ref_scale={best_ns:.3f}")
    print(f"\nWrote output/phase9_m8/")


if __name__ == "__main__":
    main()
