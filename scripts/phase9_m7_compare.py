#!/usr/bin/env python
"""Phase 9 M7 comparison report.

Reads:
  output/phase9_m7/progressive_results.csv        (hybrid routing: density>=540, static<540)
  output/phase9_m7_static/progressive_results.csv (pure Phase 8 static baseline)
  output/phase9_m6_6/density/progressive_results.csv (M6.6: all-density n_ref=10)

Writes:
  output/phase9_m7/phase9_m7_comparison.json
  output/phase9_m7/phase9_m7_comparison.md
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[1]
M7_CSV     = ROOT / "output" / "phase9_m7"         / "progressive_results.csv"
STATIC_CSV = ROOT / "output" / "phase9_m7_static"   / "progressive_results.csv"
M66_CSV    = ROOT / "output" / "phase9_m6_6" / "density" / "progressive_results.csv"
OUT_JSON   = ROOT / "output" / "phase9_m7"   / "phase9_m7_comparison.json"
OUT_MD     = ROOT / "output" / "phase9_m7"   / "phase9_m7_comparison.md"

SPEED_THRESHOLD = 540.0


def load_csv(path: Path) -> dict:
    return {r["activity_id"]: r for r in csv.DictReader(path.open(newline=""))}


def err(r): return float(r["error_hours"]) if r and r.get("error_hours") else None
def v0(r):  return float(r["initial_speed_kms"])
def obs(r): return float(r["observed_transit_hours"]) if r.get("observed_transit_hours") else None
def arr(r): return float(r["arrival_time_hours"]) if r.get("arrival_time_hours") else None


def _mae(vals):
    vs = [abs(v) for v in vals if v is not None and not math.isnan(v)]
    return sum(vs) / len(vs) if vs else None

def _rmse(vals):
    vs = [v for v in vals if v is not None and not math.isnan(v)]
    return math.sqrt(sum(v*v for v in vs) / len(vs)) if vs else None

def _bias(vals):
    vs = [v for v in vals if v is not None and not math.isnan(v)]
    return sum(vs) / len(vs) if vs else None


def main() -> None:
    m7     = load_csv(M7_CSV)
    static = load_csv(STATIC_CSV)
    m66    = load_csv(M66_CSV) if M66_CSV.exists() else {}

    events = sorted(m7.keys(), key=lambda k: m7[k]["launch_time"])

    # --- Per-event joined table ---
    rows = []
    for k in events:
        r7 = m7[k]
        rs = static.get(k, {})
        rm = m66.get(k, {})
        e7 = err(r7); es = err(rs); em = err(rm)
        speed = v0(r7)
        # For M7 hybrid, the effective mode per event:
        # density for speed >= threshold, static for speed < threshold
        mode_eff = "density" if speed >= SPEED_THRESHOLD else "static"
        rows.append({
            "activity_id":   k,
            "launch_time":   r7["launch_time"],
            "initial_speed": speed,
            "mode_effective": mode_eff,
            "obs_hours":     obs(r7),
            # M7 hybrid
            "m7_arr":   arr(r7), "m7_err":  e7, "m7_term": r7["termination_reason"],
            "m7_shock": r7.get("shock_arrived","0") == "1",
            # Phase 8 static
            "s_arr":    arr(rs), "s_err":   es, "s_term":  rs.get("termination_reason","?"),
            # M6.6 density n_ref=10
            "m66_arr":  arr(rm), "m66_err": em, "m66_term": rm.get("termination_reason","?"),
        })

    # --- Summary stats ---
    def stats_from(field, rows_, term_field="m7_term"):
        vals = [r[field] for r in rows_]
        return {"mae": _mae(vals), "rmse": _rmse(vals), "bias": _bias(vals),
                "n": sum(1 for v in vals if v is not None),
                "n_timeout": sum(1 for r in rows_ if r.get(term_field) == "timeout")}

    arrived_m7    = [r for r in rows if r["m7_err"] is not None]
    arrived_s     = [r for r in rows if r["s_err"]  is not None]
    arrived_m66   = [r for r in rows if r["m66_err"] is not None]
    fast_rows     = [r for r in rows if r["initial_speed"] >= SPEED_THRESHOLD]
    slow_rows     = [r for r in rows if r["initial_speed"] <  SPEED_THRESHOLD]
    shock_rows    = [r for r in rows if r["m7_shock"]]

    m7_errs   = [r["m7_err"]  for r in arrived_m7]
    s_errs    = [r["s_err"]   for r in arrived_s]
    m66_errs  = [r["m66_err"] for r in arrived_m66]

    # Events where all 3 produced arrivals (fair comparison)
    all3 = [r for r in rows if r["m7_err"] is not None and r["s_err"] is not None and r["m66_err"] is not None]

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "speed_threshold_kms": SPEED_THRESHOLD,
        "n_events_total": len(rows),
        "n_fast": len(fast_rows),
        "n_slow": len(slow_rows),
        "phase8_static": {
            "n_arrived": len(arrived_s), "n_timeout": len(rows) - len(arrived_s),
            "mae": _mae(s_errs), "rmse": _rmse(s_errs), "bias": _bias(s_errs),
        },
        "m6_6_density_nref10": {
            "n_arrived": len(arrived_m66), "n_timeout": len(rows) - len(arrived_m66),
            "mae": _mae(m66_errs), "rmse": _rmse(m66_errs), "bias": _bias(m66_errs),
        },
        "m7_hybrid": {
            "n_arrived": len(arrived_m7), "n_timeout": len(rows) - len(arrived_m7),
            "n_shock": len(shock_rows),
            "mae": _mae(m7_errs), "rmse": _rmse(m7_errs), "bias": _bias(m7_errs),
        },
        "matched_all3": {
            "n": len(all3),
            "m7_mae": _mae([r["m7_err"] for r in all3]),
            "s_mae":  _mae([r["s_err"]  for r in all3]),
            "m66_mae": _mae([r["m66_err"] for r in all3]),
        },
        "fast_cme_subset": {
            "n": len(fast_rows),
            "m7_arrived": sum(1 for r in fast_rows if r["m7_err"] is not None),
            "m66_arrived": sum(1 for r in fast_rows if r["m66_err"] is not None),
            "m7_mae": _mae([r["m7_err"] for r in fast_rows if r["m7_err"] is not None]),
            "m66_mae": _mae([r["m66_err"] for r in fast_rows if r["m66_err"] is not None]),
        },
        "slow_cme_subset": {
            "n": len(slow_rows),
            "m7_arrived": sum(1 for r in slow_rows if r["m7_err"] is not None),
            "s_arrived": sum(1 for r in slow_rows if r["s_err"] is not None),
            "m7_mae": _mae([r["m7_err"] for r in slow_rows if r["m7_err"] is not None]),
            "s_mae": _mae([r["s_err"] for r in slow_rows if r["s_err"] is not None]),
        },
        "per_event": rows,
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))

    # --- Markdown ---
    lines = []
    lines.append("# Phase 9 M7 — Hybrid Routing Backtest Report\n")
    lines.append(f"Generated: {summary['generated_at']}\n")
    lines.append(f"Speed threshold: {SPEED_THRESHOLD:.0f} km/s  "
                 f"(density mode for v₀ ≥ {SPEED_THRESHOLD:.0f}, static for v₀ < {SPEED_THRESHOLD:.0f})\n")

    lines.append("\n## Headline numbers (arrived-event MAE)\n")
    lines.append("| Run | n_arrived / 71 | MAE | RMSE | Bias |")
    lines.append("|---|---:|---:|---:|---:|")
    def fmt_row(label, n, mae_v, rmse_v, bias_v):
        def f(v): return f"{v:.2f}h" if v is not None else "—"
        lines.append(f"| {label} | {n}/71 | {f(mae_v)} | {f(rmse_v)} | {f(bias_v)} |")

    fmt_row("Phase 8 Static (baseline)", len(arrived_s), _mae(s_errs), _rmse(s_errs), _bias(s_errs))
    fmt_row("M6.6 All-density (n_ref=10)", len(arrived_m66), _mae(m66_errs), _rmse(m66_errs), _bias(m66_errs))
    fmt_row("M7 Hybrid (derived n_ref)", len(arrived_m7), _mae(m7_errs), _rmse(m7_errs), _bias(m7_errs))
    lines.append("")

    lines.append("\n## Speed-bucket breakdown\n")
    lines.append(f"**Fast CMEs (v₀ ≥ {SPEED_THRESHOLD:.0f} km/s, n={len(fast_rows)}):**\n")
    lines.append("| Run | arrived | MAE |")
    lines.append("|---|---:|---:|")
    lines.append(f"| M6.6 density n_ref=10 | {sum(1 for r in fast_rows if r['m66_err'] is not None)}/{len(fast_rows)} | {_mae([r['m66_err'] for r in fast_rows if r['m66_err'] is not None]):.2f}h |")
    lines.append(f"| M7 hybrid (density, derived n_ref) | {sum(1 for r in fast_rows if r['m7_err'] is not None)}/{len(fast_rows)} | {_mae([r['m7_err'] for r in fast_rows if r['m7_err'] is not None]):.2f}h |")
    lines.append("")

    lines.append(f"\n**Slow CMEs (v₀ < {SPEED_THRESHOLD:.0f} km/s, n={len(slow_rows)}):**\n")
    lines.append("| Run | arrived | MAE |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Phase 8 static | {sum(1 for r in slow_rows if r['s_err'] is not None)}/{len(slow_rows)} | {(_mae([r['s_err'] for r in slow_rows if r['s_err'] is not None]) or 0):.2f}h |")
    lines.append(f"| M7 hybrid (static) | same as Phase 8 static | — |")
    lines.append("")

    lines.append("\n## Shock-detection events\n")
    lines.append("| Launch | v₀ | obs | M7 pred | M7 err | M6.6 err | Static err |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    def fv(v): return f"{v:.2f}" if v is not None else "—"
    for r in shock_rows:
        lines.append(f"| {r['launch_time'][:19]} | {r['initial_speed']:.0f} | {fv(r['obs_hours'])} "
                     f"| {fv(r['m7_arr'])} | {fv(r['m7_err'])} | {fv(r['m66_err'])} | {fv(r['s_err'])} |")
    lines.append("")

    lines.append("\n## Analysis: why M7 hybrid is worse than M6.6 on the fast-CME subset\n")
    lines.append(
        "M6.6 used `n_ref=10.0` (global constant) whereas M7 derives n_ref per-event from the\n"
        "150h pre-launch OMNI window (ambient median ~4–5 cm⁻³ in 2026 solar maximum).\n\n"
        "With n_ref=10 the effective drag is halved: γ_eff ≈ 0.5·γ₀. This compensates for a\n"
        "systematic over-prediction of drag in Phase 8's calibrated γ₀, which was tuned against\n"
        "static mode (no density modulation). When n_ref is physically correct, γ_eff ≈ γ₀\n"
        "during typical solar wind — matching the static result — but becomes *stronger* during\n"
        "dense ambient periods (pre-event solar wind compressed by preceding events), causing\n"
        "over-deceleration and large positive transit-time errors for fast CMEs.\n\n"
        "**Root cause**: γ₀ must be re-calibrated jointly with the density-modulation formula.\n"
        "The correct next step (M8) is a joint fit of (γ₀, n_ref_target) on the training set,\n"
        "treating n_ref as a free scale parameter in the ODE, not a physics constant.\n"
    )

    lines.append("\n## Per-event table (arrived only)\n")
    lines.append("| Launch | v₀ | obs | M7 arr | M7 err | M6.6 err | Static err | mode | shock |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")
    for r in sorted(rows, key=lambda x: x["launch_time"]):
        if r["m7_err"] is None: continue
        lines.append(
            f"| {r['launch_time'][:19]} | {r['initial_speed']:.0f} | {fv(r['obs_hours'])} "
            f"| {fv(r['m7_arr'])} | {fv(r['m7_err'])} | {fv(r['m66_err'])} | {fv(r['s_err'])} "
            f"| {r['mode_effective']} | {'Y' if r['m7_shock'] else '-'} |"
        )
    lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print()
    print(f"Phase 8 static:        n={len(arrived_s):2}/71 MAE={_mae(s_errs):.2f}h RMSE={_rmse(s_errs):.2f}h bias={_bias(s_errs):+.2f}h")
    print(f"M6.6 density n_ref=10: n={len(arrived_m66):2}/71 MAE={_mae(m66_errs):.2f}h RMSE={_rmse(m66_errs):.2f}h bias={_bias(m66_errs):+.2f}h")
    print(f"M7 hybrid derived:     n={len(arrived_m7):2}/71 MAE={_mae(m7_errs):.2f}h RMSE={_rmse(m7_errs):.2f}h bias={_bias(m7_errs):+.2f}h")


if __name__ == "__main__":
    main()
