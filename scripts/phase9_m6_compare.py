#!/usr/bin/env python
"""Phase 9 M6: side-by-side comparison report.

Reads:
  output/phase8_baseline/phase8_domain_results.json  (Phase 8 per-event preds)
  output/phase9_m6/density/progressive_results.csv   (Phase 9 density mode)
  output/phase9_m6/static/progressive_results.csv    (Phase 9 static control)

Writes:
  output/phase9_m6/phase9_m6_comparison.json
  output/phase9_m6/phase9_m6_comparison.md

Scoring rules (per PHASE9_SPEC §7.1 + §9 M6):
  - OMNI-covered subset = density_coverage >= 0.8 AND termination != "timeout".
  - Events outside that subset are tagged as fall-back-to-static; their
    Phase 9 score is the Phase 8 static baseline for that event.
  - "Phase 9 effective" = density result where in-subset, Phase 8 otherwise.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
import sys
_M65 = "--m65" in sys.argv
_M66 = "--m66" in sys.argv
_TAG = "m6_6" if _M66 else ("m6_5" if _M65 else "m6")
PHASE8_JSON   = ROOT / "output" / "phase8_baseline" / "phase8_domain_results.json"
DENSITY_CSV   = ROOT / "output" / f"phase9_{_TAG}" / "density" / "progressive_results.csv"
STATIC_CSV    = ROOT / "output" / f"phase9_{_TAG}" / "static"  / "progressive_results.csv"
OUT_JSON      = ROOT / "output" / f"phase9_{_TAG}" / f"phase9_{_TAG}_comparison.json"
OUT_MD        = ROOT / "output" / f"phase9_{_TAG}" / f"phase9_{_TAG}_comparison.md"

COVERAGE_THRESHOLD = 0.80  # PHASE9_SPEC §7.1


def parse_launch(iso: str) -> datetime:
    # Accept both "2026-01-01T03:53:00.0000000Z" and "2026-01-01T03:52:32.0000000+00:00".
    s = iso.replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def nearest_match(target_ts: float, index: list[tuple[float, str]], tol_sec: float = 120.0) -> str | None:
    # Linear scan is fine for 71 events. Returns the key of the closest entry
    # within ±tol_sec, or None. Built to bridge the sub-minute skew between
    # Phase 8 per_event (seconds preserved) and the progressive CSV
    # (feature_vectors rounds to whole minutes).
    best_key, best_dt = None, math.inf
    for ts, key in index:
        d = abs(ts - target_ts)
        if d < best_dt:
            best_dt, best_key = d, key
    return best_key if best_dt <= tol_sec else None


def load_phase8() -> dict:
    data = json.loads(PHASE8_JSON.read_text())
    by_key = {}
    index: list[tuple[float, str]] = []
    for i, row in enumerate(data["per_event"]):
        dt = parse_launch(row["launch_time_utc"])
        key = f"p8_{i}"
        by_key[key] = row
        index.append((dt.timestamp(), key))
    return {"meta": data, "by_key": by_key, "index": index}


def load_progressive(path: Path) -> dict:
    rows = {}
    index: list[tuple[float, str]] = []
    with path.open(newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            dt = parse_launch(r["launch_time"])
            key = f"pg_{i}"
            rows[key] = {
                "activity_id": r["activity_id"],
                "launch_time": r["launch_time"],
                "initial_speed_kms": float(r["initial_speed_kms"]),
                "observed_transit_hours": float(r["observed_transit_hours"]) if r["observed_transit_hours"] else None,
                "arrival_time_hours": float(r["arrival_time_hours"]) if r["arrival_time_hours"] else None,
                "error_hours": float(r["error_hours"]) if r["error_hours"] else None,
                "termination_reason": r["termination_reason"],
                "shock_arrived": r["shock_arrived"] == "1",
                "density_coverage": float(r["density_coverage"]),
                "gamma_eff_mean": float(r["gamma_eff_mean"]),
            }
            index.append((dt.timestamp(), key))
    return {"by_key": rows, "index": index}


def mae(xs):
    xs = [abs(x) for x in xs if x is not None and not math.isnan(x)]
    return mean(xs) if xs else None


def rmse(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return math.sqrt(mean(x * x for x in xs)) if xs else None


def bias(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return mean(xs) if xs else None


def main() -> None:
    phase8 = load_phase8()
    density = load_progressive(DENSITY_CSV)
    static  = load_progressive(STATIC_CSV)

    # Join on launch timestamp with ±120s tolerance. Phase 8 is authoritative.
    per_event = []
    for p8_key, p8 in phase8["by_key"].items():
        p8_ts = parse_launch(p8["launch_time_utc"]).timestamp()
        d_key = nearest_match(p8_ts, density["index"])
        s_key = nearest_match(p8_ts, static["index"])
        d = density["by_key"].get(d_key) if d_key else None
        s = static["by_key"].get(s_key) if s_key else None
        if d is None or s is None:
            # Should not happen on a clean run; tag so it shows up in the report.
            per_event.append({
                "launch_time_utc": p8["launch_time_utc"],
                "obs_transit_h": p8["obs_transit_h"],
                "phase8_pred_h": p8["pred_transit_h"],
                "phase8_error_h": p8["error_transit_h"],
                "phase9_density_pred_h": None,
                "phase9_density_error_h": None,
                "phase9_static_pred_h": None,
                "phase9_static_error_h": None,
                "density_coverage": None,
                "termination": "missing_from_progressive_run",
                "in_omni_subset": False,
                "fallback_reason": "missing_from_progressive_run",
                "phase9_effective_pred_h": p8["pred_transit_h"],
                "phase9_effective_error_h": p8["error_transit_h"],
            })
            continue

        in_subset = (d["density_coverage"] >= COVERAGE_THRESHOLD
                     and d["termination_reason"] != "timeout"
                     and d["arrival_time_hours"] is not None)

        fallback_reason = None
        if not in_subset:
            if d["density_coverage"] < COVERAGE_THRESHOLD:
                fallback_reason = f"low_coverage({d['density_coverage']:.2f})"
            elif d["termination_reason"] == "timeout":
                fallback_reason = "timeout_72h"
            else:
                fallback_reason = d["termination_reason"]

        if in_subset:
            effective_pred = d["arrival_time_hours"]
            effective_err  = d["error_hours"]
        else:
            effective_pred = p8["pred_transit_h"]
            effective_err  = p8["error_transit_h"]

        per_event.append({
            "launch_time_utc": p8["launch_time_utc"],
            "obs_transit_h": p8["obs_transit_h"],
            "phase8_pred_h": p8["pred_transit_h"],
            "phase8_error_h": p8["error_transit_h"],
            "phase9_density_pred_h": d["arrival_time_hours"],
            "phase9_density_error_h": d["error_hours"],
            "phase9_static_pred_h": s["arrival_time_hours"],
            "phase9_static_error_h": s["error_hours"],
            "density_coverage": d["density_coverage"],
            "termination": d["termination_reason"],
            "shock_arrived": d["shock_arrived"],
            "gamma_eff_mean": d["gamma_eff_mean"],
            "in_omni_subset": in_subset,
            "fallback_reason": fallback_reason,
            "phase9_effective_pred_h": effective_pred,
            "phase9_effective_error_h": effective_err,
        })

    subset_events     = [e for e in per_event if e["in_omni_subset"]]
    fallback_events   = [e for e in per_event if not e["in_omni_subset"]]

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_events_total": len(per_event),
        "n_events_omni_subset": len(subset_events),
        "n_events_fallback": len(fallback_events),
        "phase8_baseline": {
            "mae_all71":  mae([e["phase8_error_h"] for e in per_event]),
            "mae_subset": mae([e["phase8_error_h"] for e in subset_events]),
            "mae_fallback": mae([e["phase8_error_h"] for e in fallback_events]),
        },
        "phase9_density": {
            "mae_subset": mae([e["phase9_density_error_h"] for e in subset_events]),
            "rmse_subset": rmse([e["phase9_density_error_h"] for e in subset_events]),
            "bias_subset": bias([e["phase9_density_error_h"] for e in subset_events]),
        },
        "phase9_static_control": {
            "mae_subset": mae([e["phase9_static_error_h"] for e in subset_events]),
        },
        "phase9_effective": {
            "mae_all71": mae([e["phase9_effective_error_h"] for e in per_event]),
            "rmse_all71": rmse([e["phase9_effective_error_h"] for e in per_event]),
            "bias_all71": bias([e["phase9_effective_error_h"] for e in per_event]),
        },
        "ship_criterion": {
            "subset_mae_no_worse_than_12_33h":
                (mae([e["phase9_effective_error_h"] for e in per_event]) or 99) <= 12.33,
        },
        "per_event": per_event,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))

    # Markdown report
    lines = []
    lines.append("# Phase 9 M6 — Full Backtest Report\n")
    lines.append(f"Generated: {summary['generated_at']}\n")
    lines.append(f"Config: `configs/phase8_live_eval.yaml`  |  Events: {summary['n_events_total']}  |  "
                 f"OMNI subset: {summary['n_events_omni_subset']}  |  Fallbacks: {summary['n_events_fallback']}\n")
    lines.append("\n## Headline numbers (MAE, transit hours)\n")
    lines.append("| Metric | Phase 8 | Phase 9 density | Phase 9 static ctrl |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| MAE on OMNI-covered subset (n={len(subset_events)}) "
                 f"| {summary['phase8_baseline']['mae_subset']:.2f}h "
                 f"| {summary['phase9_density']['mae_subset']:.2f}h "
                 f"| {summary['phase9_static_control']['mae_subset']:.2f}h |")
    lines.append(f"| MAE on 71-event set (effective; fallbacks use Phase 8) "
                 f"| {summary['phase8_baseline']['mae_all71']:.2f}h "
                 f"| {summary['phase9_effective']['mae_all71']:.2f}h "
                 f"| — |")
    lines.append(f"| RMSE subset | — | {summary['phase9_density']['rmse_subset']:.2f}h | — |")
    lines.append(f"| Bias subset | — | {summary['phase9_density']['bias_subset']:+.2f}h | — |")
    lines.append("")
    lines.append("## Ship criterion (PHASE9_SPEC §7.1)\n")
    lines.append(f"- Aggregate 71-event MAE ≤ 12.33h? **"
                 f"{'PASS' if summary['ship_criterion']['subset_mae_no_worse_than_12_33h'] else 'FAIL'}"
                 f"** ({summary['phase9_effective']['mae_all71']:.2f}h)\n")

    # OMNI subset events detail
    lines.append("## OMNI-covered subset (Phase 9 density is scored here)\n")
    lines.append("| Launch (UTC) | Obs h | P8 pred | P8 err | P9 den pred | P9 den err | Δ (P9−P8) | term | shock |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")
    for e in sorted(subset_events, key=lambda x: x["launch_time_utc"]):
        delta = abs(e["phase9_density_error_h"]) - abs(e["phase8_error_h"])
        lines.append(
            f"| {e['launch_time_utc'][:19]} "
            f"| {e['obs_transit_h']:.1f} "
            f"| {e['phase8_pred_h']:.2f} "
            f"| {e['phase8_error_h']:+.2f} "
            f"| {e['phase9_density_pred_h']:.2f} "
            f"| {e['phase9_density_error_h']:+.2f} "
            f"| {delta:+.2f} "
            f"| {e['termination']} "
            f"| {'Y' if e['shock_arrived'] else '-'} |"
        )
    lines.append("")

    # Fallback events
    lines.append("## Fall-back events (not scored against Phase 9; use Phase 8 baseline)\n")
    lines.append("| Launch (UTC) | Obs h | P8 err | density cov | termination | reason |")
    lines.append("|---|---:|---:|---:|---|---|")
    for e in sorted(fallback_events, key=lambda x: x["launch_time_utc"]):
        cov = f"{e['density_coverage']:.2f}" if e["density_coverage"] is not None else "—"
        lines.append(
            f"| {e['launch_time_utc'][:19]} "
            f"| {e['obs_transit_h']:.1f} "
            f"| {e['phase8_error_h']:+.2f} "
            f"| {cov} "
            f"| {e['termination']} "
            f"| {e['fallback_reason']} |"
        )
    lines.append("")

    # Per-event win/loss tally on subset
    wins  = sum(1 for e in subset_events if abs(e["phase9_density_error_h"]) < abs(e["phase8_error_h"]))
    losses = sum(1 for e in subset_events if abs(e["phase9_density_error_h"]) > abs(e["phase8_error_h"]))
    ties  = len(subset_events) - wins - losses
    lines.append(f"## Subset tally\n\nPhase 9 density vs Phase 8 on {len(subset_events)} subset events: "
                 f"**{wins} wins, {losses} losses, {ties} ties**.\n")

    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"subset n={len(subset_events)} p8_mae={summary['phase8_baseline']['mae_subset']:.2f}h "
          f"p9_density_mae={summary['phase9_density']['mae_subset']:.2f}h")
    print(f"71-event effective MAE p9={summary['phase9_effective']['mae_all71']:.2f}h "
          f"(p8 baseline={summary['phase8_baseline']['mae_all71']:.2f}h)")


if __name__ == "__main__":
    main()
