"""
Walk-forward backtest for the last N events in the CME catalog.

For each test event i (in the last N events with confirmed transit times):
  1. Replace the training_features VIEW cutoff to: launch_time < event[i].launch_time
     (minus gap_buffer_days to avoid leakage of near-miss events)
  2. Run: dotnet run -- train --config <yaml> --no-cache
  3. Run: dotnet run -- predict --config <yaml> --input <single-row csv> --output <tmp json>
  4. Score: error_hours = predicted - observed
  5. Restore VIEW and registry/cache to production state when done.

All failures are caught, logged, and written to logs/walkforward_errors.json — no aborts.
"""

import argparse
import sqlite3
import csv
import json
import subprocess
import os
import sys
import datetime
import math
import shutil
import tempfile

# ── Defaults (overridden by CLI args) ─────────────────────────────────────────
DB_PATH        = "data/data/output/cme_catalog.db"
YAML_CONFIG    = "configs/flux_rope_propagation_v1.yaml"
DOTNET_PROJECT = "src/SolarPipe.Host"
N_TEST_EVENTS  = 100
GAP_BUFFER_DAYS = 5
REGISTRY_DIR   = "models/registry"
CACHE_DIR      = "cache"
OUTPUT_SCORED  = "output/walkforward_backtest_scored.json"
LOG_PATH       = "logs/walkforward_backtest_run.json"
ERROR_LOG      = "logs/walkforward_errors.json"
TRAIN_TIMEOUT_S = 120
PREDICT_TIMEOUT_S = 60

# ── Helpers ───────────────────────────────────────────────────────────────────

ORIGINAL_VIEW_SQL = """CREATE VIEW training_features AS
                SELECT
                    e.event_id,
                    e.launch_time,
                    e.cme_speed          AS cme_speed_kms,
                    e.sw_speed_ambient   AS sw_speed_ambient_kms,
                    e.sw_density_ambient AS sw_density_n_cc,
                    e.sw_bt_ambient      AS sw_bt_nt,
                    e.f10_7,
                    e.quality_flag,
                    f.observed_bz_min    AS bz_gsm_proxy_nt,
                    a.transit_time_hours,
                    a.dst_min_nT         AS dst_min_nT,
                    a.kp_max,
                    a.has_in_situ_fit
                FROM cme_events e
                JOIN l1_arrivals a ON e.event_id = a.event_id
                JOIN flux_rope_fits f ON e.event_id = f.event_id
                WHERE a.transit_time_hours IS NOT NULL
                  AND e.cme_speed IS NOT NULL
                  AND e.quality_flag >= 2
                  AND e.launch_time < '2026-01-01'"""

def set_view_cutoff(conn, cutoff_iso: str):
    conn.execute("DROP VIEW IF EXISTS training_features")
    sql = ORIGINAL_VIEW_SQL.replace("AND e.launch_time < '2026-01-01'",
                                    f"AND e.launch_time < '{cutoff_iso}'")
    conn.execute(sql)
    conn.commit()

def restore_view(conn):
    conn.execute("DROP VIEW IF EXISTS training_features")
    conn.execute(ORIGINAL_VIEW_SQL)
    conn.commit()

def clear_registry_and_cache():
    """Remove all trained model artifacts so each fold starts clean."""
    for d in [REGISTRY_DIR, CACHE_DIR]:
        if os.path.isdir(d):
            for item in os.listdir(d):
                item_path = os.path.join(d, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
                elif os.path.isfile(item_path) and item != ".gitkeep":
                    os.remove(item_path)

def run_train(event_id: str) -> dict:
    """Run dotnet train. Returns {'ok': bool, 'stdout': str, 'stderr': str}."""
    result = subprocess.run(
        ["dotnet", "run", "--project", DOTNET_PROJECT, "--no-build",
         "--", "train", "--config", YAML_CONFIG, "--no-cache"],
        capture_output=True, text=True, timeout=TRAIN_TIMEOUT_S
    )
    return {
        "ok": result.returncode == 0,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "returncode": result.returncode,
    }

def build_predict_csv(row: dict, path: str):
    """Write single-row predict input CSV for this event.

    Writes all columns present in training_features_v3 so the predict command
    can satisfy any feature list a v1/v2/v3 config might require.
    Missing columns (not in row dict) are written as empty string (NULL).
    """
    def fmt(v):
        return "" if v is None else str(v)

    # All columns that may appear in any config version's feature list
    all_cols = [
        "cme_speed_kms", "bz_gsm_proxy_nt", "sw_density_n_cc",
        "sw_speed_ambient_kms", "sw_bt_nt",
        "delta_v_kms", "speed_ratio", "speed_x_bz", "speed_x_density",
        "radial_speed_km_s", "cme_speed_radial_kms",
        "delta_v_radial_kms", "speed_ratio_radial",
        "cme_half_angle_deg", "cme_abs_longitude_deg", "cme_latitude_deg",
        "cdaw_second_order_speed_final", "cdaw_accel_kms2",
        "cdaw_angular_width_deg", "cdaw_mpa_deg", "cdaw_mass_log10_g",
        "cdaw_obs",
        "usflux", "totpot", "totusjz", "r_value", "shrgt45", "meanshr",
        "sharp_obs",
        "sunspot_number", "bz_southward_nt",
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(all_cols)
        w.writerow([fmt(row.get(c)) for c in all_cols])

def run_predict(input_csv: str, output_json: str) -> dict:
    """Run dotnet predict. Returns {'ok': bool, 'value': float|None, ...}."""
    result = subprocess.run(
        ["dotnet", "run", "--project", DOTNET_PROJECT, "--no-build",
         "--", "predict", "--config", YAML_CONFIG,
         "--input", input_csv, "--output", output_json],
        capture_output=True, text=True, timeout=PREDICT_TIMEOUT_S
    )
    if result.returncode != 0:
        return {"ok": False, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(), "value": None}
    try:
        with open(output_json) as f:
            data = json.load(f)
        values = data[0]["values"]
        v = values[0] if values else None
        return {"ok": True, "value": v, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except Exception as e:
        return {"ok": False, "value": None, "error": str(e),
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global YAML_CONFIG, OUTPUT_SCORED

    parser = argparse.ArgumentParser(description="Walk-forward backtest for SolarPipe pipeline")
    parser.add_argument("--config",  default=YAML_CONFIG,   help="Pipeline YAML config path")
    parser.add_argument("--output",  default=OUTPUT_SCORED, help="Output JSON path")
    parser.add_argument("--n",       type=int, default=N_TEST_EVENTS, help="Number of test folds")
    args = parser.parse_args()
    YAML_CONFIG   = args.config
    OUTPUT_SCORED = args.output
    n_test = args.n

    os.makedirs("output", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Fetch all scoreable events with full v3 feature set for predict CSV generation.
    # The LEFT JOINs pull extended_features and cdaw_supplements if they exist;
    # missing columns resolve to NULL and are written as empty string in the CSV.
    rows = conn.execute("""
        SELECT
            e.event_id,
            e.launch_time,
            e.cme_speed            AS cme_speed_kms,
            f.observed_bz_min      AS bz_gsm_proxy_nt,
            e.sw_density_ambient   AS sw_density_n_cc,
            e.sw_speed_ambient     AS sw_speed_ambient_kms,
            e.sw_bt_ambient        AS sw_bt_nt,
            a.transit_time_hours   AS transit_hours_observed,
            -- v3 cone-corrected features (zero-fill mirrors training_features_v3 view)
            ef.cme_speed_radial_kms                                                AS radial_speed_km_s,
            ef.cme_speed_radial_kms                                                AS cme_speed_radial_kms,
            ef.cme_half_angle_deg,
            COALESCE(ef.cme_abs_longitude_deg, 0.0)                                AS cme_abs_longitude_deg,
            ef.cme_latitude_deg,
            -- v3 CDAW features (zero-fill mirrors training_features_v3 view)
            COALESCE(cs.cdaw_second_order_speed_final, 0.0)            AS cdaw_second_order_speed_final,
            COALESCE(cs.cdaw_accel_kms2,               0.0)            AS cdaw_accel_kms2,
            COALESCE(cs.cdaw_angular_width_deg,        0.0)            AS cdaw_angular_width_deg,
            cs.cdaw_mpa_deg,
            COALESCE(cs.cdaw_mass_log10_g,             0.0)            AS cdaw_mass_log10_g,
            CASE WHEN cs.event_id IS NOT NULL THEN 1.0 ELSE 0.0 END   AS cdaw_obs,
            -- v3 SHARP features (zero-fill mirrors training_features_v3 view)
            ef.usflux,
            COALESCE(ef.totpot,   0.0)                                 AS totpot,
            COALESCE(ef.totusjz,  0.0)                                 AS totusjz,
            COALESCE(ef.r_value,  0.0)                                 AS r_value,
            ef.shrgt45,
            ef.meanshr,
            CASE WHEN ef.totpot IS NOT NULL THEN 1.0 ELSE 0.0 END     AS sharp_obs,
            -- v3 solar cycle
            ef.sunspot_number,
            -- derived features (computed here; view does the same)
            (e.cme_speed - COALESCE(e.sw_speed_ambient, 420.0))                   AS delta_v_kms,
            (e.cme_speed / COALESCE(e.sw_speed_ambient, 420.0))                   AS speed_ratio,
            (e.cme_speed * f.observed_bz_min)                                     AS speed_x_bz,
            (e.cme_speed * e.sw_density_ambient)                                  AS speed_x_density,
            (ef.cme_speed_radial_kms - COALESCE(e.sw_speed_ambient, 420.0))       AS delta_v_radial_kms,
            (ef.cme_speed_radial_kms / COALESCE(e.sw_speed_ambient, 420.0))       AS speed_ratio_radial,
            CASE WHEN f.observed_bz_min < 0 THEN -f.observed_bz_min ELSE 0 END   AS bz_southward_nt
        FROM cme_events e
        JOIN l1_arrivals a ON a.event_id = e.event_id
        JOIN flux_rope_fits f ON f.event_id = e.event_id
        LEFT JOIN extended_features  ef ON ef.event_id = e.event_id
        LEFT JOIN cdaw_supplements   cs ON cs.event_id = e.event_id
        WHERE a.transit_time_hours IS NOT NULL
          AND a.transit_time_hours > 0
          AND e.cme_speed IS NOT NULL
        ORDER BY e.launch_time
    """).fetchall()

    all_events = [dict(r) for r in rows]
    test_events = all_events[-n_test:]

    print(f"Config: {YAML_CONFIG}")
    print(f"Total scoreable events: {len(all_events)}")
    print(f"Test window: {test_events[0]['launch_time']} → {test_events[-1]['launch_time']}")
    print(f"Running {n_test} walk-forward folds...\n")

    results = []
    errors = []
    tmpdir = tempfile.mkdtemp()

    try:
        for fold_i, event in enumerate(test_events):
            eid      = event["event_id"]
            launch   = event["launch_time"]   # ISO string e.g. "2025-12-24T10:36:00+00:00"
            obs      = event["transit_hours_observed"]
            speed    = event["cme_speed_kms"]

            # Gap buffer: exclude events within GAP_BUFFER_DAYS before this event's launch
            # Cutoff = launch_time - 5 days (ISO string comparison works for ISO8601)
            launch_dt = datetime.datetime.fromisoformat(launch.replace("+00:00", "+00:00"))
            cutoff_dt = launch_dt - datetime.timedelta(days=GAP_BUFFER_DAYS)
            cutoff_iso = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%S")

            # Count training events that will be available
            n_train = sum(1 for e in all_events
                          if e["launch_time"] < cutoff_iso and e["transit_hours_observed"] > 0)

            print(f"[{fold_i+1:3d}/{n_test}] {eid}  speed={speed:.0f}  obs={obs:.1f}h  "
                  f"train_n={n_train}  cutoff={cutoff_iso[:10]}", flush=True)

            if n_train < 50:
                msg = f"Insufficient training data: {n_train} events before gap cutoff"
                print(f"  SKIP: {msg}")
                errors.append({"fold": fold_i+1, "event_id": eid, "error": msg})
                results.append({
                    "fold": fold_i+1, "event_id": eid, "launch_time": launch,
                    "cme_speed_kms": speed, "transit_hours_observed": obs,
                    "predicted_transit_hours": None, "error_hours": None,
                    "abs_error_hours": None, "train_n": n_train,
                    "status": "SKIP_INSUFFICIENT_TRAIN"
                })
                continue

            # Skip events where physics ODE won't run (speed < 200)
            if speed < 200:
                print(f"  SKIP: speed {speed} < 200 km/s (ODE range RULE-032)")
                errors.append({"fold": fold_i+1, "event_id": eid,
                               "error": f"speed {speed} < 200 km/s outside DragBasedModel range"})
                results.append({
                    "fold": fold_i+1, "event_id": eid, "launch_time": launch,
                    "cme_speed_kms": speed, "transit_hours_observed": obs,
                    "predicted_transit_hours": None, "error_hours": None,
                    "abs_error_hours": None, "train_n": n_train,
                    "status": "SKIP_SPEED_OOB"
                })
                continue

            # 1. Set view cutoff
            try:
                set_view_cutoff(conn, cutoff_iso)
            except Exception as e:
                msg = f"VIEW swap failed: {e}"
                print(f"  ERROR: {msg}")
                errors.append({"fold": fold_i+1, "event_id": eid, "error": msg})
                results.append({
                    "fold": fold_i+1, "event_id": eid, "launch_time": launch,
                    "cme_speed_kms": speed, "transit_hours_observed": obs,
                    "predicted_transit_hours": None, "error_hours": None,
                    "abs_error_hours": None, "train_n": n_train, "status": "ERROR_VIEW"
                })
                continue

            # 2. Clear registry + cache so we train fresh
            clear_registry_and_cache()

            # 3. Train
            try:
                train_result = run_train(eid)
            except subprocess.TimeoutExpired:
                train_result = {"ok": False, "stderr": "TIMEOUT", "stdout": ""}

            if not train_result["ok"]:
                msg = f"Train failed rc={train_result.get('returncode')}: {train_result['stderr'][:200]}"
                print(f"  ERROR: {msg}")
                errors.append({"fold": fold_i+1, "event_id": eid, "error": msg,
                               "train_stdout": train_result["stdout"],
                               "train_stderr": train_result["stderr"]})
                results.append({
                    "fold": fold_i+1, "event_id": eid, "launch_time": launch,
                    "cme_speed_kms": speed, "transit_hours_observed": obs,
                    "predicted_transit_hours": None, "error_hours": None,
                    "abs_error_hours": None, "train_n": n_train, "status": "ERROR_TRAIN"
                })
                continue

            # 4. Build predict input CSV
            predict_csv = os.path.join(tmpdir, f"fold_{fold_i}.csv")
            pred_json   = os.path.join(tmpdir, f"fold_{fold_i}_pred.json")
            build_predict_csv(event, predict_csv)

            # 5. Predict
            try:
                pred_result = run_predict(predict_csv, pred_json)
            except subprocess.TimeoutExpired:
                pred_result = {"ok": False, "value": None, "stderr": "TIMEOUT"}

            if not pred_result["ok"] or pred_result["value"] is None:
                msg = f"Predict failed: {pred_result.get('stderr','')[:200]}"
                print(f"  ERROR: {msg}")
                errors.append({"fold": fold_i+1, "event_id": eid, "error": msg,
                               "pred_stdout": pred_result.get("stdout",""),
                               "pred_stderr": pred_result.get("stderr","")})
                results.append({
                    "fold": fold_i+1, "event_id": eid, "launch_time": launch,
                    "cme_speed_kms": speed, "transit_hours_observed": obs,
                    "predicted_transit_hours": None, "error_hours": None,
                    "abs_error_hours": None, "train_n": n_train, "status": "ERROR_PREDICT"
                })
                continue

            pred_val = pred_result["value"]
            if pred_val is None or math.isnan(pred_val) or math.isinf(pred_val):
                msg = f"Prediction value invalid: {pred_val}"
                print(f"  WARN: {msg}")
                errors.append({"fold": fold_i+1, "event_id": eid, "error": msg})
                results.append({
                    "fold": fold_i+1, "event_id": eid, "launch_time": launch,
                    "cme_speed_kms": speed, "transit_hours_observed": obs,
                    "predicted_transit_hours": None, "error_hours": None,
                    "abs_error_hours": None, "train_n": n_train, "status": "INVALID_PREDICTION"
                })
                continue

            err = pred_val - obs
            print(f"  OK  pred={pred_val:.1f}h  obs={obs:.1f}h  err={err:+.1f}h")
            results.append({
                "fold": fold_i+1, "event_id": eid, "launch_time": launch,
                "cme_speed_kms": round(speed, 1),
                "transit_hours_observed": round(obs, 2),
                "predicted_transit_hours": round(pred_val, 2),
                "error_hours": round(err, 2),
                "abs_error_hours": round(abs(err), 2),
                "train_n": n_train, "status": "SCORED"
            })

    except KeyboardInterrupt:
        print("\n\nInterrupted by user — saving partial results...")
    finally:
        # Always restore the original view and re-train production model
        print("\nRestoring original VIEW cutoff (2026-01-01)...")
        restore_view(conn)
        conn.close()
        shutil.rmtree(tmpdir, ignore_errors=True)
        print("Restoring production model (train on full dataset)...")
        clear_registry_and_cache()
        subprocess.run(
            ["dotnet", "run", "--project", DOTNET_PROJECT, "--no-build",
             "--", "train", "--config", YAML_CONFIG, "--no-cache"],
            capture_output=True, timeout=120
        )
        print("Production model restored.")

    # ── Compute metrics ────────────────────────────────────────────────────────
    scored = [r for r in results if r["status"] == "SCORED"]
    errors_vals = [r["error_hours"] for r in scored]

    metrics = {}
    if errors_vals:
        n = len(errors_vals)
        mae  = sum(abs(e) for e in errors_vals) / n
        rmse = math.sqrt(sum(e*e for e in errors_vals) / n)
        bias = sum(errors_vals) / n
        hit6  = sum(1 for e in errors_vals if abs(e) <= 6)  / n
        hit12 = sum(1 for e in errors_vals if abs(e) <= 12) / n
        hit24 = sum(1 for e in errors_vals if abs(e) <= 24) / n
        obs_vals = [r["transit_hours_observed"] for r in scored]
        obs_median = sorted(obs_vals)[n // 2]
        dbm_mae = sum(abs(v - obs_median) for v in obs_vals) / n
        skill = 1.0 - mae / dbm_mae if dbm_mae > 0 else float("nan")
        metrics = {
            "n_scored": n,
            "mae_hours": round(mae, 3),
            "rmse_hours": round(rmse, 3),
            "bias_hours": round(bias, 3),
            "hit_rate_6h": round(hit6, 4),
            "hit_rate_12h": round(hit12, 4),
            "hit_rate_24h": round(hit24, 4),
            "dbm_naive_mae_hours": round(dbm_mae, 3),
            "skill_vs_naive_dbm": round(skill, 4),
        }

    # ── Write outputs ──────────────────────────────────────────────────────────
    report = {
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "config": YAML_CONFIG,
        "description": (
            f"Walk-forward backtest: last {n_test} events. "
            "For each event, model trained exclusively on prior events "
            f"with {GAP_BUFFER_DAYS}-day gap buffer (RULE-051). True OOS predictions."
        ),
        "methodology": "expanding_window_walk_forward",
        "gap_buffer_days": GAP_BUFFER_DAYS,
        "n_test_events": n_test,
        "n_scored": len(scored),
        "n_errors": len(errors),
        "metrics": metrics,
        "events": results,
    }
    with open(OUTPUT_SCORED, "w") as f:
        json.dump(report, f, indent=2)

    with open(ERROR_LOG, "w") as f:
        json.dump({"errors": errors}, f, indent=2)

    run_log = {
        "run_timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "pipeline": YAML_CONFIG,
        "n_test_events": n_test,
        "gap_buffer_days": GAP_BUFFER_DAYS,
        "scored": len(scored),
        "skipped_or_errors": len(results) - len(scored),
        "metrics": metrics,
        "output": OUTPUT_SCORED,
        "errors_log": ERROR_LOG,
    }
    with open(LOG_PATH, "w") as f:
        json.dump(run_log, f, indent=2)

    # ── Print summary ──────────────────────────────────────────────────────────
    print()
    print("═" * 55)
    print("  WALK-FORWARD BACKTEST — RESULTS")
    print("═" * 55)
    if metrics:
        print(f"  Folds scored:     {metrics['n_scored']} / {n_test}")
        print(f"  MAE:              {metrics['mae_hours']:.2f} h")
        print(f"  RMSE:             {metrics['rmse_hours']:.2f} h")
        print(f"  Bias:             {metrics['bias_hours']:.2f} h  (+ = predicting late)")
        print(f"  Hit ±6h:          {metrics['hit_rate_6h']:.1%}")
        print(f"  Hit ±12h:         {metrics['hit_rate_12h']:.1%}")
        print(f"  Hit ±24h:         {metrics['hit_rate_24h']:.1%}")
        print(f"  Skill vs naive:   {metrics['skill_vs_naive_dbm']:.3f}")
    print(f"  Errors/skips:     {len(errors)}")
    print(f"  Output:           {OUTPUT_SCORED}")
    print(f"  Run log:          {LOG_PATH}")
    print(f"  Error log:        {ERROR_LOG}")
    print("═" * 55)

if __name__ == "__main__":
    main()
