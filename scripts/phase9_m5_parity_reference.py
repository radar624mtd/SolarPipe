"""
Phase 9 M5 parity reference — locks w=400 km/s so the numbers are directly
comparable to the C# PredictProgressiveCommand (which reads
background_speed_km_s: 400 from the drag_baseline stage hyperparameters).

This is NOT a restatement of the validation story. scripts/phase9_throwaway_4event.py
uses the pre-launch 6h observed mean for w and is the right reference for the
"does the physics work" question. THIS script answers "does C# reproduce Python
to within 0.5h when given the same (gamma0, w, nref)?" for the CCMC 4-event
benchmark.

Prints JSON on stdout so a .NET parity harness (or the integration test) can
parse it deterministically.
"""
import json
import sqlite3
import sys
from datetime import datetime, timedelta

DB = "solar_data.db"
R_SUN_KM = 695700.0
R_START = 21.5
R_TARGET = 215.0
GAMMA0 = 2.0e-8
W_FIXED = 400.0  # MUST match drag_baseline.background_speed_km_s in the C# config
N_REF = 5.0
GAMMA_MIN = 1e-9
GAMMA_MAX = 1e-6
MAX_HOURS = 72.0

EVENTS = [
    ("EVT_JAN18_X19", "2026-01-18 18:09", 1473.0, "2026-01-19 18:55", 24.77),
    ("EVT_MAR18",     "2026-03-18 09:23",  731.0, "2026-03-20 20:17", 58.90),
    ("EVT_MAR30",     "2026-03-30 03:24", 1689.0, "2026-04-01 11:29", 56.08),
    ("EVT_APR01",     "2026-04-01 23:45", 1220.0, "2026-04-03 15:02", 39.28),
]


def drag_rhs(r, v, gamma, w):
    dr = v * 3600.0 / R_SUN_KM
    dv = -gamma * (v - w) * abs(v - w) * 3600.0
    return dr, dv


def rk4_step(r, v, gamma, w, h):
    k1r, k1v = drag_rhs(r, v, gamma, w)
    k2r, k2v = drag_rhs(r + 0.5*h*k1r, v + 0.5*h*k1v, gamma, w)
    k3r, k3v = drag_rhs(r + 0.5*h*k2r, v + 0.5*h*k2v, gamma, w)
    k4r, k4v = drag_rhs(r + h*k3r, v + h*k3v, gamma, w)
    r_new = r + (h/6.0) * (k1r + 2*k2r + 2*k3r + k4r)
    v_new = v + (h/6.0) * (k1v + 2*k2v + 2*k3v + k4v)
    return r_new, v_new


def load_omni_window(conn, launch_dt, hours_after=MAX_HOURS):
    # C# L1ObservationStream indexes from floor(launch_hour), and reads rows
    # where datetime >= floor(launch) AND < floor(launch) + maxHours. Mirror that.
    start = launch_dt.replace(minute=0, second=0, microsecond=0)
    end = start + timedelta(hours=int(hours_after))
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT datetime, flow_speed, proton_density, Bz_GSM
        FROM omni_hourly
        WHERE datetime >= ? AND datetime < ?
        ORDER BY datetime
        """,
        (start.strftime("%Y-%m-%d %H:%M"), end.strftime("%Y-%m-%d %H:%M")),
    ).fetchall()
    parsed = []
    for r in rows:
        try:
            dt = datetime.strptime(r[0], "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        parsed.append((dt, r[1], r[2], r[3]))
    return start, parsed


def integrate_static(v0, gamma, w, h=1.0):
    """Plain drag ODE at hourly cadence — matches ProgressiveDragPropagator static mode."""
    r, v = R_START, v0
    t = 0.0
    while t < MAX_HOURS:
        r_prev = r
        r, v = rk4_step(r, v, gamma, w, h)
        # Linear-interp crossing detection, to match C# propagator.
        if r_prev < R_TARGET <= r:
            frac = (R_TARGET - r_prev) / (r - r_prev)
            return t + frac * h
        t += h
    return None


def integrate_density_modulated(v0, gamma0, hour0_dt, rows, n_ref=N_REF, w=W_FIXED, h=1.0):
    """Density-modulated drag with w fixed (matches C# PredictProgressiveCommand)."""
    # Build hour-indexed stream: index 0 = hour0_dt (floor of launch).
    by_hour = {}
    for row in rows:
        idx = int(round((row[0] - hour0_dt).total_seconds() / 3600.0))
        if 0 <= idx < int(MAX_HOURS):
            by_hour[idx] = row

    r, v = R_START, v0
    t = 0.0
    prev_n, prev_v_l1 = None, None
    shock_detected_t = None
    hour_index = 0

    # At each hour step, query gamma at hourIndex then integrate one hour into
    # hourIndex+1 — matching ProgressiveDragPropagator.Propagate.
    while t < MAX_HOURS:
        # C# queries γ at hourIndex starting from 1 for the segment (0 → 1).
        query_hour = hour_index + 1
        step = by_hour.get(query_hour)
        if step is None or step[2] is None or step[2] <= 0:
            gamma_eff = gamma0
            n_obs, v_obs_l1 = None, None
        else:
            n_obs = step[2]
            v_obs_l1 = step[1]
            gamma_eff = gamma0 * (n_obs / n_ref)
            gamma_eff = max(GAMMA_MIN, min(GAMMA_MAX, gamma_eff))

        r_prev = r
        r, v = rk4_step(r, v, gamma_eff, w, h)
        # Clamp speed to [200, 3000] to match C# clip (spec §6.3).
        if v < 200.0: v = 200.0
        if v > 3000.0: v = 3000.0

        # Target crossing (linear interp) — matches C#.
        if r_prev < R_TARGET <= r:
            frac = (R_TARGET - r_prev) / (r - r_prev)
            return t + frac * h, shock_detected_t

        t += h
        hour_index += 1

        # Shock detection: hour-over-hour on the observation stream. Must fire
        # AFTER the integration step, matching C# (ProgressiveDragPropagator
        # checks shockDetector after appending the new state).
        cur_obs = by_hour.get(hour_index)
        prev_obs = by_hour.get(hour_index - 1)
        if cur_obs is not None and prev_obs is not None:
            if (cur_obs[1] is not None and prev_obs[1] is not None
                and cur_obs[2] is not None and prev_obs[2] is not None
                and prev_obs[2] > 0):
                dv = cur_obs[1] - prev_obs[1]
                ratio = cur_obs[2] / prev_obs[2]
                if dv >= 200.0 and ratio >= 3.0:
                    shock_detected_t = t
                    return t, shock_detected_t

    return None, shock_detected_t


def main():
    conn = sqlite3.connect(DB)
    out = []
    for tag, launch_iso, v0, icme_iso, obs_transit_h in EVENTS:
        launch_dt = datetime.strptime(launch_iso, "%Y-%m-%d %H:%M")
        hour0, rows = load_omni_window(conn, launch_dt)

        t_static = integrate_static(v0, GAMMA0, W_FIXED)
        t_dm, shock_t = integrate_density_modulated(v0, GAMMA0, hour0, rows)

        out.append({
            "tag": tag,
            "launch": launch_iso,
            "v0_kms": v0,
            "truth_h": obs_transit_h,
            "static_w400_h": t_static,
            "density_modulated_h": t_dm,
            "shock_detected_h": shock_t,
            "n_omni_rows": len(rows),
        })
    conn.close()
    json.dump({"gamma0": GAMMA0, "w_fixed": W_FIXED, "n_ref": N_REF, "events": out},
              sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
