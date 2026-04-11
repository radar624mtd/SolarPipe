"""
Phase 9 premise validation — throwaway script.

Goal: before committing to the 4-day C# Phase 9 implementation, check whether
residual-nudge assimilation of the drag-based model against live omni_hourly
data can recover the Jan-18 X1.9 CME arrival time (truth 22.92h) substantially
better than the static Phase 8 physics baseline (42.79h).

NOT production code. Not committed logic. Hard-coded to one event. If results
look promising we formalize in C#; if not, we revise the Phase 9 spec.

Data source: ./solar_data.db (10GB, repo root)
Event: donki_cme activity_id 2026-01-18T18:09:00-CME-001
  launch 2026-01-18 19:55Z, v0=1820 km/s
Shock: donki_ips 2026-01-19 18:55Z  (linked to above CME)
Truth transit: 22.92h launch->shock
Phase 8 physics baseline: 42.79h (error +19.87h)
"""
import sqlite3
from datetime import datetime, timedelta

DB = "solar_data.db"
R_SUN_KM = 695700.0
R_START = 21.5      # solar radii
R_TARGET = 215.0    # 1 AU
GAMMA0 = 2.0e-8     # km^-1, DragBasedModel default
W_DEFAULT = 400.0   # km/s background wind
V0 = 1820.0         # km/s from donki_cme
LAUNCH = datetime(2026, 1, 18, 19, 55)
SHOCK_TRUTH = datetime(2026, 1, 19, 18, 55)
PHASE8_PHYSICS = 42.79

def drag_rhs(r, v, gamma, w):
    """dr/dt in R_sun/hr, dv/dt in km/s/hr."""
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

def integrate_static(v0, gamma, w):
    """Reference: no assimilation. Same ODE, fixed gamma/w."""
    r, v = R_START, v0
    t = 0.0
    h = 0.1
    while t < 200.0:
        if r >= R_TARGET:
            return t
        r, v = rk4_step(r, v, gamma, w, h)
        t += h
    return None

def load_omni_window(conn, t_start, t_end):
    """Return list of (datetime, flow_speed, proton_density, Bz_GSM, Dst_nT)."""
    s = t_start.strftime("%Y-%m-%d %H:%M")
    e = t_end.strftime("%Y-%m-%d %H:%M")
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT datetime, flow_speed, proton_density, Bz_GSM, Dst_nT
        FROM omni_hourly
        WHERE datetime >= ? AND datetime <= ?
        ORDER BY datetime
    """, (s, e)).fetchall()
    return [(datetime.strptime(r[0], "%Y-%m-%d %H:%M"), r[1], r[2], r[3], r[4]) for r in rows]

def integrate_assimilated(v0, gamma0, omni_rows, eta):
    """
    Residual-nudge assimilation (spec Phase9 §3.3):
        gamma(t_k) = gamma(t_{k-1}) + eta * (r_k / (v_pred * |v_pred - w|)) * dt
    where residual r_k = (v_obs_L1 - w_current) is the "extra push" signal.
    NOTE: this is a deliberately simple first try. The L1 observation is
    NOT directly comparable to the CME's bulk speed mid-transit — the obs
    is ambient wind. We use the obs as an update to the background w, not
    as a direct v_pred target. Gamma update uses (obs_w - w_prior) as the
    signal. This is the interpretation I will test.

    Strategy:
      - Initial w = median of omni flow_speed in the 6h BEFORE launch.
      - At each 1h step, peek at omni flow_speed at (launch + t).
      - Update w = 0.5*w + 0.5*obs  (simple low-pass).
      - Keep gamma fixed for now (ablation #1).
      - Detect shock criterion as safety stop: |delta_v|>200 + density 3x.
    """
    r, v = R_START, v0
    t = 0.0
    h = 1.0  # 1h steps to match omni cadence
    w = W_DEFAULT  # overwritten from pre-launch obs
    gamma = gamma0

    # Compute initial w from 6h pre-launch
    pre_vs = [row[1] for row in omni_rows if row[0] <= LAUNCH and row[1] is not None]
    if pre_vs:
        w = sum(pre_vs[-6:]) / len(pre_vs[-6:])

    # Build an hour-indexed view of post-launch omni
    post = {int((row[0] - LAUNCH).total_seconds() / 3600): row for row in omni_rows if row[0] > LAUNCH}

    trace = []
    while t < 200.0:
        if r >= R_TARGET:
            return t, trace
        step = post.get(int(t) + 1)
        if step is not None:
            obs_v, obs_n, obs_bz, _ = step[1], step[2], step[3], step[4]
            if obs_v is not None:
                # Low-pass update of background wind
                w_new = 0.5 * w + 0.5 * obs_v
                trace.append((t, r, v, w, w_new, obs_v, obs_n))
                w = w_new
        r, v = rk4_step(r, v, gamma, w, h)
        t += h
    return None, trace

def main():
    conn = sqlite3.connect(DB)
    t_end_window = LAUNCH + timedelta(hours=80)
    omni = load_omni_window(conn, LAUNCH - timedelta(hours=24), t_end_window)
    print(f"Loaded {len(omni)} omni hourly rows from launch-24h to launch+80h")
    if omni:
        print(f"  first: {omni[0]}")
        print(f"  last:  {omni[-1]}")

    # --- Reference A: static baseline with SolarPipe's default gamma/w
    t_A = integrate_static(V0, GAMMA0, W_DEFAULT)
    print(f"\n[A] Static drag (gamma={GAMMA0}, w={W_DEFAULT}): transit = {t_A:.2f} h")
    print(f"    Phase8 physics baseline reference: {PHASE8_PHYSICS:.2f} h  (match expected)")

    # --- Reference B: static with pre-launch-observed w
    pre_vs = [row[1] for row in omni if row[0] <= LAUNCH and row[1] is not None]
    w_pre = sum(pre_vs[-6:]) / len(pre_vs[-6:]) if pre_vs else W_DEFAULT
    print(f"\n    Pre-launch 6h mean flow_speed: {w_pre:.1f} km/s")
    t_B = integrate_static(V0, GAMMA0, w_pre)
    print(f"[B] Static drag with observed pre-launch w={w_pre:.1f}: transit = {t_B:.2f} h")

    # --- Reference C: w-assimilation during transit (hourly omni updates)
    t_C, trace = integrate_assimilated(V0, GAMMA0, omni, eta=0.0)
    print(f"[C] w-assimilated (eta=0, updates w only): transit = {t_C:.2f} h"
          if t_C else "[C] did not reach 1 AU")
    if trace:
        print(f"    trace length = {len(trace)} steps")
        print(f"    first 3: {trace[:3]}")
        print(f"    last  3: {trace[-3:]}")

    # --- Reference D: density-scaled gamma (gamma_eff = gamma0 * (n_obs/n_ref)^alpha)
    # This is the hypothesis the sensitivity sweep suggests.
    def integrate_density_scaled(v0, gamma0, omni_rows, n_ref, alpha):
        r, v = R_START, v0
        t = 0.0
        h = 1.0
        # pre-launch 6h mean w
        pre_vs = [row[1] for row in omni_rows if row[0] <= LAUNCH and row[1] is not None]
        w = sum(pre_vs[-6:]) / len(pre_vs[-6:]) if pre_vs else W_DEFAULT
        post = {int((row[0] - LAUNCH).total_seconds() / 3600): row for row in omni_rows if row[0] > LAUNCH}
        while t < 200.0:
            if r >= R_TARGET:
                return t
            step = post.get(int(t) + 1)
            if step is not None and step[2] is not None and step[2] > 0:
                gamma_eff = gamma0 * (step[2] / n_ref) ** alpha
            else:
                gamma_eff = gamma0
            r, v = rk4_step(r, v, gamma_eff, w, h)
            t += h
        return None

    print("\n--- Density-scaled gamma (n_ref=5, pre-launch w) ---")
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        t_D = integrate_density_scaled(V0, GAMMA0, omni, n_ref=5.0, alpha=alpha)
        err = t_D - (SHOCK_TRUTH - LAUNCH).total_seconds()/3600.0 if t_D else None
        if t_D:
            print(f"  alpha={alpha}: transit = {t_D:.2f} h  (error {err:+.2f} h)")
        else:
            print(f"  alpha={alpha}: no arrival")

    # --- Sensitivity sweep: gamma variants
    print(f"\n--- gamma sensitivity (static, w={W_DEFAULT}) ---")
    for g in [0.5e-8, 1.0e-8, 1.5e-8, 2.0e-8, 3.0e-8, 5.0e-8, 1.0e-7, 2.0e-7]:
        t_g = integrate_static(V0, g, W_DEFAULT)
        print(f"  gamma={g:.1e}: transit = {t_g:.2f} h" if t_g else f"  gamma={g:.1e}: no arrival")

    truth = (SHOCK_TRUTH - LAUNCH).total_seconds() / 3600.0
    print(f"\n=== TRUTH ===")
    print(f"Launch: {LAUNCH} -> Shock: {SHOCK_TRUTH}")
    print(f"Transit truth: {truth:.2f} h")
    print(f"[A] error: {t_A - truth:+.2f} h")
    print(f"[B] error: {t_B - truth:+.2f} h")
    if t_C:
        print(f"[C] error: {t_C - truth:+.2f} h")

    conn.close()

if __name__ == "__main__":
    main()
