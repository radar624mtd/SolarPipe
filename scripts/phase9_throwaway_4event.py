"""
Phase 9 premise validation — extended to the CCMC 4-event benchmark.

Tests the density-modulated drag rule against all 4 events Phase 8's
ccmc_comparison block tracks, using the SAME launch_time and cme_speed_kms
that Phase 8 uses (from staging.feature_vectors), so the comparison is
apples-to-apples with phase8_domain_results.json.

Ground truth from data/data/staging/staging.db.feature_vectors:
  2026-01-18 18:09  X1.9    v0=1473  icme=2026-01-19 18:55  transit=24.77h
  2026-03-18 09:23          v0= 731  icme=2026-03-20 20:17  transit=58.90h
  2026-03-30 03:24          v0=1689  icme=2026-04-01 11:29  transit=56.08h
  2026-04-01 23:45          v0=1220  icme=2026-04-03 15:02  transit=39.28h

Phase 8 baseline from output/phase8_baseline/phase8_domain_results.json:
  Jan-18: pred=43.52 err=+18.76
  Mar-18: pred=58.78 err= -1.12
  Mar-30: pred=48.62 err= -7.46
  Apr-01: pred=47.77 err= +7.66
  CCMC 4-event MAE = 8.75h

Phase 9 density-modulation rule (spec §3.3):
  gamma_eff(t) = gamma0 * (n_obs(t) / n_ref),   n_ref = 5 cm^-3
  clipped to [1e-9, 1e-6] km^-1
"""
import sqlite3
from datetime import datetime, timedelta

DB = "solar_data.db"
R_SUN_KM = 695700.0
R_START = 21.5
R_TARGET = 215.0
GAMMA0 = 2.0e-8
W_DEFAULT = 400.0
N_REF = 5.0
GAMMA_MIN = 1e-9
GAMMA_MAX = 1e-6

EVENTS = [
    # (tag, launch_iso, v0_kms, icme_iso, obs_transit_h, ph8_pred)
    ("Jan-18 X1.9", "2026-01-18 18:09", 1473.0, "2026-01-19 18:55", 24.77, 43.52),
    ("Mar-18 SIDC", "2026-03-18 09:23",  731.0, "2026-03-20 20:17", 58.90, 58.78),
    ("Mar-30",      "2026-03-30 03:24", 1689.0, "2026-04-01 11:29", 56.08, 48.62),
    ("Apr-01",      "2026-04-01 23:45", 1220.0, "2026-04-03 15:02", 39.28, 47.77),
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

def compute_pre_launch_median(conn, launch_dt, lookback_hours=150.0):
    """
    Derive n_ref from OMNI pre-event window [launch - lookback_hours, launch).
    Missing hourly slots are gap-filled via linear interpolation between the nearest
    valid readings; edge gaps use flat extrapolation.
    Returns NaN only if the entire window has zero valid readings (data ingest flag).
    Sentinels (<= 0 or >= 9999) excluded per RULE-120.
    """
    n_slots = int(round(lookback_hours))
    window_start = launch_dt - timedelta(hours=lookback_hours)
    rows = conn.execute(
        "SELECT datetime, proton_density FROM omni_hourly "
        "WHERE datetime >= ? AND datetime < ? "
        "ORDER BY datetime",
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
        print(f"  WARN EmptyPreLaunchWindow launch={launch_dt} n_ref_derived=NaN")
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
            l, r = lefts[-1], rights[0]
            t = (i - l) / (r - l)
            filled.append(readings[l] + t * (readings[r] - readings[l]))
        elif lefts:
            filled.append(readings[lefts[-1]])
        else:
            filled.append(readings[rights[0]])

    filled.sort()
    mid = len(filled) // 2
    return filled[mid] if len(filled) % 2 else (filled[mid - 1] + filled[mid]) / 2.0


def load_omni_window(conn, launch_dt, hours_after=96, hours_before=24):
    s = (launch_dt - timedelta(hours=hours_before)).strftime("%Y-%m-%d %H:%M")
    e = (launch_dt + timedelta(hours=hours_after)).strftime("%Y-%m-%d %H:%M")
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT datetime, flow_speed, proton_density, Bz_GSM, Dst_nT
        FROM omni_hourly
        WHERE datetime >= ? AND datetime <= ?
        ORDER BY datetime
    """, (s, e)).fetchall()
    parsed = []
    for r in rows:
        try:
            dt = datetime.strptime(r[0], "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        parsed.append((dt, r[1], r[2], r[3], r[4]))
    return parsed

def integrate_static(v0, gamma, w, h=0.1):
    """Plain drag ODE, no assimilation. Matches SolarPipe DragBasedModel."""
    r, v = R_START, v0
    t = 0.0
    while t < 200.0:
        if r >= R_TARGET:
            return t
        r, v = rk4_step(r, v, gamma, w, h)
        t += h
    return None

def integrate_density_modulated(v0, gamma0, omni_rows, launch_dt,
                                  n_ref=N_REF, w_override=None, h=1.0):
    """Density-modulated drag: gamma_eff(t) = gamma0 * (n_obs(t) / n_ref)."""
    # Ambient wind: pre-launch 6h mean if not overridden
    if w_override is None:
        pre_vs = [row[1] for row in omni_rows if row[0] <= launch_dt and row[1] is not None]
        w = sum(pre_vs[-6:]) / max(len(pre_vs[-6:]), 1) if pre_vs else W_DEFAULT
    else:
        w = w_override

    # Build hour-indexed post-launch stream
    post = {}
    for row in omni_rows:
        if row[0] > launch_dt:
            hr_idx = int((row[0] - launch_dt).total_seconds() // 3600)
            post[hr_idx] = row

    r, v = R_START, v0
    t = 0.0
    trace = []
    shock_detected_t = None
    prev_n, prev_v_l1 = None, None
    n_missing = 0

    while t < 72.0:  # spec-mandated cap
        if r >= R_TARGET:
            return t, trace, w, shock_detected_t, n_missing

        hr_idx = int(t) + 1  # the hour we are stepping INTO
        step = post.get(hr_idx)
        if step is None or step[2] is None or step[2] <= 0:
            gamma_eff = gamma0
            n_missing += 1
            n_obs, v_obs_l1 = None, None
        else:
            n_obs = step[2]
            v_obs_l1 = step[1]
            gamma_eff = gamma0 * (n_obs / n_ref)
            gamma_eff = max(GAMMA_MIN, min(GAMMA_MAX, gamma_eff))

            # Shock detection: delta_v > 200 AND density ratio >= 3
            if (prev_n is not None and prev_v_l1 is not None
                and v_obs_l1 is not None and n_obs is not None
                and prev_n > 0):
                if (v_obs_l1 - prev_v_l1) > 200.0 and (n_obs / prev_n) >= 3.0:
                    shock_detected_t = t + 1.0
                    trace.append((t+1.0, r, v, gamma_eff, n_obs, v_obs_l1, "SHOCK"))
                    return shock_detected_t, trace, w, shock_detected_t, n_missing

            prev_n = n_obs
            prev_v_l1 = v_obs_l1

        trace.append((t, r, v, gamma_eff, n_obs, v_obs_l1, ""))
        r, v = rk4_step(r, v, gamma_eff, w, h)
        t += h

    return None, trace, w, shock_detected_t, n_missing

def run_event(conn, tag, launch_iso, v0, icme_iso, obs_transit_h, ph8_pred):
    launch_dt = datetime.strptime(launch_iso, "%Y-%m-%d %H:%M")
    icme_dt = datetime.strptime(icme_iso, "%Y-%m-%d %H:%M")
    omni = load_omni_window(conn, launch_dt, hours_after=96)

    # Derive n_ref from the pre-launch OMNI profile (150h window)
    n_ref_derived = compute_pre_launch_median(conn, launch_dt)
    n_ref_effective = n_ref_derived if (n_ref_derived == n_ref_derived) else N_REF  # NaN guard

    # Reference: static with default params
    t_static_default = integrate_static(v0, GAMMA0, W_DEFAULT)

    # Reference: static with pre-launch observed w
    pre_vs = [row[1] for row in omni if row[0] <= launch_dt and row[1] is not None]
    w_pre = sum(pre_vs[-6:]) / max(len(pre_vs[-6:]), 1) if pre_vs else W_DEFAULT
    t_static_obs_w = integrate_static(v0, GAMMA0, w_pre)

    # Main: density-modulated with derived n_ref
    t_dm, trace, w_used, shock_t, n_missing = integrate_density_modulated(
        v0, GAMMA0, omni, launch_dt, n_ref=n_ref_effective)

    # Control: density-modulated with fixed n_ref=5 (Phase 9 default)
    t_dm_nref5, _, _, _, _ = integrate_density_modulated(
        v0, GAMMA0, omni, launch_dt, n_ref=N_REF)

    # n_ref sensitivity
    sensitivity = {}
    for n_ref in [3.0, 5.0, 7.0]:
        t_sens, _, _, _, _ = integrate_density_modulated(
            v0, GAMMA0, omni, launch_dt, n_ref=n_ref)
        sensitivity[n_ref] = t_sens

    print(f"\n{'='*70}")
    print(f"EVENT: {tag}")
    print(f"  launch={launch_iso}Z v0={v0} km/s  icme={icme_iso}Z")
    print(f"  truth transit = {obs_transit_h:.2f}h  |  phase8 pred = {ph8_pred:.2f}h (err {ph8_pred-obs_transit_h:+.2f})")
    print(f"  pre-launch 6h mean w = {w_pre:.1f} km/s  (n omni rows = {len(omni)})")
    n_ref_str = f"{n_ref_derived:.2f}" if n_ref_derived == n_ref_derived else "NaN (fallback)"
    print(f"  n_ref_derived (150h median) = {n_ref_str}  |  n_ref_effective = {n_ref_effective:.2f}")
    print(f"  static γ0=2e-8 w=400:   {t_static_default:.2f}h  (err {t_static_default-obs_transit_h:+.2f})" if t_static_default else "  static: no arrival")
    print(f"  static γ0=2e-8 w={w_pre:.0f}: {t_static_obs_w:.2f}h  (err {t_static_obs_w-obs_transit_h:+.2f})" if t_static_obs_w else "  static+obs-w: no arrival")
    if t_dm_nref5 is not None:
        print(f"  density-mod n_ref=5.0: {t_dm_nref5:.2f}h  (err {t_dm_nref5-obs_transit_h:+.2f})")
    if t_dm is not None:
        kind = "(shock detected)" if shock_t else "(1 AU crossing)"
        print(f"  density-mod n_ref_derived={n_ref_effective:.2f}: {t_dm:.2f}h  (err {t_dm-obs_transit_h:+.2f})  {kind}")
    else:
        print(f"  density-modulated:     did not arrive within 72h")
    print(f"  n_missing_hours = {n_missing}")
    print(f"  sensitivity n_ref={{3,5,7}}: {[f'{v:.2f}' if v else 'NA' for v in sensitivity.values()]}")

    return {
        "tag": tag,
        "truth": obs_transit_h,
        "ph8": ph8_pred,
        "n_ref_derived": n_ref_derived,
        "n_ref_effective": n_ref_effective,
        "static_default": t_static_default,
        "static_obs_w": t_static_obs_w,
        "density_modulated_nref5": t_dm_nref5,
        "density_modulated": t_dm,
        "shock_detected_at": shock_t,
        "n_missing": n_missing,
        "sensitivity": sensitivity,
    }

def main():
    conn = sqlite3.connect(DB)
    results = []
    for ev in EVENTS:
        results.append(run_event(conn, *ev))
    conn.close()

    print(f"\n{'='*70}")
    print(f"SUMMARY — CCMC 4-event benchmark")
    print(f"{'='*70}")
    print(f"{'Event':<18} {'Truth':>7} {'Ph8':>8} {'Ph8err':>8} {'nref':>6} {'DM5':>8} {'DM5err':>8} {'DMder':>8} {'DMerr':>8}")
    mae_ph8 = 0.0
    mae_dm5 = 0.0
    mae_dm = 0.0
    n_dm5 = n_dm = 0
    for r in results:
        dm5  = r['density_modulated_nref5']
        dm   = r['density_modulated']
        dm5_err = (dm5 - r['truth']) if dm5 is not None else None
        dm_err  = (dm  - r['truth']) if dm  is not None else None
        ph8_err = r['ph8'] - r['truth']
        mae_ph8 += abs(ph8_err)
        if dm5 is not None:
            mae_dm5 += abs(dm5_err)
            n_dm5 += 1
        if dm is not None:
            mae_dm += abs(dm_err)
            n_dm += 1
        nref_str  = f"{r['n_ref_effective']:.1f}"
        dm5_str   = f"{dm5:.2f}"  if dm5 is not None else "  NA"
        dm5e_str  = f"{dm5_err:+.2f}" if dm5_err is not None else "  NA"
        dm_str    = f"{dm:.2f}"   if dm  is not None else "  NA"
        dm_err_str = f"{dm_err:+.2f}" if dm_err is not None else "  NA"
        print(f"{r['tag']:<18} {r['truth']:>7.2f} {r['ph8']:>8.2f} {ph8_err:>+8.2f} "
              f"{nref_str:>6} {dm5_str:>8} {dm5e_str:>8} {dm_str:>8} {dm_err_str:>8}")
    print(f"\nPhase 8 MAE (4-event):              {mae_ph8/len(results):.2f}h")
    if n_dm5:
        print(f"Density-mod n_ref=5  MAE ({n_dm5}-event): {mae_dm5/n_dm5:.2f}h")
    if n_dm:
        print(f"Density-mod derived  MAE ({n_dm}-event): {mae_dm/n_dm:.2f}h")

if __name__ == "__main__":
    main()
