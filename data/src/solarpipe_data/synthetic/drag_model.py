"""Task 6.1 — Drag-Based CME Propagation Model.

Implements the aerodynamic drag equation:
    dv/dt = -γ(v - w)|v - w|

where:
    v  — CME speed (km/s)
    w  — ambient solar wind speed (km/s)
    γ  — drag parameter (km⁻¹), calibrated from ENLIL ensemble residuals

Integration uses scipy's RK45 (Dormand-Prince) solver, matching the C# side
(RULE-030 equivalent: Dormand-Prince for all ODE integration).

γ calibration: fit by minimising mean squared transit-time residual over
historical ENLIL simulations (enlil_simulations table).  Default fallback
γ = 0.2 × 10⁻³ km⁻¹ is the published Vrsnak (2013) ensemble median.

Reference distance: Sun centre to L1 ≈ 213.5 R☉ ≈ 1.485 × 10⁸ km (1 AU).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AU_KM: float = 1.495978707e8           # 1 AU in km
_L1_KM: float = _AU_KM * (1.0 - 0.01)  # L1 ≈ 0.99 AU from Sun centre
_SUN_R_KM: float = 6.957e5              # 1 R☉ in km
_R0_KM: float = 21.5 * _SUN_R_KM       # ENLIL inner boundary (21.5 R☉)
_GAMMA_DEFAULT: float = 0.2e-3          # Vrsnak (2013) median, km⁻¹

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class DragResult:
    """Outcome of a single drag-model integration."""

    transit_time_hours: float
    """Predicted Sun-to-L1 transit time in hours."""

    arrival_speed_kms: float
    """CME speed at L1 (km/s)."""

    success: bool
    """True if the integrator converged before t_max."""

    gamma: float
    """Drag parameter used (km⁻¹)."""

    ambient_wind_kms: float
    """Ambient solar wind speed used (km/s)."""


@dataclass
class DragCalibration:
    """Calibrated drag parameter with uncertainty estimate."""

    gamma: float
    """Best-fit γ (km⁻¹)."""

    gamma_std: float = 0.0
    """Standard deviation across calibration ensemble."""

    n_events: int = 0
    """Number of ENLIL events used in calibration."""

    residual_rmse_hours: float = 0.0
    """Root-mean-square transit-time residual (hours) over calibration set."""

    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ODE definition
# ---------------------------------------------------------------------------

def _drag_ode(t: float, y: list[float], gamma: float, w: float) -> list[float]:
    """Right-hand side of the drag equation.

    State vector y = [r, v]:
        dr/dt = v
        dv/dt = -γ(v - w)|v - w|

    Args:
        t: time (seconds) — unused directly but required by solve_ivp
        y: [r_km, v_kms]
        gamma: drag coefficient (km⁻¹)
        w: ambient wind speed (km/s)
    """
    _r, v = y
    dvdt = -gamma * (v - w) * abs(v - w)
    return [v, dvdt]


# ---------------------------------------------------------------------------
# Single-event integration
# ---------------------------------------------------------------------------

def propagate(
    v0_kms: float,
    w_kms: float = 450.0,
    gamma: float = _GAMMA_DEFAULT,
    r0_km: float = _R0_KM,
    r_target_km: float = _L1_KM,
    t_max_hours: float = 240.0,
) -> DragResult:
    """Integrate the drag equation from r0 to r_target.

    Args:
        v0_kms: Initial CME speed at r0 (km/s).
        w_kms: Ambient solar wind speed (km/s).
        gamma: Drag coefficient (km⁻¹).
        r0_km: Start distance from Sun centre (km). Default = 21.5 R☉.
        r_target_km: Stop distance (km). Default ≈ L1.
        t_max_hours: Maximum integration time before declaring failure.

    Returns:
        DragResult with transit_time_hours, arrival_speed_kms, success flag.
    """
    t_max_s = t_max_hours * 3600.0

    def _hit_l1(t: float, y: list[float], *args: Any) -> float:  # noqa: ANN001
        return y[0] - r_target_km

    _hit_l1.terminal = True  # type: ignore[attr-defined]
    _hit_l1.direction = 1.0  # type: ignore[attr-defined]

    y0 = [r0_km, v0_kms]

    sol = solve_ivp(
        _drag_ode,
        t_span=(0.0, t_max_s),
        y0=y0,
        args=(gamma, w_kms),
        method="RK45",
        events=_hit_l1,
        rtol=1e-6,
        atol=1e-6,
        dense_output=False,
    )

    if sol.t_events and len(sol.t_events[0]) > 0:
        t_arrival_s = float(sol.t_events[0][0])
        v_arrival = float(sol.y_events[0][0][1])
        success = True
    else:
        # Did not reach L1 — return the final state
        t_arrival_s = float(sol.t[-1])
        v_arrival = float(sol.y[1, -1])
        success = False
        logger.warning(
            "drag integrator did not reach L1 (v0=%.1f km/s, w=%.1f km/s, gamma=%.3e)",
            v0_kms, w_kms, gamma,
        )

    return DragResult(
        transit_time_hours=t_arrival_s / 3600.0,
        arrival_speed_kms=max(v_arrival, 0.0),  # no negative speeds
        success=success,
        gamma=gamma,
        ambient_wind_kms=w_kms,
    )


# ---------------------------------------------------------------------------
# γ calibration from ENLIL ensemble
# ---------------------------------------------------------------------------

def calibrate_gamma(
    observed_speeds: list[float],
    observed_winds: list[float],
    observed_transits_hours: list[float],
    gamma_grid: np.ndarray | None = None,
) -> DragCalibration:
    """Calibrate γ by grid search over ENLIL ensemble residuals.

    Minimises mean squared transit-time error:
        MSE(γ) = mean[(T_pred(γ) - T_obs)²]

    Args:
        observed_speeds: CME initial speeds (km/s) at 21.5 R☉.
        observed_winds: Ambient solar wind speeds (km/s) for each event.
        observed_transits_hours: Observed transit times (hours) from ENLIL.
        gamma_grid: Candidate γ values to test. Default: log-spaced 0.01e-3
                    to 2.0e-3 km⁻¹ (25 points).

    Returns:
        DragCalibration with best-fit γ and diagnostics.
    """
    if len(observed_speeds) != len(observed_winds) or len(observed_speeds) != len(observed_transits_hours):
        raise ValueError("calibrate_gamma: all input lists must have the same length")

    n = len(observed_speeds)
    if n == 0:
        logger.warning("calibrate_gamma: empty calibration set — using default γ")
        return DragCalibration(gamma=_GAMMA_DEFAULT, n_events=0)

    if gamma_grid is None:
        gamma_grid = np.logspace(-5, -3, 25)  # 0.01e-3 … 1.0e-3 km⁻¹

    best_gamma = _GAMMA_DEFAULT
    best_mse = float("inf")
    best_rmse = float("inf")

    for g in gamma_grid:
        errors = []
        for v0, w, t_obs in zip(observed_speeds, observed_winds, observed_transits_hours):
            result = propagate(v0_kms=v0, w_kms=w, gamma=g)
            errors.append(result.transit_time_hours - t_obs)
        mse = float(np.mean(np.array(errors) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_rmse = float(np.sqrt(mse))
            best_gamma = float(g)

    # Estimate spread: re-run at best_gamma and compute per-event errors
    errors_best = []
    for v0, w, t_obs in zip(observed_speeds, observed_winds, observed_transits_hours):
        result = propagate(v0_kms=v0, w_kms=w, gamma=best_gamma)
        errors_best.append(result.transit_time_hours - t_obs)
    gamma_std = float(np.std(errors_best))

    logger.info(
        "drag calibration: γ=%.4e km⁻¹ RMSE=%.1f h (n=%d events)",
        best_gamma, best_rmse, n,
    )

    return DragCalibration(
        gamma=best_gamma,
        gamma_std=gamma_std,
        n_events=n,
        residual_rmse_hours=best_rmse,
    )


# ---------------------------------------------------------------------------
# Calibration from staging DB
# ---------------------------------------------------------------------------

def calibrate_from_db(db_path: str) -> DragCalibration:
    """Load ENLIL simulation data from staging.db and calibrate γ.

    Uses only events where:
    - speed is available in cme_analyses (level_of_data ≥ 1)
    - enlil_simulations has a model_completion_time (proxy for convergence)
    - ambient sw_speed_ambient available from sw_ambient_context

    Falls back to default γ if fewer than 10 events pass the filter.
    """
    import sqlalchemy as sa
    from ..database.schema import make_engine

    engine = make_engine(db_path)

    query = sa.text("""
        SELECT
            ca.speed_kms,
            COALESCE(sw.sw_speed_ambient, 450.0) AS ambient_kms,
            (JULIANDAY(es.model_completion_time) - JULIANDAY(ce.start_time)) * 24.0 AS transit_hours
        FROM cme_events ce
        JOIN cme_analyses ca ON ca.activity_id = ce.activity_id
        JOIN enlil_simulations es ON INSTR(es.linked_cme_ids, ce.activity_id) > 0
        LEFT JOIN sw_ambient_context sw ON sw.activity_id = ce.activity_id
        WHERE ca.speed_kms IS NOT NULL
          AND ca.speed_kms BETWEEN 100 AND 3500
          AND es.model_completion_time IS NOT NULL
          AND ce.start_time IS NOT NULL
        ORDER BY ce.start_time
    """)

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    if len(rows) < 10:
        logger.warning(
            "calibrate_from_db: only %d usable ENLIL events — using default γ=%.3e",
            len(rows), _GAMMA_DEFAULT,
        )
        return DragCalibration(gamma=_GAMMA_DEFAULT, n_events=len(rows))

    speeds, winds, transits = [], [], []
    for row in rows:
        speed, ambient, transit = row[0], row[1], row[2]
        if transit is not None and 10.0 < transit < 200.0:
            speeds.append(float(speed))
            winds.append(float(ambient))
            transits.append(float(transit))

    return calibrate_gamma(speeds, winds, transits)
