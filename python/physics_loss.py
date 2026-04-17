"""PINN physics loss terms for CME transit time prediction (G4 gate).

Four loss components enforcing physical plausibility:

  L_pinball  : primary task loss -- pinball/quantile regression for P10/P50/P90
  L_ode      : drag ODE residual -- predicted arrival must be consistent with
               dv/dt = -gamma_eff * |v - v_sw| * (v - v_sw),
               gamma_eff(t) = gamma0 * (n_obs(t) / n_ref)
  L_mono     : monotonic deceleration hinge -- penalises speed *increases*
               while v > v_sw (physics: CME can only decelerate above ambient)
  L_bound    : transit time must lie in [T_MIN, T_MAX] = [12h, 120h]
  L_qorder   : quantile ordering -- P10 <= P50 <= P90 must hold

Total loss (weights from YAML hyperparameters):
  L = L_pinball
    + lambda_ode   * L_ode
    + lambda_mono  * L_mono
    + lambda_bound * L_bound
    + lambda_qorder * L_qorder

Design notes
------------
- L_ode uses an explicit Euler ODE integration in PyTorch (100 steps over
  24h). This is differentiable, avoids torchdiffeq as an extra dependency,
  and is fast enough for batch training.  The drag coefficient gamma_eff
  is computed from the in-situ density sequence (x_seq channel 2 = proton_density).
- L_mono is computed from x_seq speed channel (channel 1 = flow_speed).
  It is a hinge on consecutive speed differences while v > v_sw.
- All losses return scalar tensors (device-agnostic).
- lambda_ode=0 and lambda_mono=0 by default until in-transit sequences are
  confirmed well-calibrated.  lambda_bound=0.1 and lambda_qorder=0.5 are
  safe soft constraints from day 1.

References
----------
- ProgressiveDragPropagator.cs (SolarPipe.Training/Physics/) -- mirrors the
  same drag model gamma_eff = gamma0 * (n_obs / n_ref).
- CLAUDE.md: ADR-NE-002, RULE-163.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Physical constants / defaults
# ---------------------------------------------------------------------------
T_MIN_HOURS: float = 12.0    # minimum physical transit time (hours)
T_MAX_HOURS: float = 120.0   # maximum physical transit time (hours)

# Sequence channel indices (matching feature_schema.SEQUENCE_CHANNELS /
# actual Parquet column order from build_pinn_sequences.py)
_CH_SPEED:   int = 1   # flow_speed (km/s)
_CH_DENSITY: int = 2   # proton_density (cm^-3)

# Drag ODE integration defaults
_N_ODE_STEPS: int = 100    # Euler steps over the integration window
_DT_HOURS: float = 0.24    # step size: 24h total / 100 steps


# ---------------------------------------------------------------------------
# Pinball loss (primary task loss)
# ---------------------------------------------------------------------------

def pinball_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9),
) -> torch.Tensor:
    """Quantile (pinball) loss for P10/P50/P90 predictions.

    Args:
        preds:    (B, 3) float32 -- [P10, P50, P90]
        targets:  (B, 1) float32 -- observed transit time (hours)
        quantiles: tuple of 3 quantile levels

    Returns:
        Scalar loss (mean over batch and quantiles).
    """
    y = targets.expand_as(preds)       # (B, 3)
    q = torch.tensor(
        quantiles, dtype=preds.dtype, device=preds.device
    ).unsqueeze(0)                     # (1, 3)
    err = y - preds
    loss = torch.max(q * err, (q - 1.0) * err)
    return loss.mean()


# ---------------------------------------------------------------------------
# Quantile ordering loss
# ---------------------------------------------------------------------------

def quantile_ordering_loss(preds: torch.Tensor) -> torch.Tensor:
    """Penalise violations of P10 <= P50 <= P90.

    Args:
        preds: (B, 3) float32 -- [P10, P50, P90]

    Returns:
        Scalar hinge loss (0 when ordering is satisfied).
    """
    p10, p50, p90 = preds[:, 0], preds[:, 1], preds[:, 2]
    viol_10_50 = F.relu(p10 - p50)    # positive when P10 > P50
    viol_50_90 = F.relu(p50 - p90)    # positive when P50 > P90
    return (viol_10_50.pow(2) + viol_50_90.pow(2)).mean()


# ---------------------------------------------------------------------------
# Transit time bound loss
# ---------------------------------------------------------------------------

def transit_bound_loss(
    preds: torch.Tensor,
    t_min: float = T_MIN_HOURS,
    t_max: float = T_MAX_HOURS,
) -> torch.Tensor:
    """Penalise predictions outside [t_min, t_max] hours.

    Applied to all three quantile outputs so none escapes the physical range.

    Args:
        preds:  (B, 3) float32 -- [P10, P50, P90]
        t_min:  minimum plausible transit time (hours)
        t_max:  maximum plausible transit time (hours)

    Returns:
        Scalar squared-hinge loss.
    """
    below = F.relu(t_min - preds)     # positive when pred < t_min
    above = F.relu(preds - t_max)     # positive when pred > t_max
    return (below.pow(2) + above.pow(2)).mean()


# ---------------------------------------------------------------------------
# Drag ODE residual loss
# ---------------------------------------------------------------------------

def drag_ode_residual_loss(
    pred_transit_hours: torch.Tensor,
    v0_kms: torch.Tensor,
    v_sw_kms: torch.Tensor,
    density_seq: torch.Tensor,
    density_mask: torch.Tensor,
    gamma0_km_inv: float = 0.5e-7,
    n_ref_cm3: float = 5.0,
    n_ode_steps: int = _N_ODE_STEPS,
) -> torch.Tensor:
    """ODE residual: penalise deviation of predicted arrival from
    drag-integrated arrival.

    Integrates the 1-D drag ODE:
        dv/dt = -gamma_eff * |v - v_sw| * (v - v_sw)
        gamma_eff(t) = gamma0 * max(density(t), 0.01) / n_ref

    using explicit Euler with n_ode_steps steps.  The ODE integrates speed
    (km/s); the integral of v dt gives position; we find the time when
    position = 1 AU = 215 Rs (approx 1.496e8 km) and compare to pred_transit.

    Units: gamma0 is in km^-1 (NOT cm^-1) to match DragBasedModel.cs and
    ProgressiveDragPropagator.cs. Canonical Vrsnak 2013 range is
    [1e-9, 1e-6] km^-1; gamma_eff is clamped to this range each step.
    DragBasedModel default = 0.5e-7 km^-1 (line 271).

    Args:
        pred_transit_hours: (B,) float32 -- P50 predicted transit time
        v0_kms:             (B,) float32 -- CME launch speed at 21.5 Rs
        v_sw_kms:           (B,) float32 -- ambient solar wind speed (constant)
        density_seq:        (B, T) float32 -- in-situ proton density (cm^-3)
                            from x_seq[:, :, _CH_DENSITY]; NaN replaced with 0.
        density_mask:       (B, T) float32 -- 1.0 if observed, 0.0 if NULL
        gamma0_km_inv:      baseline drag coefficient (km^-1)
        n_ref_cm3:          reference density for gamma_eff scaling
        n_ode_steps:        Euler integration steps

    Returns:
        Scalar MSE loss (hours^2) between pred_transit and ODE-integrated arrival.
    """
    device = v0_kms.device

    # Physical constants
    _R_START_KM = 21.5 * 696_000.0    # 21.5 solar radii in km
    _R_STOP_KM  = 215.0 * 696_000.0   # 1 AU in km

    # Total integration window: pred_transit * 1.5 (generous upper bound)
    # Use a fixed 120h window clipped to physical range
    t_window_hrs = float(T_MAX_HOURS)
    dt_hrs = t_window_hrs / n_ode_steps
    dt_sec = dt_hrs * 3600.0

    # Interpolate density from the sequence to n_ode_steps ODE timestamps
    T_seq = density_seq.size(1)
    # Density time indices (0..T_seq-1) -> ODE time indices (0..n_ode_steps-1)
    # Use nearest-neighbour lookup for differentiability (no interp needed)
    ode_t_idx = torch.linspace(0, T_seq - 1, n_ode_steps, device=device).long()
    ode_t_idx = ode_t_idx.clamp(0, T_seq - 1)

    # density at each ODE step: (B, n_ode_steps)
    ode_density = density_seq[:, ode_t_idx]         # (B, n_ode_steps)
    ode_mask    = density_mask[:, ode_t_idx]         # (B, n_ode_steps)
    # Where density is unobserved, fall back to n_ref (neutral -- no extra drag)
    ode_density_safe = (
        ode_density * ode_mask + n_ref_cm3 * (1.0 - ode_mask)
    ).clamp(min=0.01)

    # Gamma_eff at each ODE step: (B, n_ode_steps), in km^-1
    # Clamp to ProgressiveDragPropagator.cs physical range [1e-9, 1e-6] km^-1
    gamma_eff_km = (
        gamma0_km_inv * ode_density_safe / n_ref_cm3
    ).clamp(1.0e-9, 1.0e-6)

    # Euler integration of position and speed
    v = v0_kms.clone()                  # (B,) current speed
    r = torch.full_like(v, _R_START_KM) # (B,) current position

    arrival_time_hrs = torch.full_like(v, t_window_hrs)  # default: didn't arrive

    for step in range(n_ode_steps):
        g = gamma_eff_km[:, step]       # (B,)
        dv = v_sw_kms - v               # signed delta (drag accelerates toward v_sw)
        accel = g * dv.abs() * dv       # dv/dt: negative when v > v_sw
        v_new = v + accel * dt_sec      # km/s

        # Clamp: speed cannot reverse direction relative to v_sw
        # (physical: drag never overshoots v_sw in 1-D model)
        v_new = torch.where(
            v > v_sw_kms,
            v_new.clamp(min=v_sw_kms),   # decelerating
            v_new.clamp(max=v_sw_kms),   # accelerating (v < v_sw)
        )

        r_new = r + 0.5 * (v + v_new) * dt_sec   # trapezoidal position update

        # Detect 1 AU crossing this step
        crossed = (r < _R_STOP_KM) & (r_new >= _R_STOP_KM)
        # Linear interpolation within step for crossing time
        frac = ((_R_STOP_KM - r) / (r_new - r + 1e-12)).clamp(0.0, 1.0)
        t_cross = (step + frac) * dt_hrs

        # Update arrival time only for first crossing
        not_arrived = arrival_time_hrs >= t_window_hrs
        arrival_time_hrs = torch.where(
            crossed & not_arrived, t_cross, arrival_time_hrs
        )

        v = v_new
        r = r_new

    # Loss: MSE between P50 prediction and ODE-integrated arrival
    loss = F.mse_loss(pred_transit_hours, arrival_time_hrs.detach())
    return loss


# ---------------------------------------------------------------------------
# Monotonic deceleration hinge loss
# ---------------------------------------------------------------------------

def monotonic_deceleration_loss(
    speed_seq: torch.Tensor,
    speed_mask: torch.Tensor,
    v_sw_kms: torch.Tensor,
) -> torch.Tensor:
    """Penalise speed increases while CME is faster than ambient wind.

    Physics: above ambient wind speed, the drag force can only decelerate.
    Below ambient, it can only accelerate.  This loss penalises the
    model if in-situ speed increases when it should be decelerating.

    NOTE: this loss is applied to the *observed* in-transit speed sequence,
    not to the model's internal state. It is a soft physical prior on the
    input data, regularising the model against "cheating" with unphysical
    density/speed combinations.

    Args:
        speed_seq:   (B, T) float32 -- flow_speed from x_seq[:, :, _CH_SPEED]
        speed_mask:  (B, T) float32 -- 1.0 if observed
        v_sw_kms:    (B,)   float32 -- ambient solar wind speed per event

    Returns:
        Scalar hinge loss.
    """
    # Only consider consecutive pairs where both are observed
    m_pair = speed_mask[:, :-1] * speed_mask[:, 1:]   # (B, T-1)

    speed_t   = speed_seq[:, :-1]    # v(t)
    speed_tp1 = speed_seq[:, 1:]     # v(t+1)
    v_sw = v_sw_kms.unsqueeze(1)     # (B, 1)

    # Penalise: speed increases while v > v_sw (should decelerate)
    above_ambient = (speed_t > v_sw).float()
    speed_increase = F.relu(speed_tp1 - speed_t)   # positive when increasing

    violation = above_ambient * speed_increase * m_pair
    return violation.pow(2).mean()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class PinnLoss(torch.nn.Module):
    """Combined PINN loss for the TFT+PINN model.

    Aggregates all loss components with configurable lambda weights.
    Lambdas of 0.0 disable the corresponding term without computational cost
    (gradient still flows through pinball).

    Args:
        lambda_ode:     weight for drag ODE residual loss
        lambda_mono:    weight for monotonic deceleration loss
        lambda_bound:   weight for transit time bound loss
        lambda_qorder:  weight for quantile ordering loss
        quantiles:      tuple of (p_low, p_mid, p_high) quantile levels
        gamma0_km_inv:  drag coefficient (km^-1) for ODE loss; matches
                        DragBasedModel.cs. Clamped to [1e-9, 1e-6] km^-1.
        n_ref_cm3:      reference density for gamma_eff
    """

    def __init__(
        self,
        lambda_ode: float = 0.0,
        lambda_mono: float = 0.0,
        lambda_bound: float = 0.1,
        lambda_qorder: float = 0.5,
        quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9),
        gamma0_km_inv: float = 0.5e-7,
        n_ref_cm3: float = 5.0,
    ) -> None:
        super().__init__()
        self.lambda_ode    = lambda_ode
        self.lambda_mono   = lambda_mono
        self.lambda_bound  = lambda_bound
        self.lambda_qorder = lambda_qorder
        self.quantiles     = quantiles
        self.gamma0_km_inv = gamma0_km_inv
        self.n_ref_cm3     = n_ref_cm3

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        v0_kms: torch.Tensor,
        v_sw_kms: torch.Tensor,
        x_seq: torch.Tensor,
        m_seq: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute total loss and component breakdown.

        Args:
            preds:    (B, 3) float32 -- [P10, P50, P90] transit hours
            targets:  (B, 1) float32 -- observed transit hours
            v0_kms:   (B,)   float32 -- CME launch speed
            v_sw_kms: (B,)   float32 -- ambient solar wind speed
            x_seq:    (B, T, C) float32 -- sequence values (NaN -> 0)
            m_seq:    (B, T, C) float32 -- sequence masks

        Returns:
            dict with keys: 'total', 'pinball', 'ode', 'mono', 'bound', 'qorder'
            All values are scalar tensors.
        """
        L_pinball = pinball_loss(preds, targets, self.quantiles)
        L_bound   = transit_bound_loss(preds)
        L_qorder  = quantile_ordering_loss(preds)

        total = (
            L_pinball
            + self.lambda_bound  * L_bound
            + self.lambda_qorder * L_qorder
        )

        # ODE and monotonicity losses only when lambda > 0
        L_ode  = torch.zeros(1, device=preds.device, dtype=preds.dtype)
        L_mono = torch.zeros(1, device=preds.device, dtype=preds.dtype)

        if self.lambda_ode > 0.0:
            density_seq  = x_seq[:, :, _CH_DENSITY]
            density_mask = m_seq[:, :, _CH_DENSITY]
            L_ode = drag_ode_residual_loss(
                preds[:, 1],   # P50
                v0_kms,
                v_sw_kms,
                density_seq,
                density_mask,
                gamma0_km_inv=self.gamma0_km_inv,
                n_ref_cm3=self.n_ref_cm3,
            )
            total = total + self.lambda_ode * L_ode

        if self.lambda_mono > 0.0:
            speed_seq  = x_seq[:, :, _CH_SPEED]
            speed_mask = m_seq[:, :, _CH_SPEED]
            L_mono = monotonic_deceleration_loss(speed_seq, speed_mask, v_sw_kms)
            total = total + self.lambda_mono * L_mono

        return {
            "total":   total,
            "pinball": L_pinball.detach(),
            "ode":     L_ode.detach(),
            "mono":    L_mono.detach(),
            "bound":   L_bound.detach(),
            "qorder":  L_qorder.detach(),
        }


# ---------------------------------------------------------------------------
# Factory: build from YAML hyperparameters
# ---------------------------------------------------------------------------

def build_pinn_loss(hp: dict | None = None) -> PinnLoss:
    """Instantiate PinnLoss from a hyperparameters dict (YAML-sourced).

    YAML key `drag_gamma_km_inv` is the drag coefficient in km^-1
    (matches DragBasedModel.cs). DragBasedModel default = 0.5e-7 km^-1.
    """
    hp = hp or {}
    return PinnLoss(
        lambda_ode=float(hp.get("lambda_ode", 0.0)),
        lambda_mono=float(hp.get("lambda_mono", 0.0)),
        lambda_bound=float(hp.get("lambda_bound", 0.1)),
        lambda_qorder=float(hp.get("lambda_qorder", 0.5)),
        gamma0_km_inv=float(hp.get("drag_gamma_km_inv", 0.5e-7)),
        n_ref_cm3=float(hp.get("n_ref_cm3", 5.0)),
    )
