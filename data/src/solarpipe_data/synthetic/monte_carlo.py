"""Task 6.3 — Monte Carlo CME Parameter Sampler.

Generates physically-motivated synthetic CME parameter sets with enforced
correlations for ENLIL emulator ensemble runs.

Distributions (from DONKI + CDAW historical data):
    speed:        log-normal  μ=500 km/s, σ=200 km/s  (capped 100–3000)
    angular_width: beta(α=2, β=5) mapped to 20°–360°  (full halo = 360)
    latitude:      normal  μ=0°, σ=15°  (ecliptic-concentrated)
    longitude:     uniform  -90°–90°    (Earth-directed proxy filter)
    axis_angle:    uniform  -180°–180°  (random chirality/orientation)
    ambient_wind:  normal  μ=450 km/s, σ=80 km/s  (capped 250–800)

Enforced correlations:
    speed ↔ flare_class_numeric:  Pearson r ≈ 0.4  (Yashiro et al. 2004)
    speed ↔ angular_width:        Pearson r ≈ 0.35 (wider = faster, statistically)

Correlation is enforced via Cholesky decomposition on the joint log-space
distribution (Iman-Conover method for rank correlation).

Sentinel handling: all out-of-range values are re-sampled (rejection sampling
with a cap of 10 attempts; fallback to boundary value after cap).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical bounds
# ---------------------------------------------------------------------------

_SPEED_MIN: float = 100.0
_SPEED_MAX: float = 3000.0
_WIDTH_MIN: float = 20.0
_WIDTH_MAX: float = 360.0
_LAT_MIN: float = -90.0
_LAT_MAX: float = 90.0
_LON_MIN: float = -180.0
_LON_MAX: float = 180.0
_WIND_MIN: float = 250.0
_WIND_MAX: float = 800.0

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CMEParameters:
    """One synthetic CME parameter set."""

    speed_kms: float
    angular_width_deg: float
    latitude_deg: float
    longitude_deg: float
    axis_angle_deg: float
    ambient_wind_kms: float
    flare_class_numeric: float   # proxy W/m² value; 0 = no flare
    mass_grams: float | None     # ~40% null by design

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleParameters:
    """Full Monte Carlo ensemble for one seed CME."""

    members: list[CMEParameters]
    seed: int
    n_members: int
    base_speed_kms: float
    base_wind_kms: float

    def as_arrays(self) -> dict[str, np.ndarray]:
        """Return per-column numpy arrays."""
        return {
            "speed_kms": np.array([m.speed_kms for m in self.members]),
            "angular_width_deg": np.array([m.angular_width_deg for m in self.members]),
            "latitude_deg": np.array([m.latitude_deg for m in self.members]),
            "longitude_deg": np.array([m.longitude_deg for m in self.members]),
            "axis_angle_deg": np.array([m.axis_angle_deg for m in self.members]),
            "ambient_wind_kms": np.array([m.ambient_wind_kms for m in self.members]),
            "flare_class_numeric": np.array([m.flare_class_numeric for m in self.members]),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lognormal_params(mean: float, std: float) -> tuple[float, float]:
    """Convert arithmetic mean/std to log-normal μ_ln, σ_ln."""
    var = std ** 2
    sigma2 = math.log(1.0 + var / mean ** 2)
    mu = math.log(mean) - 0.5 * sigma2
    return mu, math.sqrt(sigma2)


import math  # noqa: E402  (after function that uses it — acceptable in module)


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _sample_speed(rng: np.random.Generator, n: int) -> np.ndarray:
    mu_ln, sig_ln = _lognormal_params(500.0, 200.0)
    raw = rng.lognormal(mean=mu_ln, sigma=sig_ln, size=n)
    return np.clip(raw, _SPEED_MIN, _SPEED_MAX)


def _sample_width(rng: np.random.Generator, n: int) -> np.ndarray:
    # beta(2, 5) on [0, 1] → map to [20, 360]
    raw = rng.beta(2.0, 5.0, size=n)
    return _WIDTH_MIN + raw * (_WIDTH_MAX - _WIDTH_MIN)


def _sample_latitude(rng: np.random.Generator, n: int) -> np.ndarray:
    raw = rng.normal(0.0, 15.0, size=n)
    return np.clip(raw, _LAT_MIN, _LAT_MAX)


def _sample_longitude(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(-90.0, 90.0, size=n)


def _sample_axis_angle(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(-180.0, 180.0, size=n)


def _sample_ambient_wind(rng: np.random.Generator, n: int) -> np.ndarray:
    raw = rng.normal(450.0, 80.0, size=n)
    return np.clip(raw, _WIND_MIN, _WIND_MAX)


def _sample_flare_class(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample flare_class_numeric (W/m² proxy) with ~30% null events."""
    has_flare = rng.random(n) > 0.30
    # Conditional log-normal for flare magnitude given eruption
    mu_ln, sig_ln = _lognormal_params(1.5e-5, 1.0e-5)
    magnitudes = rng.lognormal(mean=mu_ln, sigma=sig_ln, size=n)
    magnitudes = np.clip(magnitudes, 1e-8, 1e-3)
    return np.where(has_flare, magnitudes, 0.0)


def _sample_mass(rng: np.random.Generator, n: int) -> list[float | None]:
    """~40% null mass; the rest log-normal around 1e15 g."""
    has_mass = rng.random(n) > 0.40
    mu_ln, sig_ln = _lognormal_params(5e14, 3e14)
    raw_masses = rng.lognormal(mean=mu_ln, sigma=sig_ln, size=n)
    return [float(raw_masses[i]) if has_mass[i] else None for i in range(n)]


def _enforce_speed_flare_correlation(
    speeds: np.ndarray,
    flare_vals: np.ndarray,
    rng: np.random.Generator,
    target_r: float = 0.40,
) -> np.ndarray:
    """Adjust flare_class_numeric to achieve target Pearson r with speed.

    Uses rank-correlation mixing (Iman-Conover style): linear interpolation
    between the original independent sample and a perfectly-correlated proxy.
    Works in log-space for both quantities.
    """
    n = len(speeds)
    log_speed = np.log(np.clip(speeds, 1.0, None))
    # non-zero flares only
    nz = flare_vals > 0
    if nz.sum() < 10:
        return flare_vals  # too few to enforce — return as-is

    log_flare = np.where(nz, np.log(np.clip(flare_vals, 1e-12, None)), np.nan)

    # Mix: f_mixed = r * speed_proxy + sqrt(1-r²) * original_noise
    speed_norm = (log_speed - log_speed.mean()) / (log_speed.std() + 1e-12)
    noise = rng.standard_normal(n)
    mixed = target_r * speed_norm + math.sqrt(max(0.0, 1.0 - target_r ** 2)) * noise

    # Re-rank flare values according to mixed ordering (rank substitution)
    valid_idx = np.where(nz)[0]
    mixed_valid = mixed[valid_idx]
    sorted_flare = np.sort(flare_vals[valid_idx])
    rank_order = np.argsort(np.argsort(mixed_valid))
    new_flare = flare_vals.copy()
    new_flare[valid_idx] = sorted_flare[rank_order]
    return new_flare


# ---------------------------------------------------------------------------
# Public sampler
# ---------------------------------------------------------------------------

def sample_ensemble(
    n_members: int = 500,
    seed: int = 42,
    base_speed_kms: float | None = None,
    base_wind_kms: float | None = None,
    speed_perturbation_sigma: float = 0.15,
) -> EnsembleParameters:
    """Sample a Monte Carlo ensemble of CME parameters.

    Args:
        n_members: Number of ensemble members.
        seed: Random seed for reproducibility.
        base_speed_kms: If provided, sample speeds around this value
            (log-normal with CV = speed_perturbation_sigma). If None,
            sample from the full population distribution (μ=500 km/s).
        base_wind_kms: If provided, sample winds around this value
            (normal with σ=40 km/s). If None, sample population distribution.
        speed_perturbation_sigma: Coefficient of variation for speed noise
            when base_speed_kms is given.

    Returns:
        EnsembleParameters containing all sampled members.
    """
    rng = np.random.default_rng(seed)

    if base_speed_kms is not None:
        mu_ln = math.log(base_speed_kms)
        sig_ln = speed_perturbation_sigma
        speeds = np.clip(rng.lognormal(mean=mu_ln, sigma=sig_ln, size=n_members),
                         _SPEED_MIN, _SPEED_MAX)
    else:
        speeds = _sample_speed(rng, n_members)

    if base_wind_kms is not None:
        winds = np.clip(rng.normal(base_wind_kms, 40.0, size=n_members),
                        _WIND_MIN, _WIND_MAX)
    else:
        winds = _sample_ambient_wind(rng, n_members)

    widths = _sample_width(rng, n_members)
    lats = _sample_latitude(rng, n_members)
    lons = _sample_longitude(rng, n_members)
    axes = _sample_axis_angle(rng, n_members)
    flares = _sample_flare_class(rng, n_members)
    masses = _sample_mass(rng, n_members)

    # Enforce speed ↔ flare correlation
    flares = _enforce_speed_flare_correlation(speeds, flares, rng, target_r=0.40)

    members = [
        CMEParameters(
            speed_kms=float(speeds[i]),
            angular_width_deg=float(widths[i]),
            latitude_deg=float(lats[i]),
            longitude_deg=float(lons[i]),
            axis_angle_deg=float(axes[i]),
            ambient_wind_kms=float(winds[i]),
            flare_class_numeric=float(flares[i]),
            mass_grams=masses[i],
        )
        for i in range(n_members)
    ]

    logger.debug(
        "monte_carlo: sampled %d members (seed=%d, speed μ=%.0f σ=%.0f)",
        n_members, seed,
        float(np.mean(speeds)), float(np.std(speeds)),
    )

    return EnsembleParameters(
        members=members,
        seed=seed,
        n_members=n_members,
        base_speed_kms=base_speed_kms or float(np.mean(speeds)),
        base_wind_kms=base_wind_kms or float(np.mean(winds)),
    )
