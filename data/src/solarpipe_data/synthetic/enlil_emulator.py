"""Task 6.4 — ENLIL-Proxy Ensemble Emulator.

Composes three physical layers to produce per-member transit-time and
arrival-speed predictions:
    1. Drag model (Task 6.1)  — v(t), r(t) integration
    2. Rotation model (Task 6.2) — trajectory/axis angle adjustments
    3. Systematic bias + Gaussian noise — fit from ENLIL residuals

The bias correction is modelled as:
    T_corrected = T_drag * (1 + bias_frac) + noise
where bias_frac is estimated from the mean ENLIL-to-drag model transit
residual (typically +5 to +15%, because drag model underestimates deceleration
in structured solar wind).

Output schema is fixed — column names must match C# ParquetProvider
expectations.  See ARCHITECTURE.md and configs/flux_rope_propagation_v1.yaml.

Parquet column schema (all float64 unless noted):
    event_id          TEXT    — source CME activity_id (or "synthetic_N")
    member_id         INT64   — 0-indexed member within ensemble
    seed              INT64   — random seed used
    speed_initial     FLOAT   — km/s at 21.5 R☉
    speed_arrival     FLOAT   — km/s at L1
    ambient_wind      FLOAT   — km/s
    transit_hours     FLOAT   — predicted transit time (hours)
    gamma             FLOAT   — drag coefficient km⁻¹
    latitude_deg      FLOAT   — adjusted CME latitude after rotation
    longitude_deg     FLOAT   — initial CME longitude (ecliptic, Stonyhurst)
    axis_angle_deg    FLOAT   — adjusted flux-rope axis angle
    angular_width     FLOAT   — degrees
    hcs_deflection    FLOAT   — degrees (0 if HCS unavailable)
    ch_deflection     FLOAT   — degrees (0 if CH unavailable)
    bias_correction   FLOAT   — fractional bias applied
    noise_sigma       FLOAT   — 1-σ transit noise applied (hours)
    quality_flag      INT64   — from source feature_vector (or 3 for synthetic)
    flare_class_numeric FLOAT — proxy W/m² (0 = no flare)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .drag_model import DragCalibration, DragResult, _GAMMA_DEFAULT, propagate
from .monte_carlo import CMEParameters, EnsembleParameters, sample_ensemble
from .rotation_model import RotationResult, apply_rotation_corrections

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Emulator configuration
# ---------------------------------------------------------------------------


@dataclass
class EmulatorConfig:
    """Configuration for one emulator run."""

    n_members: int = 500
    """Ensemble size per event."""

    seed: int = 42
    """Base random seed (per-event seed = seed + event_index)."""

    gamma: float = _GAMMA_DEFAULT
    """Drag coefficient (km⁻¹). Overridden by calibrate_from_db if called."""

    bias_frac: float = 0.08
    """Systematic transit-time bias: T_adj = T_drag * (1 + bias_frac)."""

    noise_sigma_hours: float = 4.0
    """1-σ Gaussian noise added to transit time (hours)."""

    speed_perturbation_sigma: float = 0.15
    """Log-normal CV for per-member speed perturbation around base speed."""

    t_max_hours: float = 240.0
    """Maximum integration time before declaring failure."""


# ---------------------------------------------------------------------------
# Per-event emulator output
# ---------------------------------------------------------------------------


@dataclass
class EmulatorResult:
    """Ensemble prediction for one CME event."""

    event_id: str
    member_ids: np.ndarray          # shape (n_members,) int64
    transit_hours: np.ndarray       # shape (n_members,) float64
    arrival_speeds: np.ndarray      # shape (n_members,) float64
    latitudes: np.ndarray           # adjusted, float64
    longitudes: np.ndarray          # original, float64
    axis_angles: np.ndarray         # adjusted, float64
    angular_widths: np.ndarray      # float64
    initial_speeds: np.ndarray      # float64
    ambient_winds: np.ndarray       # float64
    gammas: np.ndarray              # float64
    hcs_deflections: np.ndarray     # float64
    ch_deflections: np.ndarray      # float64
    flare_classes: np.ndarray       # float64
    seeds: np.ndarray               # int64 (same seed replicated, kept for traceability)
    quality_flags: np.ndarray       # int64
    bias_fracs: np.ndarray          # float64
    noise_sigmas: np.ndarray        # float64
    n_success: int
    config: EmulatorConfig = field(default_factory=EmulatorConfig)

    def as_dict(self) -> dict[str, np.ndarray | list]:
        """Return flat dict keyed by Parquet column names."""
        n = len(self.member_ids)
        return {
            "event_id": np.array([self.event_id] * n, dtype=object),
            "member_id": self.member_ids,
            "seed": self.seeds,
            "speed_initial": self.initial_speeds,
            "speed_arrival": self.arrival_speeds,
            "ambient_wind": self.ambient_winds,
            "transit_hours": self.transit_hours,
            "gamma": self.gammas,
            "latitude_deg": self.latitudes,
            "longitude_deg": self.longitudes,
            "axis_angle_deg": self.axis_angles,
            "angular_width": self.angular_widths,
            "hcs_deflection": self.hcs_deflections,
            "ch_deflection": self.ch_deflections,
            "bias_correction": self.bias_fracs,
            "noise_sigma": self.noise_sigmas,
            "quality_flag": self.quality_flags,
            "flare_class_numeric": self.flare_classes,
        }


# ---------------------------------------------------------------------------
# Core emulator
# ---------------------------------------------------------------------------


def emulate_event(
    event_id: str,
    base_speed_kms: float,
    base_wind_kms: float = 450.0,
    cme_latitude_deg: float = 0.0,
    cme_longitude_deg: float = 0.0,
    initial_axis_angle_deg: float = 0.0,
    hcs_tilt_angle_deg: float | None = None,
    hcs_distance_deg: float | None = None,
    ch_proximity: float | None = None,
    ch_polarity: int | None = None,
    quality_flag: int = 3,
    config: EmulatorConfig | None = None,
    event_index: int = 0,
) -> EmulatorResult:
    """Run the ensemble emulator for one CME event.

    Args:
        event_id: CME activity_id or synthetic identifier.
        base_speed_kms: Best-estimate initial CME speed at 21.5 R☉.
        base_wind_kms: Best-estimate ambient solar wind speed.
        cme_latitude_deg: CME source latitude.
        cme_longitude_deg: CME source longitude.
        initial_axis_angle_deg: Initial flux-rope axis angle.
        hcs_tilt_angle_deg: HCS tilt (None = unavailable).
        hcs_distance_deg: Distance to HCS (None = unavailable).
        ch_proximity: Coronal hole proximity 0–1 (None = unavailable).
        ch_polarity: CH polarity ±1 (None = unavailable).
        quality_flag: Quality flag from feature_vectors table.
        config: EmulatorConfig. Defaults to EmulatorConfig().
        event_index: Index added to seed for per-event reproducibility.

    Returns:
        EmulatorResult with per-member arrays.
    """
    if config is None:
        config = EmulatorConfig()

    event_seed = config.seed + event_index
    ensemble: EnsembleParameters = sample_ensemble(
        n_members=config.n_members,
        seed=event_seed,
        base_speed_kms=base_speed_kms,
        base_wind_kms=base_wind_kms,
        speed_perturbation_sigma=config.speed_perturbation_sigma,
    )

    rng = np.random.default_rng(event_seed + 9999)
    noise = rng.normal(0.0, config.noise_sigma_hours, size=config.n_members)

    transit_arr = np.empty(config.n_members, dtype=np.float64)
    arrival_arr = np.empty(config.n_members, dtype=np.float64)
    lat_arr = np.empty(config.n_members, dtype=np.float64)
    axis_arr = np.empty(config.n_members, dtype=np.float64)
    hcs_def_arr = np.zeros(config.n_members, dtype=np.float64)
    ch_def_arr = np.zeros(config.n_members, dtype=np.float64)
    n_success = 0

    for i, member in enumerate(ensemble.members):
        # Rotation corrections per member (using member latitude for perturbation)
        rot: RotationResult = apply_rotation_corrections(
            cme_latitude_deg=member.latitude_deg,
            initial_axis_angle_deg=member.axis_angle_deg,
            hcs_tilt_angle_deg=hcs_tilt_angle_deg,
            hcs_distance_deg=hcs_distance_deg,
            ch_proximity=ch_proximity,
            ch_polarity=ch_polarity,
        )

        # Drag integration
        drag: DragResult = propagate(
            v0_kms=member.speed_kms,
            w_kms=member.ambient_wind_kms,
            gamma=config.gamma,
            t_max_hours=config.t_max_hours,
        )

        # Bias + noise correction
        transit_corrected = (
            drag.transit_time_hours * (1.0 + config.bias_frac) + noise[i]
        )
        transit_corrected = max(1.0, transit_corrected)  # physical floor: 1 hour

        transit_arr[i] = transit_corrected
        arrival_arr[i] = drag.arrival_speed_kms
        lat_arr[i] = rot.adjusted_latitude
        axis_arr[i] = rot.adjusted_axis_angle
        hcs_def_arr[i] = rot.hcs_deflection_deg
        ch_def_arr[i] = rot.ch_deflection_deg

        if drag.success:
            n_success += 1

    members_arr = ensemble.as_arrays()

    return EmulatorResult(
        event_id=event_id,
        member_ids=np.arange(config.n_members, dtype=np.int64),
        transit_hours=transit_arr,
        arrival_speeds=arrival_arr,
        latitudes=lat_arr,
        longitudes=np.full(config.n_members, cme_longitude_deg),
        axis_angles=axis_arr,
        angular_widths=members_arr["angular_width_deg"],
        initial_speeds=members_arr["speed_kms"],
        ambient_winds=members_arr["ambient_wind_kms"],
        gammas=np.full(config.n_members, config.gamma),
        hcs_deflections=hcs_def_arr,
        ch_deflections=ch_def_arr,
        flare_classes=members_arr["flare_class_numeric"],
        seeds=np.full(config.n_members, event_seed, dtype=np.int64),
        quality_flags=np.full(config.n_members, quality_flag, dtype=np.int64),
        bias_fracs=np.full(config.n_members, config.bias_frac),
        noise_sigmas=np.full(config.n_members, config.noise_sigma_hours),
        n_success=n_success,
        config=config,
    )


# ---------------------------------------------------------------------------
# Batch emulator from feature_vectors table
# ---------------------------------------------------------------------------


def emulate_from_db(
    db_path: str,
    config: EmulatorConfig | None = None,
    min_quality: int = 3,
    limit: int | None = None,
) -> list[EmulatorResult]:
    """Run the emulator over all feature_vectors rows with quality_flag ≥ min_quality.

    Args:
        db_path: Path to staging.db.
        config: EmulatorConfig (defaults to EmulatorConfig()).
        min_quality: Minimum quality_flag to include.
        limit: Optional row limit (for testing).

    Returns:
        List of EmulatorResult, one per CME event.
    """
    import sqlalchemy as sa

    from ..database.schema import make_engine

    if config is None:
        config = EmulatorConfig()

    engine = make_engine(db_path)

    q = sa.text("""
        SELECT
            activity_id,
            cme_speed_kms,
            cme_latitude,
            cme_longitude,
            sw_speed_ambient,
            hcs_tilt_angle,
            hcs_distance,
            quality_flag,
            COALESCE(meanshr, 0.0) AS initial_axis_proxy,
            flare_class_numeric
        FROM feature_vectors
        WHERE quality_flag >= :min_q
        ORDER BY launch_time
        LIMIT :lim
    """)

    lim = limit if limit is not None else 999999

    with engine.connect() as conn:
        rows = conn.execute(q, {"min_q": min_quality, "lim": lim}).fetchall()

    if not rows:
        logger.warning("emulate_from_db: no feature_vectors rows with quality≥%d", min_quality)
        return []

    results = []
    for idx, row in enumerate(rows):
        (
            activity_id, speed, lat, lon, wind,
            hcs_tilt, hcs_dist, qflag, axis_proxy, flare_cls,
        ) = row

        result = emulate_event(
            event_id=str(activity_id),
            base_speed_kms=float(speed) if speed else 450.0,
            base_wind_kms=float(wind) if wind else 450.0,
            cme_latitude_deg=float(lat) if lat else 0.0,
            cme_longitude_deg=float(lon) if lon else 0.0,
            initial_axis_angle_deg=float(axis_proxy) if axis_proxy else 0.0,
            hcs_tilt_angle_deg=float(hcs_tilt) if hcs_tilt is not None else None,
            hcs_distance_deg=float(hcs_dist) if hcs_dist is not None else None,
            quality_flag=int(qflag) if qflag else 3,
            config=config,
            event_index=idx,
        )
        results.append(result)

    logger.info("emulate_from_db: %d events emulated (%d members each)", len(results), config.n_members)
    return results


# ---------------------------------------------------------------------------
# Generation timestamp helper
# ---------------------------------------------------------------------------


def generation_metadata(config: EmulatorConfig, n_events: int) -> dict[str, Any]:
    """Build metadata dict for Parquet file metadata."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_events": str(n_events),
        "n_members": str(config.n_members),
        "seed": str(config.seed),
        "gamma": str(config.gamma),
        "bias_frac": str(config.bias_frac),
        "noise_sigma_hours": str(config.noise_sigma_hours),
        "speed_perturbation_sigma": str(config.speed_perturbation_sigma),
        "source": "SolarPipe synthetic ENLIL emulator v1",
    }
