"""Task 6.2 — HCS Alignment and Coronal Hole Deflection Model.

Models two rotation effects on CME trajectory:
1. Heliospheric Current Sheet (HCS) alignment — CMEs tend to co-rotate
   with the HCS tilt, producing a systematic latitudinal deflection.
2. Coronal hole deflection — open-field coronal holes deflect CME axes
   away from the hole boundary, affecting predicted Bz polarity at L1.

Both effects are expressed as adjustments to the CME's initial axis angle,
which feeds into the Bz sign prediction in the flux-rope model.

Parameters are null-filled when GONG synoptic map data is unavailable
(hcs_tilt_angle=None, hcs_distance=None in feature_vectors).

All angles in degrees; positive = northward of ecliptic.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HCS deflection strength coefficient (degrees of deflection per degree of
# HCS tilt, fitted from ENLIL hindcasts in Isavnin et al. 2014 proxy).
_HCS_DEFLECTION_COEFF: float = 0.12

# Maximum HCS deflection magnitude (cap to avoid extrapolation blowup)
_HCS_DEFLECTION_MAX_DEG: float = 15.0

# Coronal hole deflection strength — radians per unit proximity (0–1 scale)
# Proximity = 1 → hole boundary is immediately adjacent to CME launch point
_CH_DEFLECTION_COEFF_DEG: float = 8.0  # degrees deflection at max proximity

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RotationResult:
    """Adjusted CME parameters after rotation corrections."""

    adjusted_latitude: float
    """Deflected CME latitude (degrees, ecliptic)."""

    adjusted_axis_angle: float
    """Adjusted initial flux-rope axis angle (degrees; 0 = ecliptic east)."""

    hcs_deflection_deg: float
    """Latitudinal deflection from HCS alignment (degrees)."""

    ch_deflection_deg: float
    """Deflection from coronal-hole proximity (degrees)."""

    hcs_available: bool
    """True if HCS tilt data was available (not null-filled)."""

    ch_available: bool
    """True if coronal hole proximity data was available."""


# ---------------------------------------------------------------------------
# HCS deflection
# ---------------------------------------------------------------------------

def _hcs_deflection(
    cme_latitude_deg: float,
    hcs_tilt_angle_deg: float | None,
    hcs_distance_deg: float | None,
) -> float:
    """Compute latitudinal deflection from HCS alignment.

    The CME latitude is deflected toward the HCS by a fraction proportional
    to the HCS tilt angle, attenuated by the angular distance to the HCS.

    Returns 0.0 if either HCS parameter is None (null-fill policy).
    """
    if hcs_tilt_angle_deg is None or hcs_distance_deg is None:
        return 0.0

    # Deflection points toward HCS (sign = sign of HCS tilt relative to CME)
    delta_to_hcs = hcs_tilt_angle_deg - cme_latitude_deg
    attenuation = math.exp(-abs(hcs_distance_deg) / 30.0)  # 30° e-folding scale
    raw = _HCS_DEFLECTION_COEFF * delta_to_hcs * attenuation

    return float(max(-_HCS_DEFLECTION_MAX_DEG, min(_HCS_DEFLECTION_MAX_DEG, raw)))


# ---------------------------------------------------------------------------
# Coronal-hole deflection
# ---------------------------------------------------------------------------

def _ch_deflection(
    cme_latitude_deg: float,
    ch_proximity: float | None,
    ch_polarity: int | None,
) -> float:
    """Compute deflection from adjacent coronal hole.

    The CME is pushed away from the coronal hole boundary. Polarity encodes
    direction: +1 = positive-polarity hole (deflects southward), -1 = negative
    (deflects northward). Net effect is small for distant holes.

    Returns 0.0 if proximity is None.
    """
    if ch_proximity is None:
        return 0.0

    proximity = float(max(0.0, min(1.0, ch_proximity)))

    if ch_polarity is not None and ch_polarity < 0:
        sign = +1.0   # negative-polarity hole → deflect northward
    else:
        sign = -1.0   # positive-polarity hole → deflect southward

    # Deflection away from CME's current latitude
    magnitude = _CH_DEFLECTION_COEFF_DEG * proximity
    return float(sign * magnitude)


# ---------------------------------------------------------------------------
# Axis angle adjustment
# ---------------------------------------------------------------------------

def _adjust_axis_angle(
    initial_axis_angle_deg: float,
    hcs_deflection_deg: float,
    ch_deflection_deg: float,
) -> float:
    """Adjust the flux-rope axis angle from cumulative deflections.

    The axis angle rotates by the same total deflection as the trajectory,
    preserving the chirality-axis relationship.  Result is normalised to
    (-180°, 180°].
    """
    total_deflection = hcs_deflection_deg + ch_deflection_deg
    raw = initial_axis_angle_deg + total_deflection
    # Normalise to (-180, 180]
    while raw > 180.0:
        raw -= 360.0
    while raw <= -180.0:
        raw += 360.0
    return float(raw)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_rotation_corrections(
    cme_latitude_deg: float,
    initial_axis_angle_deg: float,
    hcs_tilt_angle_deg: float | None = None,
    hcs_distance_deg: float | None = None,
    ch_proximity: float | None = None,
    ch_polarity: int | None = None,
) -> RotationResult:
    """Apply HCS alignment and coronal-hole deflection to a CME trajectory.

    Args:
        cme_latitude_deg: CME source latitude (degrees, ecliptic).
        initial_axis_angle_deg: Initial flux-rope axis angle (degrees).
        hcs_tilt_angle_deg: HCS latitude at CME longitude (degrees). None = unavailable.
        hcs_distance_deg: Angular distance from CME to HCS (degrees). None = unavailable.
        ch_proximity: Coronal hole proximity index 0–1. None = unavailable.
        ch_polarity: CH magnetic polarity (+1 or -1). None = unavailable.

    Returns:
        RotationResult with adjusted latitude and axis angle.
    """
    hcs_def = _hcs_deflection(cme_latitude_deg, hcs_tilt_angle_deg, hcs_distance_deg)
    ch_def = _ch_deflection(cme_latitude_deg, ch_proximity, ch_polarity)

    adjusted_lat = cme_latitude_deg + hcs_def + ch_def
    adjusted_axis = _adjust_axis_angle(initial_axis_angle_deg, hcs_def, ch_def)

    if hcs_tilt_angle_deg is None:
        logger.debug("rotation_model: HCS data unavailable — null-filled")
    if ch_proximity is None:
        logger.debug("rotation_model: coronal hole data unavailable — null-filled")

    return RotationResult(
        adjusted_latitude=adjusted_lat,
        adjusted_axis_angle=adjusted_axis,
        hcs_deflection_deg=hcs_def,
        ch_deflection_deg=ch_def,
        hcs_available=hcs_tilt_angle_deg is not None,
        ch_available=ch_proximity is not None,
    )
