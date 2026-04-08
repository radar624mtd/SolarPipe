"""Unit tests for synthetic/rotation_model.py (Task 6.2).

RULE-090: asyncio_mode=auto
RULE-092: @pytest.mark.unit
"""
from __future__ import annotations

import pytest

from solarpipe_data.synthetic.rotation_model import (
    RotationResult,
    apply_rotation_corrections,
)


@pytest.mark.unit
class TestApplyRotationCorrections:
    def test_no_data_returns_zero_deflections(self):
        r = apply_rotation_corrections(
            cme_latitude_deg=10.0,
            initial_axis_angle_deg=45.0,
        )
        assert r.hcs_deflection_deg == pytest.approx(0.0)
        assert r.ch_deflection_deg == pytest.approx(0.0)
        assert r.adjusted_latitude == pytest.approx(10.0)
        assert r.hcs_available is False
        assert r.ch_available is False

    def test_hcs_deflects_toward_sheet(self):
        # CME at -10°, HCS at +5° → deflection should be northward (positive)
        r = apply_rotation_corrections(
            cme_latitude_deg=-10.0,
            initial_axis_angle_deg=0.0,
            hcs_tilt_angle_deg=5.0,
            hcs_distance_deg=0.0,
        )
        assert r.hcs_deflection_deg > 0.0
        assert r.hcs_available is True

    def test_hcs_deflection_capped(self):
        # Very large HCS tilt should be capped at 15°
        r = apply_rotation_corrections(
            cme_latitude_deg=-89.0,
            initial_axis_angle_deg=0.0,
            hcs_tilt_angle_deg=89.0,
            hcs_distance_deg=0.0,
        )
        assert abs(r.hcs_deflection_deg) <= 15.0

    def test_hcs_attenuation_with_distance(self):
        # Larger distance → smaller deflection
        r_close = apply_rotation_corrections(
            cme_latitude_deg=0.0,
            initial_axis_angle_deg=0.0,
            hcs_tilt_angle_deg=30.0,
            hcs_distance_deg=5.0,
        )
        r_far = apply_rotation_corrections(
            cme_latitude_deg=0.0,
            initial_axis_angle_deg=0.0,
            hcs_tilt_angle_deg=30.0,
            hcs_distance_deg=60.0,
        )
        assert abs(r_close.hcs_deflection_deg) > abs(r_far.hcs_deflection_deg)

    def test_ch_deflection_positive_polarity(self):
        # Positive-polarity CH → southward deflection (negative)
        r = apply_rotation_corrections(
            cme_latitude_deg=5.0,
            initial_axis_angle_deg=0.0,
            ch_proximity=0.8,
            ch_polarity=1,
        )
        assert r.ch_deflection_deg < 0.0
        assert r.ch_available is True

    def test_ch_deflection_negative_polarity(self):
        # Negative-polarity CH → northward deflection (positive)
        r = apply_rotation_corrections(
            cme_latitude_deg=5.0,
            initial_axis_angle_deg=0.0,
            ch_proximity=0.8,
            ch_polarity=-1,
        )
        assert r.ch_deflection_deg > 0.0

    def test_ch_zero_proximity_no_deflection(self):
        r = apply_rotation_corrections(
            cme_latitude_deg=0.0,
            initial_axis_angle_deg=0.0,
            ch_proximity=0.0,
            ch_polarity=1,
        )
        assert r.ch_deflection_deg == pytest.approx(0.0)

    def test_axis_angle_normalised(self):
        # Large deflection should not produce |axis_angle| > 180
        r = apply_rotation_corrections(
            cme_latitude_deg=89.0,
            initial_axis_angle_deg=170.0,
            hcs_tilt_angle_deg=-50.0,
            hcs_distance_deg=0.0,
        )
        assert -180.0 < r.adjusted_axis_angle <= 180.0

    def test_returns_rotation_result_type(self):
        r = apply_rotation_corrections(0.0, 0.0)
        assert isinstance(r, RotationResult)

    def test_combined_deflections(self):
        r = apply_rotation_corrections(
            cme_latitude_deg=0.0,
            initial_axis_angle_deg=0.0,
            hcs_tilt_angle_deg=10.0,
            hcs_distance_deg=0.0,
            ch_proximity=0.5,
            ch_polarity=1,
        )
        total = r.hcs_deflection_deg + r.ch_deflection_deg
        assert r.adjusted_latitude == pytest.approx(0.0 + total)
