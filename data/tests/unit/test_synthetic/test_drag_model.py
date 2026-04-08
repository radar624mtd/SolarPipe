"""Unit tests for synthetic/drag_model.py (Task 6.1).

RULE-090: asyncio_mode=auto
RULE-092: @pytest.mark.unit
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from solarpipe_data.synthetic.drag_model import (
    DragCalibration,
    DragResult,
    _GAMMA_DEFAULT,
    _L1_KM,
    _R0_KM,
    calibrate_gamma,
    propagate,
)


@pytest.mark.unit
class TestPropagate:
    def test_fast_cme_reaches_l1(self):
        r = propagate(v0_kms=1500.0, w_kms=400.0)
        assert r.success is True
        assert 20.0 < r.transit_time_hours < 120.0

    def test_slow_cme_reaches_l1(self):
        r = propagate(v0_kms=300.0, w_kms=450.0)
        assert r.success is True
        assert r.transit_time_hours > 60.0

    def test_arrival_speed_nonnegative(self):
        r = propagate(v0_kms=200.0, w_kms=450.0)
        assert r.arrival_speed_kms >= 0.0

    def test_fast_cme_arrives_before_slow(self):
        r_fast = propagate(v0_kms=1500.0, w_kms=400.0)
        r_slow = propagate(v0_kms=400.0, w_kms=400.0)
        assert r_fast.transit_time_hours < r_slow.transit_time_hours

    def test_higher_wind_decelerates_more(self):
        r_fast_wind = propagate(v0_kms=1000.0, w_kms=600.0)
        r_slow_wind = propagate(v0_kms=1000.0, w_kms=400.0)
        # CME above wind speed: arrival speed converges toward ambient wind.
        # Higher ambient wind → higher arrival speed, shorter transit.
        assert r_fast_wind.arrival_speed_kms > r_slow_wind.arrival_speed_kms
        assert r_fast_wind.transit_time_hours < r_slow_wind.transit_time_hours

    def test_cme_at_wind_speed_nearly_constant_velocity(self):
        """CME at wind speed: drag ≈ 0, transit should be ≈ distance/speed."""
        v = 450.0
        r = propagate(v0_kms=v, w_kms=v)
        expected = (_L1_KM - _R0_KM) / (v * 3600.0)  # hours
        assert abs(r.transit_time_hours - expected) < 5.0

    def test_returns_drag_result_type(self):
        r = propagate(v0_kms=800.0)
        assert isinstance(r, DragResult)
        assert isinstance(r.transit_time_hours, float)
        assert isinstance(r.success, bool)

    def test_gamma_stored_in_result(self):
        r = propagate(v0_kms=800.0, gamma=1e-4)
        assert r.gamma == pytest.approx(1e-4)

    def test_ambient_wind_stored(self):
        r = propagate(v0_kms=800.0, w_kms=500.0)
        assert r.ambient_wind_kms == pytest.approx(500.0)

    def test_t_max_failure_returns_partial_result(self):
        """Extremely slow CME with huge drag — may not reach L1 in 1 hour."""
        r = propagate(v0_kms=101.0, w_kms=100.0, gamma=1.0, t_max_hours=0.01)
        # Should not raise — returns partial result with success=False
        assert isinstance(r, DragResult)


@pytest.mark.unit
class TestCalibrateGamma:
    def test_calibrate_returns_default_on_empty(self):
        cal = calibrate_gamma([], [], [])
        assert cal.gamma == pytest.approx(_GAMMA_DEFAULT)
        assert cal.n_events == 0

    def test_calibrate_returns_calibration_type(self):
        speeds = [800.0, 600.0, 1200.0]
        winds = [450.0, 450.0, 400.0]
        transits = [40.0, 55.0, 28.0]
        cal = calibrate_gamma(speeds, winds, transits)
        assert isinstance(cal, DragCalibration)
        assert cal.n_events == 3
        assert cal.gamma > 0

    def test_calibrate_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            calibrate_gamma([800.0], [450.0], [40.0, 50.0])

    def test_calibrate_rmse_is_finite(self):
        speeds = [700.0, 900.0, 500.0, 1100.0]
        winds = [400.0, 450.0, 450.0, 380.0]
        transits = [48.0, 38.0, 60.0, 32.0]
        cal = calibrate_gamma(speeds, winds, transits)
        assert math.isfinite(cal.residual_rmse_hours)
        assert cal.residual_rmse_hours >= 0.0

    def test_custom_gamma_grid(self):
        speeds = [800.0, 600.0]
        winds = [450.0, 450.0]
        transits = [40.0, 55.0]
        grid = np.array([1e-5, 2e-5, 5e-5])
        cal = calibrate_gamma(speeds, winds, transits, gamma_grid=grid)
        assert cal.gamma in grid
