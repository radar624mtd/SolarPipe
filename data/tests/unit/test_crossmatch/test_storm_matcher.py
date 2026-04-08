"""Unit tests for Task 5.4 — storm_matcher.py"""
import pytest
from datetime import datetime, timedelta, timezone

from solarpipe_data.crossmatch.storm_matcher import (
    _parse_dt,
    find_storm_response,
    _L1_LAG,
    _DST_STORM_THRESHOLD,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _parse_dt
# ---------------------------------------------------------------------------

def test_parse_dt_z():
    dt = _parse_dt("2016-09-07T06:00Z")
    assert dt is not None
    assert dt.hour == 6


def test_parse_dt_none():
    assert _parse_dt(None) is None


def test_parse_dt_no_tz_gets_utc():
    dt = _parse_dt("2016-09-07T06:00:00")
    assert dt is not None
    assert dt.tzinfo is not None


# ---------------------------------------------------------------------------
# find_storm_response — basic functionality
# ---------------------------------------------------------------------------

def _make_dst_rows(arrival_utc: datetime, offsets_hrs: list, values: list) -> list:
    """Build dst_rows with datetime strings at given hour offsets after arrival."""
    rows = []
    for h, v in zip(offsets_hrs, values):
        dt = (arrival_utc + _L1_LAG + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:00")
        rows.append({"datetime": dt, "dst_nt": v})
    return rows


def _make_kp_rows(arrival_utc: datetime, offsets_hrs: list, values: list) -> list:
    rows = []
    for h, v in zip(offsets_hrs, values):
        dt = (arrival_utc + _L1_LAG + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:00")
        rows.append({"datetime": dt, "kp": v})
    return rows


BASE_ARRIVAL = datetime(2016, 9, 10, 6, 0, tzinfo=timezone.utc)


def test_find_storm_dst_minimum_in_window():
    dst_rows = _make_dst_rows(BASE_ARRIVAL, [2, 12, 24], [-20, -85, -40])
    result = find_storm_response(
        BASE_ARRIVAL.strftime("%Y-%m-%dT%H:%M:%SZ"),
        dst_rows,
        [],
    )
    assert result["dst_min_nt"] == pytest.approx(-85.0)
    assert result["storm_threshold_met"] is True


def test_find_storm_threshold_not_met():
    dst_rows = _make_dst_rows(BASE_ARRIVAL, [6, 18], [-10, -25])
    result = find_storm_response(
        BASE_ARRIVAL.strftime("%Y-%m-%dT%H:%M:%SZ"),
        dst_rows,
        [],
    )
    assert result["dst_min_nt"] == pytest.approx(-25.0)
    assert result["storm_threshold_met"] is False


def test_find_storm_kp_maximum():
    kp_rows = _make_kp_rows(BASE_ARRIVAL, [6, 12, 36], [3.0, 7.0, 5.0])
    result = find_storm_response(
        BASE_ARRIVAL.strftime("%Y-%m-%dT%H:%M:%SZ"),
        [],
        kp_rows,
    )
    assert result["kp_max"] == pytest.approx(7.0)


def test_find_storm_no_dst_rows():
    result = find_storm_response(
        BASE_ARRIVAL.strftime("%Y-%m-%dT%H:%M:%SZ"),
        [],
        [],
    )
    assert result["dst_min_nt"] is None
    assert result["kp_max"] is None
    assert result["storm_threshold_met"] is None


def test_find_storm_no_arrival_time():
    result = find_storm_response(None, [], [])
    assert result["dst_min_nt"] is None
    assert result["storm_threshold_met"] is None


def test_find_storm_out_of_window_rows_ignored():
    # Dst row 60 hrs after arrival — outside the 0–48 hr window
    dst_rows = _make_dst_rows(BASE_ARRIVAL, [60], [-200])
    result = find_storm_response(
        BASE_ARRIVAL.strftime("%Y-%m-%dT%H:%M:%SZ"),
        dst_rows,
        [],
    )
    assert result["dst_min_nt"] is None


def test_find_storm_l1_lag_applied():
    """L1 lag of 45 min shifts the window — a row at exactly +0 min (pre-lag) is excluded."""
    # Row at exactly BASE_ARRIVAL (no lag offset) — should be before window_start
    raw_dt = BASE_ARRIVAL.strftime("%Y-%m-%dT%H:%M:00")
    dst_rows = [{"datetime": raw_dt, "dst_nt": -150}]
    result = find_storm_response(
        BASE_ARRIVAL.strftime("%Y-%m-%dT%H:%M:%SZ"),
        dst_rows,
        [],
    )
    # The row at exactly BASE_ARRIVAL is before window_start (BASE_ARRIVAL + 45 min)
    assert result["dst_min_nt"] is None


def test_find_storm_dst_min_time_set():
    dst_rows = _make_dst_rows(BASE_ARRIVAL, [10], [-60])
    result = find_storm_response(
        BASE_ARRIVAL.strftime("%Y-%m-%dT%H:%M:%SZ"),
        dst_rows,
        [],
    )
    assert result["dst_min_time"] is not None


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------

def test_constants():
    assert _L1_LAG.total_seconds() == pytest.approx(45 * 60)
    assert _DST_STORM_THRESHOLD == pytest.approx(-30.0)
