"""Unit tests for Task 5.2 — cme_icme_matcher.py"""
import json
import pytest
from datetime import timezone, datetime, timedelta

from solarpipe_data.crossmatch.cme_icme_matcher import (
    _parse_dt,
    _linked_cme_ids_from_ips,
    _linked_ips_ids_from_cme,
    estimate_arrival_time,
    match_cme_to_icme,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _parse_dt
# ---------------------------------------------------------------------------

def test_parse_dt_z():
    dt = _parse_dt("2016-09-06T14:18Z")
    assert dt.tzinfo is not None
    assert dt.year == 2016


def test_parse_dt_none():
    assert _parse_dt(None) is None


def test_parse_dt_empty():
    assert _parse_dt("") is None


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def test_linked_cme_ids_from_ips():
    ids = json.dumps(["2016-09-06T14:18Z-CME-001", "2016-09-06T14:20Z-IPS-001"])
    result = _linked_cme_ids_from_ips(ids)
    assert result == ["2016-09-06T14:18Z-CME-001"]


def test_linked_cme_ids_from_ips_none():
    assert _linked_cme_ids_from_ips(None) == []


def test_linked_ips_ids_from_cme():
    ids = json.dumps(["2016-09-10T00:00Z-IPS-001", "2016-09-10T00:00Z-FLR-002"])
    result = _linked_ips_ids_from_cme(ids)
    assert result == ["2016-09-10T00:00Z-IPS-001"]


def test_linked_ips_ids_from_cme_none():
    assert _linked_ips_ids_from_cme(None) == []


# ---------------------------------------------------------------------------
# estimate_arrival_time
# ---------------------------------------------------------------------------

def test_estimate_arrival_time_basic():
    # At 500 km/s, 1 AU = 1.496e8 km → ~83 hrs raw, drag-corrected slightly less
    arrival = estimate_arrival_time("2016-09-06T00:00Z", 500.0)
    assert arrival is not None
    launch = datetime(2016, 9, 6, 0, 0, tzinfo=timezone.utc)
    transit_hours = (arrival - launch).total_seconds() / 3600.0
    # Should be roughly 70–90 hours
    assert 60 < transit_hours < 100


def test_estimate_arrival_faster_cme_shorter_transit():
    slow = estimate_arrival_time("2016-09-06T00:00Z", 400.0)
    fast = estimate_arrival_time("2016-09-06T00:00Z", 2000.0)
    assert fast is not None and slow is not None
    assert fast < slow


def test_estimate_arrival_no_speed():
    assert estimate_arrival_time("2016-09-06T00:00Z", None) is None


def test_estimate_arrival_zero_speed():
    assert estimate_arrival_time("2016-09-06T00:00Z", 0.0) is None


def test_estimate_arrival_no_launch():
    assert estimate_arrival_time(None, 500.0) is None


def test_estimate_arrival_uses_ambient_speed():
    # With ambient = 600 km/s (fast wind), drag effect changes transit
    t_default = estimate_arrival_time("2016-09-06T00:00Z", 500.0, sw_speed_ambient=None)
    t_fast_wind = estimate_arrival_time("2016-09-06T00:00Z", 500.0, sw_speed_ambient=600.0)
    # Results should differ (fast ambient changes drag term)
    assert t_default != t_fast_wind


# ---------------------------------------------------------------------------
# match_cme_to_icme — priority 1: forward link
# ---------------------------------------------------------------------------

IPS_EVENTS = [
    {
        "ips_id": "2016-09-10T00:00Z-IPS-001",
        "event_time": "2016-09-10T00:00Z",
        "linked_event_ids": json.dumps(["2016-09-06T14:18Z-CME-001"]),
    },
    {
        "ips_id": "2016-09-15T12:00Z-IPS-002",
        "event_time": "2016-09-15T12:00Z",
        "linked_event_ids": None,
    },
]


def test_match_linked_forward():
    cme = {
        "activity_id": "2016-09-06T14:18Z-CME-001",
        "start_time": "2016-09-06T14:18Z",
        "speed_kms": 1200.0,
        "linked_ips_ids": json.dumps(["2016-09-10T00:00Z-IPS-001"]),
    }
    result = match_cme_to_icme(cme, IPS_EVENTS)
    assert result["icme_match_method"] == "linked"
    assert result["linked_ips_id"] == "2016-09-10T00:00Z-IPS-001"
    assert result["icme_match_confidence"] == pytest.approx(1.0)


def test_match_linked_reverse():
    """IPS.linked_event_ids references this CME — reverse link."""
    cme = {
        "activity_id": "2016-09-06T14:18Z-CME-001",
        "start_time": "2016-09-06T14:18Z",
        "speed_kms": 1200.0,
        "linked_ips_ids": None,
    }
    result = match_cme_to_icme(cme, IPS_EVENTS)
    assert result["icme_match_method"] == "linked"
    assert result["linked_ips_id"] == "2016-09-10T00:00Z-IPS-001"


# ---------------------------------------------------------------------------
# match_cme_to_icme — priority 2: transit estimate
# ---------------------------------------------------------------------------

def test_match_transit_estimate():
    # CME at 1500 km/s → transit ~25-40 hrs → arrives ~Sep 8
    cme = {
        "activity_id": "2016-09-06T00:00Z-CME-002",
        "start_time": "2016-09-06T00:00Z",
        "speed_kms": 1500.0,
        "linked_ips_ids": None,
    }
    # IPS event at estimated arrival ± a few hours
    from solarpipe_data.crossmatch.cme_icme_matcher import estimate_arrival_time
    arrival_est = estimate_arrival_time("2016-09-06T00:00Z", 1500.0)
    ips_near = {
        "ips_id": "near-ips",
        "event_time": arrival_est.strftime("%Y-%m-%dT%H:%MZ"),
        "linked_event_ids": None,
    }
    result = match_cme_to_icme(cme, [ips_near])
    assert result["icme_match_method"] == "transit_estimate"
    assert result["icme_match_confidence"] > 0.5
    assert result["transit_time_hours"] is not None


def test_match_no_speed_no_transit():
    cme = {
        "activity_id": "2016-09-06T00:00Z-CME-003",
        "start_time": "2016-09-06T00:00Z",
        "speed_kms": None,
        "linked_ips_ids": None,
    }
    result = match_cme_to_icme(cme, IPS_EVENTS)
    assert result["icme_match_method"] == "none"


def test_match_confidence_halved_multiple_candidates():
    """Multiple IPS candidates in window → confidence halved."""
    from solarpipe_data.crossmatch.cme_icme_matcher import estimate_arrival_time
    arrival_est = estimate_arrival_time("2016-09-06T00:00Z", 800.0)
    ips1 = {
        "ips_id": "ips-a",
        "event_time": arrival_est.strftime("%Y-%m-%dT%H:%MZ"),
        "linked_event_ids": None,
    }
    # Second IPS 6 hrs away — still within ±12 hr window
    ips2 = {
        "ips_id": "ips-b",
        "event_time": (arrival_est + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%MZ"),
        "linked_event_ids": None,
    }
    cme = {
        "activity_id": "2016-09-06T00:00Z-CME-004",
        "start_time": "2016-09-06T00:00Z",
        "speed_kms": 800.0,
        "linked_ips_ids": None,
    }
    result = match_cme_to_icme(cme, [ips1, ips2])
    assert result["icme_match_method"] == "transit_estimate"
    assert result["icme_match_confidence"] <= 0.5


def test_match_empty_ips():
    cme = {
        "activity_id": "x",
        "start_time": "2016-09-06T00:00Z",
        "speed_kms": 800.0,
        "linked_ips_ids": None,
    }
    result = match_cme_to_icme(cme, [])
    assert result["icme_match_method"] == "none"
    assert result["linked_ips_id"] is None


# ---------------------------------------------------------------------------
# transit_time_hours computed correctly
# ---------------------------------------------------------------------------

def test_transit_time_hours_correct():
    launch = "2016-09-06T00:00Z"
    arrival = "2016-09-09T12:00Z"
    ips_id = "2016-09-09T12:00Z-IPS-001"
    ips = {"ips_id": ips_id, "event_time": arrival, "linked_event_ids": None}
    cme = {
        "activity_id": "2016-09-06T00:00Z-CME-001",
        "start_time": launch,
        "speed_kms": 800.0,
        "linked_ips_ids": json.dumps([ips_id]),
    }
    result = match_cme_to_icme(cme, [ips])
    # 3.5 days = 84 hours
    assert result["transit_time_hours"] == pytest.approx(84.0)
