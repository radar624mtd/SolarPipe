"""Unit tests for Task 5.1 — cme_flare_matcher.py"""
import json
import pytest

from solarpipe_data.crossmatch.cme_flare_matcher import (
    _parse_location,
    _angular_sep_deg,
    _parse_dt,
    _linked_flare_from_json,
    match_cme_to_flare,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _parse_location
# ---------------------------------------------------------------------------

def test_parse_location_north_west():
    lat, lon = _parse_location("N09W11")
    assert lat == pytest.approx(9.0)
    assert lon == pytest.approx(11.0)


def test_parse_location_south_east():
    lat, lon = _parse_location("S15E30")
    assert lat == pytest.approx(-15.0)
    assert lon == pytest.approx(-30.0)


def test_parse_location_none():
    assert _parse_location(None) is None


def test_parse_location_empty():
    assert _parse_location("") is None


def test_parse_location_garbage():
    assert _parse_location("unknown") is None


def test_parse_location_case_insensitive():
    result = _parse_location("n09w11")
    assert result is not None
    assert result[0] == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# _angular_sep_deg
# ---------------------------------------------------------------------------

def test_angular_sep_same_location():
    assert _angular_sep_deg("N09W11", "N09W11") == pytest.approx(0.0)


def test_angular_sep_chebyshev():
    # Δlat=5, Δlon=12 → max=12
    sep = _angular_sep_deg("N10W20", "N15W08")
    assert sep == pytest.approx(12.0)


def test_angular_sep_none_if_parse_fails():
    assert _angular_sep_deg(None, "N09W11") is None
    assert _angular_sep_deg("N09W11", "bad") is None


# ---------------------------------------------------------------------------
# _parse_dt
# ---------------------------------------------------------------------------

def test_parse_dt_z_suffix():
    dt = _parse_dt("2016-09-06T14:18Z")
    assert dt is not None
    assert dt.year == 2016
    assert dt.hour == 14


def test_parse_dt_iso_offset():
    dt = _parse_dt("2016-09-06T14:18:00+00:00")
    assert dt is not None


def test_parse_dt_none():
    assert _parse_dt(None) is None


def test_parse_dt_empty():
    assert _parse_dt("") is None


# ---------------------------------------------------------------------------
# _linked_flare_from_json
# ---------------------------------------------------------------------------

def test_linked_flare_extracts_flr_ids():
    ids = json.dumps(["2016-09-06T14:18Z-FLR-001", "2016-09-06T14:20Z-CME-001"])
    result = _linked_flare_from_json(ids)
    assert result == ["2016-09-06T14:18Z-FLR-001"]


def test_linked_flare_empty_list():
    assert _linked_flare_from_json("[]") == []


def test_linked_flare_none():
    assert _linked_flare_from_json(None) == []


def test_linked_flare_bad_json():
    assert _linked_flare_from_json("not-json") == []


# ---------------------------------------------------------------------------
# match_cme_to_flare — priority 1: direct linked_flare_id
# ---------------------------------------------------------------------------

FLARES = [
    {
        "flare_id": "2016-09-06T12:00Z-FLR-001",
        "begin_time": "2016-09-06T12:00Z",
        "peak_time": "2016-09-06T12:05Z",
        "class_letter": "X",
        "class_magnitude": 9.3,
        "source_location": "S05W11",
        "active_region_num": 12673,
    },
    {
        "flare_id": "2016-09-06T14:00Z-FLR-002",
        "begin_time": "2016-09-06T14:00Z",
        "peak_time": "2016-09-06T14:10Z",
        "class_letter": "M",
        "class_magnitude": 2.1,
        "source_location": "S09W11",
        "active_region_num": 12673,
    },
]


def test_match_direct_linked_flare_id():
    cme = {
        "activity_id": "2016-09-06T14:18Z-CME-001",
        "start_time": "2016-09-06T14:18Z",
        "source_location": "S09W11",
        "linked_flare_id": "2016-09-06T14:00Z-FLR-002",
        "linked_event_ids": None,
    }
    result = match_cme_to_flare(cme, FLARES)
    assert result["flare_match_method"] == "linked"
    assert result["linked_flare_id"] == "2016-09-06T14:00Z-FLR-002"
    assert result["flare_class_letter"] == "M"


def test_match_linked_id_not_in_db():
    cme = {
        "activity_id": "2016-09-06T14:18Z-CME-001",
        "start_time": "2016-09-06T14:18Z",
        "source_location": None,
        "linked_flare_id": "2099-01-01T00:00Z-FLR-999",
        "linked_event_ids": None,
    }
    result = match_cme_to_flare(cme, FLARES)
    assert result["flare_match_method"] == "linked_missing"
    assert result["linked_flare_id"] == "2099-01-01T00:00Z-FLR-999"


# ---------------------------------------------------------------------------
# match_cme_to_flare — priority 1b: linked_event_ids JSON
# ---------------------------------------------------------------------------

def test_match_linked_event_ids_json():
    cme = {
        "activity_id": "2016-09-06T14:18Z-CME-001",
        "start_time": "2016-09-06T14:18Z",
        "source_location": None,
        "linked_flare_id": None,
        "linked_event_ids": json.dumps(["2016-09-06T14:00Z-FLR-002"]),
    }
    result = match_cme_to_flare(cme, FLARES)
    assert result["flare_match_method"] == "linked"
    assert result["linked_flare_id"] == "2016-09-06T14:00Z-FLR-002"


# ---------------------------------------------------------------------------
# match_cme_to_flare — priority 2: temporal
# ---------------------------------------------------------------------------

def test_match_temporal_only():
    cme = {
        "activity_id": "2016-09-06T14:20Z-CME-001",
        "start_time": "2016-09-06T14:20Z",  # 20 min after FLR-002 begin
        "source_location": None,
        "linked_flare_id": None,
        "linked_event_ids": None,
    }
    result = match_cme_to_flare(cme, FLARES)
    assert result["flare_match_method"] in ("temporal", "spatial")
    assert result["linked_flare_id"] == "2016-09-06T14:00Z-FLR-002"


def test_match_temporal_beyond_window():
    cme = {
        "activity_id": "2016-09-06T20:00Z-CME-001",
        "start_time": "2016-09-06T20:00Z",  # 6h after any flare
        "source_location": None,
        "linked_flare_id": None,
        "linked_event_ids": None,
    }
    result = match_cme_to_flare(cme, FLARES)
    assert result["flare_match_method"] == "none"
    assert result["linked_flare_id"] is None


# ---------------------------------------------------------------------------
# match_cme_to_flare — priority 3: spatial refinement
# ---------------------------------------------------------------------------

def test_match_spatial_preferred_over_temporal():
    # Two flares in the temporal window; one is spatially close
    flares_two = [
        {
            "flare_id": "2016-09-06T14:00Z-FLR-CLOSE",
            "begin_time": "2016-09-06T14:05Z",
            "peak_time": "2016-09-06T14:10Z",
            "class_letter": "M",
            "class_magnitude": 1.0,
            "source_location": "S10W12",  # close
            "active_region_num": 12673,
        },
        {
            "flare_id": "2016-09-06T14:00Z-FLR-FAR",
            "begin_time": "2016-09-06T14:00Z",  # slightly closer in time
            "peak_time": "2016-09-06T14:02Z",
            "class_letter": "C",
            "class_magnitude": 5.0,
            "source_location": "N45E45",  # far away
            "active_region_num": None,
        },
    ]
    cme = {
        "activity_id": "2016-09-06T14:15Z-CME-001",
        "start_time": "2016-09-06T14:15Z",
        "source_location": "S09W11",
        "linked_flare_id": None,
        "linked_event_ids": None,
    }
    result = match_cme_to_flare(cme, flares_two)
    assert result["flare_match_method"] == "spatial"
    assert result["linked_flare_id"] == "2016-09-06T14:00Z-FLR-CLOSE"


# ---------------------------------------------------------------------------
# match_cme_to_flare — no flares
# ---------------------------------------------------------------------------

def test_match_empty_flares():
    cme = {
        "activity_id": "2016-09-06T14:18Z-CME-001",
        "start_time": "2016-09-06T14:18Z",
        "source_location": "S09W11",
        "linked_flare_id": None,
        "linked_event_ids": None,
    }
    result = match_cme_to_flare(cme, [])
    assert result["flare_match_method"] == "none"
    assert result["linked_flare_id"] is None


def test_match_cme_no_start_time():
    cme = {
        "activity_id": "bad-cme",
        "start_time": None,
        "source_location": None,
        "linked_flare_id": None,
        "linked_event_ids": None,
    }
    result = match_cme_to_flare(cme, FLARES)
    assert result["flare_match_method"] == "none"
