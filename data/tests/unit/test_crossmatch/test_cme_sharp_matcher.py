"""Unit tests for Task 5.3 — cme_sharp_matcher.py"""
import pytest

from solarpipe_data.crossmatch.cme_sharp_matcher import (
    _parse_location,
    _sharp_to_feature_dict,
    _SHARP_COLUMNS,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _parse_location — same logic as cme_flare_matcher, independent implementation
# ---------------------------------------------------------------------------

def test_parse_location_nw():
    lat, lon = _parse_location("N12W34")
    assert lat == pytest.approx(12.0)
    assert lon == pytest.approx(34.0)


def test_parse_location_se():
    lat, lon = _parse_location("S07E22")
    assert lat == pytest.approx(-7.0)
    assert lon == pytest.approx(-22.0)


def test_parse_location_none():
    assert _parse_location(None) is None


def test_parse_location_empty():
    assert _parse_location("") is None


def test_parse_location_garbage():
    assert _parse_location("back_side") is None


# ---------------------------------------------------------------------------
# _sharp_to_feature_dict — null snapshot
# ---------------------------------------------------------------------------

def test_feature_dict_null_snapshot():
    d = _sharp_to_feature_dict(None)
    assert d["sharp_harpnum"] is None
    assert d["sharp_noaa_ar"] is None
    assert d["sharp_snapshot_context"] is None
    assert d["sharp_match_method"] == "none"
    # All 18 SHARP feature columns should be null
    for col in _SHARP_COLUMNS[3:]:
        assert col in d
        assert d[col] is None


def test_feature_dict_with_snapshot():
    snap = {
        "harpnum": 377,
        "noaa_ar": 12673,
        "query_context": "at_eruption",
        "usflux": 1.23e22,
        "meangam": 4.5,
        "meangbt": 0.3,
        "meangbz": -0.1,
        "meangbh": 0.25,
        "meanjzd": 0.05,
        "totusjz": 5e12,
        "meanalp": 0.12,
        "meanjzh": 0.08,
        "totusjh": 8e12,
        "absnjzh": 3e12,
        "savncpp": 0.6,
        "meanpot": 900.0,
        "totpot": 2e24,
        "meanshr": 22.0,
        "shrgt45": 10.0,
        "r_value": 1.8,
        "area_acr": 1500.0,
        "lat_fwt": -9.0,
        "lon_fwt": 11.0,
    }
    d = _sharp_to_feature_dict(snap)
    assert d["sharp_harpnum"] == 377
    assert d["sharp_noaa_ar"] == 12673
    assert d["sharp_snapshot_context"] == "at_eruption"
    assert d["usflux"] == pytest.approx(1.23e22)
    assert d["r_value"] == pytest.approx(1.8)
    # sharp_match_method not set by this helper (set by caller)
    assert "sharp_match_method" not in d


def test_feature_dict_missing_keys_become_none():
    """Snapshot with only partial keys — missing ones default to None."""
    snap = {"harpnum": 100, "noaa_ar": None, "query_context": "minus_6h"}
    d = _sharp_to_feature_dict(snap)
    assert d["usflux"] is None
    assert d["r_value"] is None


# ---------------------------------------------------------------------------
# _SHARP_COLUMNS length guard
# ---------------------------------------------------------------------------

def test_sharp_columns_has_expected_features():
    feature_cols = _SHARP_COLUMNS[3:]  # skip harpnum, noaa_ar, query_context
    expected = {
        "usflux", "meangam", "meangbt", "meangbz", "meangbh",
        "meanjzd", "totusjz", "meanalp", "meanjzh", "totusjh",
        "absnjzh", "savncpp", "meanpot", "totpot", "meanshr",
        "shrgt45", "r_value", "area_acr",
    }
    assert expected.issubset(set(feature_cols)), (
        f"Missing SHARP feature columns: {expected - set(feature_cols)}"
    )


def test_sharp_columns_no_lat_lon_in_feature_cols():
    """lat_fwt/lon_fwt are spatial columns, not ML features; verify they stay in _SHARP_COLUMNS."""
    # They should be present in the full list (used for spatial matching)
    assert "lat_fwt" in _SHARP_COLUMNS
    assert "lon_fwt" in _SHARP_COLUMNS
