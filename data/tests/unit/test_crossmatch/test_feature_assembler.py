"""Unit tests for Task 5.5 — feature_assembler.py"""
import pytest

from solarpipe_data.crossmatch.feature_assembler import (
    _date_from_ts,
    _lookup_f107,
    _lookup_ssn,
    _best_cme_speed,
    _best_analysis,
    assemble_feature_vector,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _date_from_ts
# ---------------------------------------------------------------------------

def test_date_from_ts_iso_z():
    assert _date_from_ts("2016-09-06T14:18Z") == "2016-09-06"


def test_date_from_ts_iso_no_tz():
    assert _date_from_ts("2016-09-06T14:18:00") == "2016-09-06"


def test_date_from_ts_none():
    assert _date_from_ts(None) is None


def test_date_from_ts_empty():
    assert _date_from_ts("") is None


# ---------------------------------------------------------------------------
# _lookup_f107 / _lookup_ssn
# ---------------------------------------------------------------------------

def test_lookup_f107_found():
    index = {"2016-09-06": 124.5, "2017-01-01": 80.0}
    assert _lookup_f107("2016-09-06", index) == 124.5


def test_lookup_f107_not_found():
    assert _lookup_f107("2020-01-01", {"2016-09-06": 124.5}) is None


def test_lookup_f107_none_date():
    assert _lookup_f107(None, {"2016-09-06": 124.5}) is None


def test_lookup_ssn_found():
    index = {"2016-09-06": 55.0}
    assert _lookup_ssn("2016-09-06", index) == 55.0


def test_lookup_ssn_none_date():
    assert _lookup_ssn(None, {}) is None


# ---------------------------------------------------------------------------
# _best_cme_speed / _best_analysis
# ---------------------------------------------------------------------------

def _make_analysis(activity_id, level, speed):
    return {
        "cme_activity_id": activity_id,
        "level_of_data": level,
        "speed_kms": speed,
        "half_angle_deg": 30.0,
        "latitude": -10.0,
        "longitude": 15.0,
    }


def test_best_speed_prefers_level2():
    cme = {"activity_id": "A1", "speed_kms": 100.0}
    analyses = [
        _make_analysis("A1", 0, 200.0),
        _make_analysis("A1", 2, 500.0),
        _make_analysis("A1", 1, 300.0),
    ]
    assert _best_cme_speed(cme, analyses) == 500.0


def test_best_speed_falls_back_to_level1():
    cme = {"activity_id": "A1", "speed_kms": 100.0}
    analyses = [_make_analysis("A1", 1, 350.0)]
    assert _best_cme_speed(cme, analyses) == 350.0


def test_best_speed_falls_back_to_cme_events():
    cme = {"activity_id": "A1", "speed_kms": 100.0}
    assert _best_cme_speed(cme, []) == 100.0


def test_best_speed_skips_other_cme():
    cme = {"activity_id": "A1", "speed_kms": 100.0}
    analyses = [_make_analysis("A2", 2, 900.0)]
    # A2 analysis must not affect A1
    assert _best_cme_speed(cme, analyses) == 100.0


def test_best_analysis_returns_none_when_empty():
    cme = {"activity_id": "A1"}
    assert _best_analysis(cme, []) is None


def test_best_analysis_prefers_level2():
    cme = {"activity_id": "A1"}
    analyses = [_make_analysis("A1", 1, 300.0), _make_analysis("A1", 2, 500.0)]
    result = _best_analysis(cme, analyses)
    assert result["level_of_data"] == 2


# ---------------------------------------------------------------------------
# assemble_feature_vector — full integration of all matchers
# ---------------------------------------------------------------------------

def _make_cme():
    return {
        "activity_id": "2016-09-10T08:00:00-CME-001",
        "start_time": "2016-09-10T08:00:00",
        "speed_kms": 750.0,
        "half_angle_deg": 25.0,
        "latitude": -10.0,
        "longitude": 15.0,
        "cme_mass_grams": None,
        "cme_angular_width_deg": None,
    }


def _make_flare_match():
    return {
        "linked_flare_id": "2016-09-10T08:06:00-FLR-001",
        "flare_class_letter": "X",
        "flare_class_numeric": 2.4,
        "flare_peak_time": "2016-09-10T08:10:00",
        "flare_active_region": 12781,
        "flare_match_method": "linked",
    }


def _make_icme_match():
    return {
        "linked_ips_id": "2016-09-13T07:00:00-IPS-001",
        "icme_arrival_time": "2016-09-13T07:00:00",
        "transit_time_hours": 71.0,
        "icme_match_method": "linked",
        "icme_match_confidence": 1.0,
    }


def _make_sharp_match():
    return {
        "sharp_harpnum": 4994,
        "sharp_noaa_ar": 12781,
        "sharp_snapshot_context": "at_eruption",
        "usflux": 1.23e22,
        "meangam": 12.5,
        "meangbt": 400.0,
        "meangbz": 200.0,
        "meangbh": 350.0,
        "meanjzd": 5.0,
        "totusjz": 1e13,
        "meanalp": 0.05,
        "meanjzh": 2e10,
        "totusjh": 3e15,
        "absnjzh": 4e15,
        "savncpp": 30.0,
        "meanpot": 5000.0,
        "totpot": 1e32,
        "meanshr": 45.0,
        "shrgt45": 40.0,
        "r_value": 3.5,
        "area_acr": 2500.0,
        "sharp_match_method": "noaa_ar",
    }


def _make_storm_match():
    return {
        "dst_min_nt": -70.0,
        "dst_min_time": "2016-09-13T23:00:00",
        "kp_max": 7.0,
        "storm_threshold_met": True,
    }


def _make_ambient():
    return {
        "sw_speed_ambient": 380.0,
        "sw_density_ambient": 6.5,
        "sw_bt_ambient": 8.0,
        "sw_bz_ambient": -3.5,
    }


def test_assemble_identity_fields():
    cme = _make_cme()
    row = assemble_feature_vector(
        cme=cme,
        analyses=[],
        flare_match=_make_flare_match(),
        icme_match=_make_icme_match(),
        sharp_match=_make_sharp_match(),
        storm_match=_make_storm_match(),
        ambient=_make_ambient(),
        f107_index={"2016-09-10": 110.0},
        ssn_index={"2016-09-10": 80.0},
    )
    assert row["activity_id"] == cme["activity_id"]
    assert row["launch_time"] == cme["start_time"]


def test_assemble_kinematic_fields():
    cme = _make_cme()
    row = assemble_feature_vector(
        cme=cme, analyses=[], flare_match={}, icme_match={}, sharp_match={},
        storm_match={}, ambient=None, f107_index={}, ssn_index={},
    )
    assert row["cme_speed_kms"] == 750.0
    assert row["cme_latitude"] == -10.0
    assert row["cme_longitude"] == 15.0


def test_assemble_uses_analysis_speed_over_cme_events():
    cme = _make_cme()
    analyses = [_make_analysis(cme["activity_id"], 2, 900.0)]
    row = assemble_feature_vector(
        cme=cme, analyses=analyses, flare_match={}, icme_match={}, sharp_match={},
        storm_match={}, ambient=None, f107_index={}, ssn_index={},
    )
    assert row["cme_speed_kms"] == 900.0


def test_assemble_flare_fields():
    row = assemble_feature_vector(
        cme=_make_cme(), analyses=[], flare_match=_make_flare_match(),
        icme_match={}, sharp_match={}, storm_match={},
        ambient=None, f107_index={}, ssn_index={},
    )
    assert row["flare_class_letter"] == "X"
    assert row["flare_class_numeric"] == 2.4
    assert row["flare_match_method"] == "linked"


def test_assemble_sharp_fields():
    row = assemble_feature_vector(
        cme=_make_cme(), analyses=[], flare_match={},
        icme_match={}, sharp_match=_make_sharp_match(), storm_match={},
        ambient=None, f107_index={}, ssn_index={},
    )
    assert row["sharp_harpnum"] == 4994
    assert row["sharp_noaa_ar"] == 12781
    assert row["usflux"] == pytest.approx(1.23e22)
    assert row["sharp_match_method"] == "noaa_ar"


def test_assemble_ambient_fields():
    row = assemble_feature_vector(
        cme=_make_cme(), analyses=[], flare_match={},
        icme_match={}, sharp_match={}, storm_match={},
        ambient=_make_ambient(), f107_index={}, ssn_index={},
    )
    assert row["sw_speed_ambient"] == 380.0
    assert row["sw_bz_ambient"] == -3.5


def test_assemble_ambient_none():
    row = assemble_feature_vector(
        cme=_make_cme(), analyses=[], flare_match={},
        icme_match={}, sharp_match={}, storm_match={},
        ambient=None, f107_index={}, ssn_index={},
    )
    assert row["sw_speed_ambient"] is None


def test_assemble_icme_fields():
    row = assemble_feature_vector(
        cme=_make_cme(), analyses=[], flare_match={},
        icme_match=_make_icme_match(), sharp_match={}, storm_match={},
        ambient=None, f107_index={}, ssn_index={},
    )
    assert row["transit_time_hours"] == 71.0
    assert row["icme_match_confidence"] == 1.0


def test_assemble_storm_fields():
    row = assemble_feature_vector(
        cme=_make_cme(), analyses=[], flare_match={},
        icme_match={}, sharp_match={}, storm_match=_make_storm_match(),
        ambient=None, f107_index={}, ssn_index={},
    )
    assert row["dst_min_nt"] == -70.0
    assert row["storm_threshold_met"] is True
    assert row["kp_max"] == 7.0


def test_assemble_activity_context():
    row = assemble_feature_vector(
        cme=_make_cme(), analyses=[], flare_match={},
        icme_match={}, sharp_match={}, storm_match={},
        ambient=None,
        f107_index={"2016-09-10": 110.0},
        ssn_index={"2016-09-10": 80.0},
    )
    assert row["f10_7"] == 110.0
    assert row["sunspot_number"] == 80.0


def test_assemble_deferred_fields_are_null():
    row = assemble_feature_vector(
        cme=_make_cme(), analyses=[], flare_match={},
        icme_match={}, sharp_match={}, storm_match={},
        ambient=None, f107_index={}, ssn_index={},
    )
    assert row["dimming_area"] is None
    assert row["dimming_asymmetry"] is None
    assert row["hcs_tilt_angle"] is None
    assert row["hcs_distance"] is None


def test_assemble_default_quality_flag():
    row = assemble_feature_vector(
        cme=_make_cme(), analyses=[], flare_match={},
        icme_match={}, sharp_match={}, storm_match={},
        ambient=None, f107_index={}, ssn_index={},
    )
    assert row["quality_flag"] == 3


def test_assemble_provenance():
    row = assemble_feature_vector(
        cme=_make_cme(), analyses=[], flare_match={},
        icme_match={}, sharp_match={}, storm_match={},
        ambient=None, f107_index={}, ssn_index={},
    )
    assert row["source_catalog"] == "crossmatch"


def test_assemble_no_match_defaults():
    """All matchers return empty dicts — fields should be None / 'none'."""
    row = assemble_feature_vector(
        cme=_make_cme(), analyses=[], flare_match={},
        icme_match={}, sharp_match={}, storm_match={},
        ambient=None, f107_index={}, ssn_index={},
    )
    assert row["flare_match_method"] == "none"
    assert row["icme_match_method"] == "none"
    assert row["sharp_match_method"] == "none"
    assert row["linked_flare_id"] is None
    assert row["linked_ips_id"] is None
