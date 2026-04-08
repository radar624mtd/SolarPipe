"""Unit tests for Task 5.6 — quality_scorer.py"""
import pytest

from solarpipe_data.crossmatch.quality_scorer import (
    KEY_FEATURES,
    compute_quality_flag,
    lookup_cdaw_quality,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_row(
    icme_method: str = "linked",
    icme_conf: float = 1.0,
) -> dict:
    """Row with all 10 key features present."""
    row = {f: 1.0 for f in KEY_FEATURES}
    row["flare_class_letter"] = "X"   # string key — overwrite float placeholder
    row["icme_arrival_time"] = "2016-09-13T07:00:00"
    row["icme_match_method"] = icme_method
    row["icme_match_confidence"] = icme_conf
    return row


def _row_with_nulls(n_null: int) -> dict:
    """Row with the first n_null key features set to None."""
    row = _full_row()
    for key in KEY_FEATURES[:n_null]:
        row[key] = None
    return row


# ---------------------------------------------------------------------------
# KEY_FEATURES sanity
# ---------------------------------------------------------------------------

def test_key_features_count():
    assert len(KEY_FEATURES) == 10


def test_key_features_no_duplicates():
    assert len(set(KEY_FEATURES)) == len(KEY_FEATURES)


# ---------------------------------------------------------------------------
# compute_quality_flag — CDAW override
# ---------------------------------------------------------------------------

def test_cdaw_very_poor_returns_1():
    row = _full_row()
    assert compute_quality_flag(row, cdaw_quality=1) == 1


def test_cdaw_poor_returns_2():
    row = _full_row()
    assert compute_quality_flag(row, cdaw_quality=2) == 2


def test_cdaw_none_does_not_override():
    row = _full_row()
    result = compute_quality_flag(row, cdaw_quality=None)
    assert result == 5  # full row + linked match → flag 5


# ---------------------------------------------------------------------------
# compute_quality_flag — null gap scoring
# ---------------------------------------------------------------------------

def test_all_present_linked_returns_5():
    assert compute_quality_flag(_full_row(icme_method="linked")) == 5


def test_all_present_transit_high_conf_returns_5():
    row = _full_row(icme_method="transit_estimate", icme_conf=0.8)
    assert compute_quality_flag(row) == 5


def test_all_present_transit_low_conf_returns_4():
    row = _full_row(icme_method="transit_estimate", icme_conf=0.5)
    assert compute_quality_flag(row) == 4


def test_all_present_no_icme_returns_4():
    row = _full_row(icme_method="none", icme_conf=0.0)
    assert compute_quality_flag(row) == 4


def test_one_null_returns_4():
    assert compute_quality_flag(_row_with_nulls(1)) == 4


def test_two_nulls_returns_4():
    assert compute_quality_flag(_row_with_nulls(2)) == 4


def test_three_nulls_returns_3():
    assert compute_quality_flag(_row_with_nulls(3)) == 3


def test_four_nulls_returns_3():
    assert compute_quality_flag(_row_with_nulls(4)) == 3


def test_five_nulls_returns_2():
    assert compute_quality_flag(_row_with_nulls(5)) == 2


def test_ten_nulls_returns_2():
    assert compute_quality_flag(_row_with_nulls(10)) == 2


def test_cdaw_override_beats_null_gap():
    """cdaw_quality=1 overrides even a full-feature row."""
    row = _full_row()
    assert compute_quality_flag(row, cdaw_quality=1) == 1


# ---------------------------------------------------------------------------
# lookup_cdaw_quality
# ---------------------------------------------------------------------------

def test_lookup_no_index():
    assert lookup_cdaw_quality("2016-09-10T08:00:00", {}) is None


def test_lookup_none_launch_time():
    assert lookup_cdaw_quality(None, {"2016-09-10 08:00": 1}) is None


def test_lookup_exact_match():
    index = {"2016-09-10 08:00": 1}
    result = lookup_cdaw_quality("2016-09-10T08:00:00", index)
    assert result == 1


def test_lookup_within_window():
    # CDAW event at 08:15, CME at 08:00 → 15 min gap, within 30 min default
    index = {"2016-09-10 08:15": 2}
    result = lookup_cdaw_quality("2016-09-10T08:00:00", index)
    assert result == 2


def test_lookup_outside_window():
    # CDAW event at 09:00, CME at 08:00 → 60 min gap, outside 30 min window
    index = {"2016-09-10 09:00": 1}
    result = lookup_cdaw_quality("2016-09-10T08:00:00", index)
    assert result is None


def test_lookup_picks_closest():
    index = {
        "2016-09-10 07:45": 2,  # 15 min before
        "2016-09-10 08:20": 1,  # 20 min after
    }
    result = lookup_cdaw_quality("2016-09-10T08:00:00", index)
    assert result == 2  # closer event wins


def test_lookup_z_suffix_handled():
    index = {"2016-09-10 08:00": 1}
    result = lookup_cdaw_quality("2016-09-10T08:00:00Z", index)
    assert result == 1
