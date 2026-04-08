"""Unit tests for clients/jsoc.py.

Tests the parse/transform logic without any live JSOC network calls.
RULE-090: asyncio_mode=auto
RULE-092: @pytest.mark.unit
"""
from __future__ import annotations

import pytest
import pandas as pd


@pytest.mark.unit
class TestJsocTimeConversion:
    def test_to_jsoc_time(self):
        from solarpipe_data.clients.jsoc import _to_jsoc_time
        from datetime import datetime, timezone
        dt = datetime(2016, 9, 6, 14, 18, 0, tzinfo=timezone.utc)
        result = _to_jsoc_time(dt)
        assert result == "2016.09.06_14:18:00_TAI"

    def test_normalise_t_rec_standard(self):
        from solarpipe_data.clients.jsoc import _normalise_t_rec
        assert _normalise_t_rec("2016.09.06_14:12:00_TAI") == "2016-09-06 14:12"

    def test_normalise_t_rec_empty(self):
        from solarpipe_data.clients.jsoc import _normalise_t_rec
        assert _normalise_t_rec("") is None
        assert _normalise_t_rec(None) is None


@pytest.mark.unit
class TestJsocFloatHelpers:
    def test_float_or_none_nan(self):
        from solarpipe_data.clients.jsoc import _float_or_none
        import math
        assert _float_or_none(float("nan")) is None

    def test_float_or_none_sentinel_large(self):
        from solarpipe_data.clients.jsoc import _float_or_none
        assert _float_or_none(1e31) is None
        assert _float_or_none(-1e31) is None

    def test_float_or_none_valid(self):
        from solarpipe_data.clients.jsoc import _float_or_none
        assert _float_or_none("3.14") == pytest.approx(3.14)
        assert _float_or_none(0.0) == pytest.approx(0.0)

    def test_float_or_none_none(self):
        from solarpipe_data.clients.jsoc import _float_or_none
        assert _float_or_none(None) is None

    def test_int_or_none(self):
        from solarpipe_data.clients.jsoc import _int_or_none
        assert _int_or_none("12345") == 12345
        assert _int_or_none(0) == 0
        assert _int_or_none(None) is None


@pytest.mark.unit
class TestParseSharpDf:
    def _make_df(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_basic_row_parsed(self):
        from solarpipe_data.clients.jsoc import _parse_sharp_df
        df = self._make_df([{
            "HARPNUM": 4000, "NOAA_AR": 12673, "T_REC": "2016.09.06_14:12:00_TAI",
            "LON_FWT": 15.0, "LAT_FWT": -8.0,
            "USFLUX": 1.2e22, "MEANGAM": 5.3, "MEANGBT": 100.0,
            "MEANGBZ": -50.0, "MEANGBH": 80.0,
            "MEANJZD": 0.01, "TOTUSJZ": 1.5e12, "MEANALP": 0.05,
            "MEANJZH": 0.02, "TOTUSJH": 3.0e12, "ABSNJZH": 2.5e12,
            "SAVNCPP": 120.0, "MEANPOT": 1000.0, "TOTPOT": 5.0e24,
            "MEANSHR": 20.0, "SHRGT45": 0.15, "R_VALUE": 3.5,
            "AREA_ACR": 800.0, "QUALITY": 0,
        }])
        records = _parse_sharp_df(df, "at_eruption")
        assert len(records) == 1
        r = records[0]
        assert r["harpnum"] == 4000
        assert r["noaa_ar"] == 12673
        assert r["t_rec"] == "2016-09-06 14:12"
        assert r["usflux"] == pytest.approx(1.2e22)
        assert r["query_context"] == "at_eruption"
        assert r["source_catalog"] == "JSOC"

    def test_lon_fwt_filter_drops_limb(self):
        from solarpipe_data.clients.jsoc import _parse_sharp_df
        # LON_FWT > 60° should be dropped (RULE-060)
        df = self._make_df([
            {"HARPNUM": 1, "NOAA_AR": 100, "T_REC": "2016.09.06_14:12:00_TAI",
             "LON_FWT": 65.0, "LAT_FWT": 10.0},
            {"HARPNUM": 2, "NOAA_AR": 200, "T_REC": "2016.09.06_14:12:00_TAI",
             "LON_FWT": 30.0, "LAT_FWT": 5.0},
        ])
        records = _parse_sharp_df(df, "at_eruption")
        assert len(records) == 1
        assert records[0]["harpnum"] == 2

    def test_lon_fwt_negative_limb_dropped(self):
        from solarpipe_data.clients.jsoc import _parse_sharp_df
        df = self._make_df([{
            "HARPNUM": 1, "NOAA_AR": 100,
            "T_REC": "2016.09.06_14:12:00_TAI",
            "LON_FWT": -70.0, "LAT_FWT": 5.0,
        }])
        records = _parse_sharp_df(df, "at_eruption")
        assert len(records) == 0

    def test_noaa_ar_zero_becomes_none(self):
        from solarpipe_data.clients.jsoc import _parse_sharp_df
        # RULE-063: NOAA_AR = 0 → None
        df = self._make_df([{
            "HARPNUM": 5000, "NOAA_AR": 0,
            "T_REC": "2016.09.06_14:12:00_TAI",
            "LON_FWT": 10.0, "LAT_FWT": 5.0,
        }])
        records = _parse_sharp_df(df, "at_eruption")
        assert len(records) == 1
        assert records[0]["noaa_ar"] is None

    def test_date_obs_fallback_when_t_rec_missing(self):
        from solarpipe_data.clients.jsoc import _parse_sharp_df
        # RULE-064: DATE__OBS fallback
        df = self._make_df([{
            "HARPNUM": 6000, "NOAA_AR": 12345,
            "T_REC": None, "DATE__OBS": "2016.09.06_14:00:00_TAI",
            "LON_FWT": 5.0, "LAT_FWT": 3.0,
        }])
        records = _parse_sharp_df(df, "minus_6h")
        assert records[0]["t_rec"] == "2016-09-06 14:00"

    def test_empty_df_returns_empty(self):
        from solarpipe_data.clients.jsoc import _parse_sharp_df
        records = _parse_sharp_df(pd.DataFrame(), "at_eruption")
        assert records == []


@pytest.mark.unit
class TestEarthDirectedProxy:
    def test_low_lat_lon_is_earth_directed(self):
        from solarpipe_data.ingestion.ingest_sharps import _is_earth_directed
        assert _is_earth_directed(10.0, 20.0) is True
        assert _is_earth_directed(-30.0, -40.0) is True

    def test_high_lon_not_earth_directed(self):
        from solarpipe_data.ingestion.ingest_sharps import _is_earth_directed
        assert _is_earth_directed(10.0, 50.0) is False   # >45°
        assert _is_earth_directed(10.0, -50.0) is False

    def test_high_lat_not_earth_directed(self):
        from solarpipe_data.ingestion.ingest_sharps import _is_earth_directed
        assert _is_earth_directed(50.0, 10.0) is False

    def test_none_values_return_false(self):
        from solarpipe_data.ingestion.ingest_sharps import _is_earth_directed
        assert _is_earth_directed(None, 10.0) is False
        assert _is_earth_directed(10.0, None) is False
        assert _is_earth_directed(None, None) is False

    def test_hmi_start_filter(self):
        from solarpipe_data.ingestion.ingest_sharps import _is_post_hmi
        from datetime import datetime, timezone
        assert _is_post_hmi(datetime(2010, 5, 2, tzinfo=timezone.utc)) is True
        assert _is_post_hmi(datetime(2010, 4, 30, tzinfo=timezone.utc)) is False
        assert _is_post_hmi(None) is False


@pytest.mark.unit
class TestOptimalSnapshotSelection:
    def test_context_preference_order(self):
        from solarpipe_data.ingestion.select_sharp_features import _CONTEXT_PREFERENCE
        assert _CONTEXT_PREFERENCE[0] == "at_eruption"
        assert _CONTEXT_PREFERENCE[1] == "minus_6h"
        assert _CONTEXT_PREFERENCE[2] == "minus_12h"
        # plus_6h is post-eruption — not in preference list
        assert "plus_6h" not in _CONTEXT_PREFERENCE

    def test_get_best_snapshot_returns_none_no_data(self):
        from solarpipe_data.ingestion.select_sharp_features import get_best_sharp_snapshot
        from solarpipe_data.database.schema import init_db
        engine = init_db(":memory:")
        result = get_best_sharp_snapshot(engine, noaa_ar=99999, harpnum=None, t_eruption="")
        assert result is None
        engine.dispose()

    def test_get_best_snapshot_no_ar_no_harp_returns_none(self):
        from solarpipe_data.ingestion.select_sharp_features import get_best_sharp_snapshot
        from solarpipe_data.database.schema import init_db
        engine = init_db(":memory:")
        result = get_best_sharp_snapshot(engine, noaa_ar=None, harpnum=None, t_eruption="")
        assert result is None
        engine.dispose()
