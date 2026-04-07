"""Port existing data from solar_data.db into staging.db.

Run once after staging.db is initialized. Maps solar_data.db tables to the
staging schema. Safe to re-run — all inserts use INSERT OR REPLACE (upsert).

Usage:
    python scripts/port_solar_data.py [--source PATH] [--target PATH]

Defaults:
    source: C:/Users/radar/SolarPipe/solar_data.db
    target: ./data/staging/staging.db
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Resolve project root so we can import solarpipe_data
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from solarpipe_data.database.schema import init_db

FETCH_TS = datetime.now(timezone.utc).isoformat()
DATA_VERSION = "ported_from_solar_data_db"

# sentinel values → None
_SENTINELS = {9999.9, 9999, -1e31, 99999.9, 999.9}


def _clean(v):
    """Convert sentinel values and empty strings to None."""
    if v is None:
        return None
    if isinstance(v, float) and v in _SENTINELS:
        return None
    if isinstance(v, str) and v.strip() in {"", "---", "--", "9999.9", "-1e31"}:
        return None
    return v


def _row_to_dict(cursor, row):
    return {d[0]: _clean(v) for d, v in zip(cursor.description, row)}


def port_donki_cme(src: sqlite3.Connection, tgt: sqlite3.Connection) -> int:
    src_cur = src.execute(
        "SELECT activity_id, start_time, source_location, active_region_num, "
        "catalog, note, instruments, link, speed_kms, half_angle_deg, "
        "latitude, longitude, analysis_type, is_most_accurate, "
        "n_linked_events, linked_event_ids FROM donki_cme"
    )
    count = 0
    for row in src_cur:
        r = _row_to_dict(src_cur, row)
        tgt.execute(
            """INSERT OR REPLACE INTO cme_events (
                activity_id, start_time, source_location, active_region_num,
                catalog, note, instruments, link, speed_kms, half_angle_deg,
                latitude, longitude, analysis_type, is_most_accurate,
                n_linked_events, linked_event_ids,
                source_catalog, fetch_timestamp, data_version
            ) VALUES (
                :activity_id, :start_time, :source_location, :active_region_num,
                :catalog, :note, :instruments, :link, :speed_kms, :half_angle_deg,
                :latitude, :longitude, :analysis_type, :is_most_accurate,
                :n_linked_events, :linked_event_ids,
                'DONKI', :fetch_timestamp, :data_version
            )""",
            {**r, "fetch_timestamp": FETCH_TS, "data_version": DATA_VERSION},
        )
        count += 1
    return count


def port_cdaw_cme(src: sqlite3.Connection, tgt: sqlite3.Connection) -> int:
    src_cur = src.execute(
        "SELECT date, time_ut, datetime, central_pa_deg, angular_width_deg, "
        "linear_speed_kms, second_order_speed_init, second_order_speed_final, "
        "second_order_speed_20Rs, accel_kms2, mass_grams, kinetic_energy_ergs, "
        "mpa_deg, remarks FROM cdaw_cme"
    )
    count = 0
    for row in src_cur:
        r = _row_to_dict(src_cur, row)
        # Build cdaw_id from date+time_ut
        date_str = (r.get("date") or "").replace("-", "")
        time_str = (r.get("time_ut") or "").replace(":", "")
        cdaw_id = f"{date_str}.{time_str}" if date_str and time_str else None
        if not cdaw_id:
            continue
        tgt.execute(
            """INSERT OR REPLACE INTO cdaw_cme_events (
                cdaw_id, date, time_ut, datetime,
                central_pa_deg, angular_width_deg, linear_speed_kms,
                second_order_speed_init, second_order_speed_final, speed_20rs_kms,
                accel_kms2, mass_grams, kinetic_energy_ergs, mpa_deg, remarks,
                quality_flag, source_catalog, fetch_timestamp, data_version
            ) VALUES (
                :cdaw_id, :date, :time_ut, :datetime,
                :central_pa_deg, :angular_width_deg, :linear_speed_kms,
                :second_order_speed_init, :second_order_speed_final, :speed_20rs_kms,
                :accel_kms2, :mass_grams, :kinetic_energy_ergs, :mpa_deg, :remarks,
                3, 'CDAW', :fetch_timestamp, :data_version
            )""",
            {
                **r,
                "cdaw_id": cdaw_id,
                "speed_20rs_kms": r.get("second_order_speed_20Rs"),
                "fetch_timestamp": FETCH_TS,
                "data_version": DATA_VERSION,
            },
        )
        count += 1
    return count


def port_donki_flare(src: sqlite3.Connection, tgt: sqlite3.Connection) -> int:
    src_cur = src.execute(
        "SELECT flare_id, begin_time, peak_time, end_time, date, class_type, "
        "class_letter, class_magnitude, source_location, active_region_num, "
        "catalog, instruments, note, link, n_linked_events, linked_event_ids "
        "FROM donki_flare"
    )
    count = 0
    for row in src_cur:
        r = _row_to_dict(src_cur, row)
        tgt.execute(
            """INSERT OR REPLACE INTO flares (
                flare_id, begin_time, peak_time, end_time, date,
                class_type, class_letter, class_magnitude,
                source_location, active_region_num,
                catalog, instruments, note, link,
                n_linked_events, linked_event_ids,
                source_catalog, fetch_timestamp, data_version
            ) VALUES (
                :flare_id, :begin_time, :peak_time, :end_time, :date,
                :class_type, :class_letter, :class_magnitude,
                :source_location, :active_region_num,
                :catalog, :instruments, :note, :link,
                :n_linked_events, :linked_event_ids,
                'DONKI', :fetch_timestamp, :data_version
            )""",
            {**r, "fetch_timestamp": FETCH_TS, "data_version": DATA_VERSION},
        )
        count += 1
    return count


def port_omni_hourly(src: sqlite3.Connection, tgt: sqlite3.Connection) -> int:
    src_cur = src.execute(
        "SELECT datetime, date, year, doy, hour, "
        "B_scalar_avg, B_vector_mag, Bx_GSE, By_GSE, Bz_GSE, By_GSM, Bz_GSM, "
        "proton_temp_K, proton_density, flow_speed, flow_pressure, "
        "electric_field, plasma_beta, alfven_mach, "
        "Kp_x10, Dst_nT, AE_nT, ap_index, F10_7_index "
        "FROM omni_hourly"
    )
    count = 0
    for row in src_cur:
        r = _row_to_dict(src_cur, row)
        tgt.execute(
            """INSERT OR REPLACE INTO solar_wind_hourly (
                datetime, date, year, doy, hour,
                b_scalar_avg, b_vector_mag, bx_gse, by_gse, bz_gse, by_gsm, bz_gsm,
                proton_temp_k, proton_density, flow_speed, flow_pressure,
                electric_field, plasma_beta, alfven_mach,
                kp_x10, dst_nt, ae_nt, ap_index, f10_7_index,
                source_catalog, fetch_timestamp, data_version
            ) VALUES (
                :datetime, :date, :year, :doy, :hour,
                :B_scalar_avg, :B_vector_mag, :Bx_GSE, :By_GSE, :Bz_GSE, :By_GSM, :Bz_GSM,
                :proton_temp_K, :proton_density, :flow_speed, :flow_pressure,
                :electric_field, :plasma_beta, :alfven_mach,
                :Kp_x10, :Dst_nT, :AE_nT, :ap_index, :F10_7_index,
                'OMNI', :fetch_timestamp, :data_version
            )""",
            {**r, "fetch_timestamp": FETCH_TS, "data_version": DATA_VERSION},
        )
        count += 1
    return count


def port_symh(src: sqlite3.Connection, tgt: sqlite3.Connection) -> int:
    src_cur = src.execute(
        "SELECT datetime, date, hour, symh_nT, asyh_nT, symd_nT, asyd_nT FROM symh_hourly"
    )
    count = 0
    for row in src_cur:
        r = _row_to_dict(src_cur, row)
        tgt.execute(
            """INSERT OR REPLACE INTO symh_hourly (
                datetime, date, hour, symh_nt, asyh_nt, symd_nt, asyd_nt,
                source_catalog, fetch_timestamp, data_version
            ) VALUES (
                :datetime, :date, :hour, :symh_nT, :asyh_nT, :symd_nT, :asyd_nT,
                'WDC', :fetch_timestamp, :data_version
            )""",
            {**r, "fetch_timestamp": FETCH_TS, "data_version": DATA_VERSION},
        )
        count += 1
    return count


def port_gfz_kp(src: sqlite3.Connection, tgt: sqlite3.Connection) -> int:
    src_cur = src.execute(
        "SELECT date || ' ' || hour_interval AS datetime, "
        "Kp, ap, definitive, daily_Ap, daily_F10_7_obs, daily_F10_7_adj "
        "FROM gfz_kp_ap"
    )
    count = 0
    for row in src_cur:
        r = _row_to_dict(src_cur, row)
        tgt.execute(
            """INSERT OR REPLACE INTO kp_3hr (
                datetime, kp, ap, definitive,
                daily_ap, daily_f10_7_obs, daily_f10_7_adj,
                source_catalog, fetch_timestamp, data_version
            ) VALUES (
                :datetime, :Kp, :ap, :definitive,
                :daily_Ap, :daily_F10_7_obs, :daily_F10_7_adj,
                'GFZ', :fetch_timestamp, :data_version
            )""",
            {**r, "fetch_timestamp": FETCH_TS, "data_version": DATA_VERSION},
        )
        count += 1
    return count


def port_silso(src: sqlite3.Connection, tgt: sqlite3.Connection) -> int:
    src_cur = src.execute(
        "SELECT date, year, month, day, decimal_year, "
        "sunspot_number, std_dev, n_observations, provisional FROM silso_daily_ssn"
    )
    count = 0
    for row in src_cur:
        r = _row_to_dict(src_cur, row)
        tgt.execute(
            """INSERT OR REPLACE INTO silso_daily_ssn (
                date, year, month, day, decimal_year,
                sunspot_number, std_dev, n_observations, provisional,
                source_catalog, fetch_timestamp, data_version
            ) VALUES (
                :date, :year, :month, :day, :decimal_year,
                :sunspot_number, :std_dev, :n_observations, :provisional,
                'SILSO', :fetch_timestamp, :data_version
            )""",
            {**r, "fetch_timestamp": FETCH_TS, "data_version": DATA_VERSION},
        )
        count += 1
    return count


def port_donki_gst(src: sqlite3.Connection, tgt: sqlite3.Connection) -> int:
    src_cur = src.execute(
        "SELECT gst_id, start_time, kp_index_max, all_kp_values, "
        "link, n_linked_events, linked_event_ids FROM donki_gst"
    )
    count = 0
    for row in src_cur:
        r = _row_to_dict(src_cur, row)
        tgt.execute(
            """INSERT OR REPLACE INTO geomagnetic_storms (
                gst_id, start_time, kp_index_max, all_kp_values,
                link, n_linked_events, linked_event_ids,
                source_catalog, fetch_timestamp, data_version
            ) VALUES (
                :gst_id, :start_time, :kp_index_max, :all_kp_values,
                :link, :n_linked_events, :linked_event_ids,
                'DONKI', :fetch_timestamp, :data_version
            )""",
            {**r, "fetch_timestamp": FETCH_TS, "data_version": DATA_VERSION},
        )
        count += 1
    return count


def port_donki_ips(src: sqlite3.Connection, tgt: sqlite3.Connection) -> int:
    src_cur = src.execute(
        "SELECT ips_id, event_time, location, catalog, "
        "instruments, link, n_linked_events, linked_event_ids FROM donki_ips"
    )
    count = 0
    for row in src_cur:
        r = _row_to_dict(src_cur, row)
        tgt.execute(
            """INSERT OR REPLACE INTO interplanetary_shocks (
                ips_id, event_time, location, catalog,
                instruments, link, n_linked_events, linked_event_ids,
                source_catalog, fetch_timestamp, data_version
            ) VALUES (
                :ips_id, :event_time, :location, :catalog,
                :instruments, :link, :n_linked_events, :linked_event_ids,
                'DONKI', :fetch_timestamp, :data_version
            )""",
            {**r, "fetch_timestamp": FETCH_TS, "data_version": DATA_VERSION},
        )
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Port solar_data.db → staging.db")
    parser.add_argument(
        "--source",
        default="/c/Users/radar/SolarPipe/solar_data.db",
        help="Path to source solar_data.db",
    )
    parser.add_argument(
        "--target",
        default="./data/staging/staging.db",
        help="Path to target staging.db (created if absent)",
    )
    args = parser.parse_args()

    src_path = Path(args.source)
    tgt_path = Path(args.target)

    if not src_path.exists():
        print(f"ERROR: source database not found: {src_path}", file=sys.stderr)
        sys.exit(1)

    tgt_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Source : {src_path}  ({src_path.stat().st_size / 1e9:.1f} GB)")
    print(f"Target : {tgt_path}")
    print("Initializing staging schema...")

    # Init ORM schema
    init_db(str(tgt_path))

    src = sqlite3.connect(str(src_path))
    src.row_factory = sqlite3.Row
    tgt = sqlite3.connect(str(tgt_path))
    tgt.execute("PRAGMA journal_mode=WAL")
    tgt.execute("PRAGMA synchronous=NORMAL")

    tables = [
        ("donki_cme → cme_events", port_donki_cme),
        ("cdaw_cme → cdaw_cme_events", port_cdaw_cme),
        ("donki_flare → flares", port_donki_flare),
        ("omni_hourly → solar_wind_hourly", port_omni_hourly),
        ("symh_hourly → symh_hourly", port_symh),
        ("gfz_kp_ap → kp_3hr", port_gfz_kp),
        ("silso_daily_ssn → silso_daily_ssn", port_silso),
        ("donki_gst → geomagnetic_storms", port_donki_gst),
        ("donki_ips → interplanetary_shocks", port_donki_ips),
    ]

    totals = {}
    for label, fn in tables:
        print(f"  Porting {label}...", end=" ", flush=True)
        with tgt:
            n = fn(src, tgt)
        print(f"{n:,} rows")
        totals[label] = n

    src.close()
    tgt.close()

    print("\nDone.")
    for label, n in totals.items():
        print(f"  {label}: {n:,}")


if __name__ == "__main__":
    main()
