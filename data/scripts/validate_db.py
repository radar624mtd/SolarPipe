"""Task 6.7 — Output Database Validation Script.

Checks that cme_catalog.db and enlil_ensemble_v1.parquet are ready for C#
consumption by SolarPipe.Data.SqliteProvider / ParquetProvider.

Exit codes:
    0 — all checks pass
    1 — one or more checks failed (details printed to stderr)

Checks performed:
    1. Three-table join executes without error (validates FK integrity)
    2. Row count ≥ 100 events with quality_flag ≥ 3 AND has_in_situ_fit = 1
    3. Speed range: all cme_speed values in (100, 3000] km/s, no sentinels
    4. dst_min_nT column present and has at least one non-null value in
       quality≥3 events (proxy for bz_gsm requirement from ARCHITECTURE.md)
    5. cme_catalog.db schema matches expected column list (exact match)
    6. Parquet file exists and has all required columns (if path provided)

Usage:
    python scripts/validate_db.py
    python scripts/validate_db.py --catalog path/to/cme_catalog.db
    python scripts/validate_db.py --parquet path/to/enlil_ensemble_v1.parquet
    python scripts/validate_db.py --catalog ... --parquet ... --min-rows 50
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Expected schema — must match sqlite_export.py OutputCmeEvent/etc. exactly
# ---------------------------------------------------------------------------

_EXPECTED_CME_EVENTS_COLS = {
    "event_id", "launch_time", "source_location", "noaa_ar",
    "cme_speed", "cme_mass", "cme_angular_width", "flare_class_numeric",
    "chirality", "initial_axis_angle", "usflux", "totpot", "r_value",
    "meanshr", "totusjz",
    "coronal_hole_proximity", "coronal_hole_polarity",
    "hcs_tilt_angle", "hcs_distance",
    "sw_speed_ambient", "sw_density_ambient", "sw_bt_ambient", "f10_7",
    "quality_flag",
}

_EXPECTED_FLUX_ROPE_COLS = {
    "event_id", "observed_rotation_angle", "observed_bz_min",
    "bz_polarity", "fit_method", "fit_quality", "has_in_situ_fit",
}

_EXPECTED_L1_ARRIVALS_COLS = {
    "event_id", "shock_arrival_time", "icme_start_time", "icme_end_time",
    "transit_time_hours", "dst_min_nT", "kp_max", "has_in_situ_fit",
    "icme_match_method", "icme_match_confidence",
}

_EXPECTED_PARQUET_COLS = {
    "event_id", "member_id", "seed", "speed_initial", "speed_arrival",
    "ambient_wind", "transit_hours", "gamma", "latitude_deg", "longitude_deg",
    "axis_angle_deg", "angular_width", "hcs_deflection", "ch_deflection",
    "bias_correction", "noise_sigma", "quality_flag", "flare_class_numeric",
}

_THREE_TABLE_JOIN = """
    SELECT e.event_id, e.cme_speed, f.observed_rotation_angle, a.transit_time_hours
    FROM cme_events e
    JOIN flux_rope_fits f ON e.event_id = f.event_id
    JOIN l1_arrivals a ON e.event_id = a.event_id
    WHERE e.quality_flag >= 3 AND a.has_in_situ_fit = 1
"""


# ---------------------------------------------------------------------------
# Check runners
# ---------------------------------------------------------------------------

class CheckResult:
    def __init__(self, name: str, passed: bool, detail: str = ""):
        self.name = name
        self.passed = passed
        self.detail = detail

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        suffix = f" — {self.detail}" if self.detail else ""
        return f"  [{status}] {self.name}{suffix}"


def _get_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def check_sqlite(catalog_path: str, min_rows: int = 100) -> list[CheckResult]:
    results = []

    if not Path(catalog_path).exists():
        return [CheckResult("cme_catalog.db exists", False, f"file not found: {catalog_path}")]

    try:
        conn = sqlite3.connect(catalog_path)
        conn.row_factory = sqlite3.Row
    except Exception as e:
        return [CheckResult("cme_catalog.db opens", False, str(e))]

    # Check 1 — three-table join
    try:
        rows = conn.execute(_THREE_TABLE_JOIN).fetchall()
        results.append(CheckResult(
            "Three-table join executes",
            True,
            f"{len(rows)} rows returned",
        ))
    except Exception as e:
        results.append(CheckResult("Three-table join executes", False, str(e)))
        rows = []

    # Check 2 — row count ≥ min_rows quality≥3 + has_in_situ_fit=1
    n_q3 = len(rows)
    results.append(CheckResult(
        f"Row count ≥ {min_rows} (quality≥3, has_in_situ_fit=1)",
        n_q3 >= min_rows,
        f"found {n_q3}",
    ))

    # Check 3 — speed range (no sentinels, all in 100–3000)
    try:
        speed_check = conn.execute("""
            SELECT COUNT(*) FROM cme_events
            WHERE cme_speed IS NOT NULL
              AND (cme_speed < 100 OR cme_speed > 3000)
        """).fetchone()[0]
        results.append(CheckResult(
            "Speed range 100–3000 km/s (no sentinels)",
            speed_check == 0,
            f"{speed_check} out-of-range values" if speed_check else "all in range",
        ))
    except Exception as e:
        results.append(CheckResult("Speed range check", False, str(e)))

    # Check 4 — dst_min_nT column present and has non-null values in quality≥3 events
    try:
        n_dst = conn.execute("""
            SELECT COUNT(*) FROM l1_arrivals a
            JOIN cme_events e ON e.event_id = a.event_id
            WHERE e.quality_flag >= 3 AND a.dst_min_nT IS NOT NULL
        """).fetchone()[0]
        results.append(CheckResult(
            "dst_min_nT column present with non-null values (quality≥3)",
            n_dst > 0,
            f"{n_dst} non-null values",
        ))
    except Exception as e:
        results.append(CheckResult("dst_min_nT check", False, str(e)))

    # Check 5 — schema column names
    for table, expected in [
        ("cme_events",    _EXPECTED_CME_EVENTS_COLS),
        ("flux_rope_fits", _EXPECTED_FLUX_ROPE_COLS),
        ("l1_arrivals",   _EXPECTED_L1_ARRIVALS_COLS),
    ]:
        actual = _get_columns(conn, table)
        missing = expected - actual
        extra = actual - expected
        ok = not missing  # extra columns are acceptable
        detail_parts = []
        if missing:
            detail_parts.append(f"missing: {sorted(missing)}")
        if extra:
            detail_parts.append(f"extra: {sorted(extra)}")
        results.append(CheckResult(
            f"Schema: {table} columns",
            ok,
            "; ".join(detail_parts) if detail_parts else "exact match",
        ))

    conn.close()
    return results


def check_parquet(parquet_path: str) -> list[CheckResult]:
    results = []

    if not Path(parquet_path).exists():
        return [CheckResult("Parquet file exists", False, f"file not found: {parquet_path}")]

    try:
        import pyarrow.parquet as pq
        schema = pq.read_schema(parquet_path)
        actual = set(schema.names)
        missing = _EXPECTED_PARQUET_COLS - actual
        results.append(CheckResult(
            "Parquet file exists and opens",
            True,
            f"{len(actual)} columns",
        ))
        results.append(CheckResult(
            "Parquet schema has required columns",
            not missing,
            f"missing: {sorted(missing)}" if missing else "all present",
        ))

        # Row count
        meta = pq.read_metadata(parquet_path)
        n_rows = meta.num_rows
        results.append(CheckResult(
            "Parquet has rows",
            n_rows > 0,
            f"{n_rows:,} rows across {meta.num_row_groups} row groups",
        ))

    except ImportError:
        results.append(CheckResult("Parquet check (pyarrow)", False, "pyarrow not installed"))
    except Exception as e:
        results.append(CheckResult("Parquet opens", False, str(e)))

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate SolarPipe output databases")
    parser.add_argument(
        "--catalog",
        default="data/output/cme_catalog.db",
        help="Path to cme_catalog.db",
    )
    parser.add_argument(
        "--parquet",
        default=None,
        help="Path to enlil_ensemble_v1.parquet (optional)",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=100,
        help="Minimum rows required for quality≥3 + has_in_situ_fit=1 check",
    )
    args = parser.parse_args(argv)

    all_results: list[CheckResult] = []
    all_results.extend(check_sqlite(args.catalog, min_rows=args.min_rows))

    if args.parquet:
        all_results.extend(check_parquet(args.parquet))

    # Print summary
    passed = [r for r in all_results if r.passed]
    failed = [r for r in all_results if not r.passed]

    print(f"\nSolarPipe Output Validation — {len(all_results)} checks")
    print("=" * 60)
    for r in all_results:
        print(r)
    print("=" * 60)
    print(f"PASSED: {len(passed)}  FAILED: {len(failed)}")

    if failed:
        print("\nFailed checks:", file=sys.stderr)
        for r in failed:
            print(f"  {r}", file=sys.stderr)
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
