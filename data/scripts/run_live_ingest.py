"""Full live data run — populates all empty staging.db tables.

Run order (dependency-safe):
  1. DONKI CME events (refresh / incremental)
  2. DONKI CME analyses (mostAccurateOnly=true)
  3. DONKI ENLIL simulations
  4. Kyoto Dst hourly
  5. NOAA F10.7 solar flux (monthly)
  6. SHARP keywords + HARP↔NOAA map (JSOC/HMI — slow, use --sharps-limit to cap)
  7. SW ambient context (derived; no network)

Usage:
    python scripts/run_live_ingest.py
    python scripts/run_live_ingest.py --start 2010-01-01 --end 2026-04-07
    python scripts/run_live_ingest.py --sharps-limit 100   # cap JSOC CME queries
    python scripts/run_live_ingest.py --force              # re-fetch even if cached
    python scripts/run_live_ingest.py --skip-sharps        # omit slow JSOC step
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from solarpipe_data.clients.donki import DonkiClient
from solarpipe_data.config import get_settings
from solarpipe_data.database.queries import upsert
from solarpipe_data.database.schema import CmeAnalysis, make_engine
from solarpipe_data.ingestion.ingest_donki_cme import ingest_cme_batch
from solarpipe_data.ingestion.ingest_donki_enlil import ingest_enlil
from solarpipe_data.ingestion.ingest_dst import ingest_dst
from solarpipe_data.ingestion.ingest_f107 import ingest_f107
from solarpipe_data.ingestion.ingest_sharps import ingest_sharps
from solarpipe_data.ingestion.ingest_sw_ambient import compute_ambient_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_live_ingest")

_COL = 42


def _banner(msg: str) -> None:
    logger.info("=" * 60)
    logger.info("  %s", msg)
    logger.info("=" * 60)


def _ok(label: str, val: Any, elapsed: float) -> None:
    if isinstance(val, dict):
        logger.info("  %-*s  %s  (%.1fs)", _COL, label, val, elapsed)
    else:
        logger.info("  %-*s  %6d rows  (%.1fs)", _COL, label, val, elapsed)


def _err(label: str, exc: Exception) -> None:
    logger.error("  %-*s  FAILED: %s", _COL, label, exc)


# ---------------------------------------------------------------------------
# CME analyses — no standalone ingest function; build it here
# ---------------------------------------------------------------------------
_FETCH_TS = datetime.now(timezone.utc).isoformat()


def _parse_analysis(rec: dict[str, Any]) -> dict[str, Any] | None:
    cme_id = rec.get("associatedCMEID") or rec.get("activityID") or ""
    if not cme_id:
        return None
    analysis_id = f"{cme_id}_analysis"
    t21 = rec.get("time21_5") or rec.get("startTime") or None
    return {
        "analysis_id": analysis_id,
        "cme_activity_id": cme_id,
        "time21_5": t21,
        "latitude": _flt(rec.get("latitude")),
        "longitude": _flt(rec.get("longitude")),
        "half_angle_deg": _flt(rec.get("halfAngle")),
        "speed_kms": _flt(rec.get("speed")),
        "is_most_accurate": rec.get("isMostAccurate"),
        "level_of_data": rec.get("levelOfData"),
        "catalog": rec.get("catalog") or None,
        "source_catalog": "DONKI",
        "fetch_timestamp": _FETCH_TS,
        "data_version": None,
    }


def _flt(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def ingest_cme_analyses(engine, records: list[dict]) -> int:
    rows = [_parse_analysis(r) for r in records]
    rows = [r for r in rows if r is not None]
    if not rows:
        return 0
    return upsert(engine, CmeAnalysis, rows)


# ---------------------------------------------------------------------------
# DONKI annual-chunk helpers
# kauai.ccmc.gsfc.nasa.gov times out on requests spanning many years.
# Chunk by year and ingest incrementally to avoid large payloads.
# ---------------------------------------------------------------------------
def _annual_chunks(start: date, end: date) -> list[tuple[str, str]]:
    """Return list of (start_s, end_s) covering start→end in annual slices."""
    chunks = []
    y = start.year
    while True:
        chunk_start = date(y, 1, 1) if y > start.year else start
        chunk_end = date(y, 12, 31) if y < end.year else end
        if chunk_start > end:
            break
        chunks.append((chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        if y >= end.year:
            break
        y += 1
    return chunks


async def _fetch_donki_chunked(
    settings,
    fetch_fn_name: str,
    start: date,
    end: date,
    force: bool,
) -> list[dict]:
    """Fetch a DONKI endpoint in annual chunks, concatenate results."""
    chunks = _annual_chunks(start, end)
    all_records: list[dict] = []
    for chunk_start, chunk_end in chunks:
        logger.info("  DONKI %s chunk %s → %s", fetch_fn_name, chunk_start, chunk_end)
        async with DonkiClient(settings) as client:
            fn = getattr(client, fetch_fn_name)
            records = await fn(chunk_start, chunk_end, force=force)
        all_records.extend(records)
        logger.info("  chunk done: %d records (total so far: %d)", len(records), len(all_records))
    return all_records


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
async def run(
    db_path: str,
    start: date,
    end: date,
    sharps_limit: int | None,
    skip_sharps: bool,
    force: bool,
) -> None:
    settings = get_settings()
    engine = make_engine(db_path)
    results: dict[str, Any] = {}
    t_total = time.monotonic()

    # Step 1 — DONKI CME events (annual chunks)
    _banner("Step 1 — DONKI CME events")
    t0 = time.monotonic()
    try:
        raw_cmes = await _fetch_donki_chunked(settings, "fetch_cme", start, end, force)
        n = ingest_cme_batch(engine, raw_cmes)
        results["cme_events"] = n
        _ok("DONKI CME events", n, time.monotonic() - t0)
    except Exception as exc:
        _err("DONKI CME events", exc)
        results["cme_events"] = -1

    # Step 2 — DONKI CME analyses (annual chunks)
    _banner("Step 2 — DONKI CME analyses")
    t0 = time.monotonic()
    try:
        raw_analyses = await _fetch_donki_chunked(settings, "fetch_cme_analysis", start, end, force)
        n = ingest_cme_analyses(engine, raw_analyses)
        results["cme_analyses"] = n
        _ok("DONKI CME analyses", n, time.monotonic() - t0)
    except Exception as exc:
        _err("DONKI CME analyses", exc)
        results["cme_analyses"] = -1

    # Step 3 — DONKI ENLIL simulations (annual chunks)
    _banner("Step 3 — DONKI ENLIL simulations")
    t0 = time.monotonic()
    try:
        raw_enlil = await _fetch_donki_chunked(settings, "fetch_enlil", start, end, force)
        n = ingest_enlil(engine, raw_enlil)
        results["enlil_simulations"] = n
        _ok("DONKI ENLIL simulations", n, time.monotonic() - t0)
    except Exception as exc:
        _err("DONKI ENLIL simulations", exc)
        results["enlil_simulations"] = -1

    # Step 4 — Kyoto Dst hourly
    _banner("Step 4 — Kyoto Dst hourly")
    t0 = time.monotonic()
    try:
        n = await ingest_dst(db_path, force=force)
        results["dst_hourly"] = n
        _ok("Kyoto Dst hourly", n, time.monotonic() - t0)
    except Exception as exc:
        _err("Kyoto Dst hourly", exc)
        results["dst_hourly"] = -1

    # Step 5 — F10.7 solar flux
    _banner("Step 5 — F10.7 solar radio flux")
    t0 = time.monotonic()
    try:
        n = await ingest_f107(db_path, force=force)
        results["f107_daily"] = n
        _ok("F10.7 daily", n, time.monotonic() - t0)
    except Exception as exc:
        _err("F10.7 daily", exc)
        results["f107_daily"] = -1

    # Step 6 — SHARP keywords (optional, slow)
    if skip_sharps:
        logger.info("  Skipping SHARP ingest (--skip-sharps)")
        results["sharp_keywords"] = "skipped"
    else:
        _banner("Step 6 — SHARP keywords (JSOC/HMI)")
        if sharps_limit:
            logger.info("  JSOC CME query cap: %d", sharps_limit)
        t0 = time.monotonic()
        try:
            sharp_result = await ingest_sharps(db_path, force=force, max_cmes=sharps_limit)
            results["sharp_keywords"] = sharp_result
            _ok("SHARP keywords", sharp_result, time.monotonic() - t0)
        except Exception as exc:
            _err("SHARP keywords", exc)
            results["sharp_keywords"] = -1

    # Step 7 — SW ambient context (derived, no network)
    _banner("Step 7 — SW ambient context")
    t0 = time.monotonic()
    try:
        n = compute_ambient_context(db_path, force=force)
        results["sw_ambient_context"] = n
        _ok("SW ambient context", n, time.monotonic() - t0)
    except Exception as exc:
        _err("SW ambient context", exc)
        results["sw_ambient_context"] = -1

    # Summary
    elapsed = time.monotonic() - t_total
    _banner(f"COMPLETE — {elapsed:.0f}s total")
    for k, v in results.items():
        if v == -1:
            logger.info("  %-*s  FAILED", _COL, k)
        elif v == "skipped":
            logger.info("  %-*s  skipped", _COL, k)
        elif isinstance(v, dict):
            logger.info("  %-*s  %s", _COL, k, v)
        else:
            logger.info("  %-*s  %6d rows", _COL, k, v)

    # Final DB state
    logger.info("")
    logger.info("Final staging.db row counts:")
    con = sqlite3.connect(db_path)
    for (tbl,) in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall():
        if tbl in ("schema_version", "sqlite_sequence"):
            continue
        cnt = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        logger.info("  %-35s  %8d", tbl, cnt)
    con.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Full live ingest — populates all staging.db tables")
    p.add_argument("--db", default="data/staging/staging.db")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    p.add_argument("--sharps-limit", type=int, default=None,
                   help="Max CMEs to query JSOC for (omit = all; use 50 for a quick test)")
    p.add_argument("--skip-sharps", action="store_true",
                   help="Skip JSOC SHARP step entirely (fast run without HMI data)")
    p.add_argument("--force", action="store_true",
                   help="Bypass cache and force re-fetch/re-upsert")
    args = p.parse_args()

    asyncio.run(run(
        db_path=args.db,
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        sharps_limit=args.sharps_limit,
        skip_sharps=args.skip_sharps,
        force=args.force,
    ))


if __name__ == "__main__":
    main()
