"""Click CLI harness for SolarPipe data acquisition pipeline.

Commands:
  fetch   — fetch raw data from source API to file cache
  ingest  — ingest raw cache into staging.db
  crossmatch — run CME↔flare and CME↔ICME matchers
  build   — full pipeline (fetch + ingest + crossmatch)
  validate — run validate_db.py checks
  status  — print record counts per table with temporal coverage

Rules:
- RULE-013: Click has no native async — bridge via run_async decorator
- RULE-093: CliRunner with catch_exceptions=False during development
"""
from __future__ import annotations

import asyncio
import functools
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import click

from solarpipe_data.config import get_settings
from solarpipe_data.database.queries import max_timestamp, row_count
from solarpipe_data.database.schema import (
    Base,
    CdawCmeEvent,
    CmeAnalysis,
    CmeEvent,
    DstHourly,
    EnlilSimulation,
    F107Daily,
    Flare,
    GeomagneticStorm,
    InterplanetaryShock,
    Kp3hr,
    SharpKeyword,
    SilsoDailySSN,
    SolarWindHourly,
    SymhHourly,
    init_db,
)


def run_async(fn: Callable) -> Callable:
    """Decorator: run an async Click command synchronously via asyncio.run()."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(fn(*args, **kwargs))
    return wrapper


@click.group()
def cli() -> None:
    """SolarPipe data acquisition pipeline."""


@cli.command()
@click.argument("source", type=click.Choice([
    "donki-cme", "donki-flare", "donki-gst", "donki-ips", "donki-enlil",
    "cdaw", "swpc", "kyoto", "gfz", "jsoc",
], case_sensitive=False))
@click.option("--start", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", required=True, help="End date YYYY-MM-DD")
@click.option("--force", is_flag=True, default=False, help="Bypass file cache")
@run_async
async def fetch(source: str, start: str, end: str, force: bool) -> None:
    """Fetch raw data from SOURCE API to file cache."""
    click.echo(f"Fetching {source} {start} → {end} (force={force})")
    if source == "donki-cme":
        from solarpipe_data.clients.donki import DonkiClient
        settings = get_settings()
        async with DonkiClient(settings) as client:
            result = await client.fetch_cme(start, end, force=force)
        click.echo(f"  {len(result)} CME records fetched")
    else:
        raise click.ClickException(f"Source '{source}' not yet implemented")


@cli.command()
@click.argument("source", type=click.Choice([
    "donki-cme", "donki-flare", "donki-gst", "donki-ips", "donki-enlil",
    "cdaw", "swpc", "kyoto", "gfz", "jsoc",
], case_sensitive=False))
@click.option("--start", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", required=True, help="End date YYYY-MM-DD")
@run_async
async def ingest(source: str, start: str, end: str) -> None:
    """Ingest raw file cache into staging.db."""
    click.echo(f"Ingesting {source} {start} → {end}")
    settings = get_settings()
    if source == "donki-cme":
        from solarpipe_data.clients.donki import DonkiClient
        from solarpipe_data.ingestion.ingest_donki_cme import ingest_cme_batch
        async with DonkiClient(settings) as client:
            raw = await client.fetch_cme(start, end)
        engine = init_db(settings.staging_db_path)
        n = ingest_cme_batch(engine, raw)
        click.echo(f"  {n} CME records upserted")
    else:
        raise click.ClickException(f"Source '{source}' not yet implemented")


@cli.command()
@click.option("--start", default=None, help="Start date YYYY-MM-DD (optional)")
@click.option("--end", default=None, help="End date YYYY-MM-DD (optional)")
@run_async
async def crossmatch(start: str | None, end: str | None) -> None:
    """Run all CME↔flare and CME↔ICME matchers."""
    click.echo("crossmatch not yet implemented (Phase 5)")


@cli.command()
@click.option("--start", required=True, help="Start date YYYY-MM-DD")
@click.option("--no-fetch", is_flag=True, default=False, help="Skip fetch, ingest from cache only")
@run_async
async def build(start: str, no_fetch: bool) -> None:
    """Run full pipeline: fetch → ingest → crossmatch."""
    ctx = click.get_current_context()
    if not no_fetch:
        ctx.invoke(fetch, source="donki-cme", start=start, end="today", force=False)
    ctx.invoke(ingest, source="donki-cme", start=start, end="today")
    ctx.invoke(crossmatch, start=start, end=None)


@cli.command()
def validate() -> None:
    """Run validate_db.py checks."""
    script = Path(__file__).parent.parent.parent / "scripts" / "validate_db.py"
    if not script.exists():
        raise click.ClickException(f"Validation script not found: {script}")
    result = subprocess.run([sys.executable, str(script)], capture_output=False)
    sys.exit(result.returncode)


@cli.command()
def status() -> None:
    """Print record counts per table with temporal coverage."""
    settings = get_settings()
    db_path = settings.staging_db_path

    if not Path(db_path).exists():
        click.echo(f"staging.db not found at {db_path}")
        click.echo("Run: python scripts/port_solar_data.py --source <solar_data.db> --target <staging.db>")
        return

    engine = init_db(db_path)

    table_info: list[tuple[str, type, str]] = [
        ("cme_events",            CmeEvent,            "start_time"),
        ("cme_analyses",          CmeAnalysis,         "time21_5"),
        ("cdaw_cme_events",       CdawCmeEvent,        "datetime"),
        ("flares",                Flare,               "begin_time"),
        ("solar_wind_hourly",     SolarWindHourly,     "datetime"),
        ("sharp_keywords",        SharpKeyword,        "t_rec"),
        ("dst_hourly",            DstHourly,           "datetime"),
        ("symh_hourly",           SymhHourly,          "datetime"),
        ("kp_3hr",                Kp3hr,               "datetime"),
        ("f107_daily",            F107Daily,           "date"),
        ("enlil_simulations",     EnlilSimulation,     "model_completion_time"),
        ("geomagnetic_storms",    GeomagneticStorm,    "start_time"),
        ("interplanetary_shocks", InterplanetaryShock, "event_time"),
        ("silso_daily_ssn",       SilsoDailySSN,       "date"),
    ]

    click.echo(f"\n{'Table':<28} {'Rows':>8}  {'Min date':<22} {'Max date':<22}")
    click.echo("-" * 84)

    for table_name, cls, ts_col in table_info:
        try:
            count = row_count(engine, cls)
            min_ts = _min_timestamp(engine, cls, ts_col)
            max_ts = max_timestamp(engine, cls, ts_col)
            min_s = (min_ts or "—")[:19]
            max_s = (max_ts or "—")[:19]
            click.echo(f"{table_name:<28} {count:>8}  {min_s:<22} {max_s:<22}")
        except Exception as exc:
            click.echo(f"{table_name:<28} ERROR: {exc}")

    click.echo()


def _min_timestamp(engine: Any, table_class: type, col: str) -> str | None:
    import sqlalchemy as sa
    from sqlalchemy.orm import Session
    column = getattr(table_class, col)
    with Session(engine) as s:
        return s.execute(sa.select(sa.func.min(column))).scalar_one_or_none()
