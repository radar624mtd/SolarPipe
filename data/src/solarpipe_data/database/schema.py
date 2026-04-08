"""SQLAlchemy ORM schema for staging.db.

Key rules enforced here:
- WAL mode registered on every connection via event listener
- All tables include provenance columns: source_catalog, fetch_timestamp, data_version
- Upsert uses sqlalchemy.dialects.sqlite.insert (NOT sqlalchemy.insert)
- Windows-safe: callers must use Path(p).as_posix() in connection strings
"""
from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Text,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session


class Base(DeclarativeBase):
    metadata = MetaData()


# ---------------------------------------------------------------------------
# DONKI CME events — maps directly from solar_data.db donki_cme
# activity_id is the natural PK from DONKI (e.g. "2016-09-06T14:18Z-CME-001")
# ---------------------------------------------------------------------------
class CmeEvent(Base):
    __tablename__ = "cme_events"

    activity_id: str = Column(String, primary_key=True)
    start_time: str = Column(String, nullable=True)          # ISO 8601, no seconds: "2016-09-06T14:18Z"
    source_location: str = Column(String, nullable=True)     # e.g. "S09W11"
    active_region_num: int = Column(Integer, nullable=True)  # ~10-15% null; remediate via CDAW/GOES
    catalog: str = Column(String, nullable=True)
    note: str = Column(Text, nullable=True)
    instruments: str = Column(Text, nullable=True)           # JSON array as text
    link: str = Column(String, nullable=True)
    speed_kms: float = Column(Float, nullable=True)
    half_angle_deg: float = Column(Float, nullable=True)
    latitude: float = Column(Float, nullable=True)
    longitude: float = Column(Float, nullable=True)
    is_earth_directed: bool = Column(Boolean, nullable=True)
    analysis_type: str = Column(String, nullable=True)
    is_most_accurate: bool = Column(Boolean, nullable=True)
    linked_flare_id: str = Column(String, nullable=True)
    linked_ips_ids: str = Column(Text, nullable=True)        # JSON array
    linked_gst_ids: str = Column(Text, nullable=True)        # JSON array
    n_linked_events: int = Column(Integer, nullable=True)
    linked_event_ids: str = Column(Text, nullable=True)      # JSON array
    # Provenance
    source_catalog: str = Column(String, nullable=False, default="DONKI")
    fetch_timestamp: str = Column(String, nullable=True)
    data_version: str = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# DONKI CMEAnalysis — separate from cme_events; one per CME (mostAccurateOnly=true)
# time21_5 is position at 21.5 R☉ — NOT the event start time
# ---------------------------------------------------------------------------
class CmeAnalysis(Base):
    __tablename__ = "cme_analyses"

    analysis_id: str = Column(String, primary_key=True)     # activity_id + "_analysis"
    cme_activity_id: str = Column(String, nullable=False)   # FK → cme_events.activity_id
    time21_5: str = Column(String, nullable=True)           # Front at 21.5 R☉ — not event start
    latitude: float = Column(Float, nullable=True)
    longitude: float = Column(Float, nullable=True)
    half_angle_deg: float = Column(Float, nullable=True)
    speed_kms: float = Column(Float, nullable=True)
    is_most_accurate: bool = Column(Boolean, nullable=True)
    level_of_data: int = Column(Integer, nullable=True)     # 0=RT, 1=NRT, 2=definitive — prefer 2
    catalog: str = Column(String, nullable=True)
    # Provenance
    source_catalog: str = Column(String, nullable=False, default="DONKI")
    fetch_timestamp: str = Column(String, nullable=True)
    data_version: str = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# CDAW LASCO CME catalog
# cdaw_id format: "YYYYMMDD.HHMMSS" constructed from date+time_ut columns
# speed_20rs_kms is the correct speed for arrival models (NOT linear_speed_kms)
# ---------------------------------------------------------------------------
class CdawCmeEvent(Base):
    __tablename__ = "cdaw_cme_events"

    cdaw_id: str = Column(String, primary_key=True)         # "YYYYMMDD.HHMMSS"
    date: str = Column(String, nullable=True)
    time_ut: str = Column(String, nullable=True)
    datetime: str = Column(String, nullable=True)           # ISO combined
    central_pa_deg: float = Column(Float, nullable=True)    # "Halo" → None
    angular_width_deg: float = Column(Float, nullable=True) # "Halo" → 360
    linear_speed_kms: float = Column(Float, nullable=True)
    second_order_speed_init: float = Column(Float, nullable=True)
    second_order_speed_final: float = Column(Float, nullable=True)
    speed_20rs_kms: float = Column(Float, nullable=True)    # Use this for arrival models
    accel_kms2: float = Column(Float, nullable=True)
    mass_grams: float = Column(Float, nullable=True)        # ~40% null — do not impute
    kinetic_energy_ergs: float = Column(Float, nullable=True)
    mpa_deg: float = Column(Float, nullable=True)
    remarks: str = Column(Text, nullable=True)              # "Poor Event" / "Very Poor Event" here
    quality_flag: int = Column(Integer, nullable=False, default=3)
    # Provenance
    source_catalog: str = Column(String, nullable=False, default="CDAW")
    fetch_timestamp: str = Column(String, nullable=True)
    data_version: str = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# Solar flares — merged from DONKI FLR + NOAA GOES
# ---------------------------------------------------------------------------
class Flare(Base):
    __tablename__ = "flares"

    flare_id: str = Column(String, primary_key=True)
    begin_time: str = Column(String, nullable=True)
    peak_time: str = Column(String, nullable=True)
    end_time: str = Column(String, nullable=True)
    date: str = Column(String, nullable=True)
    class_type: str = Column(String, nullable=True)         # "X1.5", "M2.3", etc.
    class_letter: str = Column(String, nullable=True)
    class_magnitude: float = Column(Float, nullable=True)
    source_location: str = Column(String, nullable=True)
    active_region_num: int = Column(Integer, nullable=True)
    catalog: str = Column(String, nullable=True)
    instruments: str = Column(Text, nullable=True)
    note: str = Column(Text, nullable=True)
    link: str = Column(String, nullable=True)
    n_linked_events: int = Column(Integer, nullable=True)
    linked_event_ids: str = Column(Text, nullable=True)
    goes_satellite: str = Column(String, nullable=True)
    # Provenance
    source_catalog: str = Column(String, nullable=False, default="DONKI")
    fetch_timestamp: str = Column(String, nullable=True)
    data_version: str = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# Solar wind — hourly averages; ACE before July 2016, DSCOVR after
# Bz_GSM is the canonical field — never use Bz_GSE (RULE-031)
# Sentinel 9999.9, -1e31 converted to None at ingest
# ---------------------------------------------------------------------------
class SolarWindHourly(Base):
    __tablename__ = "solar_wind_hourly"

    datetime: str = Column(String, primary_key=True)         # ISO 8601 UTC hourly
    date: str = Column(String, nullable=True)
    year: int = Column(Integer, nullable=True)
    doy: int = Column(Integer, nullable=True)
    hour: int = Column(Integer, nullable=True)
    # OMNI-compatible columns (matches solar_data.db omni_hourly schema)
    b_scalar_avg: float = Column(Float, nullable=True)
    b_vector_mag: float = Column(Float, nullable=True)
    bx_gse: float = Column(Float, nullable=True)
    by_gse: float = Column(Float, nullable=True)
    bz_gse: float = Column(Float, nullable=True)
    by_gsm: float = Column(Float, nullable=True)
    bz_gsm: float = Column(Float, nullable=True)             # Canonical — RULE-031
    proton_temp_k: float = Column(Float, nullable=True)
    proton_density: float = Column(Float, nullable=True)
    flow_speed: float = Column(Float, nullable=True)
    flow_pressure: float = Column(Float, nullable=True)
    electric_field: float = Column(Float, nullable=True)
    plasma_beta: float = Column(Float, nullable=True)
    alfven_mach: float = Column(Float, nullable=True)
    kp_x10: int = Column(Integer, nullable=True)
    dst_nt: int = Column(Integer, nullable=True)
    ae_nt: int = Column(Integer, nullable=True)
    ap_index: int = Column(Integer, nullable=True)
    f10_7_index: float = Column(Float, nullable=True)
    spacecraft: str = Column(String, nullable=True)          # "ACE" or "DSCOVR"
    data_quality_flag: int = Column(Integer, nullable=True)
    # Provenance
    source_catalog: str = Column(String, nullable=False, default="OMNI")
    fetch_timestamp: str = Column(String, nullable=True)
    data_version: str = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# SHARP keywords — 18 space-weather keywords per HARP/time-step
# Drop at ingest if LON_FWT > 60° (disk-passage bias — RULE)
# query_context: at_eruption / minus_6h / minus_12h / plus_6h
# ---------------------------------------------------------------------------
class SharpKeyword(Base):
    __tablename__ = "sharp_keywords"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    harpnum: int = Column(Integer, nullable=False)
    noaa_ar: int = Column(Integer, nullable=True)            # 0 = no NOAA designation
    t_rec: str = Column(String, nullable=False)              # ISO timestamp
    usflux: float = Column(Float, nullable=True)
    meangam: float = Column(Float, nullable=True)
    meangbt: float = Column(Float, nullable=True)
    meangbz: float = Column(Float, nullable=True)
    meangbh: float = Column(Float, nullable=True)
    meanjzd: float = Column(Float, nullable=True)
    totusjz: float = Column(Float, nullable=True)
    meanalp: float = Column(Float, nullable=True)
    meanjzh: float = Column(Float, nullable=True)
    totusjh: float = Column(Float, nullable=True)
    absnjzh: float = Column(Float, nullable=True)
    savncpp: float = Column(Float, nullable=True)
    meanpot: float = Column(Float, nullable=True)
    totpot: float = Column(Float, nullable=True)
    meanshr: float = Column(Float, nullable=True)
    shrgt45: float = Column(Float, nullable=True)
    r_value: float = Column(Float, nullable=True)
    area_acr: float = Column(Float, nullable=True)
    lat_fwt: float = Column(Float, nullable=True)
    lon_fwt: float = Column(Float, nullable=True)            # Must be ≤60° at ingest
    query_context: str = Column(String, nullable=True)       # at_eruption / minus_6h / etc.
    activity_id: str = Column(String, nullable=True)          # FK → cme_events.activity_id (for resume)
    # Provenance
    source_catalog: str = Column(String, nullable=False, default="JSOC")
    fetch_timestamp: str = Column(String, nullable=True)
    data_version: str = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# Kyoto WDC Dst index — hourly, with data_type preference cascade
# final > provisional > realtime — never overwrite final with lower quality
# ---------------------------------------------------------------------------
class DstHourly(Base):
    __tablename__ = "dst_hourly"

    datetime: str = Column(String, primary_key=True)         # ISO 8601 UTC hourly
    dst_nt: float = Column(Float, nullable=True)             # nT; sentinel >500 or <-500 → None
    data_type: str = Column(String, nullable=False)          # "final" / "provisional" / "realtime"
    # Provenance
    source_catalog: str = Column(String, nullable=False, default="Kyoto")
    fetch_timestamp: str = Column(String, nullable=True)
    data_version: str = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# SYM-H hourly — from solar_data.db symh_hourly; higher cadence than Dst
# ---------------------------------------------------------------------------
class SymhHourly(Base):
    __tablename__ = "symh_hourly"

    datetime: str = Column(String, primary_key=True)
    date: str = Column(String, nullable=True)
    hour: int = Column(Integer, nullable=True)
    symh_nt: float = Column(Float, nullable=True)
    asyh_nt: float = Column(Float, nullable=True)
    symd_nt: float = Column(Float, nullable=True)
    asyd_nt: float = Column(Float, nullable=True)
    # Provenance
    source_catalog: str = Column(String, nullable=False, default="WDC")
    fetch_timestamp: str = Column(String, nullable=True)
    data_version: str = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# GFZ Kp 3-hour index
# ---------------------------------------------------------------------------
class Kp3hr(Base):
    __tablename__ = "kp_3hr"

    datetime: str = Column(String, primary_key=True)         # Start of 3-hr interval
    kp: float = Column(Float, nullable=True)
    ap: int = Column(Integer, nullable=True)
    definitive: bool = Column(Boolean, nullable=True)
    daily_ap: float = Column(Float, nullable=True)
    daily_f10_7_obs: float = Column(Float, nullable=True)
    daily_f10_7_adj: float = Column(Float, nullable=True)
    # Provenance
    source_catalog: str = Column(String, nullable=False, default="GFZ")
    fetch_timestamp: str = Column(String, nullable=True)
    data_version: str = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# F10.7 solar radio flux — daily
# ---------------------------------------------------------------------------
class F107Daily(Base):
    __tablename__ = "f107_daily"

    date: str = Column(String, primary_key=True)
    f10_7_obs: float = Column(Float, nullable=True)
    f10_7_adj: float = Column(Float, nullable=True)
    sunspot_number: float = Column(Float, nullable=True)
    # Provenance
    source_catalog: str = Column(String, nullable=False, default="NOAA")
    fetch_timestamp: str = Column(String, nullable=True)
    data_version: str = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# DONKI WSA-ENLIL simulations — not 1:1 with CMEs; deduplicate by first per CME
# ---------------------------------------------------------------------------
class EnlilSimulation(Base):
    __tablename__ = "enlil_simulations"

    simulation_id: str = Column(String, primary_key=True)
    model_completion_time: str = Column(String, nullable=True)
    au: float = Column(Float, nullable=True)
    linked_cme_ids: str = Column(Text, nullable=True)        # JSON array
    link: str = Column(String, nullable=True)
    # Provenance
    source_catalog: str = Column(String, nullable=False, default="DONKI")
    fetch_timestamp: str = Column(String, nullable=True)
    data_version: str = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# DONKI Geomagnetic storms
# ---------------------------------------------------------------------------
class GeomagneticStorm(Base):
    __tablename__ = "geomagnetic_storms"

    gst_id: str = Column(String, primary_key=True)
    start_time: str = Column(String, nullable=True)
    kp_index_max: float = Column(Float, nullable=True)
    all_kp_values: str = Column(Text, nullable=True)         # JSON
    link: str = Column(String, nullable=True)
    n_linked_events: int = Column(Integer, nullable=True)
    linked_event_ids: str = Column(Text, nullable=True)
    # Provenance
    source_catalog: str = Column(String, nullable=False, default="DONKI")
    fetch_timestamp: str = Column(String, nullable=True)
    data_version: str = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# DONKI Interplanetary shocks — used for CME-ICME matching
# ---------------------------------------------------------------------------
class InterplanetaryShock(Base):
    __tablename__ = "interplanetary_shocks"

    ips_id: str = Column(String, primary_key=True)
    event_time: str = Column(String, nullable=True)
    location: str = Column(String, nullable=True)
    catalog: str = Column(String, nullable=True)
    instruments: str = Column(Text, nullable=True)
    link: str = Column(String, nullable=True)
    n_linked_events: int = Column(Integer, nullable=True)
    linked_event_ids: str = Column(Text, nullable=True)
    # Provenance
    source_catalog: str = Column(String, nullable=False, default="DONKI")
    fetch_timestamp: str = Column(String, nullable=True)
    data_version: str = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# SILSO daily sunspot number
# ---------------------------------------------------------------------------
class SilsoDailySSN(Base):
    __tablename__ = "silso_daily_ssn"

    date: str = Column(String, primary_key=True)
    year: int = Column(Integer, nullable=True)
    month: int = Column(Integer, nullable=True)
    day: int = Column(Integer, nullable=True)
    decimal_year: float = Column(Float, nullable=True)
    sunspot_number: float = Column(Float, nullable=True)
    std_dev: float = Column(Float, nullable=True)
    n_observations: int = Column(Integer, nullable=True)
    provisional: bool = Column(Boolean, nullable=True)
    # Provenance
    source_catalog: str = Column(String, nullable=False, default="SILSO")
    fetch_timestamp: str = Column(String, nullable=True)
    data_version: str = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# HARP ↔ NOAA AR mapping — from hmi.sharp_720s (CCD series, has NOAA_ARS)
# Used in Phase 5 cross-matching when NOAA_AR is 0 or multi-region
# ---------------------------------------------------------------------------
class HarpNoaaMap(Base):
    __tablename__ = "harp_noaa_map"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    harpnum: int = Column(Integer, nullable=False, index=True)
    noaa_ar: int = Column(Integer, nullable=True, index=True)   # None when NOAA_AR=0
    noaa_ars: str = Column(String, nullable=True)               # tilde-separated multi-region
    t_rec: str = Column(String, nullable=True)                  # ISO snapshot time
    # Provenance
    source_catalog: str = Column(String, nullable=False, default="JSOC")
    fetch_timestamp: str = Column(String, nullable=True)
    data_version: str = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# Ambient solar wind context — 6-hour pre-arrival averages per CME
# Computed from solar_wind_hourly for the 6-hour window before each CME start_time
# ---------------------------------------------------------------------------
class SwAmbientContext(Base):
    __tablename__ = "sw_ambient_context"

    activity_id: str = Column(String, primary_key=True)  # FK → cme_events.activity_id
    window_start: str = Column(String, nullable=True)    # CME start_time - 6h
    window_end: str = Column(String, nullable=True)      # CME start_time
    n_hours: int = Column(Integer, nullable=True)        # number of hours averaged
    sw_speed_ambient: float = Column(Float, nullable=True)   # mean flow_speed km/s
    sw_density_ambient: float = Column(Float, nullable=True) # mean proton_density /cm³
    sw_bt_ambient: float = Column(Float, nullable=True)      # mean |Bt| nT
    sw_bz_ambient: float = Column(Float, nullable=True)      # mean bz_gsm nT
    # Provenance
    source_catalog: str = Column(String, nullable=False, default="OMNI")
    fetch_timestamp: str = Column(String, nullable=True)
    data_version: str = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# Schema version tracker — for migrations.py
# ---------------------------------------------------------------------------
class SchemaVersion(Base):
    __tablename__ = "schema_version"

    version: int = Column(Integer, primary_key=True)
    applied_at: str = Column(String, nullable=False)
    description: str = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# Engine factory with WAL mode + Windows-safe path
# ---------------------------------------------------------------------------
def make_engine(db_path: str):
    """Create a SQLAlchemy engine with WAL mode and Windows-safe URL."""
    from pathlib import Path

    import sqlalchemy as sa

    posix = Path(db_path).as_posix()
    url = f"sqlite:///{posix}"
    engine = sa.create_engine(url, echo=False)

    @event.listens_for(engine, "connect")
    def _set_wal(dbapi_conn, _conn_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    return engine


def init_db(db_path: str):
    """Create all tables if they do not exist."""
    from datetime import datetime, timezone

    engine = make_engine(db_path)
    Base.metadata.create_all(engine)

    with Session(engine) as s, s.begin():
        existing = s.get(SchemaVersion, 1)
        if existing is None:
            s.add(SchemaVersion(
                version=1,
                applied_at=datetime.now(timezone.utc).isoformat(),
                description="Initial schema — all staging tables",
            ))
    return engine
