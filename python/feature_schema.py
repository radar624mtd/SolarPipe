"""Feature schema contract for the neural ensemble training pipeline.

This module is the single authoritative source for every column in the
``training_features`` SQLite view (133 columns x 9,418 rows).  All training,
loading, and ONNX export code must import from here -- never hard-code column
lists elsewhere.

Column roles
------------
key      : identifier / timestamp -- never fed to the model
label    : training target (transit_time_hours)
flat     : numeric feature fed to the flat-branch MLP
drop     : 100 % NULL phantom -- discarded at load time (6 columns)
bookkeep : string/metadata columns not used as model input

Null policies (for flat columns)
---------------------------------
dense   : < 10 % NULL -- impute with column mean; mask always ~1
sparse  : 10-95 % NULL -- replace NaN->0.0, mask=0; learnable null embedding
phantom : 100 % NULL -- drop entirely; no mask or embedding passed

Sequence channels (separate Parquet branch, NOT in flat schema)
---------------------------------------------------------------
SEQUENCE_CHANNELS: 22 channels -- 20 OMNI pre-launch + 2 GOES MAG in-transit.
Expansion to 24 channels (+SME_nT, +SMR_nT) is pending SuperMAG account
activation and must be done atomically per RULE-213.

RULE-213 checklist for SuperMAG expansion:
  1. Update SEQUENCE_CHANNELS here (+2 channels, total -> 24)
  2. Update build_pinn_sequences.py to extract those channels
  3. Update n_seq_channels in python/tft_pinn_model.py
  4. Update SEQ_CHANNELS in python/solarpipe_server.py
  All four changes in ONE commit -- no intermediate 23-channel state.
"""
from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Role = Literal["key", "label", "flat", "drop", "bookkeep"]
NullPolicy = Literal["dense", "sparse", "phantom", "n/a"]
DType = Literal["float32", "int32", "str"]

# Shorthand constants used inside the table so each row fits in 88 chars.
_R = "REAL"
_I = "INTEGER"
_T = "TEXT"
_f = "float32"
_i = "int32"
_s = "str"


@dataclass(frozen=True)
class ColumnSpec:
    """Specification for a single column in training_features."""

    name: str
    db_type: str
    dtype: DType
    role: Role
    null_policy: NullPolicy
    tier: int
    notes: str = ""


# ---------------------------------------------------------------------------
# Full schema -- 133 entries, exactly matching training_features column order
# ---------------------------------------------------------------------------
# Each row: ColumnSpec(name, db_type, dtype, role, null_policy, tier [, notes])
FEATURE_SCHEMA: list[ColumnSpec] = [
    # === keys & labels ===================================================
    ColumnSpec("activity_id",        _T, _s, "key",   "n/a",   0),
    ColumnSpec("launch_time",        _T, _s, "key",   "n/a",   0),
    ColumnSpec("transit_time_hours", _R, _f, "label", "dense", 0),
    ColumnSpec("icme_arrival_time",  _T, _s, "key",   "n/a",   0),
    ColumnSpec("linked_ips_id",      _T, _s, "key",   "n/a",   0),
    ColumnSpec("split",              _T, _s, "key",   "n/a",   0),
    ColumnSpec("exclude",            _I, _i, "key",   "n/a",   0),
    # === bookkeeping (metadata -- not model inputs) =======================
    ColumnSpec("linked_flare_id",       _T, _s,  "bookkeep", "n/a", 0),
    ColumnSpec("flare_class_letter",    _T, _s,  "bookkeep", "n/a", 0),
    ColumnSpec("flare_peak_time",       _T, _s,  "bookkeep", "n/a", 0),
    ColumnSpec("flare_active_region",   _I, _s,  "bookkeep", "n/a", 0),
    ColumnSpec("flare_match_method",    _T, _s,  "bookkeep", "n/a", 0),
    ColumnSpec("sharp_harpnum",         _I, _s,  "bookkeep", "n/a", 0),
    ColumnSpec("sharp_noaa_ar",         _I, _s,  "bookkeep", "n/a", 0),
    ColumnSpec("sharp_snapshot_context",_T, _s,  "bookkeep", "n/a", 0),
    ColumnSpec("sharp_match_method",    _T, _s,  "bookkeep", "n/a", 0),
    ColumnSpec("icme_match_method",     _T, _s,  "bookkeep", "n/a", 0),
    ColumnSpec("icme_match_confidence", _R, _f,  "bookkeep", "n/a", 0),
    ColumnSpec("dst_min_time",          _T, _s,  "bookkeep", "n/a", 0),
    ColumnSpec("source_catalog",        _T, _s,  "bookkeep", "n/a", 0),
    ColumnSpec("fetch_timestamp",       _T, _s,  "bookkeep", "n/a", 0),
    ColumnSpec("quality_flag",          _I, _i,  "bookkeep", "n/a", 0),
    # === phantom drops -- 100 % NULL, never pass to model ================
    ColumnSpec(
        "dimming_area", _R, _f, "drop", "phantom", 1,
        "HEK CD FRMs do not populate quantitative area",
    ),
    ColumnSpec(
        "dimming_asymmetry", _R, _f, "drop", "phantom", 1,
        "HEK CD FRMs do not populate asymmetry",
    ),
    ColumnSpec(
        "eit_wave_speed_kms", _R, _f, "drop", "phantom", 1,
        "HEK CW FRMs do not populate wave speed",
    ),
    ColumnSpec(
        "rc_bz_min", _R, _f, "drop", "phantom", 1,
        "R&C catalog has no Bz_min column",
    ),
    ColumnSpec(
        "phase8_pred_transit_hours", _R, _f, "drop", "phantom", 0,
        "Prediction slot -- NULL until Phase 8 backfill (P6 scope)",
    ),
    ColumnSpec(
        "pinn_v1_pred_transit_hours", _R, _f, "drop", "phantom", 0,
        "Prediction slot -- NULL until PINN V1 backfill (P6 scope)",
    ),
    # === Tier 0: CME kinematics ==========================================
    ColumnSpec("cme_speed_kms",       _R, _f, "flat", "dense",  0),
    ColumnSpec("cme_half_angle_deg",  _R, _f, "flat", "dense",  0),
    ColumnSpec("cme_latitude",        _R, _f, "flat", "dense",  0),
    ColumnSpec(
        "cme_longitude", _R, _f, "flat", "sparse", 0,
        "10.7 % NULL -- source location not always known",
    ),
    ColumnSpec(
        "cme_mass_grams", _R, _f, "flat", "sparse", 0,
        "Carried from feature_vectors; ~sparse",
    ),
    ColumnSpec("cme_angular_width_deg", _R, _f, "flat", "dense",  0),
    ColumnSpec(
        "cme_tilt_angle_deg", _R, _f, "flat", "sparse", 0,
        "13.9 % NULL",
    ),
    # === Tier 0: CDAW 2nd-order kinematics (previously excluded) =========
    ColumnSpec("cdaw_linear_speed_kms",   _R, _f, "flat", "sparse", 0),
    ColumnSpec("cdaw_angular_width_deg",  _R, _f, "flat", "sparse", 0),
    ColumnSpec(
        "cdaw_mass_log10", _R, _f, "flat", "sparse", 0,
        "82.9 % NULL -- mass fails for poor plane-of-sky events",
    ),
    ColumnSpec("cdaw_ke_log10",      _R, _f, "flat", "sparse", 0, "82.9 % NULL"),
    ColumnSpec("cdaw_matched",       _I, _i, "flat", "dense",  0),
    ColumnSpec("cdaw_mass_quality",  _I, _i, "flat", "sparse", 0),
    ColumnSpec("second_order_speed_init",  _R, _f, "flat", "sparse", 0),
    ColumnSpec("second_order_speed_final", _R, _f, "flat", "sparse", 0),
    ColumnSpec("second_order_speed_20Rs",  _R, _f, "flat", "sparse", 0),
    ColumnSpec("accel_kms2", _R, _f, "flat", "sparse", 0),
    ColumnSpec("mpa_deg",    _R, _f, "flat", "sparse", 0),
    # === Tier 0: SHARP magnetic keywords (14 physics params) =============
    ColumnSpec("usflux",  _R, _f, "flat", "sparse", 0, "87.2 % NULL"),
    ColumnSpec("meangam", _R, _f, "flat", "sparse", 0),
    ColumnSpec("meangbt", _R, _f, "flat", "sparse", 0),
    ColumnSpec("meangbz", _R, _f, "flat", "sparse", 0),
    ColumnSpec("meangbh", _R, _f, "flat", "sparse", 0),
    ColumnSpec("meanjzd", _R, _f, "flat", "sparse", 0),
    ColumnSpec("totusjz", _R, _f, "flat", "sparse", 0),
    ColumnSpec("meanjzh", _R, _f, "flat", "sparse", 0),
    ColumnSpec("totusjh", _R, _f, "flat", "sparse", 0),
    ColumnSpec("absnjzh", _R, _f, "flat", "sparse", 0),
    ColumnSpec("meanalp", _R, _f, "flat", "sparse", 0),
    ColumnSpec("savncpp", _R, _f, "flat", "sparse", 0),
    ColumnSpec("meanpot", _R, _f, "flat", "sparse", 0),
    ColumnSpec("totpot",  _R, _f, "flat", "sparse", 0),
    ColumnSpec("meanshr", _R, _f, "flat", "sparse", 0),
    ColumnSpec("shrgt45", _R, _f, "flat", "sparse", 0),
    ColumnSpec("r_value", _R, _f, "flat", "sparse", 0),
    ColumnSpec("area_acr",       _R, _f, "flat", "sparse", 0),
    ColumnSpec("sharp_available",_I, _i, "flat", "dense",  0),
    # === Tier 0: ambient solar wind ======================================
    ColumnSpec("sw_speed_ambient",   _R, _f, "flat", "dense", 0),
    ColumnSpec("sw_density_ambient", _R, _f, "flat", "dense", 0),
    ColumnSpec("sw_bt_ambient",      _R, _f, "flat", "dense", 0),
    ColumnSpec("sw_bz_ambient",      _R, _f, "flat", "dense", 0),
    ColumnSpec("delta_v_kms",        _R, _f, "flat", "dense", 0),
    # === Tier 0: OMNI 24 h pre-launch aggregates =========================
    ColumnSpec("omni_24h_bz_mean",       _R, _f, "flat", "dense", 0),
    ColumnSpec("omni_24h_bz_std",        _R, _f, "flat", "dense", 0),
    ColumnSpec("omni_24h_bz_min",        _R, _f, "flat", "dense", 0),
    ColumnSpec("omni_24h_speed_mean",    _R, _f, "flat", "dense", 0),
    ColumnSpec("omni_24h_density_mean",  _R, _f, "flat", "dense", 0),
    ColumnSpec("omni_24h_pressure_mean", _R, _f, "flat", "dense", 0),
    ColumnSpec("omni_24h_ae_max",        _R, _f, "flat", "dense", 0),
    ColumnSpec("omni_24h_dst_min",       _R, _f, "flat", "dense", 0),
    ColumnSpec("omni_24h_kp_max",        _R, _f, "flat", "dense", 0),
    # === Tier 0: OMNI 48 h pre-launch aggregates =========================
    ColumnSpec("omni_48h_density_spike_max", _R, _f, "flat", "dense", 0),
    ColumnSpec("omni_48h_speed_gradient",    _R, _f, "flat", "dense", 0),
    # === Tier 0: OMNI 150 h pre-launch aggregates ========================
    ColumnSpec("omni_150h_density_median", _R, _f, "flat", "dense", 0),
    ColumnSpec("omni_150h_speed_median",   _R, _f, "flat", "dense", 0),
    # === Tier 0: space weather indices ===================================
    ColumnSpec("f10_7",               _R, _f, "flat", "dense", 0),
    ColumnSpec("sunspot_number",      _R, _f, "flat", "dense", 0),
    ColumnSpec("dst_min_nt",          _R, _f, "flat", "dense", 0),
    ColumnSpec("kp_max",              _R, _f, "flat", "dense", 0),
    ColumnSpec("storm_threshold_met", _I, _i, "flat", "dense", 0),
    # === Tier 0: flare scalars ============================================
    ColumnSpec(
        "flare_class_numeric", _R, _f, "flat", "sparse", 0,
        "73.0 % NULL -- many CMEs have no associated flare",
    ),
    ColumnSpec("has_flare",              _I, _i, "flat", "dense",  0),
    ColumnSpec(
        "flare_source_longitude", _R, _f, "flat", "sparse", 0,
        "77.9 % NULL",
    ),
    ColumnSpec(
        "goes_peak_flux_wm2", _R, _f, "flat", "sparse", 0,
        "87.2 % NULL -- proxy for shock strength",
    ),
    # === Tier 0: HCS tilt ================================================
    ColumnSpec(
        "hcs_tilt_angle", _R, _f, "flat", "sparse", 0,
        "From feature_vectors; overlaps hcs_tilt_classic_deg",
    ),
    ColumnSpec("hcs_distance",        _R, _f, "flat", "sparse", 0),
    ColumnSpec(
        "hcs_tilt_classic_deg", _R, _f, "flat", "sparse", 0,
        "WSO classical HCS tilt; NULL post CR2302 (2025-10+)",
    ),
    ColumnSpec(
        "hcs_tilt_radial_deg", _R, _f, "flat", "sparse", 0,
        "GONG PFSS radial tilt; available through current date",
    ),
    # === Tier 0: Type II radio burst ======================================
    ColumnSpec("has_type2_burst",      _I, _i, "flat", "sparse", 0, "12.5 % NULL"),
    ColumnSpec("type2_start_freq_mhz", _R, _f, "flat", "sparse", 0),
    ColumnSpec("type2_end_freq_mhz",   _R, _f, "flat", "sparse", 0),
    # === Tier 0: ENLIL model prediction ===================================
    ColumnSpec(
        "enlil_predicted_arrival_hours", _R, _f, "flat", "sparse", 0,
        "87.2 % NULL",
    ),
    ColumnSpec("enlil_au",      _R, _f, "flat", "sparse", 0),
    ColumnSpec("enlil_matched", _I, _i, "flat", "dense",  0),
    # === Tier 0: multi-CME interaction context ============================
    ColumnSpec("preceding_cme_count_48h",  _R, _f, "flat", "dense",  0),
    ColumnSpec("preceding_cme_speed_max",  _R, _f, "flat", "dense",  0),
    ColumnSpec("preceding_cme_speed_mean", _R, _f, "flat", "dense",  0),
    ColumnSpec(
        "preceding_cme_angular_sep_min", _R, _f, "flat", "sparse", 0,
        "14.6 % NULL",
    ),
    ColumnSpec("is_multi_cme", _I, _i, "flat", "dense", 0),
    # === Tier 0: clustering labels ========================================
    ColumnSpec("cluster_id_k5",     _I, _i, "flat", "dense",  0),
    ColumnSpec("cluster_id_k8",     _I, _i, "flat", "dense",  0),
    ColumnSpec("cluster_id_k12",    _I, _i, "flat", "dense",  0),
    ColumnSpec(
        "cluster_id_dbscan", _I, _i, "flat", "sparse", 0,
        "DBSCAN -1 = noise; treat as a valid class",
    ),
    ColumnSpec("cluster_assigned", _I, _i, "flat", "dense", 0),
    # === Tier 1: Richardson & Cane ICME catalog ===========================
    ColumnSpec(
        "rc_icme_type", _I, _i, "flat", "sparse", 1,
        "70.8 % NULL; 0=no cloud, 1=cloud, 2=probable, 2H=hybrid",
    ),
    ColumnSpec(
        "rc_bde_flag", _I, _i, "flat", "sparse", 1,
        "70.8 % NULL; bidirectional electron strahl flag",
    ),
    ColumnSpec("rc_v_icme",  _R, _f, "flat", "sparse", 1, "ICME bulk speed at L1"),
    ColumnSpec("rc_b_max",   _R, _f, "flat", "sparse", 1, "Max B in ICME sheath"),
    ColumnSpec("rc_matched", _I, _i, "flat", "dense",  1),
    # === Tier 1: HEK event flags (area/speed fields are phantoms above) ===
    ColumnSpec("has_coronal_dimming",   _I, _i, "flat", "dense", 1),
    ColumnSpec("has_eit_wave",          _I, _i, "flat", "dense", 1),
    ColumnSpec("has_filament_eruption", _I, _i, "flat", "dense", 1),
    ColumnSpec("has_sigmoid",           _I, _i, "flat", "dense", 1),
    # === Tier 1: DONKI SEP events =========================================
    ColumnSpec("has_sep",  _I, _i, "flat", "dense",  1),
    ColumnSpec(
        "sep_onset_delay_hours", _R, _f, "flat", "sparse", 1,
        "95.1 % NULL -- SEP rare; NULL is MNAR signal",
    ),
    # === Tier 1: PFSS open/closed field topology ==========================
    ColumnSpec("pfss_field_type",          _I, _i, "flat", "sparse", 1, "11.4 % NULL"),
    ColumnSpec("pfss_open_fraction_10deg", _R, _f, "flat", "sparse", 1),
    ColumnSpec("pfss_ch_distance_deg",     _R, _f, "flat", "sparse", 1),
    ColumnSpec("pfss_polarity",            _I, _i, "flat", "sparse", 1),
    # === Tier 2: DONKI Magnetopause Crossings =============================
    ColumnSpec("has_mpc",         _I, _i, "flat", "dense",  2),
    ColumnSpec(
        "mpc_delay_hours", _R, _f, "flat", "sparse", 2,
        "95.3 % NULL -- extreme events only; NULL is MNAR",
    ),
    # === Tier 2: STEREO-A ambient solar wind ==============================
    ColumnSpec("stereo_available",       _I, _i, "flat", "dense",  2),
    ColumnSpec(
        "stereo_speed_ambient", _R, _f, "flat", "sparse", 2,
        "89.4 % NULL -- conjunction gaps + coverage ends 2025-10",
    ),
    ColumnSpec("stereo_density_ambient", _R, _f, "flat", "sparse", 2),
    ColumnSpec("stereo_lead_hours",      _R, _f, "flat", "sparse", 2),
    # NOTE: GOES MAG (Hp, Bt) are sequence channels, NOT flat entries.
    # See SEQUENCE_CHANNELS below.
]

# ---------------------------------------------------------------------------
# Sanity-check TOTAL immediately at import -- fail fast on edits
# ---------------------------------------------------------------------------
assert len(FEATURE_SCHEMA) == 133, (
    f"FEATURE_SCHEMA must have 133 entries, got {len(FEATURE_SCHEMA)}"
)


# ---------------------------------------------------------------------------
# Derived lists -- computed once at import
# ---------------------------------------------------------------------------
def _by_role(role: Role) -> list[ColumnSpec]:
    return [c for c in FEATURE_SCHEMA if c.role == role]


KEY_COLS: list[str]      = [c.name for c in _by_role("key")]
LABEL_COL: str           = "transit_time_hours"
FLAT_COLS: list[str]     = [c.name for c in _by_role("flat")]
DROP_COLS: list[str]     = [c.name for c in _by_role("drop")]
BOOKKEEP_COLS: list[str] = [c.name for c in _by_role("bookkeep")]

FLAT_DENSE_COLS: list[str] = [
    c.name for c in _by_role("flat") if c.null_policy == "dense"
]
FLAT_SPARSE_COLS: list[str] = [
    c.name for c in _by_role("flat") if c.null_policy == "sparse"
]

# Hard totals -- unit tests assert these exact values
TOTAL_COLS: int    = len(FEATURE_SCHEMA)   # 133
FLAT_COUNT: int    = len(FLAT_COLS)        # asserted in test_feature_schema.py
PHANTOM_COUNT: int = len(DROP_COLS)        # 6


# ---------------------------------------------------------------------------
# Sequence channels (separate Parquet branch, NOT flat)
# ---------------------------------------------------------------------------
_OMNI_CHANNELS: list[str] = [
    "bz_gsm", "speed", "density", "pressure", "temperature",
    "bx_gsm", "by_gsm", "bt", "ae_index", "al_index",
    "au_index", "dst_index", "kp_index", "f10_7",
    "sigma_bz", "flow_speed", "proton_density", "flow_pressure",
    "sigma_theta", "sigma_phi",
]
assert len(_OMNI_CHANNELS) == 20

# GOES MAG in-transit channels (variable-length window, 1-h cadence)
_GOES_CHANNELS: list[str] = ["goes_hp_nt", "goes_bt_nt"]

# Combined -- 22 channels.
# SuperMAG expansion: append ["sme_nt", "smr_nt"] here AND in
# build_pinn_sequences.py / tft_pinn_model.py / solarpipe_server.py
# in ONE atomic commit (RULE-213).  No 23-channel intermediate state.
SEQUENCE_CHANNELS: list[str] = _OMNI_CHANNELS + _GOES_CHANNELS
N_SEQ_CHANNELS: int = len(SEQUENCE_CHANNELS)
assert N_SEQ_CHANNELS == 22


# ---------------------------------------------------------------------------
# DB schema drift detector -- call at training start; aborts on mismatch
# ---------------------------------------------------------------------------
def assert_schema_matches_db(db_path: str) -> None:
    """Raise RuntimeError if training_features columns diverge from schema.

    Args:
        db_path: absolute path to staging.db

    Raises:
        RuntimeError: detailed diff if any drift is detected
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("PRAGMA table_info(training_features)").fetchall()
    finally:
        conn.close()

    db_names: list[str]     = [r[1] for r in rows]
    schema_names: list[str] = [c.name for c in FEATURE_SCHEMA]
    db_set                  = set(db_names)
    schema_set              = set(schema_names)

    errors: list[str] = []
    missing_from_db     = schema_set - db_set
    missing_from_schema = db_set - schema_set
    if missing_from_db:
        errors.append(
            f"In schema but NOT in DB: {sorted(missing_from_db)}"
        )
    if missing_from_schema:
        errors.append(
            f"In DB but NOT in schema: {sorted(missing_from_schema)}"
        )
    if len(db_names) != len(schema_names):
        errors.append(
            f"Count mismatch: DB={len(db_names)}, schema={len(schema_names)}"
        )
    if errors:
        raise RuntimeError(
            "training_features schema drift -- update feature_schema.py:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------
def print_summary() -> None:
    """Print a human-readable column count breakdown."""
    role_counts = Counter(c.role for c in FEATURE_SCHEMA)
    null_counts = Counter(
        c.null_policy for c in FEATURE_SCHEMA if c.role == "flat"
    )
    tier_counts = Counter(c.tier for c in FEATURE_SCHEMA if c.role == "flat")

    sep = "=" * 60
    print(sep)
    print("feature_schema.py -- training_features contract")
    print(sep)
    print(f"Total columns  : {TOTAL_COLS}  (must be 133)")
    print(f"  key          : {role_counts['key']}")
    print(f"  label        : {role_counts['label']}")
    print(f"  flat (model) : {role_counts['flat']}")
    print(f"  drop/phantom : {role_counts['drop']}  (must be 6)")
    print(f"  bookkeeping  : {role_counts['bookkeep']}")
    print()
    print("Flat null policy breakdown:")
    print(f"  dense   (<10% NULL) : {null_counts['dense']}")
    print(f"  sparse  (10-95%)    : {null_counts['sparse']}")
    print()
    print("Flat columns by tier:")
    for t in sorted(tier_counts):
        print(f"  Tier {t} : {tier_counts[t]}")
    print()
    print(f"Sequence channels : {N_SEQ_CHANNELS}  (22 now; 24 after SuperMAG)")
    print(sep)


if __name__ == "__main__":
    print_summary()
