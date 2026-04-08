"""Task 5.7 — Physical consistency validation for assembled feature vectors.

Checks:
  1. Speed ↔ transit time correlation  (Spearman r > 0.4 expected)
  2. Dst_min ↔ Bz_min correlation      (Spearman r > 0.5 expected)
  3. Speed range gate                  (100–3000 km/s; flag outliers)
  4. Density range gate                (0.1–100 cm⁻³; flag probable sentinels)

All checks operate on plain lists/dicts so they can be unit-tested without a DB.
The `run_validation` entry point accepts an SQLAlchemy engine and returns a
`ValidationReport` summarising pass/fail and any flagged activity_ids.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import sqlalchemy as sa

from ..database.schema import FeatureVector, make_engine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

SPEED_MIN_KMS: float = 100.0
SPEED_MAX_KMS: float = 3000.0
DENSITY_MIN_CM3: float = 0.1
DENSITY_MAX_CM3: float = 100.0
CORR_SPEED_TRANSIT_MIN: float = 0.4   # Spearman r lower bound
CORR_DST_BZ_MIN: float = 0.5          # Spearman r lower bound (absolute value)
MIN_PAIRS_FOR_CORR: int = 10           # skip correlation if fewer paired rows


# ---------------------------------------------------------------------------
# Spearman rank correlation (no scipy dependency)
# ---------------------------------------------------------------------------

def _rank(values: list[float]) -> list[float]:
    """Return fractional ranks (averaged for ties)."""
    n = len(values)
    sorted_pairs = sorted(enumerate(values), key=lambda x: x[1])
    ranks: list[float] = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_pairs[j + 1][1] == sorted_pairs[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[sorted_pairs[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_r(x: list[float], y: list[float]) -> float | None:
    """Spearman rank correlation coefficient for paired lists.

    Returns None when fewer than MIN_PAIRS_FOR_CORR pairs are available.
    """
    if len(x) != len(y) or len(x) < MIN_PAIRS_FOR_CORR:
        return None
    rx = _rank(x)
    ry = _rank(y)
    n = len(rx)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_speed_range(
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Return (outlier_ids, warning_messages) for CME speeds outside valid range."""
    outliers: list[str] = []
    warnings: list[str] = []
    for r in rows:
        speed = r.get("cme_speed_kms")
        if speed is None:
            continue
        if speed < SPEED_MIN_KMS or speed > SPEED_MAX_KMS:
            aid = r["activity_id"]
            outliers.append(aid)
            warnings.append(
                f"speed outlier {aid}: {speed:.1f} km/s "
                f"(expected {SPEED_MIN_KMS}–{SPEED_MAX_KMS})"
            )
    return outliers, warnings


def check_density_range(
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Return (sentinel_ids, warning_messages) for ambient density outside valid range."""
    sentinels: list[str] = []
    warnings: list[str] = []
    for r in rows:
        density = r.get("sw_density_ambient")
        if density is None:
            continue
        if density < DENSITY_MIN_CM3 or density > DENSITY_MAX_CM3:
            aid = r["activity_id"]
            sentinels.append(aid)
            warnings.append(
                f"density sentinel {aid}: {density:.3g} cm⁻³ "
                f"(expected {DENSITY_MIN_CM3}–{DENSITY_MAX_CM3})"
            )
    return sentinels, warnings


def check_speed_transit_correlation(
    rows: list[dict[str, Any]],
) -> tuple[float | None, bool, str]:
    """Spearman r between CME speed and transit time (should be negatively correlated).

    Returns (r, passed, message).  Passed when |r| >= threshold or too few pairs.
    Note: faster CMEs arrive sooner, so r is negative; we test |r| >= threshold.
    """
    pairs = [
        (r["cme_speed_kms"], r["transit_time_hours"])
        for r in rows
        if r.get("cme_speed_kms") is not None and r.get("transit_time_hours") is not None
        and r["transit_time_hours"] > 0
    ]
    if len(pairs) < MIN_PAIRS_FOR_CORR:
        msg = (
            f"speed↔transit: skipped (only {len(pairs)} paired rows, "
            f"need ≥{MIN_PAIRS_FOR_CORR})"
        )
        return None, True, msg
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    r = spearman_r(x, y)
    if r is None:
        return None, True, "speed↔transit: could not compute correlation"
    passed = abs(r) >= CORR_SPEED_TRANSIT_MIN
    direction = "PASS" if passed else "FAIL"
    msg = (
        f"speed↔transit: Spearman r={r:.3f} ({len(pairs)} pairs) "
        f"[threshold |r|≥{CORR_SPEED_TRANSIT_MIN}] — {direction}"
    )
    return r, passed, msg


def check_dst_bz_correlation(
    rows: list[dict[str, Any]],
) -> tuple[float | None, bool, str]:
    """Spearman r between Dst_min and ambient Bz_GSM (should be positively correlated:
    more negative Bz → more negative Dst).

    Returns (r, passed, message).
    """
    pairs = [
        (r["sw_bz_ambient"], r["dst_min_nt"])
        for r in rows
        if r.get("sw_bz_ambient") is not None and r.get("dst_min_nt") is not None
    ]
    if len(pairs) < MIN_PAIRS_FOR_CORR:
        msg = (
            f"Dst↔Bz: skipped (only {len(pairs)} paired rows, "
            f"need ≥{MIN_PAIRS_FOR_CORR})"
        )
        return None, True, msg
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    r = spearman_r(x, y)
    if r is None:
        return None, True, "Dst↔Bz: could not compute correlation"
    passed = r >= CORR_DST_BZ_MIN
    direction = "PASS" if passed else "FAIL"
    msg = (
        f"Dst↔Bz: Spearman r={r:.3f} ({len(pairs)} pairs) "
        f"[threshold r≥{CORR_DST_BZ_MIN}] — {direction}"
    )
    return r, passed, msg


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    speed_transit_r: float | None = None
    speed_transit_passed: bool = True
    speed_transit_msg: str = ""

    dst_bz_r: float | None = None
    dst_bz_passed: bool = True
    dst_bz_msg: str = ""

    speed_outlier_ids: list[str] = field(default_factory=list)
    density_sentinel_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    n_rows_checked: int = 0

    @property
    def passed(self) -> bool:
        return self.speed_transit_passed and self.dst_bz_passed

    def summary(self) -> str:
        lines = [
            f"=== Physical Consistency Validation ({self.n_rows_checked} rows) ===",
            self.speed_transit_msg,
            self.dst_bz_msg,
            f"Speed outliers: {len(self.speed_outlier_ids)}",
            f"Density sentinels: {len(self.density_sentinel_ids)}",
            f"Overall: {'PASS' if self.passed else 'FAIL'}",
        ]
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  {w}" for w in self.warnings[:20])
            if len(self.warnings) > 20:
                lines.append(f"  … and {len(self.warnings) - 20} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core validation (operates on list[dict])
# ---------------------------------------------------------------------------

def validate_feature_vectors(rows: list[dict[str, Any]]) -> ValidationReport:
    """Run all four consistency checks on a list of feature_vector dicts.

    This function is DB-agnostic: pass any list of dicts with the right keys.
    """
    report = ValidationReport(n_rows_checked=len(rows))

    # Correlation checks
    r, passed, msg = check_speed_transit_correlation(rows)
    report.speed_transit_r = r
    report.speed_transit_passed = passed
    report.speed_transit_msg = msg

    r, passed, msg = check_dst_bz_correlation(rows)
    report.dst_bz_r = r
    report.dst_bz_passed = passed
    report.dst_bz_msg = msg

    # Range checks
    outliers, warns = check_speed_range(rows)
    report.speed_outlier_ids = outliers
    report.warnings.extend(warns)

    sentinels, warns = check_density_range(rows)
    report.density_sentinel_ids = sentinels
    report.warnings.extend(warns)

    return report


# ---------------------------------------------------------------------------
# DB entry point
# ---------------------------------------------------------------------------

def run_validation(engine: Any) -> ValidationReport:
    """Load feature_vectors from DB and run all checks.

    Accepts an SQLAlchemy engine (sync).  Returns a ValidationReport.
    """
    cols = [
        FeatureVector.activity_id,
        FeatureVector.cme_speed_kms,
        FeatureVector.transit_time_hours,
        FeatureVector.dst_min_nt,
        FeatureVector.sw_bz_ambient,
        FeatureVector.sw_density_ambient,
    ]
    with engine.connect() as conn:
        result = conn.execute(sa.select(*cols))
        rows = [dict(row._mapping) for row in result]

    logger.info("run_validation: loaded %d feature_vector rows", len(rows))
    report = validate_feature_vectors(rows)

    if report.warnings:
        for w in report.warnings:
            logger.warning(w)
    logger.info(report.summary())
    return report
