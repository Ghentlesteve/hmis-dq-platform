"""Data quality thresholds, following the WHO Data Quality Review (DQR) toolkit.

Kept in one place so every number in a report can be traced to a rule here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DQRules:
    # Completeness of reporting: share of expected reports received
    completeness_target: float = 0.80  # WHO DQR benchmark for facilities/districts
    completeness_critical: float = 0.50

    # Outliers, per facility x indicator over its own history
    min_months_for_stats: int = 6  # fewer points than this: no outlier test
    extreme_sd: float = 3.0  # WHO DQR "extreme outlier": >= 3 SD from the mean
    robust_modified_z: float = 3.5  # Iglewicz & Hoaglin modified z-score (median/MAD)


DEFAULT_RULES = DQRules()


class Dimension:
    """WHO DQR dimensions each check belongs to."""

    COMPLETENESS = "completeness"
    TIMELINESS = "timeliness"
    OUTLIERS = "outliers"
    CONSISTENCY_TIME = "consistency_over_time"
    CONSISTENCY_INTERNAL = "internal_consistency"
    SYSTEM = "system"


class Severity:
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
