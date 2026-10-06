"""Data quality thresholds, following the WHO Data Quality Review (DQR) toolkit.

Kept in one place so every number in a report can be traced to a rule here.
"""

from dataclasses import dataclass
from enum import StrEnum


class PairKind(StrEnum):
    DROPOUT = "dropout"  # the second can't exceed the first (e.g. Penta3 <= Penta1)
    RATIO = "ratio"  # the ratio should match the national ratio (e.g. OPV1 ~ Penta1)


@dataclass(frozen=True)
class IndicatorPair:
    check: str
    first: str  # data element names, as in DHIS2
    second: str
    kind: PairKind


# Related indicators compared by internal-consistency checks
INDICATOR_PAIRS = (
    IndicatorPair(
        "penta1_penta3_dropout", "Penta1 doses given", "Penta3 doses given", PairKind.DROPOUT
    ),
    IndicatorPair("anc1_anc4_dropout", "ANC 1st visit", "ANC 4th or more visits", PairKind.DROPOUT),
    # given at the same visit, so the counts should move together
    IndicatorPair("penta1_opv1_ratio", "Penta1 doses given", "OPV1 doses given", PairKind.RATIO),
)

# Tracer indicators for consistency over time (WHO DQR picks a few per programme)
TRACER_INDICATORS = (
    "ANC 1st visit",
    "ANC 4th or more visits",
    "Live births",
    "BCG doses given",
    "Penta1 doses given",
    "Penta3 doses given",
    "Measles doses given",
)


@dataclass(frozen=True)
class DQRules:
    # Completeness of reporting: share of expected reports received
    completeness_target: float = 0.80  # WHO DQR benchmark for facilities/districts
    completeness_critical: float = 0.50

    # Outliers, per facility x indicator over its own history
    min_months_for_stats: int = 6  # fewer points than this: no outlier test
    extreme_sd: float = 3.0  # WHO DQR "extreme outlier": >= 3 SD from the mean
    robust_modified_z: float = 3.5  # Iglewicz & Hoaglin modified z-score (median/MAD)

    # Consistency over time: district yearly total vs mean of up to 3 previous years
    time_consistency_tolerance: float = 0.33  # WHO DQR: flag beyond +/-33%
    time_consistency_years: int = 3
    min_months_compared: int = 6

    # Internal consistency
    ratio_tolerance: float = 0.10  # district ratio vs national ratio, +/-10%
    min_months_per_year: int = 3  # months with both indicators reported

    # Repeated values (signs of copied rather than counted data)
    repeat_min_months: int = 3  # same value this many months in a row
    repeat_high_months: int = 6
    repeat_min_value: float = 5  # small counts repeat naturally (1, 2, 3...)
    copy_min_months: int = 6  # months comparable with the same month last year
    copy_share: float = 0.5  # share of values identical to last year's
    copy_share_high: float = 0.8


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
