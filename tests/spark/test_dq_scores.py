"""Score tests, with every expected number worked out by hand in the comments."""

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from hmis_dq.spark.dq import scores

pytestmark = pytest.mark.spark

RATES = (
    "dataset_id string, district_id string, district string, org_unit_id string, "
    "facility string, reports_expected long, reports_received long, completeness double"
)
FINDINGS = (
    "check string, dimension string, severity string, dataset_id string, "
    "org_unit_id string, facility string, period string"
)
KEYED = "dataset_id string, org_unit_id string"


@pytest.fixture
def scored(spark: SparkSession) -> dict[str, dict[str, object]]:
    rates = spark.createDataFrame(
        [
            ("ds1", "d1", "Bo", "full", "Full CHC", 10, 10, 1.0),
            ("ds1", "d1", "Bo", "never", "Never MCHP", 30, 0, 0.0),
        ],
        RATES,
    )
    # 100 values over 2 years for "full"; nothing for "never"
    months = spark.createDataFrame(
        [("ds1", "full", 2024 + i % 2) for i in range(100)], f"{KEYED}, year int"
    )
    outlier = ("outlier", "outliers")
    findings = spark.createDataFrame(
        [(*outlier, "high", "ds1", "full", "Full CHC", "202501")]
        + [(*outlier, "medium", "ds1", "full", "Full CHC", "202502")] * 2
        + [(*outlier, "low", "ds1", "full", "Full CHC", "202503")] * 10
        # one inconsistent year (counted once even with two pair checks)
        + [("penta", "internal_consistency", "medium", "ds1", "full", "Full CHC", "2025")]
        + [("anc", "internal_consistency", "medium", "ds1", "full", "Full CHC", "2025")]
        # district-level row: no facility, must not count against "full"
        + [("penta", "internal_consistency", "high", "ds1", None, None, "2024")],
        FINDINGS,
    )
    copies = spark.createDataFrame(
        [("ds1", "full", 30, 60)], f"{KEYED}, identical long, compared long"
    )
    timestamps = spark.createDataFrame([("ds1", "full", 20, 100)], f"{KEYED}, bad long, total long")
    early = spark.createDataFrame([("ds1", "full", 10, 10)], f"{KEYED}, early long, received long")

    result = scores.facility_scores(
        rates, months, findings, copies=copies, timestamps=timestamps, early_entries=early
    )
    return {r["org_unit_id"]: r.asDict() for r in result.collect()}


def test_each_dimension_is_explainable_from_counts(scored: dict[str, dict[str, object]]) -> None:
    full = scored["full"]

    assert full["completeness"] == 100.0  # 10 of 10 reports
    assert full["accuracy"] == 97.0  # outlier weight 1 + 2x0.5 + 10x0.1 = 3 of 100 values
    assert full["consistency"] == 50.0  # 1 inconsistent year of 2 reported
    assert full["integrity"] == 43.3  # 1 - mean(30/60, 20/100, 10/10) = 1 - 0.5667


def test_overall_is_the_weighted_mean_and_graded(scored: dict[str, dict[str, object]]) -> None:
    # 100x0.35 + 97x0.25 + 50x0.20 + 43.3x0.20 = 77.91
    assert scored["full"]["overall"] == 77.9
    assert scored["full"]["grade"] == "B"


def test_never_reporting_facility_scores_zero_without_fake_perfect_dimensions(
    scored: dict[str, dict[str, object]],
) -> None:
    never = scored["never"]

    assert never["completeness"] == 0.0
    assert never["accuracy"] is None  # nothing to measure, not 100
    assert never["consistency"] is None
    assert never["integrity"] is None
    assert (never["overall"], never["grade"]) == (0.0, "D")


def test_district_rollup_weights_by_expected_reports_and_skips_unmeasured(
    spark: SparkSession, scored: dict[str, dict[str, object]]
) -> None:
    facilities = spark.createDataFrame(list(scored.values()))

    (district,) = scores.district_scores(facilities).collect()

    assert district["completeness"] == 25.0  # (10 + 0) of (10 + 30), from the totals
    assert district["accuracy"] == 97.0  # "never" has no accuracy: not averaged in as 0 or 100
    assert (district["facilities"], district["facilities_grade_d"]) == (2, 1)


@pytest.mark.parametrize(
    ("score", "expected"), [(95.0, "A"), (90.0, "A"), (89.9, "B"), (60.0, "C"), (59.9, "D")]
)
def test_grade_thresholds(spark: SparkSession, score: float, expected: str) -> None:
    frame: DataFrame = spark.createDataFrame([(score,)], "s double")

    assert frame.select(scores.grade(F.col("s"))).first()[0] == expected  # type: ignore[index]
