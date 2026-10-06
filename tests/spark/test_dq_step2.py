"""DQ engine tests: internal consistency, consistency over time, system checks."""

from collections.abc import Iterable
from datetime import datetime

import pytest
from pyspark.sql import DataFrame, SparkSession

from hmis_dq.spark.dq import consistency, system
from hmis_dq.spark.dq.rules import IndicatorPair, PairKind, Severity

pytestmark = pytest.mark.spark

FM_SCHEMA = (
    "dataset_id string, district_id string, district string, org_unit_id string, "
    "facility string, data_element_id string, data_element string, period string, "
    "year int, value double"
)
PENTA = IndicatorPair("penta_dropout", "Penta1", "Penta3", PairKind.DROPOUT)
OPV = IndicatorPair("opv_ratio", "Penta1", "OPV1", PairKind.RATIO)


def fm_rows(
    ou: str, element: str, values: Iterable[float | None], *, year: int = 2025, district: str = "d1"
) -> list[tuple[object, ...]]:
    return [
        (
            "ds1",
            district,
            district.upper(),
            ou,
            ou.upper(),
            element,
            element,
            f"{year}{m:02d}",
            year,
            float(v),
        )
        for m, v in enumerate(values, start=1)
        if v is not None
    ]


def frame(spark: SparkSession, *groups: list[tuple[object, ...]]) -> DataFrame:
    return spark.createDataFrame([row for group in groups for row in group], FM_SCHEMA)


# ------------------------------------------------------- internal consistency


def test_negative_dropout_over_a_year_is_flagged(spark: SparkSession) -> None:
    fm = frame(
        spark,
        fm_rows("f1", "Penta1", [10, 10, 10, 10]),
        fm_rows("f1", "Penta3", [12, 12, 12, 12]),  # 48 finished, only 40 started
    )

    found = consistency.dropout_findings(fm, PENTA).collect()

    levels = {(r["facility"], r["severity"]) for r in found}
    assert levels == {("F1", Severity.MEDIUM), (None, Severity.HIGH)}  # facility + district
    facility = next(r for r in found if r["facility"])
    assert (facility["value"], facility["expected"], facility["period"]) == (48.0, 40.0, "2025")
    assert facility["score"] == pytest.approx(-0.2)
    assert "exceeds" in facility["message"]


def test_one_month_above_is_fine_if_the_year_adds_up(spark: SparkSession) -> None:
    fm = frame(
        spark,
        fm_rows("f1", "Penta1", [10, 10, 10, 10]),
        fm_rows("f1", "Penta3", [14, 8, 8, 8]),  # January high, the year is 38 <= 40
    )

    assert consistency.dropout_findings(fm, PENTA).count() == 0


def test_months_missing_one_indicator_are_not_counted(spark: SparkSession) -> None:
    fm = frame(
        spark,
        fm_rows("f1", "Penta1", [10, 10, 10, None]),  # April Penta1 report missing
        fm_rows("f1", "Penta3", [9, 9, 9, 30]),
    )

    # with April compared (40 vs 57) this would be a false negative drop-out
    assert consistency.dropout_findings(fm, PENTA).count() == 0


def test_ratio_far_from_national_is_flagged(spark: SparkSession) -> None:
    fm = frame(
        spark,
        fm_rows("a", "Penta1", [100, 100, 100], district="d1"),
        fm_rows("a", "OPV1", [100, 100, 100], district="d1"),
        fm_rows("b", "Penta1", [100, 100, 100], district="d2"),
        fm_rows("b", "OPV1", [101, 99, 100], district="d2"),
        fm_rows("c", "Penta1", [100, 100, 100], district="d3"),
        fm_rows("c", "OPV1", [80, 80, 80], district="d3"),  # 0.8 vs national 0.93: -14%
    )

    found = consistency.ratio_findings(fm, OPV).collect()

    assert [r["district_id"] for r in found] == ["d3"]
    assert found[0]["value"] == pytest.approx(0.8)


# --------------------------------------------------- consistency over time

DM_SCHEMA = (
    "dataset_id string, district_id string, district string, data_element_id string, "
    "data_element string, period string, year int, value double"
)


def dm_rows(year: int, values: Iterable[float], district: str = "d1") -> list[tuple[object, ...]]:
    return [
        (
            "ds1",
            district,
            "Bo",
            "de1",
            "Penta1 doses given",
            f"{year}{m:02d}",
            year,
            float(v) if v is not None else None,
        )
        for m, v in enumerate(values, start=1)
    ]


def test_year_far_above_previous_years_is_flagged(spark: SparkSession) -> None:
    dm = spark.createDataFrame(
        dm_rows(2023, [100] * 12) + dm_rows(2024, [100] * 12) + dm_rows(2025, [150] * 12),
        DM_SCHEMA,
    )

    found = consistency.time_consistency_findings(dm).collect()

    assert [r["period"] for r in found] == ["2025"]
    assert found[0]["score"] == pytest.approx(1.5)
    assert "2 previous year(s)" in found[0]["message"]


def test_partial_year_is_compared_on_the_same_months(spark: SparkSession) -> None:
    # 2026 has only 6 months; comparing with 12 months of 2025 would look like -50%
    dm = spark.createDataFrame(dm_rows(2025, [100] * 12) + dm_rows(2026, [100] * 6), DM_SCHEMA)

    assert consistency.time_consistency_findings(dm).count() == 0


# ------------------------------------------------------------------ system


def test_repeated_runs_of_the_same_value(spark: SparkSession) -> None:
    fm = frame(
        spark,
        fm_rows("f1", "BCG", [24, 24, 24, 24, 31, 20]),  # 4 months of 24
        fm_rows("f2", "BCG", [2, 2, 2, 2, 2, 2]),  # small counts repeat naturally
        fm_rows("f3", "BCG", [9, 9, None, 9, 9, 9]),  # gap in March: runs of 2 and 3
    )

    found = {r["org_unit_id"]: r for r in system.repeated_values_findings(fm).collect()}

    assert set(found) == {"f1", "f3"}
    assert (found["f1"]["score"], found["f1"]["severity"]) == (4.0, Severity.MEDIUM)
    assert found["f1"]["message"] == "BCG = 24 for 4 months in a row (202501 to 202504)"
    assert found["f3"]["period"] == "202504"


def test_long_runs_are_high_severity(spark: SparkSession) -> None:
    fm = frame(spark, fm_rows("f1", "BCG", [24] * 7))

    (found,) = system.repeated_values_findings(fm).collect()

    assert found["severity"] == Severity.HIGH


def test_values_copied_from_last_year(spark: SparkSession) -> None:
    pattern = [25.0, 19, 20, 14, 20, 9, 22, 24]
    fm = frame(
        spark,
        fm_rows("copy", "Penta1", pattern, year=2024),
        fm_rows("copy", "Penta1", pattern, year=2025),  # identical: copied
        fm_rows("real", "Penta1", pattern, year=2024),
        fm_rows("real", "Penta1", [v + 1 for v in pattern], year=2025),
    )

    found = system.repeats_last_year_findings(fm).collect()

    assert [(r["org_unit_id"], r["period"], r["severity"]) for r in found] == [
        ("copy", "2025", Severity.HIGH)
    ]
    assert (
        found[0]["message"]
        == "2025: 8 of 8 values (100%) are identical to the same month last year"
    )


def test_last_updated_before_created(spark: SparkSession) -> None:
    values = spark.createDataFrame(
        [
            ("ds1", "f1", datetime(2022, 9, 5), datetime(2010, 3, 5)),
            ("ds1", "f1", datetime(2025, 2, 1), datetime(2025, 2, 3)),
        ],
        "dataset_id string, org_unit_id string, created timestamp, last_updated timestamp",
    )
    units = spark.createDataFrame(
        [("f1", "Facility 1", "d1", "Bo")],
        "org_unit_id string, name string, district_id string, district string",
    )

    (found,) = system.last_updated_before_created_findings(values, units).collect()

    assert (found["facility"], found["score"]) == ("Facility 1", 0.5)
    assert found["message"] == "1 of 2 values were last updated before they were created"


def test_entered_before_period_end(spark: SparkSession) -> None:
    reports = spark.createDataFrame(
        [
            ("ds1", "f1", "F1", "d1", "Bo", True, True, -5870),
            ("ds1", "f1", "F1", "d1", "Bo", True, False, 10),
            ("ds1", "f1", "F1", "d1", "Bo", False, None, None),
        ],
        "dataset_id string, org_unit_id string, facility string, district_id string, "
        "district string, reported boolean, entered_before_period_end boolean, "
        "days_after_period_end int",
    )

    (found,) = system.entered_before_period_end_findings(reports).collect()

    assert (found["value"], found["expected"]) == (1.0, 2.0)
    assert "up to 5870 days early" in found["message"]


def test_missing_coordinates_only_for_facilities_expected_to_report(spark: SparkSession) -> None:
    units = spark.createDataFrame(
        [
            ("f1", "Has GPS", "d1", "Bo", True, -12.0),
            ("f2", "No GPS", "d1", "Bo", True, None),
            ("f3", "No GPS, not assigned", "d1", "Bo", True, None),
            ("d1", "Bo", "d1", "Bo", False, None),
        ],
        "org_unit_id string, name string, district_id string, district string, "
        "is_facility boolean, longitude double",
    )
    expected = spark.createDataFrame([("f1",), ("f2",)], "org_unit_id string")

    found = system.missing_coordinates_findings(units, expected).collect()

    assert [(r["org_unit_id"], r["facility"]) for r in found] == [("f2", "No GPS")]
