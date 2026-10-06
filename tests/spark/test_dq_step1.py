"""DQ engine tests: findings format, completeness/timeliness, outliers."""

from collections.abc import Sequence

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from hmis_dq.spark.dq.findings import FINDING_COLUMNS, to_findings
from hmis_dq.spark.dq.outliers import outlier_findings, outlier_scores
from hmis_dq.spark.dq.reporting import facility_reporting, reporting_findings
from hmis_dq.spark.dq.rules import DQRules, Severity

pytestmark = pytest.mark.spark

REPORTS_SCHEMA = (
    "dataset_id string, district_id string, district string, org_unit_id string, "
    "facility string, year int, period string, reported boolean, on_time boolean"
)
FM_SCHEMA = (
    "dataset_id string, district_id string, district string, org_unit_id string, "
    "facility string, data_element_id string, data_element string, period string, value double"
)


def reports(spark: SparkSession, rows: Sequence[tuple[str, int, bool, bool | None]]) -> DataFrame:
    """rows: (facility, month, reported, on_time)"""
    data = [
        ("ds1", "d1", "District 1", ou, f"Facility {ou}", 2025, f"2025{m:02d}", rep, on_time)
        for ou, m, rep, on_time in rows
    ]
    return spark.createDataFrame(data, REPORTS_SCHEMA)


def series(spark: SparkSession, values: list[float], ou: str = "f1") -> DataFrame:
    data = [
        ("ds1", "d1", "District 1", ou, "Facility 1", "de1", "Penta1", f"2025{m:02d}", v)
        for m, v in enumerate(values, start=1)
    ]
    return spark.createDataFrame(data, FM_SCHEMA)


# -------------------------------------------------------------- findings


def test_to_findings_fills_missing_columns_with_typed_nulls(spark: SparkSession) -> None:
    frame = spark.createDataFrame([("f1", "x")], "org_unit_id string, unrelated string")

    out = to_findings(
        frame, check="c", dimension="d", severity="low", message=F.lit("m"), score=F.lit(1)
    )

    assert out.columns == list(FINDING_COLUMNS)
    (row,) = out.collect()
    assert (row["org_unit_id"], row["district"], row["score"]) == ("f1", None, 1.0)
    assert dict(out.dtypes)["score"] == "double"


# ------------------------------------------------- completeness/timeliness


def test_facility_rates(spark: SparkSession) -> None:
    rows = [
        ("f1", 1, True, True),
        ("f1", 2, True, False),
        ("f1", 3, True, None),  # received, impossible entry date
        ("f1", 4, False, False),
    ]

    (rate,) = facility_reporting(reports(spark, rows)).collect()

    assert (rate["reports_expected"], rate["reports_received"]) == (4, 3)
    assert rate["completeness"] == 0.75
    assert rate["timeliness_unknown"] == 1
    assert rate["timeliness"] == 0.5  # 1 on time of the 2 with a usable date


def test_timeliness_is_null_when_no_date_is_usable(spark: SparkSession) -> None:
    (rate,) = facility_reporting(reports(spark, [("f1", 1, True, None)])).collect()

    assert rate["timeliness"] is None


def test_reporting_findings_split_never_and_low(spark: SparkSession) -> None:
    rows = (
        [("good", m, True, True) for m in range(1, 11)]  # 100%
        + [("never", m, False, False) for m in range(1, 11)]  # 0%
        + [("low", m, m <= 3, None) for m in range(1, 11)]  # 30%: below critical 50%
        + [("medium", m, m <= 6, None) for m in range(1, 11)]  # 60%: below target 80%
    )
    rates = facility_reporting(reports(spark, rows))

    found = {r["org_unit_id"]: r for r in reporting_findings(rates).collect()}

    assert set(found) == {"never", "low", "medium"}
    assert (found["never"]["check"], found["never"]["severity"]) == ("never_reported", "high")
    assert found["low"]["severity"] == Severity.HIGH
    assert found["medium"]["severity"] == Severity.MEDIUM
    assert found["medium"]["message"] == "Sent 6 of 10 expected reports (60%)"
    assert found["medium"]["expected"] == 0.8


# --------------------------------------------------------------- outliers


STEADY = [30.0, 32.0, 29.0, 31.0, 30.0, 28.0, 33.0, 30.0, 31.0, 29.0, 30.0]


def test_a_spike_is_flagged_by_both_methods(spark: SparkSession) -> None:
    found = outlier_findings(series(spark, [*STEADY, 900.0])).collect()

    assert len(found) == 1
    (spike,) = found
    assert (spike["value"], spike["period"], spike["severity"]) == (900.0, "202512", "high")
    assert spike["expected"] == 30.0  # the median: typical value for this facility
    assert spike["message"].startswith("Penta1 = 900, typical 30")


def test_robust_method_catches_what_the_mean_misses(spark: SparkSession) -> None:
    # Two huge values inflate the mean and SD so neither is 3 SD away,
    # but the median/MAD barely move, so both are still flagged.
    scored = {
        r["period"]: r
        for r in outlier_scores(series(spark, [*STEADY[:10], 500.0, 520.0])).collect()
    }

    spike = scored["202512"]
    assert abs(spike["z"]) < 3
    assert spike["modified_z"] > 3.5
    found = outlier_findings(series(spark, [*STEADY[:10], 500.0, 520.0])).collect()
    assert {r["period"] for r in found} == {"202511", "202512"}
    assert {r["severity"] for r in found} == {Severity.LOW}  # robust method only


def test_steady_series_has_no_outliers(spark: SparkSession) -> None:
    assert outlier_findings(series(spark, STEADY)).count() == 0


def test_short_series_is_not_tested(spark: SparkSession) -> None:
    rules = DQRules(min_months_for_stats=6)

    assert outlier_findings(series(spark, [30.0, 31.0, 900.0]), rules).count() == 0


def test_uncomputable_score_reads_na_not_zero(spark: SparkSession) -> None:
    # 10 identical values make the MAD 0; the spike is still 3+ SD from the mean
    (found,) = outlier_findings(series(spark, [*([30.0] * 10), 90.0])).collect()

    assert found["severity"] == Severity.MEDIUM  # SD method only
    assert found["message"].endswith("modified z n/a)")


def test_constant_series_does_not_divide_by_zero(spark: SparkSession) -> None:
    scored = outlier_scores(series(spark, [5.0] * 8)).collect()

    assert all(r["z"] is None and r["modified_z"] is None for r in scored)
