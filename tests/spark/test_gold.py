"""Gold transform tests on small silver-shaped tables."""

from datetime import date, datetime
from typing import Any

import pytest
from pyspark.sql import DataFrame, SparkSession

from hmis_dq.spark import gold

pytestmark = pytest.mark.spark

VALUES_SCHEMA = (
    "dataset_id string, data_element_id string, period string, period_start date, year int, "
    "org_unit_id string, category_option_combo_id string, value double, value_status string, "
    "created timestamp, last_updated timestamp"
)
UNITS_SCHEMA = (
    "org_unit_id string, name string, chiefdom_id string, chiefdom string, district_id string, "
    "district string, opening_date date, closed_date date, is_facility boolean"
)


def value(
    ou: str, month: int, val: float | None, *, de: str = "de1", coc: str = "c1", **kw: Any
) -> tuple[Any, ...]:
    entered = kw.get(
        "created", datetime(2025, month + 1, 5) if month < 12 else datetime(2026, 1, 5)
    )
    return (
        kw.get("ds", "ds1"),
        de,
        f"2025{month:02d}",
        date(2025, month, 1),
        2025,
        ou,
        coc,
        val,
        kw.get("status", "ok" if val is not None else "missing"),
        entered,
        entered,
    )


@pytest.fixture
def units(spark: SparkSession) -> DataFrame:
    rows = [
        ("f1", "Facility 1", "c1", "Chiefdom 1", "d1", "District 1", None, None, True),
        ("f2", "Facility 2", "c1", "Chiefdom 1", "d1", "District 1", None, None, True),
        # opened in March: not expected to report before then
        ("f3", "Facility 3", "c2", "Chiefdom 2", "d1", "District 1", date(2025, 3, 15), None, True),
        # the national unit, wrongly assigned the dataset: never expected to report
        ("nat", "Sierra Leone", None, None, None, None, None, None, False),
    ]
    return spark.createDataFrame(rows, UNITS_SCHEMA)


@pytest.fixture
def elements(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        [("de1", "Penta1"), ("de2", "BCG")], "data_element_id string, name string"
    )


@pytest.fixture
def assignments(spark: SparkSession) -> DataFrame:
    rows = [("ds1", "f1"), ("ds1", "f2"), ("ds1", "f3"), ("ds1", "nat")]
    return spark.createDataFrame(rows, "dataset_id string, org_unit_id string")


@pytest.fixture
def datasets(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame([("ds1", 0)], "dataset_id string, timely_days int")


def test_facility_month_sums_disaggregations_and_adds_names(
    spark: SparkSession, elements: DataFrame, units: DataFrame
) -> None:
    values = spark.createDataFrame(
        [
            value("f1", 1, 5.0, coc="male"),
            value("f1", 1, 7.0, coc="female"),
            value("f1", 1, None, coc="unknown"),  # not ok: excluded from the sum
        ],
        VALUES_SCHEMA,
    )

    (row,) = gold.facility_month(values, elements, units).collect()

    assert row["value"] == 12.0
    assert row["disaggregations_reported"] == 2
    assert (row["data_element"], row["facility"], row["district"]) == (
        "Penta1",
        "Facility 1",
        "District 1",
    )


def test_reporting_has_a_row_for_every_expected_report(
    spark: SparkSession, units: DataFrame, assignments: DataFrame, datasets: DataFrame
) -> None:
    # data for Jan-Apr; f1 reports every month, f2 only in January, f3 never
    values = spark.createDataFrame(
        [value("f1", m, 1.0) for m in (1, 2, 3, 4)] + [value("f2", 1, 1.0)], VALUES_SCHEMA
    )

    rows = gold.reporting(values, assignments, units, datasets).collect()
    grid = {(r["org_unit_id"], r["period"]): r["reported"] for r in rows}

    # f1, f2: 4 months each; f3 opened 15 March, so expected for March and April only
    assert len(rows) == 10
    assert grid[("f2", "202502")] is False  # a missing report is a row, not an absence
    assert grid[("f3", "202503")] is False
    assert ("f3", "202502") not in grid
    assert sum(grid.values()) == 5
    missing = next(r for r in rows if (r["org_unit_id"], r["period"]) == ("f2", "202502"))
    assert missing["on_time"] is False  # expected but never sent: not on time


def test_reporting_window_starts_at_the_datasets_first_month(
    spark: SparkSession, units: DataFrame, assignments: DataFrame, datasets: DataFrame
) -> None:
    # ds1's first data is in March, so January and February aren't expected
    values = spark.createDataFrame([value("f1", 3, 1.0), value("f1", 4, 1.0)], VALUES_SCHEMA)

    periods = {r["period"] for r in gold.reporting(values, assignments, units, datasets).collect()}

    assert periods == {"202503", "202504"}


def test_on_time_uses_the_default_deadline_when_unset(
    spark: SparkSession, units: DataFrame, assignments: DataFrame, datasets: DataFrame
) -> None:
    values = spark.createDataFrame(
        [
            value("f1", 1, 1.0, created=datetime(2025, 2, 10)),  # 10 days after period end
            value("f2", 1, 1.0, created=datetime(2025, 3, 20)),  # 48 days after
        ],
        VALUES_SCHEMA,
    )

    rows = {
        r["org_unit_id"]: r for r in gold.reporting(values, assignments, units, datasets).collect()
    }

    assert rows["f1"]["deadline_days"] == gold.DHIS2_DEFAULT_TIMELY_DAYS
    assert (rows["f1"]["days_after_period_end"], rows["f1"]["on_time"]) == (10, True)
    assert (rows["f2"]["days_after_period_end"], rows["f2"]["on_time"]) == (48, False)


def test_entry_before_the_period_ended_makes_timeliness_unknown(
    spark: SparkSession, units: DataFrame, assignments: DataFrame, datasets: DataFrame
) -> None:
    # the demo data's pattern: a January 2025 value "created" in 2022
    values = spark.createDataFrame(
        [value("f1", 1, 1.0, created=datetime(2022, 9, 5))], VALUES_SCHEMA
    )

    rows = {
        r["org_unit_id"]: r for r in gold.reporting(values, assignments, units, datasets).collect()
    }

    assert rows["f1"]["entered_before_period_end"] is True
    assert rows["f1"]["on_time"] is None  # not "on time": we can't know


def test_district_month_totals_and_completeness(
    spark: SparkSession,
    elements: DataFrame,
    units: DataFrame,
    assignments: DataFrame,
    datasets: DataFrame,
) -> None:
    values = spark.createDataFrame(
        [value("f1", 1, 10.0), value("f2", 1, 5.0), value("f1", 2, 8.0)], VALUES_SCHEMA
    )
    fm = gold.facility_month(values, elements, units)
    rep = gold.reporting(values, assignments, units, datasets)

    rows = {r["period"]: r for r in gold.district_month(fm, rep).collect()}

    jan, feb = rows["202501"], rows["202502"]
    assert (jan["value"], jan["facilities_reporting_element"]) == (15.0, 2)
    # f3 isn't open yet, so 2 reports expected each month
    assert (jan["reports_expected"], jan["reports_received"], jan["completeness"]) == (2, 2, 1.0)
    assert (feb["reports_received"], feb["completeness"]) == (1, 0.5)
