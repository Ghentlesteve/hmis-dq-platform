"""Gold layer: analysis-ready tables built from silver.

- ``facility_month``: facility x indicator x month, disaggregations summed, with names.
- ``reporting``: one row for every report a facility was *expected* to send
  (dataset x facility x month), whether or not it arrived. Missing reports are
  rows with ``reported = false``, which is what makes completeness computable.
- ``district_month``: district totals per indicator and month, with the
  dataset's reporting completeness alongside, for the map and the forecasts.

Reporting window: only facilities are expected to report. A dataset is expected
from the first month it has any data
(the demo database starts some datasets later than others) up to the latest
month extracted, and only from facilities that were open in that month.

Timeliness proxy: DHIS2 measures timeliness from the form's "complete"
registration, which isn't extracted. Here a report counts as on time when its
first value was entered within the dataset's deadline after the period ended.
An entry date *before* the period ended is impossible, so timeliness is unknown
(null) for that report and ``entered_before_period_end`` says why.
"""

from dataclasses import dataclass

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from hmis_dq.config import Settings
from hmis_dq.spark.session import build_spark, s3a_url
from hmis_dq.spark.silver import ValueStatus

DHIS2_DEFAULT_TIMELY_DAYS = 15  # DHIS2 uses 15 days when a dataset doesn't set its own


def _facility_columns() -> list[Column | str]:
    # a function, not a constant: Spark 4 can only build Columns once a session exists
    return [
        "org_unit_id",
        F.col("name").alias("facility"),
        "chiefdom_id",
        "chiefdom",
        "district_id",
        "district",
    ]


def facility_month(values: DataFrame, elements: DataFrame, units: DataFrame) -> DataFrame:
    """Sum every indicator over its disaggregations, per facility and month."""
    usable = values.filter(F.col("value_status") == ValueStatus.OK)
    totals = usable.groupBy(
        "dataset_id", "data_element_id", "org_unit_id", "period", "period_start", "year"
    ).agg(
        F.sum("value").alias("value"),
        F.count("*").alias("disaggregations_reported"),
        F.min("created").alias("first_entered_at"),
        F.max("last_updated").alias("last_updated_at"),
    )
    names = elements.select("data_element_id", F.col("name").alias("data_element"))
    return totals.join(F.broadcast(names), "data_element_id", "left").join(
        F.broadcast(units.select(*_facility_columns())), "org_unit_id", "left"
    )


def _expected_reports(
    values: DataFrame, assignments: DataFrame, units: DataFrame, datasets: DataFrame
) -> DataFrame:
    """dataset x facility x month for every report that should exist."""
    last_month = values.agg(F.max("period_start").alias("last_month"))
    windows = (
        values.groupBy("dataset_id")
        .agg(F.min("period_start").alias("first_month"))
        .crossJoin(last_month)
        .select(
            "dataset_id",
            F.explode(F.sequence("first_month", "last_month", F.expr("interval 1 month"))).alias(
                "period_start"
            ),
        )
    )
    # Only facilities send reports. DHIS2 configs sometimes assign a dataset to a
    # district or the whole country too; those are reported as a DQ finding instead.
    open_units = units.filter(F.col("is_facility")).select(
        *_facility_columns(), "opening_date", "closed_date"
    )
    period_end = F.last_day("period_start")
    deadlines = datasets.select(
        "dataset_id",
        F.when(F.col("timely_days") > 0, F.col("timely_days"))
        .otherwise(DHIS2_DEFAULT_TIMELY_DAYS)
        .alias("deadline_days"),
    )
    return (
        windows.join(assignments, "dataset_id")
        .join(open_units, "org_unit_id")
        .filter(F.col("opening_date").isNull() | (F.col("opening_date") <= period_end))
        .filter(F.col("closed_date").isNull() | (F.col("closed_date") >= F.col("period_start")))
        .drop("opening_date", "closed_date")
        .join(F.broadcast(deadlines), "dataset_id", "left")
        .withColumn("period", F.date_format("period_start", "yyyyMM"))
        .withColumn("year", F.year("period_start"))
    )


def reporting(
    values: DataFrame, assignments: DataFrame, units: DataFrame, datasets: DataFrame
) -> DataFrame:
    expected = _expected_reports(values, assignments, units, datasets)
    received = values.groupBy("dataset_id", "org_unit_id", "period").agg(
        F.count("*").alias("values_reported"),
        F.countDistinct("data_element_id").alias("data_elements_reported"),
        F.min("created").alias("first_entered_at"),
        F.max("last_updated").alias("last_updated_at"),
    )
    days_late = F.datediff(F.to_date("first_entered_at"), F.last_day("period_start"))
    return (
        expected.join(received, ["dataset_id", "org_unit_id", "period"], "left")
        .withColumn("reported", F.col("values_reported").isNotNull())
        .fillna(0, subset=["values_reported", "data_elements_reported"])
        .withColumn("days_after_period_end", days_late)
        .withColumn("entered_before_period_end", F.col("days_after_period_end") < 0)
        .withColumn(
            "on_time",
            F.when(~F.col("reported"), False)
            .when(F.col("entered_before_period_end"), None)  # impossible date: unknown
            .otherwise(F.col("days_after_period_end") <= F.col("deadline_days")),
        )
    )


def district_month(facility_months: DataFrame, reports: DataFrame) -> DataFrame:
    totals = facility_months.groupBy(
        "dataset_id",
        "district_id",
        "district",
        "data_element_id",
        "data_element",
        "period",
        "period_start",
        "year",
    ).agg(
        F.sum("value").alias("value"),
        F.countDistinct("org_unit_id").alias("facilities_reporting_element"),
    )
    completeness = reports.groupBy("dataset_id", "district_id", "period").agg(
        F.count("*").alias("reports_expected"),
        F.sum(F.col("reported").cast("int")).alias("reports_received"),
        F.sum(F.col("on_time").cast("int")).alias("reports_on_time"),
        F.sum(F.col("on_time").isNull().cast("int")).alias("reports_timeliness_unknown"),
    )
    return totals.join(completeness, ["dataset_id", "district_id", "period"], "left").withColumn(
        "completeness", F.round(F.col("reports_received") / F.col("reports_expected"), 4)
    )


# ----------------------------------------------------------------------- IO


@dataclass(frozen=True)
class GoldResult:
    row_counts: dict[str, int]
    completeness: list[tuple[str, int, int, float]]  # dataset, expected, received, rate


def run_gold(settings: Settings) -> GoldResult:
    spark = build_spark(settings, "hmis-gold")

    def silver(table: str) -> DataFrame:
        return spark.read.parquet(s3a_url(settings.silver_bucket, f"dhis2/{table}/"))

    def gold_path(table: str) -> str:
        return s3a_url(settings.gold_bucket, f"dhis2/{table}/")

    values = silver("data_values").cache()
    units = silver("org_units").cache()
    elements, datasets = silver("data_elements"), silver("data_sets")
    assignments = silver("dataset_org_units")

    fm = facility_month(values, elements, units)
    fm.write.mode("overwrite").partitionBy("dataset_id", "year").parquet(
        gold_path("facility_month")
    )
    fm = spark.read.parquet(gold_path("facility_month"))

    rep = reporting(values, assignments, units, datasets)
    rep.write.mode("overwrite").partitionBy("dataset_id", "year").parquet(gold_path("reporting"))
    rep = spark.read.parquet(gold_path("reporting"))

    district_month(fm, rep).coalesce(1).write.mode("overwrite").parquet(gold_path("district_month"))

    counts = {
        table: spark.read.parquet(gold_path(table)).count()
        for table in ("facility_month", "reporting", "district_month")
    }
    summary = (
        rep.groupBy("dataset_id")
        .agg(F.count("*").alias("expected"), F.sum(F.col("reported").cast("int")).alias("received"))
        .orderBy("dataset_id")
        .collect()
    )
    completeness = [
        (r["dataset_id"], r["expected"], r["received"], r["received"] / r["expected"])
        for r in summary
    ]
    return GoldResult(row_counts=counts, completeness=completeness)
