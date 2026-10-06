"""System-level checks: signs that values weren't captured the way they should be.

- last_updated_before_created: a value edited before it existed (import/sync errors)
- entered_before_period_end: a report entered before its month was over
- missing_coordinates: facilities that can't be placed on a map
- non_facility_assignment: a dataset assigned to a district/country, not a facility
- repeated_values: the same value several months running (copy-forward)
- repeats_last_year: most of a year's values identical to the same month last year
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from hmis_dq.spark.dq.findings import to_findings
from hmis_dq.spark.dq.rules import DEFAULT_RULES, Dimension, DQRules, Severity

FACILITY_NAMES = ("org_unit_id", "facility", "district_id", "district")


def _facility_names(units: DataFrame) -> DataFrame:
    return units.select("org_unit_id", F.col("name").alias("facility"), "district_id", "district")


def last_updated_counts(values: DataFrame) -> DataFrame:
    """Per facility and dataset: values whose last edit predates their creation."""
    return values.groupBy("dataset_id", "org_unit_id").agg(
        F.count("*").alias("total"),
        F.sum((F.col("last_updated") < F.col("created")).cast("int")).alias("bad"),
    )


def last_updated_before_created_findings(values: DataFrame, units: DataFrame) -> DataFrame:
    counts = (
        last_updated_counts(values)
        .filter(F.col("bad") > 0)
        .join(F.broadcast(_facility_names(units)), "org_unit_id", "left")
    )
    return to_findings(
        counts,
        check="last_updated_before_created",
        dimension=Dimension.SYSTEM,
        severity=Severity.LOW,
        message=F.format_string(
            "%d of %d values were last updated before they were created", "bad", "total"
        ),
        value=F.col("bad"),
        expected=F.col("total"),
        score=F.col("bad") / F.col("total"),
    )


def early_entry_counts(reports: DataFrame) -> DataFrame:
    """Per facility and dataset: received reports entered before their month ended."""
    return (
        reports.filter("reported")
        .groupBy("dataset_id", *FACILITY_NAMES)
        .agg(
            F.count("*").alias("received"),
            F.sum(F.col("entered_before_period_end").cast("int")).alias("early"),
            F.min("days_after_period_end").alias("earliest"),
        )
    )


def entered_before_period_end_findings(reports: DataFrame) -> DataFrame:
    counts = early_entry_counts(reports).filter(F.col("early") > 0)
    return to_findings(
        counts,
        check="entered_before_period_end",
        dimension=Dimension.SYSTEM,
        severity=Severity.MEDIUM,
        message=F.format_string(
            "%d of %d reports were entered before their month ended (up to %d days early); "
            "timeliness can't be measured",
            "early",
            "received",
            -F.col("earliest"),
        ),
        value=F.col("early"),
        expected=F.col("received"),
        score=F.col("early") / F.col("received"),
    )


def missing_coordinates_findings(units: DataFrame, assignments: DataFrame) -> DataFrame:
    """Facilities expected to report something, but without a GPS point."""
    expected = assignments.select("org_unit_id").distinct()
    missing = units.filter("is_facility and longitude is null").join(expected, "org_unit_id")
    return to_findings(
        missing.withColumn("facility", F.col("name")),
        check="missing_coordinates",
        dimension=Dimension.SYSTEM,
        severity=Severity.LOW,
        message=F.lit("No GPS coordinates: the facility can't be shown on the map"),
    )


def non_facility_assignment_findings(
    assignments: DataFrame, units: DataFrame, datasets_in_scope: DataFrame
) -> DataFrame:
    """Datasets assigned to org units that aren't facilities (a configuration error)."""
    wrong = (
        assignments.join(datasets_in_scope, "dataset_id")
        .join(units.filter(~F.col("is_facility")), "org_unit_id")
        .withColumn("facility", F.col("name"))
    )
    return to_findings(
        wrong,
        check="non_facility_assignment",
        dimension=Dimension.SYSTEM,
        severity=Severity.LOW,
        message=F.format_string(
            "Dataset assigned to %s (level %d), which isn't a facility; excluded from completeness",
            "name",
            "level",
        ),
    )


def repeated_values_findings(
    facility_months: DataFrame, rules: DQRules = DEFAULT_RULES
) -> DataFrame:
    """Runs of the same non-trivial value in consecutive months ("gaps and islands")."""
    series = ["dataset_id", "org_unit_id", "data_element_id"]
    month_index = F.col("year") * 12 + F.substring("period", 5, 2).cast("int")
    ordered = Window.partitionBy(*series).orderBy("month_index")
    runs = (
        facility_months.withColumn("month_index", month_index)
        .withColumn(
            "new_run",
            (F.lag("value").over(ordered).isNull())
            | (F.lag("value").over(ordered) != F.col("value"))
            | (F.lag("month_index").over(ordered) != F.col("month_index") - 1),
        )
        .withColumn("run_id", F.sum(F.col("new_run").cast("int")).over(ordered))
        .groupBy(*series, "run_id", *FACILITY_NAMES, "data_element", "value")
        .agg(
            F.count("*").alias("months"),
            F.min("period").alias("first_period"),
            F.max("period").alias("last_period"),
        )
        .filter(
            (F.col("months") >= rules.repeat_min_months)
            & (F.col("value") >= rules.repeat_min_value)
        )
        .withColumn("period", F.col("first_period"))
    )
    return to_findings(
        runs,
        check="repeated_values",
        dimension=Dimension.SYSTEM,
        severity=F.when(F.col("months") >= rules.repeat_high_months, Severity.HIGH).otherwise(
            Severity.MEDIUM
        ),
        message=F.format_string(
            "%s = %.0f for %d months in a row (%s to %s)",
            "data_element",
            "value",
            "months",
            "first_period",
            "last_period",
        ),
        value=F.col("value"),
        score=F.col("months"),
    )


def copy_counts(facility_months: DataFrame) -> DataFrame:
    """Per facility, dataset and year: non-zero values identical to the same month last year."""
    keys = ["dataset_id", "org_unit_id", "data_element_id", "month"]
    with_month = facility_months.withColumn("month", F.substring("period", 5, 2))
    last_year = with_month.select(
        *keys, (F.col("year") + 1).alias("year"), F.col("value").alias("last_year_value")
    )
    return (
        with_month.join(last_year, [*keys, "year"])
        .filter(F.col("value") > 0)  # zeros repeat naturally
        .groupBy("dataset_id", *FACILITY_NAMES, "year")
        .agg(
            F.count("*").alias("compared"),
            F.sum((F.col("value") == F.col("last_year_value")).cast("int")).alias("identical"),
        )
    )


def repeats_last_year_findings(
    facility_months: DataFrame, rules: DQRules = DEFAULT_RULES
) -> DataFrame:
    """Facility-years where most values equal the same month of the previous year."""
    compared = (
        copy_counts(facility_months)
        .withColumn("share", F.col("identical") / F.col("compared"))
        .filter((F.col("compared") >= rules.copy_min_months) & (F.col("share") >= rules.copy_share))
        .withColumn("period", F.col("year").cast("string"))
    )
    return to_findings(
        compared,
        check="repeats_last_year",
        dimension=Dimension.SYSTEM,
        severity=F.when(F.col("share") >= rules.copy_share_high, Severity.HIGH).otherwise(
            Severity.MEDIUM
        ),
        message=F.format_string(
            "%d: %d of %d values (%.0f%%) are identical to the same month last year",
            "year",
            "identical",
            "compared",
            F.col("share") * 100,
        ),
        value=F.col("identical"),
        expected=F.col("compared"),
        score=F.col("share"),
    )
