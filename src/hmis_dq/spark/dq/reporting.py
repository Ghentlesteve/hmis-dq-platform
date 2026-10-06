"""Completeness and timeliness of reporting (WHO DQR dimension 1).

Built on gold.reporting, which has one row per *expected* report:
- completeness = received / expected
- timeliness = on time / received with a usable entry date. Reports whose entry
  date is impossible are counted separately as ``timeliness_unknown`` instead of
  being treated as on time or late.
"""

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from hmis_dq.spark.dq.findings import to_findings
from hmis_dq.spark.dq.rules import DEFAULT_RULES, Dimension, DQRules, Severity


def _rates(reports: DataFrame, *keys: str) -> DataFrame:
    def count_true(condition: Column) -> Column:
        return F.sum(F.when(condition, 1).otherwise(0))

    received = F.col("reported")
    return (
        reports.groupBy(*keys)
        .agg(
            F.count("*").alias("reports_expected"),
            count_true(received).alias("reports_received"),
            count_true(F.col("on_time")).alias("reports_on_time"),
            count_true(received & F.col("on_time").isNull()).alias("timeliness_unknown"),
        )
        .withColumn("completeness", F.col("reports_received") / F.col("reports_expected"))
        .withColumn(
            "timeliness",
            F.try_divide(
                F.col("reports_on_time"),
                F.col("reports_received") - F.col("timeliness_unknown"),
            ),
        )
    )


def facility_reporting(reports: DataFrame) -> DataFrame:
    """Completeness and timeliness per dataset and facility over the whole window."""
    return _rates(reports, "dataset_id", "district_id", "district", "org_unit_id", "facility")


def district_reporting(reports: DataFrame) -> DataFrame:
    """Completeness and timeliness per dataset, district and year."""
    return _rates(reports, "dataset_id", "district_id", "district", "year")


def reporting_findings(facility_rates: DataFrame, rules: DQRules = DEFAULT_RULES) -> DataFrame:
    """Facilities below the completeness benchmark (never-reporting ones separately)."""
    sent = F.format_string(
        "Sent %d of %d expected reports (%.0f%%)",
        "reports_received",
        "reports_expected",
        F.col("completeness") * 100,
    )

    never = to_findings(
        facility_rates.filter(F.col("reports_received") == 0),
        check="never_reported",
        dimension=Dimension.COMPLETENESS,
        severity=Severity.HIGH,
        message=F.format_string(
            "Assigned this dataset but sent none of %d expected reports", "reports_expected"
        ),
        value=F.col("completeness"),
        expected=F.lit(rules.completeness_target),
    )
    low = to_findings(
        facility_rates.filter(
            (F.col("reports_received") > 0) & (F.col("completeness") < rules.completeness_target)
        ),
        check="low_reporting_completeness",
        dimension=Dimension.COMPLETENESS,
        severity=F.when(
            F.col("completeness") < rules.completeness_critical, Severity.HIGH
        ).otherwise(Severity.MEDIUM),
        message=sent,
        value=F.col("completeness"),
        expected=F.lit(rules.completeness_target),
    )
    return never.unionByName(low)
