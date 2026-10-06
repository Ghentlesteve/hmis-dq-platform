"""Run every data quality check over the gold layer and write the results.

Outputs (gold bucket, under dhis2/dq/):
- findings: one row per problem, from every check, partitioned by check
- facility_reporting / district_reporting: completeness and timeliness rates
"""

from dataclasses import dataclass
from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from hmis_dq.config import Settings
from hmis_dq.spark.dq.outliers import outlier_findings
from hmis_dq.spark.dq.reporting import district_reporting, facility_reporting, reporting_findings
from hmis_dq.spark.dq.rules import DEFAULT_RULES, DQRules
from hmis_dq.spark.session import build_spark, s3a_url


@dataclass(frozen=True)
class DQResult:
    findings_by_check: list[tuple[str, str, int]]  # check, severity, count
    completeness: list[tuple[str, float, float | None]]  # dataset, completeness, timeliness


def run_dq(settings: Settings, rules: DQRules = DEFAULT_RULES) -> DQResult:
    spark = build_spark(settings, "hmis-dq")

    def gold(table: str) -> DataFrame:
        return spark.read.parquet(s3a_url(settings.gold_bucket, f"dhis2/{table}/"))

    def dq_path(table: str) -> str:
        return s3a_url(settings.gold_bucket, f"dhis2/dq/{table}/")

    reports = gold("reporting").cache()
    facility_months = gold("facility_month")

    facility_rates = facility_reporting(reports).cache()
    facility_rates.coalesce(1).write.mode("overwrite").parquet(dq_path("facility_reporting"))
    district_reporting(reports).coalesce(1).write.mode("overwrite").parquet(
        dq_path("district_reporting")
    )

    findings = reduce(
        DataFrame.unionByName,
        [
            reporting_findings(facility_rates, rules),
            outlier_findings(facility_months, rules),
        ],
    )
    # Overwrite the whole table: a check that finds nothing this run must not
    # leave last run's findings behind (dynamic partition overwrite would).
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "static")
    findings.repartition("check").write.mode("overwrite").partitionBy("check").parquet(
        dq_path("findings")
    )

    written = spark.read.parquet(dq_path("findings"))
    by_check = [
        (r["check"], r["severity"], r["count"])
        for r in written.groupBy("check", "severity").count().orderBy("check", "severity").collect()
    ]
    overall = (
        reports.groupBy("dataset_id")
        .agg(
            (F.sum(F.col("reported").cast("int")) / F.count("*")).alias("completeness"),
            F.try_divide(
                F.sum(F.col("on_time").cast("int")),
                F.sum((F.col("reported") & F.col("on_time").isNotNull()).cast("int")),
            ).alias("timeliness"),
        )
        .orderBy("dataset_id")
        .collect()
    )
    return DQResult(
        findings_by_check=by_check,
        completeness=[(r["dataset_id"], r["completeness"], r["timeliness"]) for r in overall],
    )
