"""Run every data quality check over the gold layer and write the results.

Outputs (gold bucket, under dhis2/dq/):
- findings: one row per problem, from every check, partitioned by check
- facility_reporting / district_reporting: completeness and timeliness rates
- facility_scores / district_scores / national_scores: 0-100 scores per dimension
"""

from dataclasses import dataclass
from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from hmis_dq.config import Settings
from hmis_dq.spark.dq.consistency import (
    internal_consistency_findings,
    time_consistency_findings,
)
from hmis_dq.spark.dq.outliers import outlier_findings
from hmis_dq.spark.dq.reporting import district_reporting, facility_reporting, reporting_findings
from hmis_dq.spark.dq.rules import DEFAULT_RULES, INDICATOR_PAIRS, DQRules
from hmis_dq.spark.dq.scores import district_scores, facility_scores, national_scores
from hmis_dq.spark.dq.system import (
    copy_counts,
    early_entry_counts,
    entered_before_period_end_findings,
    last_updated_before_created_findings,
    last_updated_counts,
    missing_coordinates_findings,
    non_facility_assignment_findings,
    repeated_values_findings,
    repeats_last_year_findings,
)
from hmis_dq.spark.session import build_spark, s3a_url

# Every check the engine runs, so a check with no findings still shows up as 0
ALL_CHECKS = (
    "never_reported",
    "low_reporting_completeness",
    "outlier",
    *(pair.check for pair in INDICATOR_PAIRS),
    "consistency_over_time",
    "repeats_last_year",
    "repeated_values",
    "entered_before_period_end",
    "last_updated_before_created",
    "missing_coordinates",
    "non_facility_assignment",
)


@dataclass(frozen=True)
class DQResult:
    findings_by_check: dict[str, dict[str, int]]  # check -> severity -> count
    completeness: list[tuple[str, float, float | None]]  # dataset, completeness, timeliness
    national: list[dict[str, object]]  # one row per dataset
    worst_districts: list[dict[str, object]]


def run_dq(settings: Settings, rules: DQRules = DEFAULT_RULES) -> DQResult:
    spark = build_spark(settings, "hmis-dq")

    def gold(table: str) -> DataFrame:
        return spark.read.parquet(s3a_url(settings.gold_bucket, f"dhis2/{table}/"))

    def dq_path(table: str) -> str:
        return s3a_url(settings.gold_bucket, f"dhis2/dq/{table}/")

    def silver(table: str) -> DataFrame:
        return spark.read.parquet(s3a_url(settings.silver_bucket, f"dhis2/{table}/"))

    reports = gold("reporting").cache()
    facility_months = gold("facility_month").cache()
    units = silver("org_units").cache()

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
            internal_consistency_findings(facility_months, rules),
            time_consistency_findings(gold("district_month"), rules),
            last_updated_before_created_findings(silver("data_values"), units),
            entered_before_period_end_findings(reports),
            missing_coordinates_findings(units, reports.select("org_unit_id").distinct()),
            non_facility_assignment_findings(
                silver("dataset_org_units"), units, reports.select("dataset_id").distinct()
            ),
            repeated_values_findings(facility_months, rules),
            repeats_last_year_findings(facility_months, rules),
        ],
    )
    # Overwrite the whole table: a check that finds nothing this run must not
    # leave last run's findings behind (dynamic partition overwrite would).
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "static")
    findings.repartition("check").write.mode("overwrite").partitionBy("check").parquet(
        dq_path("findings")
    )

    written = spark.read.parquet(dq_path("findings")).cache()

    facilities = facility_scores(
        facility_rates,
        facility_months,
        written,
        copies=copy_counts(facility_months),
        timestamps=last_updated_counts(silver("data_values")),
        early_entries=early_entry_counts(reports),
    )
    facilities.coalesce(1).write.mode("overwrite").parquet(dq_path("facility_scores"))
    facilities = spark.read.parquet(dq_path("facility_scores"))
    districts = district_scores(facilities)
    districts.coalesce(1).write.mode("overwrite").parquet(dq_path("district_scores"))
    national = national_scores(facilities)
    national.coalesce(1).write.mode("overwrite").parquet(dq_path("national_scores"))

    by_check: dict[str, dict[str, int]] = {check: {} for check in ALL_CHECKS}
    for r in written.groupBy("check", "severity").count().collect():
        by_check.setdefault(r["check"], {})[r["severity"]] = r["count"]
    score_columns = [
        "dataset_id",
        "completeness",
        "accuracy",
        "consistency",
        "integrity",
        "overall",
        "grade",
    ]
    national_rows = [
        r.asDict() for r in national.select(*score_columns).orderBy("dataset_id").collect()
    ]
    worst = [
        r.asDict()
        for r in districts.select("district", *score_columns)
        .filter(F.col("district").isNotNull())
        .orderBy("overall")
        .limit(5)
        .collect()
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
        national=national_rows,
        worst_districts=worst,
    )
