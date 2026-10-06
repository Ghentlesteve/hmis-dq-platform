"""Data quality scores (0-100) per facility and district, for each dataset.

Each dimension score is 100 x (1 - share of the facility's data affected),
so every number can be explained from counts:

- completeness: reports received / reports expected
- accuracy:     1 - weighted share of values flagged as outliers
                (high 1, medium 0.5, low 0.1: low flags are robust-method-only)
- consistency:  1 - share of reported years with a negative drop-out
- integrity:    1 - mean of: share of values identical to an earlier year's,
                share with impossible timestamps, share of reports entered
                before the period ended

Overall = weighted mean of the dimensions that can be measured. A facility that
never reported has no values to check, so only completeness counts (and it's 0);
an unmeasurable dimension is left out, never counted as a perfect 100.

District scores weight each facility by the reports it was expected to send.
"""

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from hmis_dq.spark.dq.rules import Dimension, Severity

DIMENSION_WEIGHTS: dict[str, float] = {
    "completeness": 0.35,
    "accuracy": 0.25,
    "consistency": 0.20,
    "integrity": 0.20,
}
OUTLIER_WEIGHTS = {Severity.HIGH: 1.0, Severity.MEDIUM: 0.5, Severity.LOW: 0.1}
GRADES = ((90, "A"), (75, "B"), (60, "C"))  # below the last: D

KEY = ["dataset_id", "org_unit_id"]


def _share_score(affected: Column, total: Column) -> Column:
    """100 x (1 - affected/total); null when there is nothing to measure."""
    return F.when(total > 0, F.round(100 * (1 - affected / total), 1))


def _weighted_overall(prefix: str = "") -> Column:
    numerator = sum(
        (F.coalesce(F.col(f"{prefix}{d}") * w, F.lit(0.0)) for d, w in DIMENSION_WEIGHTS.items()),
        start=F.lit(0.0),
    )
    denominator = sum(
        (
            F.when(F.col(f"{prefix}{d}").isNotNull(), w).otherwise(0.0)
            for d, w in DIMENSION_WEIGHTS.items()
        ),
        start=F.lit(0.0),
    )
    return F.round(numerator / denominator, 1)


def grade(score: Column) -> Column:
    graded = F.when(score.isNull(), None)
    for threshold, letter in GRADES:
        graded = graded.when(score >= threshold, letter)
    return graded.otherwise("D")


def facility_scores(
    facility_rates: DataFrame,
    facility_months: DataFrame,
    findings: DataFrame,
    *,
    copies: DataFrame,
    timestamps: DataFrame,
    early_entries: DataFrame,
) -> DataFrame:
    values = facility_months.groupBy(*KEY).agg(
        F.count("*").alias("values"), F.countDistinct("year").alias("years_reported")
    )

    outlier_weight = F.lit(0.0)
    for severity, weight in OUTLIER_WEIGHTS.items():
        outlier_weight = F.when(F.col("severity") == severity, weight).otherwise(outlier_weight)
    outliers = (
        findings.filter(F.col("check") == "outlier")
        .groupBy(*KEY)
        .agg(F.sum(outlier_weight).alias("outlier_weight"))
    )
    inconsistent = (
        findings.filter(
            (F.col("dimension") == Dimension.CONSISTENCY_INTERNAL) & F.col("facility").isNotNull()
        )
        .groupBy(*KEY)
        .agg(F.countDistinct("period").alias("inconsistent_years"))
    )
    copied = copies.groupBy(*KEY).agg(
        (F.sum("identical") / F.sum("compared")).alias("copied_share")
    )
    bad_timestamps = timestamps.select(
        *KEY, (F.col("bad") / F.col("total")).alias("timestamp_share")
    )
    early = early_entries.select(*KEY, (F.col("early") / F.col("received")).alias("early_share"))

    shares = [F.col("copied_share"), F.col("timestamp_share"), F.col("early_share")]
    known = sum((F.when(s.isNotNull(), 1).otherwise(0) for s in shares), start=F.lit(0))
    mean_share = sum((F.coalesce(s, F.lit(0.0)) for s in shares), start=F.lit(0.0)) / known

    scored = (
        facility_rates.select(
            *KEY,
            "facility",
            "district_id",
            "district",
            "reports_expected",
            "reports_received",
            F.round(100 * F.col("completeness"), 1).alias("completeness"),
        )
        .join(values, KEY, "left")
        .join(outliers, KEY, "left")
        .join(inconsistent, KEY, "left")
        .join(copied, KEY, "left")
        .join(bad_timestamps, KEY, "left")
        .join(early, KEY, "left")
        .fillna(0, subset=["outlier_weight", "inconsistent_years"])
        .withColumns(
            {
                "accuracy": _share_score(F.col("outlier_weight"), F.col("values")),
                "consistency": _share_score(
                    F.col("inconsistent_years").cast("double"), F.col("years_reported")
                ),
                "integrity": F.when(
                    (F.col("values") > 0) & (known > 0), F.round(100 * (1 - mean_share), 1)
                ),
            }
        )
    )
    return scored.withColumn("overall", _weighted_overall()).withColumn(
        "grade", grade(F.col("overall"))
    )


def rollup_scores(facilities: DataFrame, *keys: str) -> DataFrame:
    """Facility scores rolled up, weighted by the reports each was expected to send."""
    weight = F.col("reports_expected")

    def weighted(dimension: str) -> Column:
        has = F.col(dimension).isNotNull()
        return F.round(
            F.sum(F.when(has, F.col(dimension) * weight)) / F.sum(F.when(has, weight)),
            1,
        ).alias(dimension)

    rolled = facilities.groupBy(*keys).agg(
        F.count("*").alias("facilities"),
        F.sum((F.col("grade") == "D").cast("int")).alias("facilities_grade_d"),
        F.sum("reports_expected").alias("reports_expected"),
        F.sum("reports_received").alias("reports_received"),
        *(weighted(d) for d in DIMENSION_WEIGHTS),
    )
    # completeness is exact from the totals rather than a weighted average
    rolled = rolled.withColumn(
        "completeness",
        F.round(100 * F.col("reports_received") / F.col("reports_expected"), 1),
    )
    return rolled.withColumn("overall", _weighted_overall()).withColumn(
        "grade", grade(F.col("overall"))
    )


def district_scores(facilities: DataFrame) -> DataFrame:
    return rollup_scores(facilities, "dataset_id", "district_id", "district")


def national_scores(facilities: DataFrame) -> DataFrame:
    return rollup_scores(facilities, "dataset_id")
