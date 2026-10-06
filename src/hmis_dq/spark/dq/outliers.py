"""Outliers (WHO DQR dimension 2): values far from a facility's own history.

Two methods, side by side:
- Standard deviations from the mean (WHO DQR "extreme outlier" at 3 SD).
  Simple and widely understood, but one huge value inflates the mean and SD
  enough to hide itself.
- Modified z-score from the median and MAD (Iglewicz & Hoaglin, flag at 3.5).
  Robust: the median and median absolute deviation barely move for one bad value.

Severity follows WHO DQR, where 3 SD is the reference "extreme outlier":
high when both methods agree, medium when only the SD method flags it, low
when only the robust method does. Routine counts are often heavy-tailed
(tightly clustered with occasional bursts), which makes the MAD small and the
robust method flag many values; grading them low keeps them visible without
letting them dominate facility scores.
"""

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from hmis_dq.spark.dq.findings import to_findings
from hmis_dq.spark.dq.rules import DEFAULT_RULES, Dimension, DQRules, Severity

SERIES_KEY = ("dataset_id", "org_unit_id", "data_element_id")
MAD_TO_SD = 0.6745  # makes the modified z-score comparable to a z-score for normal data


def outlier_scores(facility_months: DataFrame, rules: DQRules = DEFAULT_RULES) -> DataFrame:
    """Every value with enough history, scored by both methods."""
    stats = facility_months.groupBy(*SERIES_KEY).agg(
        F.count("value").alias("months"),
        F.mean("value").alias("mean"),
        F.stddev_samp("value").alias("sd"),
        F.median("value").alias("median"),
    )
    scored = facility_months.join(stats, list(SERIES_KEY)).filter(
        F.col("months") >= rules.min_months_for_stats
    )
    mad = (
        scored.withColumn("abs_dev", F.abs(F.col("value") - F.col("median")))
        .groupBy(*SERIES_KEY)
        .agg(F.median("abs_dev").alias("mad"))
    )
    return (
        scored.join(mad, list(SERIES_KEY))
        .withColumn("z", F.when(F.col("sd") > 0, (F.col("value") - F.col("mean")) / F.col("sd")))
        .withColumn(
            "modified_z",
            F.when(F.col("mad") > 0, MAD_TO_SD * (F.col("value") - F.col("median")) / F.col("mad")),
        )
    )


def outlier_findings(facility_months: DataFrame, rules: DQRules = DEFAULT_RULES) -> DataFrame:
    scored = outlier_scores(facility_months, rules)
    extreme = F.coalesce(F.abs("z") >= rules.extreme_sd, F.lit(False))
    robust = F.coalesce(F.abs("modified_z") >= rules.robust_modified_z, F.lit(False))
    flagged = scored.filter(extreme | robust)

    def shown(score: str) -> Column:
        # a score that couldn't be computed (SD or MAD of 0) reads "n/a", not 0.0
        return F.when(F.col(score).isNull(), F.lit("n/a")).otherwise(F.format_string("%.1f", score))

    message = F.format_string(
        "%s = %.0f, typical %.0f (%s SD from mean, modified z %s)",
        "data_element",
        "value",
        "median",
        shown("z"),
        shown("modified_z"),
    )
    return to_findings(
        flagged,
        check="outlier",
        dimension=Dimension.OUTLIERS,
        severity=F.when(extreme & robust, Severity.HIGH)
        .when(extreme, Severity.MEDIUM)
        .otherwise(Severity.LOW),
        message=message,
        value=F.col("value"),
        expected=F.col("median"),
        score=F.coalesce("modified_z", "z"),
    )
