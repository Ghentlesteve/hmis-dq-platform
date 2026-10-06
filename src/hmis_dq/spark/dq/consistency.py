"""Consistency checks (WHO DQR dimensions 3 and 4).

Internal consistency: related indicators must agree.
- Drop-out pairs (Penta1 -> Penta3, ANC1 -> ANC4): over a year, more children
  can't finish a series than started it, so a negative drop-out is a data error.
  Compared per year, not per month: Penta3 is given weeks after Penta1, so one
  month's Penta3 can legitimately exceed that month's Penta1.
  Only months where the facility reported both indicators are counted, so a
  missing report isn't mistaken for a drop-out.
- Ratio pairs (OPV1 vs Penta1, given at the same visit): each district's
  ratio should be within +/-10% of the national ratio.

Consistency over time: each district's yearly total for a tracer indicator,
against the mean of up to three previous years over the *same months*
(so a partial current year compares fairly), flagged beyond +/-33%.
"""

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from hmis_dq.spark.dq.findings import to_findings
from hmis_dq.spark.dq.rules import (
    DEFAULT_RULES,
    INDICATOR_PAIRS,
    TRACER_INDICATORS,
    Dimension,
    DQRules,
    IndicatorPair,
    PairKind,
    Severity,
)

# ----------------------------------------------------------- internal


def _paired_months(facility_months: DataFrame, pair: IndicatorPair) -> DataFrame:
    """Facility-months where both indicators of a pair were reported."""

    def side(name: str, suffix: str) -> DataFrame:
        return facility_months.filter(F.col("data_element") == name).select(
            "dataset_id",
            "district_id",
            "district",
            "org_unit_id",
            "facility",
            "period",
            "year",
            F.col("data_element_id").alias(f"data_element_id_{suffix}"),
            F.col("value").alias(f"value_{suffix}"),
        )

    keys = ["dataset_id", "district_id", "district", "org_unit_id", "facility", "period", "year"]
    return side(pair.first, "first").join(side(pair.second, "second"), keys)


def _yearly(paired: DataFrame, *keys: str) -> DataFrame:
    return paired.groupBy(*keys, "year", "data_element_id_second").agg(
        F.sum("value_first").alias("first_total"),
        F.sum("value_second").alias("second_total"),
        F.countDistinct("period").alias("months"),
    )


def dropout_findings(
    facility_months: DataFrame, pair: IndicatorPair, rules: DQRules = DEFAULT_RULES
) -> DataFrame:
    paired = _paired_months(facility_months, pair)
    dropout = (F.col("first_total") - F.col("second_total")) / F.col("first_total")
    message = F.format_string(
        "%s %d: %s (%.0f) exceeds %s (%.0f), drop-out %.0f%%",
        F.coalesce("facility", "district"),
        "year",
        F.lit(pair.second),
        "second_total",
        F.lit(pair.first),
        "first_total",
        dropout * 100,
    )

    def flagged(frame: DataFrame) -> DataFrame:
        if "facility" not in frame.columns:  # district level: no facility
            frame = frame.withColumn("facility", F.lit(None).cast("string"))
        return frame.filter(
            (F.col("months") >= rules.min_months_per_year)
            & (F.col("first_total") > 0)
            & (F.col("second_total") > F.col("first_total"))
        ).withColumns(
            {
                "period": F.col("year").cast("string"),
                "data_element_id": F.col("data_element_id_second"),
                "data_element": F.lit(pair.second),
            }
        )

    def as_findings(frame: DataFrame, severity: str) -> DataFrame:
        return to_findings(
            frame,
            check=pair.check,
            dimension=Dimension.CONSISTENCY_INTERNAL,
            severity=severity,
            message=message,
            value=F.col("second_total"),
            expected=F.col("first_total"),
            score=dropout,
        )

    facility_keys = ("dataset_id", "district_id", "district", "org_unit_id", "facility")
    facilities = flagged(_yearly(paired, *facility_keys))
    districts = flagged(_yearly(paired, "dataset_id", "district_id", "district"))
    return as_findings(facilities, Severity.MEDIUM).unionByName(
        as_findings(districts, Severity.HIGH)
    )


def ratio_findings(
    facility_months: DataFrame, pair: IndicatorPair, rules: DQRules = DEFAULT_RULES
) -> DataFrame:
    paired = _paired_months(facility_months, pair)
    districts = _yearly(paired, "dataset_id", "district_id", "district").filter(
        (F.col("months") >= rules.min_months_per_year) & (F.col("first_total") > 0)
    )
    national = districts.groupBy("year").agg(
        (F.sum("second_total") / F.sum("first_total")).alias("national_ratio")
    )
    ratio = F.col("second_total") / F.col("first_total")
    difference = ratio / F.col("national_ratio") - 1
    flagged = (
        districts.join(national, "year")
        .filter(F.abs(difference) > rules.ratio_tolerance)
        .withColumns(
            {
                "period": F.col("year").cast("string"),
                "data_element_id": F.col("data_element_id_second"),
                "data_element": F.lit(pair.second),
            }
        )
    )
    return to_findings(
        flagged,
        check=pair.check,
        dimension=Dimension.CONSISTENCY_INTERNAL,
        severity=Severity.MEDIUM,
        message=F.format_string(
            "%s %d: %s / %s = %.2f vs national %.2f (%+.0f%%)",
            "district",
            "year",
            F.lit(pair.second),
            F.lit(pair.first),
            ratio,
            "national_ratio",
            difference * 100,
        ),
        value=ratio,
        expected=F.col("national_ratio"),
        score=difference,
    )


def internal_consistency_findings(
    facility_months: DataFrame,
    rules: DQRules = DEFAULT_RULES,
    pairs: tuple[IndicatorPair, ...] = INDICATOR_PAIRS,
) -> DataFrame:
    frames = [
        dropout_findings(facility_months, p, rules)
        if p.kind is PairKind.DROPOUT
        else ratio_findings(facility_months, p, rules)
        for p in pairs
    ]
    result = frames[0]
    for frame in frames[1:]:
        result = result.unionByName(frame)
    return result


# --------------------------------------------------------- over time


def time_consistency_findings(
    district_months: DataFrame,
    rules: DQRules = DEFAULT_RULES,
    tracers: tuple[str, ...] = TRACER_INDICATORS,
) -> DataFrame:
    keys = ["dataset_id", "district_id", "data_element_id", "month"]
    base = district_months.filter(F.col("data_element").isin(*tracers)).withColumn(
        "month", F.substring("period", 5, 2)
    )
    current = base.select(*keys, "district", "data_element", "year", "value")
    previous = base.select(*keys, F.col("year").alias("prev_year"), F.col("value").alias("prev"))

    in_window: Column = (F.col("prev_year") < F.col("year")) & (
        F.col("prev_year") >= F.col("year") - rules.time_consistency_years
    )
    # one row per (current year, previous year) on the months both have
    per_prev_year = (
        current.join(previous, keys)
        .filter(in_window)
        .groupBy(
            "dataset_id",
            "district_id",
            "district",
            "data_element_id",
            "data_element",
            "year",
            "prev_year",
        )
        .agg(
            F.sum("value").alias("current_total"),
            F.sum("prev").alias("prev_total"),
            F.count("*").alias("months"),
        )
        .filter(F.col("months") >= rules.min_months_compared)
    )
    yearly = per_prev_year.groupBy(
        "dataset_id", "district_id", "district", "data_element_id", "data_element", "year"
    ).agg(
        (F.sum("current_total") / F.sum("prev_total")).alias("ratio"),
        F.avg("current_total").alias("current_total"),
        F.avg("prev_total").alias("prev_mean"),
        F.count("*").alias("years_compared"),
    )
    flagged = yearly.filter(
        (F.col("prev_mean") > 0) & (F.abs(F.col("ratio") - 1) > rules.time_consistency_tolerance)
    ).withColumn("period", F.col("year").cast("string"))
    return to_findings(
        flagged,
        check="consistency_over_time",
        dimension=Dimension.CONSISTENCY_TIME,
        severity=Severity.MEDIUM,
        message=F.format_string(
            "%s %d: %s %.0f vs %.0f average of %d previous year(s), same months (%+.0f%%)",
            "district",
            "year",
            "data_element",
            "current_total",
            "prev_mean",
            "years_compared",
            (F.col("ratio") - 1) * 100,
        ),
        value=F.col("current_total"),
        expected=F.col("prev_mean"),
        score=F.col("ratio"),
    )
