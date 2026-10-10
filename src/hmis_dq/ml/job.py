"""ML jobs over the gold layer -> gold/dhis2/ml/ (via DuckDB, no Spark): forecast
backtest and anomaly detection.

The series are small (districts x tracer indicators, at most a few years of
months), so pandas and scikit-learn are the right tools here; Spark is kept for
the heavy bronze -> silver -> gold processing.
"""

from dataclasses import dataclass

import duckdb
import pandas as pd

from hmis_dq.config import Settings
from hmis_dq.explore import gold, lake
from hmis_dq.ml.anomaly import ANTIGENS, deviation_profiles, explain, score_anomalies
from hmis_dq.ml.forecast import (
    LOCAL_MODELS,
    backtest_gbm,
    backtest_series,
    series_metrics,
    summarise,
)
from hmis_dq.spark.dq.rules import TRACER_INDICATORS

SERIES_KEYS = ["dataset_id", "district_id", "district", "data_element_id", "data_element"]


def load_series(db: duckdb.DuckDBPyConnection, tracers: tuple[str, ...]) -> pd.DataFrame:
    """District x tracer monthly totals, with every month of the dataset's window
    present (a month with no data is NaN rather than missing)."""
    placeholders = ", ".join("?" for _ in tracers)
    observed = db.execute(
        f"""
        SELECT dataset_id, district_id, district, data_element_id, data_element,
               CAST(period_start AS DATE) AS period_start, value
        FROM {gold("district_month")}
        WHERE district IS NOT NULL AND data_element IN ({placeholders})
        """,
        list(tracers),
    ).df()
    observed["period_start"] = pd.to_datetime(observed["period_start"])

    complete = []
    for _, rows in observed.groupby("dataset_id"):
        months = pd.date_range(rows["period_start"].min(), rows["period_start"].max(), freq="MS")
        series = rows[SERIES_KEYS].drop_duplicates()
        grid = series.merge(pd.DataFrame({"period_start": months}), how="cross")
        complete.append(grid.merge(rows, on=[*SERIES_KEYS, "period_start"], how="left"))
    return pd.concat(complete, ignore_index=True)


def run_backtest(panel: pd.DataFrame, test_months: int = 12) -> pd.DataFrame:
    """Per-series models for every series, plus the global gradient-boosting model."""
    local = []
    for key, rows in panel.groupby(SERIES_KEYS, sort=False):
        values = rows.set_index("period_start")["value"].sort_index()
        result = backtest_series(values, LOCAL_MODELS, test_months)
        for column, value in zip(SERIES_KEYS, key, strict=True):
            result[column] = str(value)  # series keys are IDs and names
        local.append(result)
    global_models = [
        backtest_gbm(panel, SERIES_KEYS, test_months),
        backtest_gbm(panel, SERIES_KEYS, test_months, seasonal_residual=True),
    ]
    combined = pd.concat([*local, *global_models], ignore_index=True)
    return combined


@dataclass(frozen=True)
class ForecastResult:
    series: int
    summary: pd.DataFrame  # per dataset and model


def _write(
    db: duckdb.DuckDBPyConnection, frame: pd.DataFrame, settings: Settings, name: str
) -> None:
    db.register("frame", frame)
    db.sql(f"COPY frame TO 's3://{settings.gold_bucket}/dhis2/ml/{name}.parquet' (FORMAT parquet)")
    db.unregister("frame")


def run_forecast_backtest(settings: Settings, test_months: int = 12) -> ForecastResult:
    db = lake(settings)
    panel = load_series(db, TRACER_INDICATORS)
    backtest = run_backtest(panel, test_months)
    metrics = series_metrics(backtest, SERIES_KEYS)
    summary = summarise(metrics, by=["dataset_id"])

    _write(db, backtest, settings, "forecast_backtest")
    _write(db, metrics, settings, "forecast_metrics")
    _write(db, summary, settings, "forecast_summary")
    return ForecastResult(series=panel.groupby(SERIES_KEYS).ngroups, summary=summary)


# ---------------------------------------------------------------- anomalies


@dataclass(frozen=True)
class AnomalyResult:
    profiles: int
    anomalies: pd.DataFrame  # flagged facility-months, most unusual first
    caught_by_rules: float  # share of anomalies with any rule-based outlier that month
    caught_strongly: float  # ... with a high/medium outlier


def compare_with_rules(anomalies: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
    """Mark each anomaly with the strongest rule-based outlier at the same facility-month."""
    rank = {"high": 3, "medium": 2, "low": 1}
    strongest = (
        outliers.assign(rank=outliers["severity"].map(rank))
        .sort_values("rank", ascending=False)
        .drop_duplicates(["org_unit_id", "period"])[["org_unit_id", "period", "severity"]]
        .rename(columns={"severity": "rule_severity"})
    )
    return anomalies.merge(strongest, on=["org_unit_id", "period"], how="left")


def run_anomaly_detection(settings: Settings) -> AnomalyResult:
    db = lake(settings)
    placeholders = ", ".join("?" for _ in ANTIGENS)
    facility_months = db.execute(
        f"""
        SELECT org_unit_id, facility, district, period, data_element, value
        FROM {gold("facility_month")} WHERE data_element IN ({placeholders})
        """,
        list(ANTIGENS),
    ).df()
    outliers = db.execute(
        f"""
        SELECT org_unit_id, period, severity FROM {gold("dq/findings")}
        WHERE "check" = 'outlier' AND data_element IN ({placeholders})
        """,
        list(ANTIGENS),
    ).df()

    scored = score_anomalies(deviation_profiles(facility_months))
    flagged = scored[scored["is_anomaly"]].copy()
    flagged["explanation"] = flagged.apply(explain, axis=1)
    flagged = compare_with_rules(flagged, outliers)

    _write(db, scored, settings, "anomaly_scores")
    _write(db, flagged, settings, "anomalies")
    caught = flagged["rule_severity"].notna()
    strong = flagged["rule_severity"].isin(["high", "medium"])
    return AnomalyResult(
        profiles=len(scored),
        anomalies=flagged,
        caught_by_rules=float(caught.mean()) if len(flagged) else 0.0,
        caught_strongly=float(strong.mean()) if len(flagged) else 0.0,
    )
