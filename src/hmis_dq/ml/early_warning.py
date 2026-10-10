"""Early warning: districts whose latest month falls below the expected range.

Expected range: the best forecaster in the backtest ("same month last year"),
widened by how wrong that forecast has actually been for the same indicator,
measured only on months *before* the one being checked:

    log((actual + 1) / (forecast + 1))  over earlier months, pooled across districts
    range = (forecast + 1) * exp(5th..95th percentile of those errors) - 1

Only shortfalls are flagged: early warning is about service disruption.
Each warning says whether the district's reporting also dropped that month, since
"fewer facilities reported" and "fewer services delivered" need different responses.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hmis_dq.ml.forecast import backtest_series, seasonal_naive


@dataclass(frozen=True)
class WarningRules:
    # A 5th-percentile lower bound means ~1 in 20 normal district-months falls below it
    # by chance; those are usually small shortfalls, graded medium.
    lower_quantile: float = 0.05
    upper_quantile: float = 0.95
    min_errors: int = 20  # past forecast errors needed for a trustworthy range
    high_shortfall: float = 0.25  # >= 25% below the lower bound: high severity
    reporting_drop: float = 0.80  # reports received < 80% of usual: a reporting problem
    recent_months: int = 3  # facility anomalies this recent are included


DEFAULT_WARNING_RULES = WarningRules()


def latest_forecasts(
    panel: pd.DataFrame, series_keys: list[str], history_months: int = 13
) -> pd.DataFrame:
    """Seasonal-naive forecasts for each series' recent months (latest one included)."""
    out = []
    for key, rows in panel.groupby(series_keys, sort=False):
        values = rows.set_index("period_start")["value"].sort_index()
        result = backtest_series(values, {"seasonal_naive": seasonal_naive}, history_months)
        for column, value in zip(series_keys, key, strict=True):
            result[column] = str(value)
        out.append(result)
    return pd.concat(out, ignore_index=True)


def error_bounds(
    forecasts: pd.DataFrame, before: pd.Timestamp, rules: WarningRules = DEFAULT_WARNING_RULES
) -> pd.DataFrame:
    """Per dataset and indicator: quantiles of past log errors, from months before ``before``."""
    past = forecasts[(forecasts["period_start"] < before) & forecasts["forecast"].notna()].copy()
    past["log_error"] = np.log1p(past["actual"]) - np.log1p(past["forecast"])
    bounds = past.groupby(["dataset_id", "data_element"]).agg(
        errors=("log_error", "size"),
        q_low=("log_error", lambda e: e.quantile(rules.lower_quantile)),
        q_high=("log_error", lambda e: e.quantile(rules.upper_quantile)),
    )
    return bounds[bounds["errors"] >= rules.min_errors].reset_index()


def district_warnings(
    forecasts: pd.DataFrame,
    reporting: pd.DataFrame,
    rules: WarningRules = DEFAULT_WARNING_RULES,
) -> pd.DataFrame:
    """Latest month per dataset, below the expected range, with the likely cause.

    ``reporting``: dataset_id, district_id, period_start, reports_received.
    """
    warnings = []
    for dataset_id, rows in forecasts.groupby("dataset_id"):
        latest = rows["period_start"].max()
        bounds = error_bounds(rows, latest, rules)
        current = rows[(rows["period_start"] == latest) & rows["forecast"].notna()]
        checked = current.merge(bounds, on=["dataset_id", "data_element"])
        checked = checked.assign(
            expected_low=(checked["forecast"] + 1) * np.exp(checked["q_low"]) - 1,
            expected_high=(checked["forecast"] + 1) * np.exp(checked["q_high"]) - 1,
        )
        below = checked[checked["actual"] < checked["expected_low"]].copy()
        if below.empty:
            continue
        below["shortfall"] = 1 - below["actual"] / below["expected_low"]

        dataset_reporting = reporting[reporting["dataset_id"] == dataset_id]
        usual = (
            dataset_reporting[dataset_reporting["period_start"] < latest]
            .groupby("district_id")["reports_received"]
            .median()
            .rename("usual_reports")
        )
        now = (
            dataset_reporting[dataset_reporting["period_start"] == latest]
            .set_index("district_id")["reports_received"]
            .rename("reports_received")
        )
        below = below.join(usual, on="district_id").join(now, on="district_id")
        warnings.append(below)

    if not warnings:
        return pd.DataFrame()
    result = pd.concat(warnings, ignore_index=True)
    result["reports_received"] = result["reports_received"].fillna(0)
    reporting_dropped = result["reports_received"] < rules.reporting_drop * result["usual_reports"]
    result["likely_cause"] = np.where(reporting_dropped, "reporting drop", "service decline")
    result["severity"] = np.where(result["shortfall"] >= rules.high_shortfall, "high", "medium")
    result["message"] = result.apply(_message, axis=1)
    ordered: pd.DataFrame = result.sort_values(["severity", "shortfall"], ascending=[True, False])
    return ordered.reset_index(drop=True)


def warning_history(
    forecasts: pd.DataFrame,
    reporting: pd.DataFrame,
    months: int = 12,
    rules: WarningRules = DEFAULT_WARNING_RULES,
) -> pd.DataFrame:
    """Replay the warnings as if each of the last ``months`` months had been the latest.

    Each replay only sees data up to that month, exactly as it would have run then.
    Shows whether the system fires when the data changes, and feeds "warnings over time".
    """
    replays = []
    for month in sorted(forecasts["period_start"].unique())[-months:]:
        found = district_warnings(forecasts[forecasts["period_start"] <= month], reporting, rules)
        if not found.empty:
            replays.append(found[found["period_start"] == month])
    return pd.concat(replays, ignore_index=True) if replays else pd.DataFrame()


def _message(row: pd.Series) -> str:
    month = row["period_start"].strftime("%B %Y")
    reports = (
        f"{row['reports_received']:.0f} reports vs usual {row['usual_reports']:.0f}"
        if pd.notna(row["usual_reports"])
        else "reporting history unknown"
    )
    return (
        f"{row['district']}, {row['data_element']}, {month}: {row['actual']:.0f}, "
        f"expected {row['expected_low']:.0f}-{row['expected_high']:.0f} "
        f"({row['shortfall']:.0%} below the range). {reports}: likely {row['likely_cause']}"
    )


def recent_facility_drops(
    anomalies: pd.DataFrame, latest_period: str, rules: WarningRules = DEFAULT_WARNING_RULES
) -> pd.DataFrame:
    """Facility anomalies of the "far below usual" kind in the last few months."""
    latest = pd.Period(latest_period, freq="M")
    window = {str(latest - n).replace("-", "") for n in range(rules.recent_months)}
    drops = anomalies[
        anomalies["period"].isin(window) & anomalies["explanation"].str.contains("far below usual")
    ]
    return drops.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
