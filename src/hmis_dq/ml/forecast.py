"""One-month-ahead forecasting of monthly series, with a rolling-origin backtest.

Every model has the same shape: it gets a series' history (monthly, contiguous,
NaN where no data) and returns its forecast for the next month, or NaN when it
doesn't have enough history.

The backtest replays the past: for each of the last ``test_months`` months, each
model sees only the data *before* that month, so no model ever sees the answer.
Models are judged against ``seasonal_naive`` ("same month last year"), the
standard benchmark for monthly service data:

    skill = 1 - MAE(model) / MAE(seasonal_naive), on the same months

> 0 means the model beats the benchmark; < 0 means copying last year was better.
"""

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

Forecaster = Callable[[pd.Series], float]

SEASON = 12
BENCHMARK = "seasonal_naive"


# ------------------------------------------------------------------ models


def naive(history: pd.Series) -> float:
    """Last observed value."""
    observed = history.dropna()
    return float(observed.iloc[-1]) if len(observed) else np.nan


def mean_3(history: pd.Series) -> float:
    """Mean of the last three months that have data."""
    recent = history.iloc[-3:].dropna()
    return float(recent.mean()) if len(recent) else np.nan


def seasonal_naive(history: pd.Series, lag: int = SEASON) -> float:
    """Same month, ``lag`` months earlier (12 = last year)."""
    return float(history.iloc[-lag]) if len(history) >= lag else np.nan


def seasonal_naive_2y(history: pd.Series) -> float:
    """Same month two years earlier: a diagnostic for data copied from two years back."""
    return seasonal_naive(history, lag=2 * SEASON)


def ets(history: pd.Series) -> float:
    """Holt-Winters exponential smoothing, additive yearly seasonality, no trend.

    Fitted on the stretch of history after the last missing month, which must
    cover two full seasons to estimate the seasonal pattern.
    """
    gapless = history.iloc[_last_gap(history) :]
    if len(gapless) < 2 * SEASON:
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ExponentialSmoothing(
                gapless.to_numpy(dtype=float),
                seasonal="add",
                seasonal_periods=SEASON,
                trend=None,
                initialization_method="estimated",
            ).fit()
        return max(float(fitted.forecast(1)[0]), 0.0)
    except (ValueError, np.linalg.LinAlgError):
        return np.nan


def _last_gap(history: pd.Series) -> int:
    """Index just after the last missing value (0 when there are none)."""
    missing = np.flatnonzero(history.isna().to_numpy())
    return int(missing[-1]) + 1 if len(missing) else 0


LOCAL_MODELS: dict[str, Forecaster] = {
    "naive": naive,
    "mean_3": mean_3,
    "seasonal_naive": seasonal_naive,
    "seasonal_naive_2y": seasonal_naive_2y,
    "ets": ets,
}


# ---------------------------------------------------------------- backtest


def backtest_series(
    values: pd.Series, models: dict[str, Forecaster] = LOCAL_MODELS, test_months: int = 12
) -> pd.DataFrame:
    """Rolling-origin backtest of the per-series models on one series.

    ``values`` is indexed by month start, contiguous. Returns one row per
    (month predicted, model) with the actual value and the forecast.
    """
    rows = []
    for position in range(max(len(values) - test_months, 1), len(values)):
        actual = values.iloc[position]
        if pd.isna(actual):
            continue
        history = values.iloc[:position]  # strictly before the month predicted
        for name, model in models.items():
            rows.append(
                {
                    "period_start": values.index[position],
                    "model": name,
                    "actual": float(actual),
                    "forecast": model(history),
                }
            )
    return pd.DataFrame(rows, columns=["period_start", "model", "actual", "forecast"])


# --------------------------------------------------------- global ML model

LAGS = (1, 2, 3, 12, 24)


def lag_features(panel: pd.DataFrame, series_keys: list[str]) -> pd.DataFrame:
    """Lag features per series, built only from earlier months (no leakage).

    Values are log-transformed so districts of very different sizes share one model.
    """
    panel = panel.sort_values([*series_keys, "period_start"]).copy()
    logged = np.log1p(panel["value"])
    grouped = logged.groupby([panel[k] for k in series_keys])
    for lag in LAGS:
        panel[f"lag_{lag}"] = grouped.shift(lag)
    panel["mean_3"] = grouped.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    panel["month"] = panel["period_start"].dt.month
    panel["target"] = logged
    return panel


FEATURES = [*(f"lag_{lag}" for lag in LAGS), "mean_3", "month"]


def backtest_gbm(
    panel: pd.DataFrame,
    series_keys: list[str],
    test_months: int = 12,
    min_train_rows: int = 50,
    *,
    seasonal_residual: bool = False,
) -> pd.DataFrame:
    """Rolling-origin backtest of one gradient-boosting model trained on all series.

    For each month predicted, the model is refit on every row from earlier months
    across all series, then predicts that month for every series.

    With ``seasonal_residual`` the model ("gbm_seasonal") predicts only the change
    from the same month last year, so the seasonal-naive benchmark is built in and
    the trees learn when to deviate from it. Months without last year's value are
    skipped, since there is nothing to build on.
    """
    features = lag_features(panel, series_keys)
    base = features["lag_12"] if seasonal_residual else pd.Series(0.0, index=features.index)
    usable = features["target"].notna() & base.notna()
    months = sorted(features["period_start"].unique())[-test_months:]
    name = "gbm_seasonal" if seasonal_residual else "gbm"
    out = []
    for month in months:
        train = features[(features["period_start"] < month) & usable]
        test = features[(features["period_start"] == month) & usable]
        if len(train) < min_train_rows or test.empty:
            continue
        model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0)
        model.fit(train[FEATURES], train["target"] - base[train.index])
        logged = base[test.index] + model.predict(test[FEATURES])
        out.append(
            test[[*series_keys, "period_start"]].assign(
                model=name,
                actual=np.expm1(test["target"]),
                forecast=np.clip(np.expm1(logged), 0, None),
            )
        )
    columns = [*series_keys, "period_start", "model", "actual", "forecast"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=columns)


# ----------------------------------------------------------------- metrics


def smape(actual: pd.Series, forecast: pd.Series) -> float:
    """Symmetric MAPE in %, defined as 0 when both are 0."""
    denominator = (actual.abs() + forecast.abs()) / 2
    ratio = (actual - forecast).abs() / denominator.where(denominator > 0)
    return float(100 * ratio.fillna(0).mean())


def series_metrics(backtest: pd.DataFrame, series_keys: list[str]) -> pd.DataFrame:
    """MAE, sMAPE and skill vs the benchmark per series and model.

    Each model is compared with the benchmark on the months where *both* made a
    forecast, so a model is never judged on easier months than the benchmark.
    """
    rows = []
    for key, group in backtest.groupby(series_keys, sort=False):
        wide = group.pivot_table(
            index="period_start", columns="model", values="forecast", aggfunc="first"
        )
        actual = group.drop_duplicates("period_start").set_index("period_start")["actual"]
        if BENCHMARK not in wide:
            continue
        for model in wide.columns:
            both = wide[[model, BENCHMARK]].dropna().index
            if len(both) == 0:
                continue
            error = (actual[both] - wide.loc[both, model]).abs()
            benchmark_error = (actual[both] - wide.loc[both, BENCHMARK]).abs()
            benchmark_mae = benchmark_error.mean()
            rows.append(
                {
                    **dict(
                        zip(series_keys, key if isinstance(key, tuple) else (key,), strict=True)
                    ),
                    "model": model,
                    "forecasts": len(both),
                    "mae": error.mean(),
                    "smape": smape(actual[both], wide.loc[both, model]),
                    "skill": 1 - error.mean() / benchmark_mae if benchmark_mae > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarise(metrics: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Per model: median skill, share of series where it beats the benchmark, median sMAPE."""
    keys = [*(by or []), "model"]
    summary = metrics.groupby(keys).agg(
        series=("skill", "size"),
        median_skill=("skill", "median"),
        beats_benchmark=("skill", lambda s: float((s > 0).mean())),
        median_smape=("smape", "median"),
    )
    return summary.reset_index().sort_values([*(by or []), "median_skill"], ascending=False)
