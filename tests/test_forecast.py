"""Forecasting tests: models, leakage-free backtest, global model, metrics. No Spark."""

import duckdb
import numpy as np
import pandas as pd
import pytest

from hmis_dq.ml import forecast, job
from hmis_dq.ml.forecast import (
    backtest_gbm,
    backtest_series,
    ets,
    mean_3,
    naive,
    seasonal_naive,
    seasonal_naive_2y,
    series_metrics,
    summarise,
)

PATTERN = [20.0, 18, 22, 25, 30, 35, 40, 38, 30, 26, 22, 21]


def monthly(values: list[float], start: str = "2023-01-01") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="MS"))


# ------------------------------------------------------------------ models


def test_baselines() -> None:
    history = monthly([*PATTERN, 10.0, np.nan, 14.0])

    assert naive(history) == 14.0
    assert mean_3(history) == 12.0  # mean of 10 and 14, ignoring the gap
    assert seasonal_naive(history) == PATTERN[3]  # same month last year (April)


def test_seasonal_naive_needs_a_full_year() -> None:
    assert np.isnan(seasonal_naive(monthly(PATTERN[:11])))
    assert np.isnan(seasonal_naive_2y(monthly(PATTERN * 2)[:23]))
    assert seasonal_naive_2y(monthly(PATTERN * 2)) == PATTERN[0]


def test_ets_learns_a_clean_seasonal_pattern() -> None:
    forecast_value = ets(monthly(PATTERN * 3))

    assert forecast_value == pytest.approx(PATTERN[0], abs=1.0)


def test_ets_needs_two_gapless_years() -> None:
    with_gap = PATTERN * 3
    with_gap[-20] = np.nan  # leaves only 19 months after the gap

    assert np.isnan(ets(monthly(PATTERN * 2)[:-1]))
    assert np.isnan(ets(monthly(with_gap)))


# ---------------------------------------------------------------- backtest


def test_backtest_never_shows_a_model_the_month_it_predicts() -> None:
    values = monthly(list(range(1, 31)), "2023-01-01")
    seen: list[tuple[pd.Timestamp, int]] = []

    def spy(history: pd.Series) -> float:
        seen.append((history.index[-1], len(history)))
        return 0.0

    result = backtest_series(values, {"spy": spy}, test_months=6)

    assert len(result) == 6
    for (last_seen, _), predicted in zip(seen, result["period_start"], strict=True):
        assert last_seen < predicted


def test_backtest_skips_months_with_no_actual() -> None:
    values = monthly([*PATTERN, *PATTERN[:5], np.nan, *PATTERN[6:]])

    result = backtest_series(values, {"naive": naive}, test_months=12)

    assert len(result) == 11


def synthetic_panel(series: int = 4, years: int = 3) -> pd.DataFrame:
    rows = []
    for s in range(series):
        scale = 10 * (s + 1)
        for i, month in enumerate(pd.date_range("2023-01-01", periods=12 * years, freq="MS")):
            rows.append(
                {
                    "district_id": f"d{s}",
                    "period_start": month,
                    "value": scale * PATTERN[i % 12] / 20,
                }
            )
    return pd.DataFrame(rows)


def test_global_model_trains_only_on_earlier_months() -> None:
    panel = synthetic_panel()

    result = backtest_gbm(panel, ["district_id"], test_months=6, min_train_rows=20)

    assert sorted(result["period_start"].unique()) == list(
        pd.date_range("2025-07-01", periods=6, freq="MS")
    )
    assert len(result) == 6 * 4
    assert (result["forecast"] >= 0).all()
    # a clean repeating pattern is easy: forecasts land close to the truth
    assert ((result["forecast"] - result["actual"]).abs() / result["actual"]).median() < 0.15


def test_seasonal_residual_model_builds_on_last_year() -> None:
    panel = synthetic_panel()

    result = backtest_gbm(
        panel, ["district_id"], test_months=6, min_train_rows=20, seasonal_residual=True
    )

    assert set(result["model"]) == {"gbm_seasonal"}
    # an exactly repeating year is "no change from last year": near-perfect forecasts
    assert ((result["forecast"] - result["actual"]).abs() / result["actual"]).max() < 0.02


def test_lag_features_use_only_the_past() -> None:
    panel = synthetic_panel(series=1, years=2)

    features = forecast.lag_features(panel, ["district_id"])

    january_2024 = features[features["period_start"] == "2024-01-01"].iloc[0]
    assert january_2024["lag_1"] == pytest.approx(np.log1p(10 * PATTERN[11] / 20))
    assert january_2024["lag_12"] == pytest.approx(np.log1p(10 * PATTERN[0] / 20))
    assert np.isnan(january_2024["lag_24"])


# ----------------------------------------------------------------- metrics


def test_skill_compares_with_the_benchmark_on_the_same_months() -> None:
    months = pd.date_range("2025-01-01", periods=3, freq="MS")
    backtest = pd.DataFrame(
        {
            "district_id": "d1",
            "period_start": list(months) * 2 + [months[0]],
            "model": ["seasonal_naive"] * 3 + ["good"] * 3 + ["sparse"],
            "actual": [10.0, 10, 10] * 2 + [10],
            # benchmark errors 4,4,4; "good" errors 1,1,1; "sparse" only Jan, error 0
            "forecast": [14.0, 6, 14, 11, 9, 11, 10],
        }
    )

    metrics = series_metrics(backtest, ["district_id"]).set_index("model")

    assert metrics.loc["good", "mae"] == 1.0
    assert metrics.loc["good", "skill"] == pytest.approx(0.75)  # 1 - 1/4
    assert metrics.loc["seasonal_naive", "skill"] == 0.0
    assert metrics.loc["sparse", "forecasts"] == 1  # judged on January only
    assert metrics.loc["sparse", "skill"] == 1.0

    summary = summarise(metrics.reset_index()).set_index("model")
    assert summary.loc["good", "beats_benchmark"] == 1.0


# ------------------------------------------------------------------- data


def test_load_series_fills_missing_months(monkeypatch: pytest.MonkeyPatch) -> None:
    db = duckdb.connect()
    db.sql(
        """
        CREATE TABLE dm AS SELECT * FROM (VALUES
            ('ds1', 'd1', 'Bo', 'de1', 'Penta1 doses given', DATE '2025-01-01', 10.0),
            ('ds1', 'd1', 'Bo', 'de1', 'Penta1 doses given', DATE '2025-03-01', 12.0),
            ('ds1', 'd2', 'Kono', 'de1', 'Penta1 doses given', DATE '2025-02-01', 7.0),
            ('ds1', NULL, NULL, 'de1', 'Penta1 doses given', DATE '2025-02-01', 1.0),
            ('ds1', 'd1', 'Bo', 'de9', 'Not a tracer', DATE '2025-02-01', 99.0)
        ) t(dataset_id, district_id, district, data_element_id, data_element, period_start, value)
        """
    )
    monkeypatch.setattr(job, "gold", lambda table: "dm")

    panel = job.load_series(db, ("Penta1 doses given",))

    assert len(panel) == 6  # 2 districts x 3 months (Jan-Mar), no NULL district or non-tracer
    bo = panel[panel["district"] == "Bo"].sort_values("period_start")["value"].tolist()
    assert bo[0] == 10.0 and np.isnan(bo[1]) and bo[2] == 12.0
