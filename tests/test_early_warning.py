"""Early-warning tests: honest ranges, shortfalls, and the likely cause. No Spark."""

import numpy as np
import pandas as pd
import pytest

from hmis_dq.ml.early_warning import (
    WarningRules,
    district_warnings,
    error_bounds,
    recent_facility_drops,
    warning_history,
)

MONTHS = pd.date_range("2025-09-01", periods=13, freq="MS")  # Sep 2025 .. Sep 2026
LATEST = MONTHS[-1]
DISTRICTS = [f"d{i}" for i in range(13)]
RNG = np.random.default_rng(7)


def forecasts(latest_actuals: dict[str, float] | None = None) -> pd.DataFrame:
    """Seasonal-naive forecasts of 100 with ~5% noise in past months, 13 districts."""
    latest_actuals = latest_actuals or {}
    rows = []
    for district in DISTRICTS:
        for month in MONTHS:
            actual = float(round(100 * RNG.normal(1.0, 0.05)))
            if month == LATEST:
                # only planted districts deviate in the month checked; random noise
                # would put ~1 in 20 below a 5th-percentile bound by chance
                actual = latest_actuals.get(district, 100.0)
            rows.append(
                {
                    "dataset_id": "ds1",
                    "district_id": district,
                    "district": district.upper(),
                    "data_element_id": "de1",
                    "data_element": "Penta1 doses given",
                    "period_start": month,
                    "model": "seasonal_naive",
                    "actual": float(actual),
                    "forecast": 100.0,
                }
            )
    return pd.DataFrame(rows)


def reporting(latest_received: dict[str, float] | None = None) -> pd.DataFrame:
    latest_received = latest_received or {}
    return pd.DataFrame(
        [
            {
                "dataset_id": "ds1",
                "district_id": district,
                "period_start": month,
                "reports_received": latest_received.get(district, 50.0)
                if month == LATEST
                else 50.0,
            }
            for district in DISTRICTS
            for month in MONTHS
        ]
    )


def test_ranges_use_only_months_before_the_one_checked() -> None:
    data = forecasts({"d0": 5000.0})  # an absurd latest value must not widen the range

    bounds = error_bounds(data, LATEST).iloc[0]

    assert bounds["errors"] == 13 * 12
    assert bounds["q_high"] < 0.2  # ~5% noise, not the latest month's 50x


def test_too_few_past_errors_gives_no_range() -> None:
    bounds = error_bounds(forecasts(), LATEST, WarningRules(min_errors=1000))

    assert bounds.empty


def test_normal_month_raises_no_warning() -> None:
    assert district_warnings(forecasts(), reporting()).empty


def test_shortfall_with_stable_reporting_is_a_service_decline() -> None:
    warnings = district_warnings(forecasts({"d3": 50.0}), reporting())

    (row,) = warnings.to_dict("records")
    assert row["district_id"] == "d3"
    assert row["likely_cause"] == "service decline"
    assert row["severity"] == "high"  # far more than 25% below the range
    assert row["message"].startswith("D3, Penta1 doses given, September 2026: 50, expected")
    assert "50 reports vs usual 50: likely service decline" in row["message"]


def test_shortfall_with_missing_reports_is_a_reporting_drop() -> None:
    warnings = district_warnings(forecasts({"d5": 60.0}), reporting({"d5": 30.0}))

    (row,) = warnings.to_dict("records")
    assert row["likely_cause"] == "reporting drop"
    assert "30 reports vs usual 50" in row["message"]


def test_small_shortfall_is_medium_and_warnings_are_ranked() -> None:
    warnings = district_warnings(forecasts({"d1": 50.0, "d2": 85.0}), reporting())

    assert warnings["district_id"].tolist() == ["d1", "d2"]
    assert warnings["severity"].tolist() == ["high", "medium"]
    assert warnings["shortfall"].iloc[0] > warnings["shortfall"].iloc[1]


def test_recent_facility_drops_keep_only_the_last_months() -> None:
    anomalies = pd.DataFrame(
        {
            "facility": ["A", "B", "C", "D"],
            "period": ["202609", "202607", "202606", "202609"],
            "anomaly_score": [0.7, 0.8, 0.9, 0.6],
            "explanation": [
                "8 of 9 antigens far below usual (...)",
                "7 of 9 antigens far below usual (...)",
                "8 of 9 antigens far below usual (...)",  # 4 months ago: too old
                "Antigens out of step with each other (...)",  # not a drop
            ],
        }
    )

    drops = recent_facility_drops(anomalies, "202609")

    assert drops["facility"].tolist() == ["B", "A"]  # most unusual first


@pytest.mark.parametrize(
    ("received", "cause"), [(40.0, "service decline"), (39.0, "reporting drop")]
)
def test_reporting_drop_threshold(received: float, cause: str) -> None:
    # 80% of usual (50) is 40: at 40 reporting counts as normal
    warnings = district_warnings(forecasts({"d0": 40.0}), reporting({"d0": received}))

    assert warnings.loc[0, "likely_cause"] == cause


def test_history_replays_each_month_with_only_the_data_known_then() -> None:
    data = forecasts()
    data.loc[data["period_start"] >= "2026-07-01", "actual"] = 100.0  # no chance shortfalls
    # a shortfall in July 2026 at d4, recovered by the latest month
    july_d4 = (data["district_id"] == "d4") & (data["period_start"] == "2026-07-01")
    data.loc[july_d4, "actual"] = 40.0

    history = warning_history(data, reporting(), months=3)

    assert history[["district_id", "period_start"]].values.tolist() == [
        ["d4", pd.Timestamp("2026-07-01")]
    ]
    assert district_warnings(data, reporting()).empty  # the latest month alone misses it
