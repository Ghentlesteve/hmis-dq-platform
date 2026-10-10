"""Anomaly detection tests on synthetic facilities with planted problems. No Spark."""

import numpy as np
import pandas as pd
import pytest

from hmis_dq.ml.anomaly import (
    ANTIGENS,
    deviation_profiles,
    explain,
    score_anomalies,
    shape_features,
)
from hmis_dq.ml.job import compare_with_rules

RNG = np.random.default_rng(42)


def facility_months(
    facilities: int = 40,
    months: int = 24,
    *,
    plant: dict[tuple[str, str], dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Facilities of different sizes reporting every antigen with small noise.

    ``plant`` overrides values: {(facility, period): {antigen: value}}.
    """
    plant = plant or {}
    rows = []
    for f in range(facilities):
        size = 10 * (1 + f % 8)  # sizes from 10 to 80 doses a month
        for m in range(months):
            period = f"{2024 + m // 12}{m % 12 + 1:02d}"
            for antigen in ANTIGENS:
                value = max(0.0, round(size * RNG.normal(1.0, 0.08)))
                value = plant.get((f"f{f}", period), {}).get(antigen, value)
                rows.append(
                    {
                        "org_unit_id": f"f{f}",
                        "facility": f"Facility {f}",
                        "district": "Bo",
                        "period": period,
                        "data_element": antigen,
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


def test_profiles_are_scale_free() -> None:
    profiles = deviation_profiles(facility_months(facilities=8, months=12))

    # a facility reporting its usual numbers sits near 0, whatever its size
    assert profiles[list(ANTIGENS)].abs().median().max() < 0.1


def test_everything_dropping_at_once_is_the_top_anomaly() -> None:
    stockout = {antigen: 1.0 for antigen in ANTIGENS}  # a 60-dose facility reports ~1 of each
    data = facility_months(plant={("f5", "202503"): stockout})

    scored = score_anomalies(deviation_profiles(data))
    top = scored.iloc[0]

    assert (top["org_unit_id"], top["period"]) == ("f5", "202503")
    assert top["is_anomaly"]
    assert "far below usual" in explain(top)
    assert "stock-out" in explain(top)


def test_one_antigen_out_of_step_is_explained_by_name() -> None:
    data = facility_months(plant={("f3", "202410"): {"Penta3 doses given": 400.0}})

    scored = score_anomalies(deviation_profiles(data))
    top = scored.iloc[0]

    assert (top["org_unit_id"], top["period"]) == ("f3", "202410")
    reason = explain(top)
    assert reason.startswith("Antigens out of step")
    assert "Penta3" in reason


def test_thin_profiles_are_not_scored() -> None:
    data = facility_months(facilities=10, months=12)
    thin = data[~((data["org_unit_id"] == "f1") & ~data["data_element"].isin(ANTIGENS[:3]))]

    scored = score_anomalies(deviation_profiles(thin))

    assert "f1" not in set(scored["org_unit_id"])  # only 3 antigens: below the minimum


def test_share_controls_how_many_are_flagged() -> None:
    scored = score_anomalies(deviation_profiles(facility_months()), share=0.05)

    assert scored["is_anomaly"].mean() == pytest.approx(0.05, abs=0.01)


def test_explain_mild_combination() -> None:
    row = pd.Series({antigen: 0.2 for antigen in ANTIGENS})

    assert explain(row) == "Unusual combination of small changes across antigens"


def test_compare_with_rules_keeps_the_strongest_outlier() -> None:
    anomalies = pd.DataFrame({"org_unit_id": ["a", "b"], "period": ["202501", "202501"]})
    outliers = pd.DataFrame(
        {
            "org_unit_id": ["a", "a"],
            "period": ["202501", "202501"],
            "severity": ["low", "high"],
        }
    )

    merged = compare_with_rules(anomalies, outliers).set_index("org_unit_id")

    assert merged.loc["a", "rule_severity"] == "high"
    assert pd.isna(merged.loc["b", "rule_severity"])  # new: no rule caught it


def test_shape_features_summarise_the_profile() -> None:
    deviations = pd.DataFrame(
        {"a": [-2.0, 0.0], "b": [-2.0, 3.0], "c": [-2.0, np.nan]}  # all down / one far off
    )

    features = shape_features(deviations)

    assert features["mean_deviation"].tolist() == [-2.0, 1.5]
    assert features["max_deviation"].tolist() == [2.0, 3.0]
    assert features["spread"][0] == 0.0  # all moved together
    assert features["c"][1] == 0.0  # missing antigen treated as usual
