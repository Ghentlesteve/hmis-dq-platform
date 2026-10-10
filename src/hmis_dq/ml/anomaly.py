"""Multivariate anomaly detection on facility-months (Isolation Forest).

The rule-based outlier checks look at one indicator at a time. Here each
facility-month is a *profile* across related indicators (the immunisation
antigens), and the Isolation Forest flags unusual combinations, e.g. every
antigen dropping at once (stock-out, missed outreach, half-filled form) or one
antigen jumping while the rest stay normal (a typing error).

Features are scale-free: each antigen is expressed as log(value / the facility's
own median for that antigen), so a big hospital and a small health post with
their usual numbers both look "normal", and the model compares *shapes*.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Child Health antigens, by DHIS2 data element name
ANTIGENS = (
    "BCG doses given",
    "OPV1 doses given",
    "OPV3 doses given",
    "Penta1 doses given",
    "Penta2 doses given",
    "Penta3 doses given",
    "PCV1 doses given",
    "PCV3 doses given",
    "Measles doses given",
)

FACILITY_KEYS = ["org_unit_id", "facility", "district"]
MIN_ANTIGENS = 5  # fewer present than this: the profile is too thin to judge
ANOMALY_SHARE = 0.01  # flag the most unusual 1% of facility-months
LARGE_CHANGE = np.log(2)  # "doubled or halved" in log terms


def deviation_profiles(
    facility_months: pd.DataFrame, antigens: tuple[str, ...] = ANTIGENS
) -> pd.DataFrame:
    """One row per facility-month, one column per antigen: log(value / facility's usual).

    Missing antigens stay NaN here; ``score_anomalies`` decides how to handle them.
    """
    rows = facility_months[facility_months["data_element"].isin(antigens)].copy()
    usual = rows.groupby(["org_unit_id", "data_element"])["value"].transform("median")
    rows["deviation"] = np.log1p(rows["value"]) - np.log1p(usual)
    wide = rows.pivot_table(
        index=[*FACILITY_KEYS, "period"],
        columns="data_element",
        values="deviation",
        aggfunc="first",
    )
    return wide.reindex(columns=list(antigens)).reset_index()


def shape_features(deviations: pd.DataFrame) -> pd.DataFrame:
    """The antigen deviations plus three summaries of the profile's shape.

    An Isolation Forest splits on one random column at a time, so a pattern spread
    over many columns (all antigens down) or hidden in one of nine (a single typo)
    takes many splits to isolate. Summaries put each pattern in a single column:
    mean (all up / all down), max (one far off) and spread (out of step).
    """
    filled = deviations.fillna(0.0)
    return filled.assign(
        mean_deviation=deviations.mean(axis=1),
        max_deviation=deviations.abs().max(axis=1),
        spread=deviations.std(axis=1).fillna(0.0),
    )


def score_anomalies(
    profiles: pd.DataFrame,
    antigens: tuple[str, ...] = ANTIGENS,
    *,
    share: float = ANOMALY_SHARE,
    min_antigens: int = MIN_ANTIGENS,
    random_state: int = 0,
) -> pd.DataFrame:
    """Score every profile with an Isolation Forest; the top ``share`` are anomalies.

    A missing antigen is treated as "usual" (deviation 0) and counted in
    ``antigens_missing``, which is also a feature: a form with half the antigens
    blank is itself unusual.
    """
    columns = list(antigens)
    present = profiles[columns].notna().sum(axis=1)
    usable = profiles[present >= min_antigens].copy()
    usable["antigens_missing"] = len(columns) - present[usable.index]

    features = shape_features(usable[columns]).assign(antigens_missing=usable["antigens_missing"])
    forest = IsolationForest(n_estimators=300, contamination=share, random_state=random_state).fit(
        features
    )
    usable["anomaly_score"] = -forest.score_samples(features)  # higher = more unusual
    usable["is_anomaly"] = forest.predict(features) == -1
    return usable.sort_values("anomaly_score", ascending=False).reset_index(drop=True)


def explain(row: pd.Series, antigens: tuple[str, ...] = ANTIGENS) -> str:
    """Plain-language reason: which antigens moved, and the overall pattern.

    "2.0x" means (value + 1) / (usual + 1), i.e. about twice the facility's usual level.
    """
    deviations = row[list(antigens)].dropna().astype(float)
    ratios = pd.Series(np.exp(deviations.to_numpy()), index=deviations.index)
    big = deviations[deviations.abs() >= LARGE_CHANGE]
    if big.empty:
        return "Unusual combination of small changes across antigens"

    down, up = big[big < 0], big[big > 0]
    most = 0.6 * len(deviations)
    if len(down) >= most:
        return (
            f"{len(down)} of {len(deviations)} antigens far below usual "
            f"({_ratio_text(ratios, down)}): possible stock-out, missed outreach or partial report"
        )
    if len(up) >= most:
        return (
            f"{len(up)} of {len(deviations)} antigens far above usual "
            f"({_ratio_text(ratios, up)}): possible double entry or catch-up campaign"
        )
    parts = [f"up: {_ratio_text(ratios, up)}"] if len(up) else []
    parts += [f"down: {_ratio_text(ratios, down)}"] if len(down) else []
    return "Antigens out of step with each other (" + "; ".join(parts) + ")"


def _ratio_text(ratios: pd.Series, chosen: pd.Series, limit: int = 3) -> str:
    strongest = chosen.abs().sort_values(ascending=False).index[:limit]
    return ", ".join(
        f"{str(name).replace(' doses given', '')} {ratios[name]:.1f}x" for name in strongest
    )
