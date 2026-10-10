"""Build the LinkedIn Part 2 graphic from the results in the lake.

python docs/make_linkedin_part2.py      (lake running)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from hmis_dq.explore import lake

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#c9c8c2"
CHILD_HEALTH = "BfMAe6Itzgt"
OUT = Path(__file__).parent / "images" / "linkedin-part2.png"
MIN_LABELLED = 3  # segments narrower than this get no count label

MODEL_LABELS = {
    "seasonal_naive": "Same month last year (benchmark)",
    "seasonal_naive_2y": "Same month 2 years earlier",
    "ets": "Holt-Winters smoothing (28 series*)",
    "gbm_seasonal": "Gradient boosting (change from last year)",
    "gbm": "Gradient boosting (lag features)",
    "mean_3": "Average of last 3 months",
    "naive": "Last month",
}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    db = lake()
    summary = db.sql(
        "SELECT * FROM read_parquet('s3://gold/dhis2/ml/forecast_summary.parquet')"
    ).df()
    anomalies = db.sql("SELECT * FROM read_parquet('s3://gold/dhis2/ml/anomalies.parquet')").df()
    return summary[summary["dataset_id"] == CHILD_HEALTH], anomalies


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=11)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def forecast_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    rows = summary.sort_values("median_smape", ascending=False)
    labels = [MODEL_LABELS[m] for m in rows["model"]]
    colours = [BLUE if m == "seasonal_naive" else GREY for m in rows["model"]]
    bars = ax.barh(labels, rows["median_smape"], color=colours, height=0.62)
    for bar, value, model in zip(bars, rows["median_smape"], rows["model"], strict=True):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=11,
            color=INK if model == "seasonal_naive" else INK_2,
            fontweight="bold" if model == "seasonal_naive" else "normal",
        )
    style(ax)
    ax.set_xlim(0, rows["median_smape"].max() * 1.18)
    ax.set_xlabel("Median forecast error (sMAPE, %)  ·  lower is better", color=INK_2, fontsize=11)
    ax.set_title(
        "1. The simplest forecast wins",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
        pad=24,
    )
    ax.text(
        0,
        1.02,
        "Child Health, 52 district series, last 12 months each predicted without seeing it",
        transform=ax.transAxes,
        fontsize=11,
        color=INK_2,
    )


def anomaly_panel(ax: plt.Axes, anomalies: pd.DataFrame) -> None:
    pattern = anomalies["explanation"].str.extract(
        r"^(?:\d+ of \d+ antigens far (below|above)|(Antigens out of step))"
    )
    anomalies = anomalies.assign(
        pattern=pattern[0]
        .map({"below": "Most vaccines far BELOW usual", "above": "Most vaccines far ABOVE usual"})
        .fillna(pattern[1].map({"Antigens out of step": "Vaccines out of step"})),
        rules=anomalies["rule_severity"].fillna("missed").replace({"medium": "high"}),
    ).dropna(subset=["pattern"])
    counts = anomalies.groupby(["pattern", "rules"]).size().unstack(fill_value=0)
    order = [
        "Vaccines out of step",
        "Most vaccines far ABOVE usual",
        "Most vaccines far BELOW usual",
    ]
    counts = counts.reindex(order).fillna(0)

    segments = [
        ("high", "Rules flagged it (high)", BLUE),
        ("low", "Rules flagged it (low)", AQUA),
        ("missed", "Rules MISSED it", ORANGE),
    ]
    left = pd.Series(0.0, index=counts.index)
    for column, label, colour in segments:
        values = counts.get(column, pd.Series(0, index=counts.index)).astype(float)
        bars = ax.barh(
            counts.index,
            values,
            left=left,
            color=colour,
            height=0.62,
            label=label,
            edgecolor=SURFACE,
            linewidth=2,
        )
        for bar, value in zip(bars, values, strict=True):
            if value >= MIN_LABELLED:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white" if colour != AQUA else INK,
                    fontweight="bold",
                )
        left += values
    style(ax)
    ax.set_xlabel("Unusual facility-months flagged by the anomaly model", color=INK_2, fontsize=11)
    ax.set_title(
        "2. Machine learning finds the drops the rules miss",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
        pad=24,
    )
    ax.text(
        0,
        1.02,
        "Isolation Forest over 9 vaccines at once, 9,289 facility-months, top 1% flagged",
        transform=ax.transAxes,
        fontsize=11,
        color=INK_2,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=10.5, labelcolor=INK_2)


def main() -> None:
    summary, anomalies = load()
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10.8, 12), gridspec_kw={"height_ratios": [7, 4.2], "hspace": 0.55}
    )
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(
        "My best model lost to a one-line rule",
        x=0.03,
        y=0.985,
        ha="left",
        fontsize=21,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.03,
        0.948,
        "Building a DHIS2 data quality platform in public · Part 2",
        fontsize=12.5,
        color=INK_2,
    )
    forecast_panel(top, summary)
    anomaly_panel(bottom, anomalies)
    fig.text(
        0.03,
        0.015,
        "*needs 2 gap-free years.  Data: public DHIS2 demo database (synthetic).  "
        "github.com/Ghentlesteve/hmis-dq-platform",
        fontsize=10,
        color=INK_2,
    )
    fig.subplots_adjust(left=0.36, right=0.95, top=0.87, bottom=0.08)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
