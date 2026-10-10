"""Plotly figures for the dashboard. Pure functions: DataFrames in, figures out.

Grades are a status (good -> critical) and use the fixed status colours; anything
coloured is also labelled, so colour is never the only way to read a chart.
"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from hmis_dq.dashboard.data import bounds, district_geojson, label_point
from hmis_dq.spark.dq.rules import DIMENSION_WEIGHTS, GRADES

GRADE_COLOURS = {"A": "#0ca30c", "B": "#fab219", "C": "#ec835a", "D": "#d03b3b"}


def _grade_meanings() -> dict[str, str]:
    """{"A": "A (90+)", "B": "B (75-89)", ..., "D": "D (below 60)"} from GRADES."""
    meanings, upper = {}, None
    for threshold, letter in GRADES:
        span = f"{threshold}+" if upper is None else f"{threshold}-{upper - 1}"
        meanings[letter] = f"{letter} ({span})"
        upper = threshold
    meanings["D"] = f"D (below {upper})"
    return meanings


GRADE_MEANING = _grade_meanings()
NO_DATA = "#c9c8c2"
INK, INK_DARK = "#0b0b0b", "#fcfcfb"
BELOW, ABOVE = "#e34948", "#2a78d6"  # diverging poles: worse / better than national
MIDPOINT_LIGHT, MIDPOINT_DARK = "#f0efec", "#383835"
BLUE = "#2a78d6"
BORDER = "#fcfcfb"

DIMENSIONS = {
    "completeness": "Completeness",
    "accuracy": "Accuracy (outliers)",
    "consistency": "Internal consistency",
    "integrity": "Integrity (copies, timestamps)",
}

_HOVER = (
    "<b>%{customdata[0]}</b><br>Overall %{customdata[1]:.1f} (grade %{customdata[2]})"
    "<br>Completeness %{customdata[3]:.1f}%<br>Facilities graded D: %{customdata[4]}"
    "<extra></extra>"
)


def district_map(
    districts: pd.DataFrame,
    shapes: pd.DataFrame,
    national_score: float,
    *,
    dark: bool = False,
    height: int = 560,
) -> go.Figure:
    """Districts coloured by how far their overall score is from the national one.

    Diverging: red below national, blue above, grey at national. On the demo data
    every Child Health district is graded D, so colouring by grade would paint one
    flat colour; distance from national shows where to look first. Each district
    is labelled with its score and grade, so colour is never needed to read it.
    """
    geojson = district_geojson(shapes)
    merged = shapes[["district_id", "geometry"]].merge(districts, on="district_id", how="left")
    scored = merged.dropna(subset=["overall"])
    gap = scored["overall"] - national_score
    reach = max(1.0, float(np.ceil(gap.abs().max()))) if not gap.empty else 1.0
    middle = MIDPOINT_DARK if dark else MIDPOINT_LIGHT
    ink = INK_DARK if dark else INK

    fig = go.Figure(
        go.Choropleth(
            geojson=geojson,
            locations=scored["district_id"],
            z=gap,
            zmin=-reach,
            zmax=reach,
            colorscale=[[0, BELOW], [0.5, middle], [1, ABOVE]],
            marker={"line": {"color": middle if dark else BORDER, "width": 1.5}},
            customdata=scored[
                ["district", "overall", "grade", "completeness", "facilities_grade_d"]
            ].to_numpy(),
            hovertemplate=_HOVER,
            colorbar={
                "title": {
                    "text": f"Points below / above the national score ({national_score:.1f})",
                    "side": "top",
                },
                "orientation": "h",
                "x": 0.5,
                "xanchor": "center",
                "y": 0,
                "yanchor": "top",
                "len": 0.7,
                "thickness": 12,
                "outlinewidth": 0,
                "tickvals": [-reach, 0, reach],
                "ticktext": [f"-{reach:.0f}", "national", f"+{reach:.0f}"],
            },
        )
    )
    unscored = merged[merged["overall"].isna()]
    if not unscored.empty:
        fig.add_trace(
            go.Choropleth(
                geojson=geojson,
                locations=unscored["district_id"],
                z=[0] * len(unscored),
                colorscale=[[0, NO_DATA], [1, NO_DATA]],
                showscale=False,
                hoverinfo="skip",
            )
        )

    points = [label_point(json.loads(g)) for g in scored["geometry"]]
    fig.add_trace(
        go.Scattergeo(
            lon=[p[0] for p in points],
            lat=[p[1] for p in points],
            text=[
                f"<b>{row.district}</b><br>{row.overall:.1f} · {row.grade}"
                for row in scored.itertuples()
            ],
            mode="text",
            textfont={"size": 11, "color": ink, "shadow": "auto"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    (west, south), (east, north) = bounds(geojson)
    pad = 0.03 * max(east - west, north - south)
    fig.update_geos(
        projection_type="mercator",
        lonaxis_range=[west - pad, east + pad],
        lataxis_range=[south - pad, north + pad],
        visible=False,
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        height=height,
        margin={"l": 0, "r": 0, "t": 0, "b": 70},
        paper_bgcolor="rgba(0,0,0,0)",
        dragmode=False,
    )
    return fig


def dimension_bars(scores: pd.Series) -> go.Figure:
    """The four dimension scores, and how many overall points each one costs."""
    rows = [
        (label, float(scores[column]), DIMENSION_WEIGHTS[column] * (100 - float(scores[column])))
        for column, label in DIMENSIONS.items()
        if pd.notna(scores[column])
    ]
    labels, values, lost = zip(*rows, strict=True)
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": BLUE, "cornerradius": 4},
            width=0.55,
            text=[f"{v:.1f}  (-{c:.1f} pts)" for v, c in zip(values, lost, strict=True)],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1f}<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, 135], showgrid=True, tickvals=[0, 25, 50, 75, 100])
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        height=230,
        margin={"l": 0, "r": 10, "t": 10, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
