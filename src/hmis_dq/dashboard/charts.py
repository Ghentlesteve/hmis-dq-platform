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


# ------------------------------------------------------------------ district

SEVERITY_COLOURS = {"high": "#184f95", "medium": "#3987e5", "low": "#86b6ef"}  # dark = worse
WARNING = GRADE_COLOURS["D"]  # status "critical"
LAST_YEAR = "#8c8b86"

_FACILITY_HOVER = (
    "<b>%{customdata[0]}</b><br>Overall %{customdata[1]:.1f} (grade %{customdata[2]})"
    "<br>Completeness %{customdata[3]:.1f}%<extra></extra>"
)


def facility_map(
    outline: pd.DataFrame, facilities: pd.DataFrame, *, dark: bool = False, height: int = 460
) -> go.Figure:
    """The district outline (one row of district_shapes) with its facilities as
    points coloured by grade."""
    geojson = district_geojson(outline)
    middle = MIDPOINT_DARK if dark else MIDPOINT_LIGHT
    fig = go.Figure(
        go.Choropleth(
            geojson=geojson,
            locations=outline["district_id"],
            z=[0],
            colorscale=[[0, middle], [1, middle]],
            showscale=False,
            marker={"line": {"color": NO_DATA, "width": 1}},
            hoverinfo="skip",
        )
    )
    placed = facilities.dropna(subset=["longitude", "latitude"])
    for grade, colour in GRADE_COLOURS.items():
        rows = placed[placed["grade"] == grade]
        if rows.empty:
            continue
        fig.add_trace(
            go.Scattergeo(
                lon=rows["longitude"],
                lat=rows["latitude"],
                mode="markers",
                name=GRADE_MEANING[grade],
                marker={
                    "size": 10,
                    "color": colour,
                    "line": {"color": middle, "width": 1.5},
                },
                customdata=rows[["facility", "overall", "grade", "completeness"]].to_numpy(),
                hovertemplate=_FACILITY_HOVER,
            )
        )
    (west, south), (east, north) = bounds(geojson)
    pad = 0.05 * max(east - west, north - south)
    fig.update_geos(
        projection_type="mercator",
        lonaxis_range=[west - pad, east + pad],
        lataxis_range=[south - pad, north + pad],
        visible=False,
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        height=height,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        legend={"title": {"text": "Facility grade"}, "orientation": "h", "y": 0, "x": 0},
        dragmode=False,
    )
    return fig


def findings_bars(counts: pd.DataFrame) -> go.Figure:
    """Findings per check, split by severity, most common check on top."""
    totals = counts.groupby("label")["findings"].sum().sort_values()
    fig = go.Figure()
    for severity, colour in SEVERITY_COLOURS.items():
        rows = counts[counts["severity"] == severity].set_index("label")["findings"]
        values = rows.reindex(totals.index).fillna(0)
        fig.add_trace(
            go.Bar(
                x=values,
                y=totals.index,
                orientation="h",
                name=severity,
                marker={"color": colour, "line": {"color": "rgba(0,0,0,0)", "width": 0}},
                hovertemplate="%{y}: %{x:,} " + severity + "<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=totals,
            y=totals.index,
            mode="text",
            text=[f"  {total:,}" for total in totals],
            textposition="middle right",
            hoverinfo="skip",
            showlegend=False,
            cliponaxis=False,
        )
    )
    fig.update_xaxes(range=[0, totals.max() * 1.18 if len(totals) else 1], showgrid=True)
    fig.update_layout(
        barmode="stack",
        bargap=0.35,
        height=60 + 32 * len(totals),
        margin={"l": 0, "r": 10, "t": 10, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"title": {"text": "Severity"}, "orientation": "h", "y": 1.02, "yanchor": "bottom"},
    )
    return fig


def trend_chart(trend: pd.DataFrame, warnings: pd.DataFrame, indicator: str) -> go.Figure:
    """An indicator's monthly totals, the same month a year earlier, and the months
    the early warning fired."""
    fig = go.Figure(
        [
            go.Scatter(
                x=trend["period_start"],
                y=trend["last_year"],
                name="Same month last year",
                mode="lines",
                line={"color": LAST_YEAR, "width": 2, "dash": "dot"},
                hovertemplate="%{x|%b %Y} last year: %{y:,.0f}<extra></extra>",
            ),
            go.Scatter(
                x=trend["period_start"],
                y=trend["value"],
                name=indicator,
                mode="lines",
                line={"color": BLUE, "width": 2},
                customdata=trend[["reports_received"]].to_numpy(),
                hovertemplate=("%{x|%b %Y}: %{y:,.0f} (%{customdata[0]} reports)<extra></extra>"),
            ),
        ]
    )
    if not warnings.empty:
        fig.add_trace(
            go.Scatter(
                x=warnings["period_start"],
                y=warnings["actual"],
                name="Early warning",
                mode="markers",
                marker={
                    "color": WARNING,
                    "size": 11,
                    "symbol": "triangle-down",
                    "line": {"color": BORDER, "width": 1.5},
                },
                customdata=warnings[["message"]].to_numpy(),
                hovertemplate="%{customdata[0]}<extra></extra>",
            )
        )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(rangemode="tozero", showgrid=True)
    fig.update_layout(
        height=340,
        margin={"l": 0, "r": 10, "t": 30, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0},
    )
    return fig
