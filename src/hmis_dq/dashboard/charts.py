"""Plotly figures for the dashboard. Pure functions: DataFrames in, figures out.

Grades are a status (good -> critical) and use the fixed status colours; anything
coloured is also labelled, so colour is never the only way to read a chart.
"""

import json
import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

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


def _outline(geometry: dict[str, Any]) -> tuple[list[float | None], list[float | None]]:
    """x (lon) and y (lat) of every polygon's outer ring, separated by gaps, for a
    filled Scatter trace."""
    polygons = (
        geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
    )
    xs: list[float | None] = []
    ys: list[float | None] = []
    for polygon in polygons:
        for lon, lat, *_ in polygon[0]:
            xs.append(lon)
            ys.append(lat)
        xs.append(None)
        ys.append(None)
    return xs, ys


def _as_map(fig: go.Figure, geojson: dict[str, Any], pad: float, height: int) -> go.Figure:
    """Plain x/y axes as a map: longitude across, latitude up, true proportions.

    Shapes are drawn as filled outlines rather than with Plotly's geo maps, which
    download a world map from the internet before drawing anything: slow on a
    weak connection, and blank with none.
    """
    (west, south), (east, north) = bounds(geojson)
    margin = pad * max(east - west, north - south)
    hidden = {"visible": False, "fixedrange": True}
    fig.update_xaxes(range=[west - margin, east + margin], **hidden)
    fig.update_yaxes(
        range=[south - margin, north + margin],
        # a degree of longitude is shorter than one of latitude away from the equator
        scaleanchor="x",
        scaleratio=1 / math.cos(math.radians((south + north) / 2)),
        **hidden,
    )
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        dragmode=False,
        hovermode="closest",
    )
    return fig


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
    Clicking a district's label selects it: each label point carries the
    district_id as customdata.
    """
    geojson = district_geojson(shapes)
    merged = shapes[["district_id", "geometry"]].merge(districts, on="district_id", how="left")
    scored = merged.dropna(subset=["overall"])
    gap = scored["overall"] - national_score
    reach = max(1.0, float(np.ceil(gap.abs().max()))) if not gap.empty else 1.0
    middle = MIDPOINT_DARK if dark else MIDPOINT_LIGHT
    border = middle if dark else BORDER
    scale = [[0.0, BELOW], [0.5, middle], [1.0, ABOVE]]
    fills = sample_colorscale(scale, ((gap + reach) / (2 * reach)).clip(0, 1).tolist())
    hover = [
        f"<b>{row.district}</b><br>Overall {row.overall:.1f} (grade {row.grade})"
        f"<br>Completeness {row.completeness:.1f}%<br>Facilities graded D: "
        f"{row.facilities_grade_d}"
        for row in scored.itertuples()
    ]

    fig = go.Figure()
    for row, colour, text in zip(scored.itertuples(), fills, hover, strict=True):
        xs, ys = _outline(json.loads(str(row.geometry)))
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                fill="toself",
                fillcolor=colour,
                mode="lines",
                line={"color": border, "width": 1.5},
                hoveron="fills",
                text=text,
                hoverinfo="text",
                showlegend=False,
            )
        )
    for row in merged[merged["overall"].isna()].itertuples():
        xs, ys = _outline(json.loads(str(row.geometry)))
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                fill="toself",
                fillcolor=NO_DATA,
                mode="lines",
                line={"color": border, "width": 1.5},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    points = [label_point(json.loads(g)) for g in scored["geometry"]]
    fig.add_trace(
        go.Scatter(
            x=[p[0] for p in points],
            y=[p[1] for p in points],
            text=[
                f"<b>{row.district}</b><br>{row.overall:.1f} · {row.grade}"
                for row in scored.itertuples()
            ],
            mode="markers+text",
            # an invisible marker under each label, so the label can be clicked
            marker={"size": 34, "opacity": 0},
            textfont={"size": 11, "color": INK_DARK if dark else INK, "shadow": "auto"},
            customdata=scored[["district_id"]].to_numpy(),
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        )
    )
    # the colour key: an empty trace that only draws its colour bar
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={
                "colorscale": scale,
                "cmin": -reach,
                "cmax": reach,
                "color": [0],
                "showscale": True,
                "colorbar": {
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
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 70})
    return _as_map(fig, geojson, pad=0.03, height=height)


def clicked_district(point: dict[str, Any]) -> str | None:
    """The district_id a click on the district map points at (its label carries it)."""
    data = point.get("customdata")
    if isinstance(data, list):
        data = data[0] if data else None
    return str(data) if data else None


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
    xs, ys = _outline(geojson["features"][0]["geometry"])
    fig = go.Figure(
        go.Scatter(
            x=xs,
            y=ys,
            fill="toself",
            fillcolor=middle,
            mode="lines",
            line={"color": NO_DATA, "width": 1},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    placed = facilities.dropna(subset=["longitude", "latitude"])
    for grade, colour in GRADE_COLOURS.items():
        rows = placed[placed["grade"] == grade]
        if rows.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=rows["longitude"],
                y=rows["latitude"],
                mode="markers",
                name=GRADE_MEANING[grade],
                marker={"size": 10, "color": colour, "line": {"color": middle, "width": 1.5}},
                customdata=rows[["facility", "overall", "grade", "completeness"]].to_numpy(),
                hovertemplate=_FACILITY_HOVER,
            )
        )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend={"title": {"text": "Facility grade"}, "orientation": "h", "y": 0, "x": 0},
    )
    return _as_map(fig, geojson, pad=0.05, height=height)


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
