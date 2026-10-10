"""HMIS data quality dashboard (Streamlit).

    hmis-dq dashboard          (lake running; opens http://localhost:8501)

Streamlit reruns this script top to bottom on every interaction; the lake is
read through cached functions, so only the first view of a dataset queries it.
"""

import pandas as pd
import streamlit as st

from hmis_dq.dashboard.charts import (
    GRADE_MEANING,
    dimension_bars,
    district_map,
    facility_map,
    findings_bars,
    trend_chart,
)
from hmis_dq.dashboard.data import FINDINGS_SHOWN, DashboardData, open_dashboard_data
from hmis_dq.extract.catalog import DATASETS

CACHE_SECONDS = 600
REPO = "https://github.com/Ghentlesteve/hmis-dq-platform"
NO_TOOLBAR = {"displayModeBar": False}


@st.cache_resource
def data() -> DashboardData:
    return open_dashboard_data()


@st.cache_data(ttl=CACHE_SECONDS, show_spinner="Reading the lake...")
def datasets() -> pd.DataFrame:
    return data().datasets()


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def national(dataset_id: str) -> pd.Series:
    return data().national(dataset_id)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner="Reading the lake...")
def districts(dataset_id: str) -> pd.DataFrame:
    return data().districts(dataset_id)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def shapes() -> pd.DataFrame:
    return data().district_shapes()


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def latest_month(dataset_id: str) -> pd.Timestamp:
    return data().latest_month(dataset_id)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def anomalies() -> pd.DataFrame:
    return data().anomalies()


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def warnings(dataset_id: str) -> pd.DataFrame:
    return data().warnings(dataset_id)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner="Reading the district...")
def facilities(dataset_id: str, district_id: str) -> pd.DataFrame:
    return data().facilities(dataset_id, district_id)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def finding_counts(dataset_id: str, district_id: str) -> pd.DataFrame:
    return data().finding_counts(dataset_id, district_id)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def findings(dataset_id: str, district_id: str, check: str) -> pd.DataFrame:
    return data().findings(dataset_id, district_id, check)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def indicators(dataset_id: str) -> list[str]:
    return data().indicators(dataset_id)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def trend(dataset_id: str, district_id: str, data_element: str) -> pd.DataFrame:
    return data().trend(dataset_id, district_id, data_element)


def dark_mode() -> bool:
    return st.context.theme.type == "dark"


def sidebar() -> str:
    """Dataset picker and the notes a first-time viewer needs. Returns the dataset id."""
    options = datasets()
    names = dict(zip(options["dataset_id"], options["name"], strict=True))
    with st.sidebar:
        dataset_id: str = st.radio(
            "Dataset", list(names), format_func=names.__getitem__, key="dataset"
        )
        st.markdown(
            "**How scores work.** Each facility gets four WHO Data Quality Review "
            "dimension scores (100 = no problems found), combined 35/25/20/20 into "
            "an overall score. Districts weight facilities by the reports they owe.\n\n"
            + "\n".join(f"- {meaning}" for meaning in GRADE_MEANING.values())
        )
        st.caption(
            f"Data: the public DHIS2 Sierra Leone demo database (synthetic). [Source code]({REPO})"
        )
    return dataset_id


# ------------------------------------------------------------------ national


def headline(dataset_id: str) -> None:
    scores = national(dataset_id)
    month = latest_month(dataset_id)
    table = districts(dataset_id)
    flagged = anomalies()

    tiles = st.columns(5)
    tiles[0].metric(
        "Overall score",
        f"{scores['overall']:.1f} · {scores['grade']}",
        help="Weighted mean of the four dimension scores below.",
    )
    tiles[1].metric(
        "Reporting completeness",
        f"{scores['completeness']:.1f}%",
        help=f"{scores['reports_received']:,} of {scores['reports_expected']:,} expected "
        "monthly reports received.",
    )
    tiles[2].metric(
        "Facilities graded D",
        f"{scores['facilities_grade_d']:,}",
        help=f"Out of {scores['facilities']:,} facilities assigned this dataset.",
    )
    tiles[3].metric(
        f"Early warnings, {month:%b %Y}",
        f"{int(table['warnings'].sum())}",
        help="District x indicator values below their expected range in the latest month.",
    )
    is_child_health = dataset_id == DATASETS["child_health"]
    tiles[4].metric(
        "Anomalous facility-months",
        f"{len(flagged):,}" if is_child_health and not flagged.empty else "n/a",
        help="Isolation Forest over 9 vaccines at once (Child Health only).",
    )


def overview(dataset_id: str) -> None:
    table = districts(dataset_id)
    scores = national(dataset_id)
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Districts against the national score")
        figure = district_map(table, shapes(), float(scores["overall"]), dark=dark_mode())
        clicked = st.plotly_chart(
            figure, config=NO_TOOLBAR, key="map", on_select="rerun", selection_mode="points"
        )
        # Clicking a district opens it below. The map keeps its selection across
        # reruns, so act only on a new click, or it would undo the picker below.
        points = clicked.selection.points if clicked else []
        picked = points[0].get("location") if points else None
        if picked and picked != st.session_state.get("map_pick"):
            st.session_state["district"] = picked
        st.session_state["map_pick"] = picked
        st.caption("Click a district to open it below.")
    with right:
        st.subheader("Ranking, worst first")
        columns = ["district", "grade", "overall", "completeness", "warnings"]
        st.dataframe(
            table[columns],
            hide_index=True,
            height=36 + 35 * min(len(table), 10),
            column_config={
                "district": "District",
                "grade": st.column_config.TextColumn("Grade", width="small"),
                "overall": st.column_config.ProgressColumn(
                    "Overall", min_value=0, max_value=100, format="%.1f", width="medium"
                ),
                "completeness": st.column_config.NumberColumn(
                    "Complete %", format="%.1f", width="small"
                ),
                "warnings": st.column_config.NumberColumn(
                    "Warnings", help="Early warnings in the latest month", width="small"
                ),
            },
        )
        st.subheader("Where the national score is lost")
        st.plotly_chart(dimension_bars(scores), config=NO_TOOLBAR, key="dims")


# ------------------------------------------------------------------ district


def district_view(dataset_id: str) -> None:
    table = districts(dataset_id).sort_values("district")
    names = dict(zip(table["district_id"], table["district"], strict=True))
    if st.session_state.get("district") not in names:
        # open on the worst district
        st.session_state["district"] = table.sort_values("overall")["district_id"].iloc[0]

    st.divider()
    district_id: str = st.selectbox(
        "District", list(names), format_func=names.__getitem__, key="district"
    )
    row = table.set_index("district_id").loc[district_id]
    st.header(f"{row['district']} district")

    tiles = st.columns(4)
    tiles[0].metric(
        "Overall score",
        f"{row['overall']:.1f} · {row['grade']}",
        delta=f"{row['overall'] - national(dataset_id)['overall']:+.1f} vs national",
    )
    tiles[1].metric(
        "Reporting completeness",
        f"{row['completeness']:.1f}%",
        help=f"{row['reports_received']:,} of {row['reports_expected']:,} expected reports.",
    )
    tiles[2].metric("Facilities graded D", f"{row['facilities_grade_d']} of {row['facilities']}")
    tiles[3].metric(f"Early warnings, {latest_month(dataset_id):%b %Y}", f"{row['warnings']}")

    district_facilities(dataset_id, district_id)
    district_findings(dataset_id, district_id)
    district_trends(dataset_id, district_id)
    district_anomalies(dataset_id, str(row["district"]))


def district_facilities(dataset_id: str, district_id: str) -> None:
    rows = facilities(dataset_id, district_id)
    outlines = shapes()
    outline = outlines[outlines["district_id"] == district_id]
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Facilities")
        if not outline.empty:
            st.plotly_chart(
                facility_map(outline, rows, dark=dark_mode()),
                config=NO_TOOLBAR,
                key="facility_map",
            )
        unplaced = int(rows["latitude"].isna().sum())
        if unplaced:
            st.caption(
                f"{unplaced} of {len(rows)} facilities have no GPS coordinates "
                "and can't be shown on the map (a finding in itself)."
            )
    with right:
        st.subheader("Worst first")
        st.dataframe(
            rows[["facility", "grade", "overall", "completeness", "integrity", "values"]],
            hide_index=True,
            height=460,
            column_config={
                "facility": "Facility",
                "grade": st.column_config.TextColumn("Grade", width="small"),
                "overall": st.column_config.ProgressColumn(
                    "Overall", min_value=0, max_value=100, format="%.1f", width="medium"
                ),
                "completeness": st.column_config.NumberColumn(
                    "Complete %", format="%.1f", width="small"
                ),
                "integrity": st.column_config.NumberColumn(
                    "Integrity", format="%.1f", width="small"
                ),
                "values": st.column_config.NumberColumn(
                    "Values", help="Data values reported", width="small"
                ),
            },
        )


def district_findings(dataset_id: str, district_id: str) -> None:
    counts = finding_counts(dataset_id, district_id)
    st.subheader("What the checks found")
    if counts.empty:
        st.info("No findings in this district.")
        return
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.plotly_chart(findings_bars(counts), config=NO_TOOLBAR, key="findings")
    with right:
        totals = counts.groupby(["check", "label"])["findings"].sum().reset_index()
        totals = totals.sort_values("findings", ascending=False)
        labels = {str(row.check): f"{row.label} ({row.findings:,})" for row in totals.itertuples()}
        check: str = st.selectbox(
            "Show findings for", list(labels), format_func=labels.__getitem__, key="check"
        )
        rows = findings(dataset_id, district_id, check)
        st.dataframe(
            rows.dropna(axis="columns", how="all"),
            hide_index=True,
            height=360,
            column_config={
                "severity": st.column_config.TextColumn("Severity", width="small"),
                "facility": "Facility",
                "period": st.column_config.TextColumn("Month", width="small"),
                "data_element": "Indicator",
                "message": st.column_config.TextColumn("What was found", width="large"),
            },
        )
        if len(rows) == FINDINGS_SHOWN:
            st.caption(f"Showing the first {FINDINGS_SHOWN:,}, most severe first.")


def district_trends(dataset_id: str, district_id: str) -> None:
    options = indicators(dataset_id)
    if not options:
        return
    st.subheader("Trends and early warnings")
    indicator: str = st.selectbox("Indicator", options, key="indicator")
    found = warnings(dataset_id)
    if not found.empty:
        found = found[(found["district_id"] == district_id) & (found["data_element"] == indicator)]
    st.plotly_chart(
        trend_chart(trend(dataset_id, district_id, indicator), found, indicator),
        config=NO_TOOLBAR,
        key="trend",
    )
    if found.empty:
        st.caption("No early warning for this indicator in the last 12 months.")
    else:
        for message in found.sort_values("period_start")["message"]:
            st.markdown(f"- :red[**Warning**] {message}")


def district_anomalies(dataset_id: str, district: str) -> None:
    if dataset_id != DATASETS["child_health"]:
        return
    flagged = anomalies()
    if flagged.empty:
        return
    rows = flagged[flagged["district"] == district]
    st.subheader("Unusual facility-months (anomaly model)")
    if rows.empty:
        st.caption("The anomaly model flagged no facility-month in this district.")
        return
    shown = rows.assign(rules=rows["rule_severity"].fillna("missed by the rules"))
    st.dataframe(
        shown[["facility", "period", "anomaly_score", "rules", "explanation"]],
        hide_index=True,
        column_config={
            "facility": "Facility",
            "period": st.column_config.TextColumn("Month", width="small"),
            "anomaly_score": st.column_config.NumberColumn(
                "Score", format="%.2f", width="small", help="Higher is more unusual"
            ),
            "rules": st.column_config.TextColumn(
                "Rule-based outlier", help="Strongest outlier flag the same month"
            ),
            "explanation": st.column_config.TextColumn("Why", width="large"),
        },
    )


def main() -> None:
    st.set_page_config(
        page_title="HMIS Data Quality", page_icon=":material/monitor_heart:", layout="wide"
    )
    st.title("HMIS data quality and early warning")
    dataset_id = sidebar()
    headline(dataset_id)
    overview(dataset_id)
    district_view(dataset_id)


main()
