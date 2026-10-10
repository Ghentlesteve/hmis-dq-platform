"""HMIS data quality dashboard (Streamlit).

    hmis-dq dashboard          (lake running; opens http://localhost:8501)

Streamlit reruns this script top to bottom on every interaction; the lake is
read through cached functions, so only the first view of a dataset queries it.
"""

import pandas as pd
import streamlit as st

from hmis_dq.dashboard.charts import GRADE_MEANING, dimension_bars, district_map
from hmis_dq.dashboard.data import DashboardData, open_dashboard_data
from hmis_dq.extract.catalog import DATASETS

CACHE_SECONDS = 600
REPO = "https://github.com/Ghentlesteve/hmis-dq-platform"


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
        figure = district_map(
            table, shapes(), float(scores["overall"]), dark=st.context.theme.type == "dark"
        )
        st.plotly_chart(figure, config={"displayModeBar": False}, key="map")
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
        st.plotly_chart(dimension_bars(scores), config={"displayModeBar": False}, key="dims")


def main() -> None:
    st.set_page_config(
        page_title="HMIS Data Quality", page_icon=":material/monitor_heart:", layout="wide"
    )
    st.title("HMIS data quality and early warning")
    dataset_id = sidebar()
    headline(dataset_id)
    overview(dataset_id)


main()
