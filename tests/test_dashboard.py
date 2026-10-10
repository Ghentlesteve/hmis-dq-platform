"""Dashboard tests: queries over a tiny in-memory lake, figures, and the app itself."""

import json
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from hmis_dq.dashboard import data as dashboard_data
from hmis_dq.dashboard.charts import (
    GRADE_MEANING,
    NO_DATA,
    dimension_bars,
    district_map,
)
from hmis_dq.dashboard.data import (
    DashboardData,
    district_geojson,
    label_point,
)

CH, RH = "BfMAe6Itzgt", "QX4ZTUbOt3a"


def square(x: float, y: float, size: float = 1.0) -> list[list[float]]:
    return [[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]]


def scores(district_id: str, district: str | None, overall: float, grade: str) -> dict[str, object]:
    return {
        "dataset_id": CH,
        "district_id": district_id,
        "district": district,
        "facilities": 10,
        "facilities_grade_d": 3,
        "reports_expected": 120,
        "reports_received": 60,
        "completeness": 50.0,
        "accuracy": 99.0,
        "consistency": 80.0,
        "integrity": 10.0,
        "overall": overall,
        "grade": grade,
    }


TABLES: dict[str, pd.DataFrame] = {
    "national_scores": pd.DataFrame(
        [
            {**scores("", None, 48.7, "D"), "dataset_id": CH},
            {**scores("", None, 80.3, "B"), "dataset_id": "unnamedXXXX"},
        ]
    ).drop(columns=["district_id", "district"]),
    "data_sets": pd.DataFrame({"dataset_id": [CH], "name": ["Child Health"]}),
    "district_scores": pd.DataFrame(
        [scores("d1", "Bo", 70.0, "C"), scores("d2", "Kono", 40.0, "D"), scores("x", None, 1, "D")]
    ),
    "district_month": pd.DataFrame(
        {
            "dataset_id": [CH, CH],
            "period_start": pd.to_datetime(["2026-08-01", "2026-09-01"]),
        }
    ),
    "org_units": pd.DataFrame(
        {
            "org_unit_id": ["d1", "d2", "d3", "c1"],
            "name": ["Bo", "Kono", "PTT region", "Chiefdom"],
            "level": [2, 2, 2, 3],
            "geometry_type": ["Polygon", "MultiPolygon", None, "Polygon"],
            "geometry": [
                json.dumps({"type": "Polygon", "coordinates": [square(0, 0)]}),
                json.dumps(
                    {"type": "MultiPolygon", "coordinates": [[square(2, 0, 2)], [square(9, 9)]]}
                ),
                None,
                json.dumps({"type": "Polygon", "coordinates": [square(0, 0)]}),
            ],
        }
    ),
    "warnings": pd.DataFrame(
        {
            "dataset_id": [CH, CH, CH, RH],
            "district_id": ["d2", "d2", "d1", "d1"],
            "period_start": pd.to_datetime(
                ["2026-09-01", "2026-09-01", "2026-08-01", "2026-09-01"]
            ),
        }
    ),
    "anomalies": pd.DataFrame({"facility": ["A", "B"], "anomaly_score": [0.6, 0.9]}),
}


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    for name, frame in TABLES.items():
        connection.register("frame", frame)
        connection.execute(f"CREATE TABLE {name} AS SELECT * FROM frame")
        connection.unregister("frame")
    yield connection
    connection.close()


@pytest.fixture
def dashboard(db: duckdb.DuckDBPyConnection) -> DashboardData:
    return DashboardData(db, {name: name for name in TABLES})


@pytest.fixture
def without_ml(db: duckdb.DuckDBPyConnection, tmp_path: Path) -> DashboardData:
    """ML tables not written yet: reading them fails like a missing lake file."""
    missing = f"read_parquet('{(tmp_path / 'missing.parquet').as_posix()}')"
    return DashboardData(db, {**{n: n for n in TABLES}, "warnings": missing, "anomalies": missing})


def test_datasets_have_names_or_fall_back_to_the_id(dashboard: DashboardData) -> None:
    assert dashboard.datasets().values.tolist() == [[CH, "Child Health"], [*["unnamedXXXX"] * 2]]


def test_districts_ranked_worst_first_with_latest_month_warnings(
    dashboard: DashboardData,
) -> None:
    table = dashboard.districts(CH)

    assert table["district"].tolist() == ["Kono", "Bo"]  # the unnamed district is left out
    # Kono: 2 warnings in Sep 2026; Bo's was in August, not the latest month
    assert table["warnings"].tolist() == [2, 0]


def test_missing_ml_tables_mean_no_warnings_or_anomalies(without_ml: DashboardData) -> None:
    assert without_ml.districts(CH)["warnings"].tolist() == [0, 0]
    assert without_ml.anomalies().empty


def test_only_district_polygons_are_shapes(dashboard: DashboardData) -> None:
    assert dashboard.district_shapes()["district"].tolist() == ["Bo", "Kono"]


def test_geojson_features_are_keyed_by_district(dashboard: DashboardData) -> None:
    geojson = district_geojson(dashboard.district_shapes())

    assert [f["id"] for f in geojson["features"]] == ["d1", "d2"]
    assert geojson["features"][1]["geometry"]["type"] == "MultiPolygon"


def test_label_sits_on_the_largest_polygon() -> None:
    assert label_point({"type": "Polygon", "coordinates": [square(0, 0)]}) == (0.5, 0.5)
    island = {"type": "MultiPolygon", "coordinates": [[square(9, 9)], [square(2, 0, 2)]]}
    assert label_point(island) == (3.0, 1.0)


def test_map_colours_districts_by_distance_from_national(dashboard: DashboardData) -> None:
    fig = district_map(dashboard.districts(CH), dashboard.district_shapes(), national_score=50.0)

    fill, labels = fig.data
    assert list(fill.locations) == ["d1", "d2"]
    assert list(fill.z) == [20.0, -10.0]  # Bo 70, Kono 40 vs national 50
    assert (fill.zmin, fill.zmax) == (-20.0, 20.0)  # symmetric: grey sits at national
    assert list(fill.colorbar.ticktext) == ["-20", "national", "+20"]
    assert fig.layout.geo.lonaxis.range == (-0.3, 10.3)  # all shapes, 3% padding
    assert list(labels.text) == ["<b>Bo</b><br>70.0 · C", "<b>Kono</b><br>40.0 · D"]


def test_map_greys_out_districts_without_a_score(dashboard: DashboardData) -> None:
    only_bo = dashboard.districts(CH).query("district == 'Bo'")

    fig = district_map(only_bo, dashboard.district_shapes(), national_score=50.0, dark=True)

    assert list(fig.data[1].locations) == ["d2"]
    assert fig.data[1].colorscale[0][1] == NO_DATA
    assert fig.data[2].textfont.color == "#fcfcfb"  # light labels in dark mode


def test_grade_meanings_follow_the_thresholds() -> None:
    assert GRADE_MEANING == {
        "A": "A (90+)",
        "B": "B (75-89)",
        "C": "C (60-74)",
        "D": "D (below 60)",
    }


def test_dimension_bars_show_points_lost(dashboard: DashboardData) -> None:
    fig = dimension_bars(dashboard.national(CH))

    # completeness 50 at weight 0.35 costs 17.5 of the 100 overall points
    assert fig.data[0].text[0] == "50.0  (-17.5 pts)"


def test_app_renders_from_the_lake_tables(
    dashboard: DashboardData, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415

    monkeypatch.setattr(dashboard_data, "open_dashboard_data", lambda: dashboard)
    app = AppTest.from_file(
        str(Path(dashboard_data.__file__).parent / "app.py"), default_timeout=30
    )

    app.run()

    assert not app.exception
    assert app.metric[0].value == "48.7 · D"
    assert app.metric[3].label == "Early warnings, Sep 2026"
    assert app.metric[3].value == "2"
    assert app.metric[4].value == "2"  # anomalies (Child Health)
