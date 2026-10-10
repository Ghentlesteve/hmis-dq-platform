"""Everything the dashboard reads, as plain DataFrames (DuckDB over the lake).

The queries take their table locations from a ``Tables`` mapping, so tests can
point them at small local Parquet files instead of the lake.
"""

import json
import math
from collections.abc import Mapping
from itertools import pairwise
from typing import Any

import duckdb
import pandas as pd

from hmis_dq.config import Settings
from hmis_dq.explore import find_env_file, gold, lake, silver
from hmis_dq.spark.dq.rules import TRACER_INDICATORS

Tables = Mapping[str, str]  # logical name -> DuckDB table expression

DISTRICT_LEVEL = 2
POLYGONS = ("Polygon", "MultiPolygon")
FINDINGS_SHOWN = 500

# Plain-language names for the checks in gold/dq/findings
CHECK_LABELS = {
    "never_reported": "Never reported",
    "low_reporting_completeness": "Low reporting completeness",
    "outlier": "Outlier values",
    "penta1_penta3_dropout": "Penta3 above Penta1 (negative drop-out)",
    "anc1_anc4_dropout": "ANC4 above ANC1 (negative drop-out)",
    "consistency_over_time": "Inconsistent with earlier years",
    "repeats_earlier_year": "Values copied from an earlier year",
    "repeated_values": "Same value month after month",
    "entered_before_period_end": "Entered before the month ended",
    "last_updated_before_created": "Impossible timestamps",
    "missing_coordinates": "No GPS coordinates",
    "non_facility_assignment": "Dataset assigned to a non-facility",
}


def lake_tables(settings: Settings) -> dict[str, str]:
    def ml(name: str) -> str:
        return f"read_parquet('s3://{settings.gold_bucket}/dhis2/ml/{name}.parquet')"

    return {
        "org_units": silver("org_units"),
        "data_sets": silver("data_sets"),
        "district_month": gold("district_month"),
        "national_scores": gold("dq/national_scores"),
        "district_scores": gold("dq/district_scores"),
        "facility_scores": gold("dq/facility_scores"),
        "findings": gold("dq/findings"),
        "anomalies": ml("anomalies"),
        "warnings": ml("early_warning_history"),
    }


def open_dashboard_data(settings: Settings | None = None) -> "DashboardData":
    """The dashboard's data, read from the lake configured in .env."""
    settings = settings or Settings(_env_file=find_env_file())
    return DashboardData(lake(settings), lake_tables(settings))


class DashboardData:
    def __init__(self, db: duckdb.DuckDBPyConnection, tables: Tables) -> None:
        self.db = db
        self.tables = tables

    def _query(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        # one cursor per query: Streamlit serves each viewer from its own thread
        with self.db.cursor() as cursor:
            return cursor.execute(sql.format(**self.tables), params or []).df()

    def _optional(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        """For tables a later pipeline step writes (ML): empty if not there yet."""
        try:
            return self._query(sql, params)
        except duckdb.IOException:
            return pd.DataFrame()

    # ------------------------------------------------------------- national

    def datasets(self) -> pd.DataFrame:
        """The scored datasets, with their names."""
        return self._query(
            """
            SELECT n.dataset_id, coalesce(d.name, n.dataset_id) AS name
            FROM {national_scores} n LEFT JOIN {data_sets} d USING (dataset_id)
            ORDER BY name
            """
        )

    def national(self, dataset_id: str) -> pd.Series:
        rows = self._query("SELECT * FROM {national_scores} WHERE dataset_id = ?", [dataset_id])
        return rows.iloc[0]

    def latest_month(self, dataset_id: str) -> pd.Timestamp:
        rows = self._query(
            "SELECT max(period_start) AS latest FROM {district_month} WHERE dataset_id = ?",
            [dataset_id],
        )
        latest: pd.Timestamp = pd.to_datetime(rows["latest"]).iloc[0]
        return latest

    def districts(self, dataset_id: str) -> pd.DataFrame:
        """District scores, plus warnings in the latest month and anomalies (any time)."""
        scores = self._query(
            """
            SELECT * FROM {district_scores}
            WHERE dataset_id = ? AND district IS NOT NULL
            ORDER BY overall
            """,
            [dataset_id],
        )
        latest = self.latest_month(dataset_id)
        warnings = self.warnings(dataset_id)
        current = warnings[warnings["period_start"] == latest] if not warnings.empty else warnings
        scores["warnings"] = (
            scores["district_id"].map(current["district_id"].value_counts()).fillna(0).astype(int)
            if not current.empty
            else 0
        )
        return scores

    def district_shapes(self) -> pd.DataFrame:
        placeholders = ", ".join("?" for _ in POLYGONS)
        return self._query(
            f"""
            SELECT org_unit_id AS district_id, name AS district, geometry
            FROM {{org_units}}
            WHERE level = ? AND geometry_type IN ({placeholders})
            """,
            [DISTRICT_LEVEL, *POLYGONS],
        )

    def warnings(self, dataset_id: str) -> pd.DataFrame:
        """Early warnings, each month replayed as if it had been the latest."""
        rows = self._optional("SELECT * FROM {warnings} WHERE dataset_id = ?", [dataset_id])
        if not rows.empty:
            rows["period_start"] = pd.to_datetime(rows["period_start"])
        return rows

    def anomalies(self) -> pd.DataFrame:
        """Facility-months the anomaly model flagged (Child Health antigens)."""
        return self._optional("SELECT * FROM {anomalies} ORDER BY anomaly_score DESC")

    # ------------------------------------------------------------- district

    def facilities(self, dataset_id: str, district_id: str) -> pd.DataFrame:
        """Facility scores in a district, worst first, with coordinates when known."""
        return self._query(
            """
            SELECT f.*, o.longitude, o.latitude
            FROM {facility_scores} f LEFT JOIN {org_units} o USING (org_unit_id)
            WHERE f.dataset_id = ? AND f.district_id = ?
            ORDER BY f.overall, f.facility
            """,
            [dataset_id, district_id],
        )

    def finding_counts(self, dataset_id: str, district_id: str) -> pd.DataFrame:
        """Findings per check and severity. Checks that aren't about one dataset
        (missing coordinates) have no dataset_id and are always included."""
        counts = self._query(
            """
            SELECT "check", severity, count(*) AS findings
            FROM {findings}
            WHERE (dataset_id = ? OR dataset_id IS NULL) AND district_id = ?
            GROUP BY ALL
            """,
            [dataset_id, district_id],
        )
        counts["label"] = counts["check"].map(CHECK_LABELS).fillna(counts["check"])
        return counts

    def findings(
        self, dataset_id: str, district_id: str, check: str, limit: int = FINDINGS_SHOWN
    ) -> pd.DataFrame:
        """One check's findings in a district, most severe first."""
        return self._query(
            """
            SELECT severity, facility, period, data_element, message
            FROM {findings}
            WHERE (dataset_id = ? OR dataset_id IS NULL) AND district_id = ? AND "check" = ?
            ORDER BY CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                     facility, period
            LIMIT ?
            """,
            [dataset_id, district_id, check, limit],
        )

    def indicators(self, dataset_id: str) -> list[str]:
        """The tracer indicators this dataset reports, in the order of TRACER_INDICATORS."""
        found = set(
            self._query(
                "SELECT DISTINCT data_element FROM {district_month} WHERE dataset_id = ?",
                [dataset_id],
            )["data_element"]
        )
        return [name for name in TRACER_INDICATORS if name in found]

    def trend(self, dataset_id: str, district_id: str, data_element: str) -> pd.DataFrame:
        """Monthly district totals of one indicator, next to the same month a year
        earlier (the forecast the early warning is built on)."""
        rows = self._query(
            """
            SELECT CAST(period_start AS DATE) AS period_start, value, reports_received
            FROM {district_month}
            WHERE dataset_id = ? AND district_id = ? AND data_element = ?
            ORDER BY period_start
            """,
            [dataset_id, district_id, data_element],
        )
        rows["period_start"] = pd.to_datetime(rows["period_start"])
        last_year = rows[["period_start", "value"]].assign(
            period_start=rows["period_start"] + pd.DateOffset(years=1)
        )
        return rows.merge(
            last_year.rename(columns={"value": "last_year"}), on="period_start", how="left"
        )


# ------------------------------------------------------------------ geometry


def district_geojson(shapes: pd.DataFrame) -> dict[str, Any]:
    """A GeoJSON FeatureCollection keyed by district_id, for a choropleth."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": row["district_id"],
                "properties": {"district": row["district"]},
                "geometry": json.loads(row["geometry"]),
            }
            for row in shapes.to_dict("records")
        ],
    }


def _ring_centroid(ring: list[list[float]]) -> tuple[float, float, float]:
    """Area and centroid of a closed ring (shoelace formula)."""
    area = cx = cy = 0.0
    for (x0, y0), (x1, y1) in pairwise(ring):
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area /= 2
    if math.isclose(area, 0.0):
        xs, ys = zip(*ring, strict=True)
        return 0.0, sum(xs) / len(xs), sum(ys) / len(ys)
    return abs(area), cx / (6 * area), cy / (6 * area)


def label_point(geometry: dict[str, Any]) -> tuple[float, float]:
    """(lon, lat) to put a district's label: the centroid of its largest polygon,
    so islands don't drag the label into the sea."""
    polygons = (
        geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
    )
    _, lon, lat = max(_ring_centroid(polygon[0]) for polygon in polygons)
    return lon, lat


def bounds(geojson: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    """((west, south), (east, north)) around every polygon in a FeatureCollection."""
    lons, lats = [], []
    for feature in geojson["features"]:
        geometry = feature["geometry"]
        polygons = (
            geometry["coordinates"]
            if geometry["type"] == "MultiPolygon"
            else [geometry["coordinates"]]
        )
        for polygon in polygons:
            for lon, lat, *_ in polygon[0]:
                lons.append(lon)
                lats.append(lat)
    return (min(lons), min(lats)), (max(lons), max(lats))
