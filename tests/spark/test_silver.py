"""Silver transform tests on small samples shaped exactly like the bronze files."""

from datetime import date, datetime
from functools import reduce
from typing import Any

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from hmis_dq.spark import silver
from hmis_dq.spark.schemas import DATA_VALUE_FILE, metadata_file_schema
from hmis_dq.spark.silver import ValueStatus
from tests.spark.conftest import json_frame

pytestmark = pytest.mark.spark


def metadata(spark: SparkSession, resource: str, *snapshots: tuple[str, list[Any]]) -> DataFrame:
    """Metadata items as they come out of _read_metadata (snapshot column + exploded items)."""
    schema = metadata_file_schema(resource)
    frames = [
        json_frame(spark, schema, [{"meta": {}, "payload": {resource: items}}]).withColumn(
            "snapshot", F.lit(label)
        )
        for label, items in snapshots
    ]
    files = reduce(DataFrame.unionByName, frames)
    return silver.explode_items(silver.latest_snapshot(files), resource)


def value(de: str, ou: str, val: str | None, **extra: Any) -> dict[str, Any]:
    return {
        "dataElement": de,
        "period": extra.pop("period", "202501"),
        "orgUnit": ou,
        "categoryOptionCombo": "coc1",
        "attributeOptionCombo": "aoc1",
        "value": val,
        "lastUpdated": extra.pop("lastUpdated", "2025-02-01T10:00:00.000+0000"),
        "created": "2025-02-01T09:00:00.000+0000",
        **extra,
    }


def value_file(dataset: str, values: list[dict[str, Any]], extracted: str) -> dict[str, Any]:
    return {
        "meta": {"data_set": dataset, "extracted_at": extracted},
        "payload": {"dataSet": dataset, "dataValues": values},
    }


@pytest.fixture
def elements(spark: SparkSession) -> DataFrame:
    items = [
        {"id": "deNum", "name": "Penta1", "valueType": "INTEGER", "categoryCombo": {"id": "cc"}},
        {"id": "deTxt", "name": "Remarks", "valueType": "TEXT", "categoryCombo": {"id": "cc"}},
    ]
    return silver.data_elements(metadata(spark, "dataElements", ("s1", items)))


# ------------------------------------------------------------- metadata


def test_latest_snapshot_wins(spark: SparkSession) -> None:
    old = [{"id": "deNum", "name": "Old name", "valueType": "INTEGER"}]
    new = [{"id": "deNum", "name": "New name", "valueType": "INTEGER"}]

    items = metadata(spark, "dataElements", ("20250101T000000Z", old), ("20250201T000000Z", new))

    assert [r["name"] for r in items.collect()] == ["New name"]


def test_org_units_resolve_hierarchy_names_and_coordinates(spark: SparkSession) -> None:
    point = {"type": "Point", "coordinates": [-12.9487, 9.0131]}
    polygon = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
    items = [
        {"id": "nat", "name": "Sierra Leone", "level": 1, "path": "/nat"},
        {"id": "dis", "name": "Bo", "level": 2, "path": "/nat/dis", "geometry": polygon},
        {"id": "chf", "name": "Badjia", "level": 3, "path": "/nat/dis/chf"},
        {
            "id": "fac",
            "name": "Ngelehun CHC",
            "level": 4,
            "path": "/nat/dis/chf/fac",
            "parent": {"id": "chf"},
            "openingDate": "1970-01-01T00:00:00.000",
            "geometry": point,
        },
        {"id": "nogeo", "name": "No GPS MCHP", "level": 4, "path": "/nat/dis/chf/nogeo"},
    ]

    units = silver.org_units(metadata(spark, "organisationUnits", ("s1", items)))
    rows = {r["org_unit_id"]: r for r in units.collect()}

    fac = rows["fac"]
    assert (fac["district"], fac["chiefdom"], fac["parent_id"]) == ("Bo", "Badjia", "chf")
    assert fac["is_facility"]
    assert (fac["longitude"], fac["latitude"]) == (-12.9487, 9.0131)
    assert fac["opening_date"] == date(1970, 1, 1)
    assert rows["dis"]["geometry_type"] == "Polygon"
    assert rows["dis"]["longitude"] is None  # only points get coordinates
    assert rows["dis"]["district"] == "Bo"
    assert rows["dis"]["chiefdom"] is None
    assert rows["nat"]["district_id"] is None
    assert rows["nogeo"]["geometry_type"] is None


def test_data_elements_flag_numeric_types(elements: DataFrame) -> None:
    flags = {r["data_element_id"]: r["is_numeric"] for r in elements.collect()}

    assert flags == {"deNum": True, "deTxt": False}


def test_data_sets_and_assignments(spark: SparkSession) -> None:
    items = [
        {
            "id": "ds1",
            "name": "Child Health",
            "periodType": "Monthly",
            "timelyDays": 15.0,
            "dataSetElements": [{"dataElement": {"id": "deNum"}}],
            "organisationUnits": [{"id": "fac1"}, {"id": "fac2"}],
        }
    ]
    ds_items = metadata(spark, "dataSets", ("s1", items))

    (ds,) = silver.data_sets(ds_items).collect()
    assigned = {r["org_unit_id"] for r in silver.dataset_org_units(ds_items).collect()}
    (element,) = silver.dataset_elements(ds_items).collect()

    assert (ds["timely_days"], ds["org_unit_count"], ds["data_element_count"]) == (15, 2, 1)
    assert assigned == {"fac1", "fac2"}
    assert element["data_element_id"] == "deNum"


def test_category_option_combos_get_their_combo_name(spark: SparkSession) -> None:
    cocs = metadata(
        spark,
        "categoryOptionCombos",
        ("s1", [{"id": "c1", "name": "<1y", "categoryCombo": {"id": "cc"}}]),
    )
    combos = metadata(spark, "categoryCombos", ("s1", [{"id": "cc", "name": "Age"}]))

    (row,) = silver.category_option_combos(cocs, combos).collect()

    assert (row["name"], row["category_combo"]) == ("<1y", "Age")


# ---------------------------------------------------------------- values


def test_values_are_parsed_with_a_status_never_dropped(
    spark: SparkSession, elements: DataFrame
) -> None:
    files = json_frame(
        spark,
        DATA_VALUE_FILE,
        [
            value_file(
                "ds1",
                [
                    value("deNum", "ou1", "13"),
                    value("deNum", "ou2", " 7 "),
                    value("deNum", "ou3", "twelve"),
                    value("deNum", "ou4", ""),
                    value("deTxt", "ou5", "stock-out"),
                    value("deGhost", "ou6", "4"),
                ],
                "2025-03-01T00:00:00+00:00",
            )
        ],
    )

    rows = {r["org_unit_id"]: r for r in silver.data_values(files, elements).collect()}

    assert len(rows) == 6
    assert (rows["ou1"]["value"], rows["ou1"]["value_status"]) == (13.0, ValueStatus.OK)
    assert rows["ou2"]["value"] == 7.0
    assert (rows["ou3"]["value"], rows["ou3"]["value_status"]) == (None, ValueStatus.NOT_NUMERIC)
    assert rows["ou3"]["value_raw"] == "twelve"  # kept for the DQ report
    assert rows["ou4"]["value_status"] == ValueStatus.MISSING
    assert rows["ou5"]["value_status"] == ValueStatus.TEXT_TYPE
    assert rows["ou6"]["value_status"] == ValueStatus.UNKNOWN_ELEMENT


def test_periods_and_timestamps_are_typed(spark: SparkSession, elements: DataFrame) -> None:
    files = json_frame(
        spark,
        DATA_VALUE_FILE,
        [
            value_file(
                "ds1",
                [
                    value(
                        "deNum",
                        "ou1",
                        "1",
                        period="202412",
                        lastUpdated="2010-03-05T00:00:00.000+0000",
                    )
                ],
                "2025-03-01T00:00:00+00:00",
            )
        ],
    )

    (row,) = silver.data_values(files, elements).collect()

    assert (row["period_start"], row["year"]) == (date(2024, 12, 1), 2024)
    # the impossible demo-data date survives unchanged for the DQ engine to flag
    assert row["last_updated"] == datetime(2010, 3, 5)
    assert row["created"] > row["last_updated"]
    assert row["dataset_id"] == "ds1"


def test_duplicates_keep_the_newest_version(spark: SparkSession, elements: DataFrame) -> None:
    files = json_frame(
        spark,
        DATA_VALUE_FILE,
        [
            value_file("ds1", [value("deNum", "ou1", "5")], "2025-03-01T00:00:00+00:00"),
            value_file(
                "ds1",
                [value("deNum", "ou1", "6", lastUpdated="2025-02-20T10:00:00.000+0000")],
                "2025-03-02T00:00:00+00:00",
            ),
        ],
    )

    rows = silver.data_values(files, elements).collect()

    assert [r["value"] for r in rows] == [6.0]


def test_same_value_in_two_datasets_is_not_a_duplicate(
    spark: SparkSession, elements: DataFrame
) -> None:
    files = json_frame(
        spark,
        DATA_VALUE_FILE,
        [
            value_file("ds1", [value("deNum", "ou1", "5")], "2025-03-01T00:00:00+00:00"),
            value_file("ds2", [value("deNum", "ou1", "5")], "2025-03-01T00:00:00+00:00"),
        ],
    )

    assert silver.data_values(files, elements).count() == 2


def test_empty_chunks_produce_no_rows(spark: SparkSession, elements: DataFrame) -> None:
    files = json_frame(spark, DATA_VALUE_FILE, [value_file("ds1", [], "2025-03-01T00:00:00+00:00")])

    assert silver.data_values(files, elements).count() == 0
