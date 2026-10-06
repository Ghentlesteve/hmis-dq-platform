"""Silver layer: raw bronze JSON -> clean, typed Parquet tables.

Transforms are pure ``DataFrame -> DataFrame`` functions, tested on small
in-memory samples; ``run_silver`` does the reading and writing.

Principles:
- Never silently drop data. A value that can't be parsed is kept, with a
  ``value_status`` saying why; the DQ engine (phase 4) reports on it.
- Metadata comes from the most recent snapshot, so every table in one run
  describes the same version of the DHIS2 hierarchy.
- Re-running is safe: tables are overwritten, data_values partition by partition.
"""

from dataclasses import dataclass

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from hmis_dq.config import Settings
from hmis_dq.spark.schemas import DATA_VALUE_FILE, metadata_file_schema
from hmis_dq.spark.session import build_spark, s3a_url

# DHIS2 value types that hold a number; everything else stays text.
NUMERIC_VALUE_TYPES = frozenset(
    {
        "INTEGER",
        "INTEGER_POSITIVE",
        "INTEGER_NEGATIVE",
        "INTEGER_ZERO_OR_POSITIVE",
        "NUMBER",
        "PERCENTAGE",
        "UNIT_INTERVAL",
    }
)

DHIS2_TIMESTAMP = "yyyy-MM-dd'T'HH:mm:ss.SSSZ"  # e.g. 2010-03-05T00:00:00.000+0000
DATA_VALUE_KEY = (
    "dataset_id",
    "data_element_id",
    "period",
    "org_unit_id",
    "category_option_combo_id",
    "attribute_option_combo_id",
)


class ValueStatus:
    OK = "ok"
    MISSING = "missing"  # null or blank
    NOT_NUMERIC = "not_numeric"  # numeric data element, but the text isn't a number
    TEXT_TYPE = "text_type"  # data element isn't numeric (TEXT, BOOLEAN, DATE, ...)
    UNKNOWN_ELEMENT = "unknown_element"  # no metadata for the data element


# ------------------------------------------------------------------ helpers


def with_snapshot(files: DataFrame) -> DataFrame:
    """Add the snapshot id from the file name (``.../snapshot=<run id>.json.gz``)."""
    return files.withColumn(
        "snapshot", F.regexp_extract("_metadata.file_path", r"snapshot=([^/]+?)\.json", 1)
    )


def latest_snapshot(files: DataFrame) -> DataFrame:
    """Keep only rows from the most recent snapshot."""
    newest = files.agg(F.max("snapshot").alias("snapshot"))
    return files.join(newest, "snapshot")


def explode_items(files: DataFrame, resource: str) -> DataFrame:
    """One row per metadata item, plus the snapshot it came from."""
    return files.select(F.col("snapshot"), F.explode(f"payload.{resource}").alias("item")).select(
        "snapshot", "item.*"
    )


def _dhis2_date(column: str) -> Column:
    # "1970-01-01T00:00:00.000" -> date; the time part is always midnight
    return F.to_date(F.try_to_timestamp(F.substring(F.col(column), 1, 10), F.lit("yyyy-MM-dd")))


# ------------------------------------------------------------- dimensions


def org_units(items: DataFrame, *, facility_level: int = 4) -> DataFrame:
    """Org units with their district and chiefdom resolved from ``path``.

    ``path`` is ``/national/district/chiefdom/facility``, so splitting it gives
    every ancestor without walking the parent chain.
    """
    parts = F.split("path", "/")  # ["", national, district, chiefdom, facility]
    geometry_type = F.get_json_object("geometry", "$.type")
    is_point = geometry_type == "Point"

    units = items.select(
        F.col("id").alias("org_unit_id"),
        "name",
        F.col("shortName").alias("short_name"),
        "level",
        F.col("parent.id").alias("parent_id"),
        "path",
        F.when(F.col("level") >= 2, F.element_at(parts, 3)).alias("district_id"),  # noqa: PLR2004
        F.when(F.col("level") >= 3, F.element_at(parts, 4)).alias("chiefdom_id"),  # noqa: PLR2004
        (F.col("level") == facility_level).alias("is_facility"),
        _dhis2_date("openingDate").alias("opening_date"),
        _dhis2_date("closedDate").alias("closed_date"),
        geometry_type.alias("geometry_type"),
        F.when(
            is_point, F.get_json_object("geometry", "$.coordinates[0]").try_cast("double")
        ).alias("longitude"),
        F.when(
            is_point, F.get_json_object("geometry", "$.coordinates[1]").try_cast("double")
        ).alias("latitude"),
        "geometry",
        "snapshot",
    )

    names = units.select("org_unit_id", "name")
    district_names = names.withColumnsRenamed({"org_unit_id": "district_id", "name": "district"})
    chiefdom_names = names.withColumnsRenamed({"org_unit_id": "chiefdom_id", "name": "chiefdom"})
    return units.join(F.broadcast(district_names), "district_id", "left").join(
        F.broadcast(chiefdom_names), "chiefdom_id", "left"
    )


def data_elements(items: DataFrame) -> DataFrame:
    return items.select(
        F.col("id").alias("data_element_id"),
        "name",
        F.col("shortName").alias("short_name"),
        F.col("valueType").alias("value_type"),
        F.col("aggregationType").alias("aggregation_type"),
        F.col("domainType").alias("domain_type"),
        F.col("categoryCombo.id").alias("category_combo_id"),
        F.col("valueType").isin(*NUMERIC_VALUE_TYPES).alias("is_numeric"),
        "snapshot",
    )


def category_option_combos(items: DataFrame, combos: DataFrame) -> DataFrame:
    combo_names = combos.select(
        F.col("id").alias("category_combo_id"), F.col("name").alias("category_combo")
    )
    return items.select(
        F.col("id").alias("category_option_combo_id"),
        "name",
        F.col("categoryCombo.id").alias("category_combo_id"),
        "snapshot",
    ).join(combo_names, "category_combo_id", "left")


def data_sets(items: DataFrame) -> DataFrame:
    return items.select(
        F.col("id").alias("dataset_id"),
        "name",
        F.col("periodType").alias("period_type"),
        F.col("timelyDays").cast("int").alias("timely_days"),
        F.col("expiryDays").cast("int").alias("expiry_days"),
        F.col("openFuturePeriods").alias("open_future_periods"),
        F.size("dataSetElements").alias("data_element_count"),
        F.size("organisationUnits").alias("org_unit_count"),
        "snapshot",
    )


def dataset_org_units(items: DataFrame) -> DataFrame:
    """Which org units are expected to report each dataset (the completeness denominator)."""
    return items.select(
        F.col("id").alias("dataset_id"),
        F.explode("organisationUnits.id").alias("org_unit_id"),
    )


def dataset_elements(items: DataFrame) -> DataFrame:
    return items.select(
        F.col("id").alias("dataset_id"),
        F.explode("dataSetElements.dataElement.id").alias("data_element_id"),
    )


# ------------------------------------------------------------------- facts


def data_values(files: DataFrame, elements: DataFrame) -> DataFrame:
    """One clean row per reported value, deduplicated, with a parse status."""
    values = files.select(
        F.col("meta.data_set").alias("dataset_id"),
        F.try_to_timestamp("meta.extracted_at").alias("extracted_at"),
        F.explode("payload.dataValues").alias("dv"),
    ).select(
        "dataset_id",
        F.col("dv.dataElement").alias("data_element_id"),
        F.col("dv.period").alias("period"),
        F.col("dv.orgUnit").alias("org_unit_id"),
        F.col("dv.categoryOptionCombo").alias("category_option_combo_id"),
        F.col("dv.attributeOptionCombo").alias("attribute_option_combo_id"),
        F.col("dv.value").alias("value_raw"),
        F.col("dv.storedBy").alias("stored_by"),
        F.try_to_timestamp("dv.created", F.lit(DHIS2_TIMESTAMP)).alias("created"),
        F.try_to_timestamp("dv.lastUpdated", F.lit(DHIS2_TIMESTAMP)).alias("last_updated"),
        F.col("dv.comment").alias("comment"),
        F.coalesce("dv.followup", F.lit(False)).alias("followup"),
        "extracted_at",
    )

    # Same value delivered twice (e.g. overlapping extracts): keep the newest version
    newest_first = Window.partitionBy(*DATA_VALUE_KEY).orderBy(
        F.col("last_updated").desc_nulls_last(), F.col("extracted_at").desc_nulls_last()
    )
    values = (
        values.withColumn("_rank", F.row_number().over(newest_first))
        .filter("_rank = 1")
        .drop("_rank")
    )

    types = elements.select("data_element_id", "value_type", "is_numeric")
    values = values.join(F.broadcast(types), "data_element_id", "left")

    number = F.trim("value_raw").try_cast("double")
    blank = F.col("value_raw").isNull() | (F.trim("value_raw") == "")
    status = (
        F.when(blank, ValueStatus.MISSING)
        .when(F.col("is_numeric").isNull(), ValueStatus.UNKNOWN_ELEMENT)
        .when(~F.col("is_numeric"), ValueStatus.TEXT_TYPE)
        .when(number.isNull(), ValueStatus.NOT_NUMERIC)
        .otherwise(ValueStatus.OK)
    )
    period_start = F.to_date(F.try_to_timestamp(F.concat("period", F.lit("01")), F.lit("yyyyMMdd")))

    return values.select(
        "dataset_id",
        "data_element_id",
        "period",
        period_start.alias("period_start"),
        F.year(period_start).alias("year"),
        "org_unit_id",
        "category_option_combo_id",
        "attribute_option_combo_id",
        "value_raw",
        F.when(status == ValueStatus.OK, number).alias("value"),
        status.alias("value_status"),
        "value_type",
        "stored_by",
        "created",
        "last_updated",
        "comment",
        "followup",
        "extracted_at",
    )


# ----------------------------------------------------------------------- IO


@dataclass(frozen=True)
class SilverResult:
    row_counts: dict[str, int]
    value_status_counts: dict[str, int]
    duplicates_removed: int


def _read_metadata(spark: SparkSession, settings: Settings, resource: str) -> DataFrame:
    path = s3a_url(settings.bronze_bucket, f"dhis2/metadata/{resource}/")
    files = with_snapshot(spark.read.schema(metadata_file_schema(resource)).json(path))
    return explode_items(latest_snapshot(files), resource)


def run_silver(settings: Settings) -> SilverResult:
    spark = build_spark(settings, "hmis-silver")
    spark.conf.set("spark.sql.sources.partitionColumnTypeInference.enabled", "false")

    def read(resource: str) -> DataFrame:
        return _read_metadata(spark, settings, resource).cache()

    ou_items, de_items, ds_items = read("organisationUnits"), read("dataElements"), read("dataSets")
    tables: dict[str, DataFrame] = {
        "org_units": org_units(ou_items),
        "data_elements": data_elements(de_items),
        "category_option_combos": category_option_combos(
            read("categoryOptionCombos"), read("categoryCombos")
        ),
        "data_sets": data_sets(ds_items),
        "dataset_org_units": dataset_org_units(ds_items),
        "dataset_elements": dataset_elements(ds_items),
    }

    def silver_path(table: str) -> str:
        return s3a_url(settings.silver_bucket, f"dhis2/{table}/")

    counts: dict[str, int] = {}
    for name, frame in tables.items():
        frame.coalesce(1).write.mode("overwrite").parquet(silver_path(name))
        counts[name] = spark.read.parquet(silver_path(name)).count()

    files = spark.read.schema(DATA_VALUE_FILE).json(
        s3a_url(settings.bronze_bucket, "dhis2/data_value_sets/")
    )
    raw_count = files.select(F.explode("payload.dataValues")).count()
    (
        data_values(files, tables["data_elements"])
        .repartition("dataset_id", "year")
        .write.mode("overwrite")
        .partitionBy("dataset_id", "year")
        .parquet(silver_path("data_values"))
    )

    written = spark.read.parquet(silver_path("data_values"))
    counts["data_values"] = written.count()
    statuses = {
        r["value_status"]: r["count"] for r in written.groupBy("value_status").count().collect()
    }
    return SilverResult(
        row_counts=counts,
        value_status_counts=statuses,
        duplicates_removed=raw_count - counts["data_values"],
    )
