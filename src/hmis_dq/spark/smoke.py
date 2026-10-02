"""Smoke job: prove Spark can read the bronze layer straight from the lake.

Each bronze file is one JSON document on one line, so Spark reads it as one row.
The ``dataset=`` / ``period=`` / ``org_unit=`` folders become columns automatically
(partition discovery), and ``explode`` turns each file's list of values into rows.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from hmis_dq.config import Settings
from hmis_dq.spark.session import build_spark, s3a_url


def bronze_value_counts(settings: Settings) -> DataFrame:
    spark = build_spark(settings, "hmis-smoke")
    files = spark.read.json(s3a_url(settings.bronze_bucket, "dhis2/data_value_sets/"))
    values = files.select(
        "dataset",
        F.col("period").cast("string").alias("period"),
        F.explode("payload.dataValues").alias("dv"),
    )
    return (
        values.groupBy("dataset", F.substring("period", 1, 4).alias("year"))
        .agg(F.count("*").alias("values"), F.countDistinct("dv.orgUnit").alias("facilities"))
        .orderBy("dataset", "year")
    )
