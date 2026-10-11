"""session_setting: a step's Spark setting must not leak into the next step.

Regression test: silver turned off partition type inference for the whole session,
so in `hmis-dq spark build` gold's `year=2026` folders came back as the text
"2026" and the data quality messages ("%d") failed. Found on Kubernetes, where the
pipeline first ran as one job.
"""

from pathlib import Path

from pyspark.sql import SparkSession

from hmis_dq.spark.session import session_setting

INFERENCE = "spark.sql.sources.partitionColumnTypeInference.enabled"


def year_type(spark: SparkSession, path: Path) -> str:
    return spark.read.parquet(str(path)).schema["year"].dataType.simpleString()


def test_partition_types_come_back_after_the_block(spark: SparkSession, tmp_path: Path) -> None:
    table = tmp_path / "facility_month"
    spark.createDataFrame([(1.0, 2026)], "value double, year int").write.partitionBy(
        "year"
    ).parquet(str(table))

    assert year_type(spark, table) == "int"
    with session_setting(spark, INFERENCE, "false"):
        assert year_type(spark, table) == "string"  # what silver wants for bronze
    assert year_type(spark, table) == "int"  # what gold and the checks need


def test_a_setting_that_was_set_is_restored_even_after_an_error(spark: SparkSession) -> None:
    key = "spark.sql.shuffle.partitions"
    before = spark.conf.get(key)
    try:
        with session_setting(spark, key, "7"):
            assert spark.conf.get(key) == "7"
            raise RuntimeError("the step failed")
    except RuntimeError:
        pass

    assert spark.conf.get(key) == before
