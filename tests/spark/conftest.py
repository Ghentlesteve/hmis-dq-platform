"""Shared Spark session for the Spark tests.

Runs anywhere Spark can start (Linux, the spark-test container, CI). Where it
can't (some Windows setups), the tests are skipped with a pointer to Docker:

    docker compose run --rm spark-test
"""

import json
from collections.abc import Iterator
from typing import Any

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

_startup_error: str | None = None


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    global _startup_error  # noqa: PLW0603
    if _startup_error:
        pytest.skip(_startup_error)
    try:
        session = (
            SparkSession.builder.master("local[2]")
            .appName("hmis-tests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )
    except Exception as exc:
        _startup_error = (
            f"Spark can't start here ({type(exc).__name__}); "
            "run these tests with `docker compose run --rm spark-test`"
        )
        pytest.skip(_startup_error)
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def json_frame(spark: SparkSession, schema: StructType, docs: list[dict[str, Any]]) -> DataFrame:
    """Build a DataFrame the same way the jobs do: JSON text parsed with an explicit schema."""
    lines = spark.sparkContext.parallelize([json.dumps(d) for d in docs])
    return spark.read.schema(schema).json(lines)
