"""SparkSession setup for reading and writing the S3 data lake through S3A.

``spark_conf`` is a pure function (settings in, dict out) so the configuration
can be unit-tested without starting a JVM.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from pyspark.sql import SparkSession

from hmis_dq.config import Settings


def s3a_url(bucket: str, path: str = "") -> str:
    return f"s3a://{bucket}/{path.lstrip('/')}"


def spark_conf(settings: Settings) -> dict[str, str]:
    conf = {
        "spark.sql.session.timeZone": "UTC",
        "spark.sql.shuffle.partitions": str(settings.spark_shuffle_partitions),
        "spark.driver.memory": settings.spark_driver_memory,
        # Parquet written for analytics: compressed, and readable by pandas/DuckDB too
        "spark.sql.parquet.compression.codec": "zstd",
        # Overwrite only the partitions a job writes, not the whole table
        "spark.sql.sources.partitionOverwriteMode": "dynamic",
        "spark.ui.showConsoleProgress": "false",
    }

    s3 = "spark.hadoop.fs.s3a."
    conf[s3 + "path.style.access"] = "true"
    conf[s3 + "endpoint.region"] = settings.s3_region
    if settings.s3_endpoint_url is not None:
        endpoint = str(settings.s3_endpoint_url).rstrip("/")
        conf[s3 + "endpoint"] = endpoint
        conf[s3 + "connection.ssl.enabled"] = str(endpoint.startswith("https")).lower()
    if settings.s3_access_key and settings.s3_secret_key:
        conf[s3 + "aws.credentials.provider"] = (
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"
        )
        conf[s3 + "access.key"] = settings.s3_access_key
        conf[s3 + "secret.key"] = settings.s3_secret_key.get_secret_value()
    return conf


def build_spark(settings: Settings, app_name: str) -> SparkSession:
    builder = SparkSession.builder.appName(app_name).master(settings.spark_master)
    for key, value in spark_conf(settings).items():
        builder = builder.config(key, value)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


@contextmanager
def session_setting(spark: SparkSession, key: str, value: str) -> Iterator[None]:
    """Change a Spark SQL setting for the ``with`` block only.

    `hmis-dq spark build` runs silver, gold and the checks in one session, so a
    setting one step changes and leaves behind silently changes the next steps:
    silver's "keep partition values as text" once turned gold's year into text
    and broke the checks' "%d" messages.
    """
    previous = spark.conf.get(key, None)
    spark.conf.set(key, value)
    try:
        yield
    finally:
        if previous is None:
            spark.conf.unset(key)
        else:
            spark.conf.set(key, previous)
