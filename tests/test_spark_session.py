"""Spark configuration tests. No JVM needed: spark_conf is a pure function."""

from typing import Any

from hmis_dq.config import Settings
from hmis_dq.spark.session import s3a_url, spark_conf


def make_settings(**overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        dhis2_base_url="https://dhis2.test",
        dhis2_username="u",
        dhis2_password="p",
        **overrides,
    )


def test_local_lake_uses_path_style_http_and_static_keys() -> None:
    settings = make_settings(
        s3_endpoint_url="http://lake:8333",
        s3_access_key="key",
        s3_secret_key="secret",
    )

    conf = spark_conf(settings)

    assert conf["spark.hadoop.fs.s3a.endpoint"] == "http://lake:8333"
    assert conf["spark.hadoop.fs.s3a.connection.ssl.enabled"] == "false"
    assert conf["spark.hadoop.fs.s3a.path.style.access"] == "true"
    assert conf["spark.hadoop.fs.s3a.access.key"] == "key"
    assert conf["spark.hadoop.fs.s3a.secret.key"] == "secret"


def test_aws_s3_uses_default_endpoint_and_credential_chain() -> None:
    conf = spark_conf(make_settings())

    assert "spark.hadoop.fs.s3a.endpoint" not in conf
    assert "spark.hadoop.fs.s3a.access.key" not in conf  # e.g. an IAM role in the cloud


def test_https_endpoint_enables_ssl() -> None:
    conf = spark_conf(make_settings(s3_endpoint_url="https://s3.example.org"))

    assert conf["spark.hadoop.fs.s3a.connection.ssl.enabled"] == "true"


def test_job_defaults() -> None:
    conf = spark_conf(make_settings())

    assert conf["spark.sql.session.timeZone"] == "UTC"
    assert conf["spark.sql.sources.partitionOverwriteMode"] == "dynamic"


def test_s3a_url() -> None:
    assert s3a_url("silver", "/data_values/") == "s3a://silver/data_values/"
