"""Explore the lake from Python with DuckDB: no Spark or Docker needed for reading.

DuckDB reads Parquet and gzipped JSON straight from S3, so every layer can be
queried with plain SQL from a notebook or a script:

    from hmis_dq.explore import lake
    db = lake()
    db.sql(f"SELECT * FROM {silver('org_units')} LIMIT 5").df()
"""

import contextlib
from pathlib import Path
from urllib.parse import urlsplit

import duckdb

from hmis_dq.config import Settings


def find_env_file(start: Path | None = None) -> Path | None:
    """The nearest .env in this folder or a parent (notebooks run from notebooks/)."""
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        candidate = folder / ".env"
        if candidate.is_file():
            return candidate
    return None


def s3_secret_sql(settings: Settings) -> str:
    """DuckDB statement giving it access to the lake (the local one or AWS S3)."""
    options = ["TYPE s3", f"REGION '{settings.s3_region}'", "URL_STYLE 'path'"]
    if settings.s3_access_key and settings.s3_secret_key:
        options += [
            f"KEY_ID '{settings.s3_access_key}'",
            f"SECRET '{settings.s3_secret_key.get_secret_value()}'",
        ]
    if settings.s3_endpoint_url is not None:
        url = urlsplit(str(settings.s3_endpoint_url))
        options += [f"ENDPOINT '{url.netloc}'", f"USE_SSL {str(url.scheme == 'https').lower()}"]
    return f"CREATE OR REPLACE SECRET lake ({', '.join(options)})"


def lake(settings: Settings | None = None) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB connection that can read every bucket in the lake."""
    if settings is None:
        settings = Settings(_env_file=find_env_file())
    db = duckdb.connect()
    # A text progress bar floods script output. Inside Jupyter, DuckDB refuses to
    # change this setting without ipywidgets, but there it doesn't print text anyway.
    with contextlib.suppress(duckdb.InvalidInputException):
        db.sql("SET enable_progress_bar = false")
    db.sql("INSTALL httpfs")
    db.sql("LOAD httpfs")
    db.sql(s3_secret_sql(settings))
    return db


def parquet_table(bucket: str, table: str) -> str:
    """SQL table expression for a Parquet table, including its partition columns."""
    return f"read_parquet('s3://{bucket}/dhis2/{table}/**/*.parquet', hive_partitioning = true)"


def silver(table: str) -> str:
    return parquet_table("silver", table)


def gold(table: str) -> str:
    return parquet_table("gold", table)
