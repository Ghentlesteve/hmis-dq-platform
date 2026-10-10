"""Runtime configuration, loaded from environment variables or a local .env file.

Every setting is prefixed with ``HMIS_`` so it can't clash with other tools,
e.g. ``HMIS_DHIS2_BASE_URL``. Nothing secret is hard-coded in the repo.
"""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class StoreKind(StrEnum):
    LAKE = "lake"  # S3 API: the docker compose lake, or AWS S3
    LOCAL = "local"  # plain folder under data_dir, no Docker needed


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HMIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # DHIS2 source
    dhis2_base_url: HttpUrl
    dhis2_username: str
    dhis2_password: SecretStr
    dhis2_timeout_seconds: float = Field(default=60.0, gt=0)

    # Where the bronze layer is written by default
    store: StoreKind = StoreKind.LAKE

    # Local storage for raw pulls (used with store=local)
    data_dir: Path = Path("data")

    # Data lake (S3 API). Leave the endpoint empty to use AWS S3 itself.
    s3_endpoint_url: HttpUrl | None = None
    s3_access_key: str | None = None
    s3_secret_key: SecretStr | None = None
    s3_region: str = "us-east-1"
    bronze_bucket: str = "bronze"
    silver_bucket: str = "silver"
    gold_bucket: str = "gold"

    # Spark (local mode inside the job container; a cluster URL in Kubernetes later)
    spark_master: str = "local[*]"
    spark_driver_memory: str = "3g"
    spark_shuffle_partitions: int = Field(default=16, gt=0)

    # Apache NiFi (docker compose --profile nifi). Its HTTPS certificate is
    # self-signed, so it isn't verified by default: fine for localhost only.
    nifi_url: HttpUrl = HttpUrl("https://localhost:18443")
    nifi_username: str | None = None
    nifi_password: SecretStr | None = None
    nifi_verify_tls: bool = False
    # the lake as NiFi reaches it, inside the compose network
    nifi_s3_endpoint_url: HttpUrl = HttpUrl("http://lake:8333")
    nifi_refresh_months: int = Field(default=3, gt=0)
    nifi_schedule: str = "0 0 2 * * ?"  # Quartz cron: every day at 02:00

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


@lru_cache
def get_settings() -> Settings:
    """Return the settings, read once and cached for the life of the process."""
    return Settings()
