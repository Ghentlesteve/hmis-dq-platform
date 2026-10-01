"""Runtime configuration, loaded from environment variables or a local .env file.

Every setting is prefixed with ``HMIS_`` so it can't clash with other tools,
e.g. ``HMIS_DHIS2_BASE_URL``. Nothing secret is hard-coded in the repo.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Local storage for raw pulls (replaced by S3/MinIO in stage 2)
    data_dir: Path = Path("data")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


@lru_cache
def get_settings() -> Settings:
    """Return the settings, read once and cached for the life of the process."""
    return Settings()
