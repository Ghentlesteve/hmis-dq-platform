from pathlib import Path

import pytest
from pydantic import ValidationError

from hmis_dq.config import Settings


@pytest.fixture
def dhis2_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HMIS_DHIS2_BASE_URL", "https://dhis2.example.org")
    monkeypatch.setenv("HMIS_DHIS2_USERNAME", "reader")
    monkeypatch.setenv("HMIS_DHIS2_PASSWORD", "not-a-real-password")


def test_settings_load_from_environment(dhis2_env: None) -> None:
    settings = Settings(_env_file=None)

    assert str(settings.dhis2_base_url) == "https://dhis2.example.org/"
    assert settings.raw_dir == Path("data") / "raw"


def test_password_is_hidden_when_printed(dhis2_env: None) -> None:
    settings = Settings(_env_file=None)

    assert "not-a-real-password" not in repr(settings)
    assert settings.dhis2_password.get_secret_value() == "not-a-real-password"


def test_missing_url_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HMIS_DHIS2_BASE_URL", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
