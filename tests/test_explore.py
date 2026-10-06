from pathlib import Path

import pytest

from hmis_dq.config import Settings
from hmis_dq.explore import find_env_file, gold, s3_secret_sql, silver


def settings(**overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        dhis2_base_url="https://dhis2.test",
        dhis2_username="u",
        dhis2_password="p",
        **overrides,  # type: ignore[arg-type]
    )


def test_finds_env_file_in_a_parent_folder(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("HMIS_X=1")
    notebooks = tmp_path / "notebooks"
    notebooks.mkdir()

    assert find_env_file(notebooks) == tmp_path / ".env"


def test_no_env_file_anywhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lonely = tmp_path / "a" / "b"
    lonely.mkdir(parents=True)
    # stop the search from reaching a real .env above the temp folder
    monkeypatch.setattr(Path, "is_file", lambda self: False)

    assert find_env_file(lonely) is None


def test_secret_for_the_local_lake() -> None:
    sql = s3_secret_sql(
        settings(s3_endpoint_url="http://localhost:8333", s3_access_key="k", s3_secret_key="s")
    )

    assert "ENDPOINT 'localhost:8333'" in sql
    assert "USE_SSL false" in sql
    assert "URL_STYLE 'path'" in sql
    assert "KEY_ID 'k'" in sql


def test_secret_for_aws_s3_uses_default_endpoint_and_credentials() -> None:
    sql = s3_secret_sql(settings())

    assert "ENDPOINT" not in sql
    assert "KEY_ID" not in sql


def test_table_expressions_read_partitions() -> None:
    assert silver("data_values") == (
        "read_parquet('s3://silver/dhis2/data_values/**/*.parquet', hive_partitioning = true)"
    )
    assert "s3://gold/dhis2/dq/findings/" in gold("dq/findings")
