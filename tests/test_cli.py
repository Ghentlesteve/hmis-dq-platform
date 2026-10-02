"""End-to-end CLI tests: real command, fake DHIS2 server, temp data directory."""

from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from hmis_dq import cli
from hmis_dq.config import Settings
from hmis_dq.dhis2 import DHIS2Client
from hmis_dq.extract.store import LocalRawStore
from tests.test_extract_job import FakeDHIS2

runner = CliRunner()


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeDHIS2:
    fake = FakeDHIS2()
    settings = Settings(
        _env_file=None,
        dhis2_base_url="https://dhis2.test",
        dhis2_username="u",
        dhis2_password="p",
        data_dir=tmp_path,
    )

    def from_settings(_: Settings, **kwargs: Any) -> DHIS2Client:
        return DHIS2Client(
            "https://dhis2.test", "u", "p", max_attempts=1, transport=httpx.MockTransport(fake)
        )

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(DHIS2Client, "from_settings", from_settings)
    return fake


def test_ping(fake: FakeDHIS2) -> None:
    result = runner.invoke(cli.app, ["ping"])

    assert result.exit_code == 0, result.output
    assert "2.43.1" in result.output


def test_extract_end_to_end(fake: FakeDHIS2, tmp_path: Path) -> None:
    # the fake returns one org unit for every metadata query, used as the only "district"
    args = ["extract", "--start", "2025-01", "--end", "2025-02", "-d", "child_health"]
    result = runner.invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    assert "2 chunks" in result.output
    store = LocalRawStore(tmp_path / "raw")
    key = (
        "dhis2/data_value_sets/dataset=BfMAe6Itzgt/period=202501/"
        "org_unit=organisationUnits-1.json.gz"
    )
    assert store.exists(key)
    assert list((tmp_path / "raw" / "dhis2" / "metadata").iterdir())


def test_extract_exits_nonzero_when_chunks_fail(fake: FakeDHIS2) -> None:
    fake.failing_org_unit = "organisationUnits-1"

    result = runner.invoke(cli.app, ["extract", "--start", "2025-01", "--end", "2025-01"])

    assert result.exit_code == 1
    assert "Re-run" in result.output
