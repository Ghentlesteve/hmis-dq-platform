"""End-to-end CLI tests: real command, fake DHIS2 server, temp data directory."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import boto3
import httpx
import pytest
from moto import mock_aws
from typer.testing import CliRunner

from hmis_dq import cli
from hmis_dq.config import Settings, StoreKind
from hmis_dq.dhis2 import DHIS2Client
from hmis_dq.extract.store import LocalRawStore, S3RawStore
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
        store=StoreKind.LOCAL,
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


# ------------------------------------------------------------ lake commands


@pytest.fixture
def lake_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Settings]:
    """Settings pointing at a moto (fake) S3 with an empty bronze bucket."""
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="bronze")
        settings = Settings(
            _env_file=None,
            dhis2_base_url="https://dhis2.test",
            dhis2_username="u",
            dhis2_password="p",
            data_dir=tmp_path,
            store=StoreKind.LAKE,
        )
        monkeypatch.setattr(cli, "get_settings", lambda: settings)
        monkeypatch.setattr(S3RawStore, "ensure_reachable", lambda self: None)
        yield settings


def test_lake_upload_copies_then_skips(lake_settings: Settings) -> None:
    local = LocalRawStore(lake_settings.raw_dir)
    local.put_json("dhis2/data_value_sets/dataset=ds/period=202501/org_unit=a.json.gz", {"x": 1})
    local.put_json("dhis2/metadata/dataSets/snapshot=r1.json.gz", {"y": 2})

    first = runner.invoke(cli.app, ["lake", "upload"])
    second = runner.invoke(cli.app, ["lake", "upload"])

    assert first.exit_code == 0, first.output
    assert "2 copied, 0 already in the lake" in first.output
    assert "0 copied, 2 already in the lake" in second.output


def test_lake_status_counts_objects(lake_settings: Settings) -> None:
    local = LocalRawStore(lake_settings.raw_dir)
    local.put_json("dhis2/data_value_sets/dataset=ds/period=202501/org_unit=a.json.gz", {})
    runner.invoke(cli.app, ["lake", "upload"])

    result = runner.invoke(cli.app, ["lake", "status"])

    assert result.exit_code == 0, result.output
    assert "dhis2/data_value_sets" in result.output
    assert "dataset=ds" in result.output


def test_extract_writes_to_the_lake(
    lake_settings: Settings, fake: FakeDHIS2, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the `fake` fixture points get_settings at store=local; point it back at the lake
    monkeypatch.setattr(cli, "get_settings", lambda: lake_settings)

    result = runner.invoke(cli.app, ["extract", "--start", "2025-01", "--end", "2025-01"])

    assert result.exit_code == 0, result.output
    assert "s3://bronze" in result.output
    keys = list(S3RawStore.from_settings(lake_settings).list_keys("dhis2/data_value_sets/"))
    assert len(keys) == 2  # 2 default datasets x 1 district x 1 month


def test_lake_unavailable_gives_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(
        _env_file=None,
        dhis2_base_url="https://dhis2.test",
        dhis2_username="u",
        dhis2_password="p",
        s3_endpoint_url="http://127.0.0.1:1",  # nothing listens on port 1
        s3_access_key="k",
        s3_secret_key="s",
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    result = runner.invoke(cli.app, ["lake", "status"])

    assert result.exit_code == 2
    assert "docker compose up -d" in result.output
