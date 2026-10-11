from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
import pytest
from moto import mock_aws

from hmis_dq.extract.store import LocalRawStore
from hmis_dq.extract.sync import mirror_buckets, sync_stores

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


def stores(tmp_path: Path) -> tuple[LocalRawStore, LocalRawStore]:
    source = LocalRawStore(tmp_path / "src")
    for key in ["dhis2/a/1.json.gz", "dhis2/a/2.json.gz", "dhis2/b/1.json.gz"]:
        source.put_json(key, {"key": key})
    return source, LocalRawStore(tmp_path / "dst")


def test_copies_everything_once(tmp_path: Path) -> None:
    source, destination = stores(tmp_path)

    first = sync_stores(source, destination)
    second = sync_stores(source, destination)

    assert (first.copied, first.skipped) == (3, 0)
    assert (second.copied, second.skipped) == (0, 3)
    assert destination.get_json("dhis2/b/1.json.gz") == {"key": "dhis2/b/1.json.gz"}


def test_prefix_limits_what_is_copied(tmp_path: Path) -> None:
    source, destination = stores(tmp_path)

    result = sync_stores(source, destination, prefix="dhis2/a/")

    assert result.copied == 2
    assert list(destination.list_keys()) == ["dhis2/a/1.json.gz", "dhis2/a/2.json.gz"]


def test_overwrite_copies_existing_keys_again(tmp_path: Path) -> None:
    source, destination = stores(tmp_path)
    sync_stores(source, destination)
    source.put_json("dhis2/a/1.json.gz", {"key": "changed"})

    result = sync_stores(source, destination, overwrite=True)

    assert result.copied == 3
    assert destination.get_json("dhis2/a/1.json.gz") == {"key": "changed"}


def test_reports_progress_per_key(tmp_path: Path) -> None:
    source, destination = stores(tmp_path)
    seen: list[str] = []
    planned: list[int] = []

    sync_stores(source, destination, on_planned=planned.append, on_copied=seen.append)

    assert planned == [3]
    assert sorted(seen) == sorted(source.list_keys())


# ----------------------------------------------------------- bucket mirror


@pytest.fixture
def s3() -> Iterator["S3Client"]:
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="src-gold")
        client.put_object(Bucket="src-gold", Key="dhis2/a.parquet", Body=b"\x00PAR1 binary")
        client.put_object(Bucket="src-gold", Key="dhis2/b.json.gz", Body=b"\x1f\x8b gzip")
        yield client


def test_mirror_copies_any_object_byte_for_byte(s3: "S3Client") -> None:
    result = mirror_buckets(s3, s3, [("src-gold", "dst-gold")])

    assert (result.copied, result.skipped) == (2, 0)
    body = s3.get_object(Bucket="dst-gold", Key="dhis2/a.parquet")["Body"].read()
    assert body == b"\x00PAR1 binary"  # created the missing bucket, bytes untouched


def test_mirror_copies_only_what_changed(s3: "S3Client") -> None:
    mirror_buckets(s3, s3, [("src-gold", "dst-gold")])
    s3.put_object(Bucket="src-gold", Key="dhis2/a.parquet", Body=b"\x00PAR1 rewritten, longer")

    again = mirror_buckets(s3, s3, [("src-gold", "dst-gold")])

    assert (again.copied, again.skipped) == (1, 1)
