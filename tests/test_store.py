"""Storage tests. The contract tests run against every RawStore implementation."""

from collections.abc import Iterator
from pathlib import Path

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from hmis_dq.extract.store import LocalRawStore, RawStore, S3RawStore

KEY = "dhis2/data_value_sets/dataset=a/period=202501/org_unit=b.json.gz"


@pytest.fixture
def s3_store() -> Iterator[S3RawStore]:
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="bronze")
        yield S3RawStore(client, "bronze")


@pytest.fixture(params=["local", "s3"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> RawStore:
    if request.param == "local":
        return LocalRawStore(tmp_path)
    s3: RawStore = request.getfixturevalue("s3_store")
    return s3


# ------------------------------------------------- contract (both stores)


def test_round_trip(store: RawStore) -> None:
    assert not store.exists(KEY)
    written = store.put_json(KEY, {"payload": {"value": "13"}})

    assert store.exists(KEY)
    assert written > 0
    assert store.get_json(KEY) == {"payload": {"value": "13"}}


def test_overwrite_replaces_the_document(store: RawStore) -> None:
    store.put_json(KEY, {"v": 1})
    store.put_json(KEY, {"v": 2})

    assert store.get_json(KEY) == {"v": 2}


def test_list_keys_filters_by_prefix_in_sorted_order(store: RawStore) -> None:
    for key in ["a/2.json.gz", "b/1.json.gz", "a/1.json.gz"]:
        store.put_json(key, {})

    assert list(store.list_keys("a/")) == ["a/1.json.gz", "a/2.json.gz"]
    assert len(list(store.list_keys())) == 3


def test_list_keys_on_empty_store(store: RawStore) -> None:
    assert list(store.list_keys()) == []


@pytest.mark.parametrize("key", ["../escape.json.gz", "/abs/path.json.gz", ""])
def test_rejects_keys_outside_the_store(store: RawStore, key: str) -> None:
    with pytest.raises(ValueError):
        store.put_json(key, {})


# ----------------------------------------------------- local-store specifics


def test_no_temp_files_left_behind(tmp_path: Path) -> None:
    store = LocalRawStore(tmp_path)
    store.put_json("a/b.json.gz", {"x": 1})

    files = [p.name for p in tmp_path.rglob("*") if p.is_file()]
    assert files == ["b.json.gz"]


def test_same_content_gives_same_bytes(tmp_path: Path) -> None:
    store = LocalRawStore(tmp_path)
    store.put_json("one.json.gz", {"x": 1})
    store.put_json("two.json.gz", {"x": 1})

    assert (tmp_path / "one.json.gz").read_bytes() == (tmp_path / "two.json.gz").read_bytes()


def test_list_keys_on_missing_folder(tmp_path: Path) -> None:
    assert list(LocalRawStore(tmp_path / "nope").list_keys()) == []


# -------------------------------------------------------- S3-store specifics


def test_s3_objects_are_gzip(s3_store: S3RawStore) -> None:
    s3_store.put_json(KEY, {"x": 1})

    head = s3_store.client.head_object(Bucket="bronze", Key=KEY)
    assert head["ContentType"] == "application/gzip"


def test_s3_list_keys_follows_pagination(s3_store: S3RawStore) -> None:
    for i in range(1_005):  # more than one 1,000-key page
        s3_store.client.put_object(Bucket="bronze", Key=f"k/{i:05d}", Body=b"")

    assert len(list(s3_store.list_keys("k/"))) == 1_005


def test_s3_errors_other_than_not_found_are_raised(s3_store: S3RawStore) -> None:
    missing_bucket = S3RawStore(s3_store.client, "no-such-bucket")

    with pytest.raises(ClientError) as excinfo:
        missing_bucket.get_json(KEY)

    assert excinfo.value.response["Error"]["Code"] == "NoSuchBucket"
