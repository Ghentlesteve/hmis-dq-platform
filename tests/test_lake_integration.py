"""Integration test against the real local lake (docker compose).

Skipped automatically when the lake isn't running or isn't configured.
Run only these with:  pytest -m integration
"""

import socket
import uuid
from collections.abc import Iterator

import pytest
from botocore.exceptions import BotoCoreError, ClientError

from hmis_dq.config import Settings
from hmis_dq.extract.store import S3RawStore

pytestmark = pytest.mark.integration


@pytest.fixture
def lake() -> Iterator[S3RawStore]:
    try:
        settings = Settings()
    except Exception:
        pytest.skip("settings not configured (.env missing)")
    if settings.s3_endpoint_url is None:
        pytest.skip("HMIS_S3_ENDPOINT_URL not set")

    # Fail fast if nothing is listening, instead of waiting for boto3's retries.
    host, port = settings.s3_endpoint_url.host, settings.s3_endpoint_url.port
    try:
        socket.create_connection((host or "localhost", port or 80), timeout=1).close()
    except OSError:
        pytest.skip("lake not running; start it with `docker compose up -d`")

    store = S3RawStore.from_settings(settings)
    try:
        store.client.head_bucket(Bucket=store.bucket)
    except (BotoCoreError, ClientError) as exc:
        pytest.skip(f"lake not reachable ({type(exc).__name__}); run `docker compose up -d`")
    yield store


def test_round_trip_against_the_real_lake(lake: S3RawStore) -> None:
    key = f"_tests/{uuid.uuid4()}.json.gz"
    try:
        lake.put_json(key, {"hello": "lake"})

        assert lake.exists(key)
        assert lake.get_json(key) == {"hello": "lake"}
        assert key in list(lake.list_keys("_tests/"))
    finally:
        lake.client.delete_object(Bucket=lake.bucket, Key=key)

    assert not lake.exists(key)
