"""Storage for the bronze (raw) layer.

Files are gzipped JSON addressed by a key such as
``dhis2/data_value_sets/dataset=X/period=202501/org_unit=Y.json.gz``.
The ``name=value`` folders are Hive-style partitions, which Spark reads natively.

``RawStore`` is a protocol with two implementations: ``LocalRawStore`` (a folder
on disk) and ``S3RawStore`` (any S3 API: the local SeaweedFS lake or AWS S3).
The extraction code works with either.
"""

import gzip
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol, Self

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from hmis_dq.config import Settings

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


class RawStore(Protocol):
    def exists(self, key: str) -> bool: ...

    def put_json(self, key: str, document: Mapping[str, Any]) -> int:
        """Write a document under ``key`` and return the number of bytes stored."""
        ...

    def get_json(self, key: str) -> Any: ...

    def list_keys(self, prefix: str = "") -> Iterator[str]:
        """Yield every stored key starting with ``prefix``, in sorted order."""
        ...


def encode_json_gz(document: Mapping[str, Any]) -> bytes:
    raw = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return gzip.compress(raw, mtime=0)  # fixed mtime: same content -> same bytes


def validate_key(key: str) -> PurePosixPath:
    path = PurePosixPath(key)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"invalid storage key: {key!r}")
    return path


class LocalRawStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root.joinpath(*validate_key(key).parts)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def put_json(self, key: str, document: Mapping[str, Any]) -> int:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = encode_json_gz(document)

        # Write to a temp file then rename: a crash mid-write never leaves a
        # half-written file that a resumed run would mistake for a finished chunk.
        fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".tmp-", suffix=".part")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp_name, target)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return len(data)

    def get_json(self, key: str) -> Any:
        return json.loads(gzip.decompress(self._path(key).read_bytes()))

    def list_keys(self, prefix: str = "") -> Iterator[str]:
        if not self.root.is_dir():
            return
        keys = (
            p.relative_to(self.root).as_posix()
            for p in self.root.rglob("*")
            if p.is_file() and not p.name.startswith(".tmp-")
        )
        yield from sorted(k for k in keys if k.startswith(prefix))


class S3RawStore:
    """Stores documents as objects in an S3 bucket.

    An S3 PUT is atomic (readers see the old object or the new one, never half),
    so no temp-file dance is needed here.
    """

    def __init__(self, client: "S3Client", bucket: str) -> None:
        self.client = client
        self.bucket = bucket

    @classmethod
    def from_settings(cls, settings: Settings, bucket: str | None = None) -> Self:
        secret = settings.s3_secret_key.get_secret_value() if settings.s3_secret_key else None
        client = boto3.client(
            "s3",
            endpoint_url=str(settings.s3_endpoint_url) if settings.s3_endpoint_url else None,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=secret,
            region_name=settings.s3_region,
            config=Config(
                retries={"max_attempts": 5, "mode": "standard"},
                # path-style URLs (endpoint/bucket/key) work with every S3-compatible server
                s3={"addressing_style": "path"},
                max_pool_connections=32,
            ),
        )
        return cls(client, bucket or settings.bronze_bucket)

    def exists(self, key: str) -> bool:
        validate_key(key)
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise
        return True

    def put_json(self, key: str, document: Mapping[str, Any]) -> int:
        validate_key(key)
        data = encode_json_gz(document)
        self.client.put_object(
            Bucket=self.bucket, Key=key, Body=data, ContentType="application/gzip"
        )
        return len(data)

    def get_json(self, key: str) -> Any:
        validate_key(key)
        body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        return json.loads(gzip.decompress(body))

    def list_keys(self, prefix: str = "") -> Iterator[str]:
        # S3 returns at most 1,000 keys per call; the paginator follows the continuation tokens.
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]
