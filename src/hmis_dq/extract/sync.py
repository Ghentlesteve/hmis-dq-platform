"""Copy documents between stores, e.g. the local bronze folder into the lake,
or whole buckets between two lakes (e.g. Docker Compose's and Kubernetes').

Idempotent: keys already in the destination are skipped (one listing call up
front instead of one existence check per key), so it is safe to re-run.
"""

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from botocore.exceptions import ClientError

from hmis_dq.extract.store import RawStore

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


@dataclass(frozen=True)
class SyncResult:
    copied: int
    skipped: int
    bytes_written: int


def sync_stores(
    source: RawStore,
    destination: RawStore,
    *,
    prefix: str = "",
    overwrite: bool = False,
    workers: int = 8,
    on_planned: Callable[[int], None] | None = None,
    on_copied: Callable[[str], None] | None = None,
) -> SyncResult:
    keys = list(source.list_keys(prefix))
    existing = set() if overwrite else set(destination.list_keys(prefix))
    to_copy = [k for k in keys if k not in existing]
    if on_planned is not None:
        on_planned(len(to_copy))

    def copy(key: str) -> int:
        written = destination.put_json(key, source.get_json(key))
        if on_copied is not None:
            on_copied(key)
        return written

    with ThreadPoolExecutor(max_workers=workers) as pool:
        written = sum(pool.map(copy, to_copy))

    return SyncResult(copied=len(to_copy), skipped=len(keys) - len(to_copy), bytes_written=written)


# ------------------------------------------------------------- whole lakes


def _sizes(client: "S3Client", bucket: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        for item in page.get("Contents", []):
            sizes[item["Key"]] = item["Size"]
    return sizes


def mirror_buckets(
    source: "S3Client",
    destination: "S3Client",
    buckets: Iterable[tuple[str, str]],
    *,
    workers: int = 8,
    on_planned: Callable[[int], None] | None = None,
    on_copied: Callable[[str], None] | None = None,
) -> SyncResult:
    """Copy every object, byte for byte, from one lake to another (any format:
    bronze JSON, silver and gold Parquet). ``buckets`` pairs source and destination
    bucket names. An object already there with the same size is skipped; missing
    destination buckets are created."""
    work: list[tuple[str, str, str]] = []
    skipped = 0
    for source_bucket, destination_bucket in buckets:
        try:
            destination.head_bucket(Bucket=destination_bucket)
        except ClientError:
            destination.create_bucket(Bucket=destination_bucket)
        there = _sizes(destination, destination_bucket)
        for key, size in _sizes(source, source_bucket).items():
            if there.get(key) == size:
                skipped += 1
            else:
                work.append((source_bucket, destination_bucket, key))
    if on_planned is not None:
        on_planned(len(work))

    def copy(item: tuple[str, str, str]) -> int:
        source_bucket, destination_bucket, key = item
        body = source.get_object(Bucket=source_bucket, Key=key)["Body"].read()
        destination.put_object(Bucket=destination_bucket, Key=key, Body=body)
        if on_copied is not None:
            on_copied(key)
        return len(body)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        written = sum(pool.map(copy, work))
    return SyncResult(copied=len(work), skipped=skipped, bytes_written=written)
