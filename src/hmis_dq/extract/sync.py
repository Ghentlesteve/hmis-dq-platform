"""Copy documents between stores, e.g. the local bronze folder into the lake.

Idempotent: keys already in the destination are skipped (one listing call up
front instead of one existence check per key), so it is safe to re-run.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from hmis_dq.extract.store import RawStore


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
