"""Storage for the bronze (raw) layer.

Files are gzipped JSON addressed by a key such as
``dhis2/data_value_sets/dataset=X/period=202501/org_unit=Y.json.gz``.
The ``name=value`` folders are Hive-style partitions, which Spark reads natively.

``RawStore`` is a protocol so the local-disk store used now can be swapped for an
S3/MinIO store in stage 2 without touching the extraction code.
"""

import gzip
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Protocol


class RawStore(Protocol):
    def exists(self, key: str) -> bool: ...

    def put_json(self, key: str, document: Mapping[str, Any]) -> int:
        """Write a document under ``key`` and return the number of bytes stored."""
        ...

    def get_json(self, key: str) -> Any: ...


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
