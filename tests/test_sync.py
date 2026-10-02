from pathlib import Path

from hmis_dq.extract.store import LocalRawStore
from hmis_dq.extract.sync import sync_stores


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

    sync_stores(source, destination, on_copied=seen.append)

    assert sorted(seen) == sorted(source.list_keys())
