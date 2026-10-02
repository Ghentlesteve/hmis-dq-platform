from pathlib import Path

import pytest

from hmis_dq.extract.store import LocalRawStore


def test_round_trip(tmp_path: Path) -> None:
    store = LocalRawStore(tmp_path)
    key = "dhis2/data_value_sets/dataset=a/period=202501/org_unit=b.json.gz"

    assert not store.exists(key)
    written = store.put_json(key, {"payload": {"value": "13"}})

    assert store.exists(key)
    assert written > 0
    assert store.get_json(key) == {"payload": {"value": "13"}}


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


@pytest.mark.parametrize("key", ["../escape.json.gz", "/abs/path.json.gz", ""])
def test_rejects_keys_outside_the_store(tmp_path: Path, key: str) -> None:
    with pytest.raises(ValueError):
        LocalRawStore(tmp_path).put_json(key, {})
