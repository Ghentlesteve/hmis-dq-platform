"""Extraction job tests against a fake DHIS2 server and a temp-dir store."""

import threading
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from hmis_dq.dhis2 import DHIS2Client
from hmis_dq.extract.job import METADATA_RESOURCES, Chunk, Extractor
from hmis_dq.extract.periods import Month, month_range
from hmis_dq.extract.store import LocalRawStore

TODAY = date(2025, 6, 15)


class FakeDHIS2:
    """Records data value requests; can be told to fail for one org unit."""

    def __init__(self) -> None:
        self.data_requests: list[tuple[str, str, str]] = []
        self.failing_org_unit: str | None = None
        self._lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/api/")
        params = request.url.params

        if path == "system/info":
            return httpx.Response(200, json={"version": "2.43.1"})

        if path == "dataValueSets":
            ds, ou, start = params["dataSet"], params["orgUnit"], params["startDate"]
            with self._lock:
                self.data_requests.append((ds, ou, start))
            if ou == self.failing_org_unit:
                return httpx.Response(404)
            period = start[:7].replace("-", "")
            return httpx.Response(200, json={"dataSet": ds, "dataValues": [value(ou, period)] * 2})

        if path in METADATA_RESOURCES:
            pager = {"page": 1, "pageCount": 1, "total": 1, "pageSize": 500}
            item: dict[str, Any] = {"id": f"{path}-1"}
            if path == "organisationUnits":
                item |= {"name": "District 1", "level": 2, "path": f"/root/{path}-1"}
            return httpx.Response(200, json={"pager": pager, path: [item]})

        return httpx.Response(404)


def value(org_unit: str, period: str) -> dict[str, Any]:
    return {
        "dataElement": "de1",
        "period": period,
        "orgUnit": org_unit,
        "categoryOptionCombo": "coc",
        "attributeOptionCombo": "aoc",
        "value": "5",
    }


@pytest.fixture
def fake() -> FakeDHIS2:
    return FakeDHIS2()


@pytest.fixture
def store(tmp_path: Path) -> LocalRawStore:
    return LocalRawStore(tmp_path)


def make_extractor(fake: FakeDHIS2, store: LocalRawStore, **kwargs: Any) -> Extractor:
    client = DHIS2Client(
        "https://dhis2.test",
        "u",
        "p",
        max_attempts=1,
        transport=httpx.MockTransport(fake),
    )
    defaults: dict[str, Any] = {"workers": 2, "refresh_recent_months": 0, "today": TODAY}
    return Extractor(client, store, source_url="https://dhis2.test", **{**defaults, **kwargs})


MONTHS = month_range(Month(2025, 1), Month(2025, 3))


def test_every_chunk_is_fetched_and_stored(fake: FakeDHIS2, store: LocalRawStore) -> None:
    extractor = make_extractor(fake, store)

    summary = extractor.extract_data_values(["dsA", "dsB"], ["ou1", "ou2"], MONTHS, run_id="r1")

    assert summary.planned == summary.fetched == 12
    assert summary.records == 24
    assert summary.server_version == "2.43.1"
    assert summary.ok

    doc = store.get_json(Chunk("dsA", "ou1", Month(2025, 2)).key)
    assert doc["meta"]["period"] == "202502"
    assert doc["meta"]["record_count"] == 2
    assert doc["meta"]["source_url"] == "https://dhis2.test"
    assert doc["payload"]["dataValues"][0]["orgUnit"] == "ou1"  # raw, untouched


def test_chunk_keys_are_hive_partitioned() -> None:
    key = Chunk("BfMAe6Itzgt", "O6uvpzGd5pu", Month(2025, 1)).key
    assert key == (
        "dhis2/data_value_sets/dataset=BfMAe6Itzgt/period=202501/org_unit=O6uvpzGd5pu.json.gz"
    )


def test_rerun_skips_chunks_already_stored(fake: FakeDHIS2, store: LocalRawStore) -> None:
    make_extractor(fake, store).extract_data_values(["dsA"], ["ou1"], MONTHS, run_id="r1")
    fake.data_requests.clear()

    summary = make_extractor(fake, store).extract_data_values(["dsA"], ["ou1"], MONTHS, run_id="r2")

    assert summary.skipped == 3
    assert summary.fetched == 0
    assert fake.data_requests == []


def test_recent_months_are_always_refreshed(fake: FakeDHIS2, store: LocalRawStore) -> None:
    months = month_range(Month(2025, 3), Month(2025, 6))
    make_extractor(fake, store).extract_data_values(["dsA"], ["ou1"], months, run_id="r1")
    fake.data_requests.clear()

    # today is 2025-06-15: the last 2 complete months are April and May,
    # plus June itself, which is still in progress
    summary = make_extractor(fake, store, refresh_recent_months=2).extract_data_values(
        ["dsA"], ["ou1"], months, run_id="r2"
    )

    assert summary.skipped == 1  # only March
    assert sorted(start for _, _, start in fake.data_requests) == [
        "2025-04-01",
        "2025-05-01",
        "2025-06-01",
    ]


def test_force_refetches_everything(fake: FakeDHIS2, store: LocalRawStore) -> None:
    make_extractor(fake, store).extract_data_values(["dsA"], ["ou1"], MONTHS, run_id="r1")

    summary = make_extractor(fake, store).extract_data_values(
        ["dsA"], ["ou1"], MONTHS, run_id="r2", force=True
    )

    assert summary.fetched == 3


def test_a_failing_chunk_does_not_stop_the_run(fake: FakeDHIS2, store: LocalRawStore) -> None:
    fake.failing_org_unit = "bad"

    summary = make_extractor(fake, store).extract_data_values(
        ["dsA"], ["ou1", "bad"], MONTHS, run_id="r1"
    )

    assert summary.fetched == 3
    assert summary.failed == 3
    assert not summary.ok
    assert all("org_unit=bad" in f["key"] for f in summary.failures)
    assert not store.exists(Chunk("dsA", "bad", Month(2025, 1)).key)

    # the failed chunks are retried on the next run, the good ones skipped
    fake.failing_org_unit = None
    retry = make_extractor(fake, store).extract_data_values(
        ["dsA"], ["ou1", "bad"], MONTHS, run_id="r2"
    )
    assert (retry.fetched, retry.skipped, retry.failed) == (3, 3, 0)


def test_run_summary_is_written_to_the_store(fake: FakeDHIS2, store: LocalRawStore) -> None:
    make_extractor(fake, store).extract_data_values(["dsA"], ["ou1"], MONTHS, run_id="r1")

    summary = store.get_json("dhis2/_runs/run=r1.json.gz")
    assert summary["fetched"] == 3
    assert summary["finished_at"] is not None


def test_metadata_snapshot_writes_every_resource(fake: FakeDHIS2, store: LocalRawStore) -> None:
    counts = make_extractor(fake, store).extract_metadata("r1")

    assert set(counts) == set(METADATA_RESOURCES)
    doc = store.get_json("dhis2/metadata/dataElements/snapshot=r1.json.gz")
    assert doc["payload"] == {"dataElements": [{"id": "dataElements-1"}]}
    assert doc["meta"]["resource"] == "dataElements"


def test_rejects_zero_workers(fake: FakeDHIS2, store: LocalRawStore) -> None:
    with pytest.raises(ValueError):
        make_extractor(fake, store, workers=0)
