"""The custom NiFi processors (nifi/python_extensions), run without NiFi.

They are standard-library only, because they run inside NiFi's own Python where
hmis_dq isn't installed. These tests pin them to hmis_dq: NiFi must write the same
bronze keys and the same envelope as `hmis-dq extract`, or Spark would read two
different formats.
"""

import gzip
import importlib.util
import json
from datetime import UTC, date, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from hmis_dq.extract import job
from hmis_dq.extract.job import Chunk, Extractor
from hmis_dq.extract.periods import Month
from hmis_dq.extract.store import encode_json_gz

EXTENSIONS = Path(__file__).parents[1] / "nifi" / "python_extensions"
DISTRICTS = {"organisationUnits": [{"id": "O6uvpzGd5pu", "name": "Bo"}, {"id": "fdc6uOvgoji"}]}


def load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, EXTENSIONS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def chunks() -> Any:
    return load("ListDhis2Chunks")


@pytest.fixture
def wrap() -> Any:
    return load("WrapDhis2Envelope")


# ------------------------------------------------------------- planning


def test_plans_last_complete_months_for_every_dataset_and_district(chunks: Any) -> None:
    planned = chunks.plan_chunks(DISTRICTS, ["ds1", "ds2"], 3, date(2026, 1, 15))

    assert len(planned) == 2 * 2 * 3
    assert sorted({c["period"] for c in planned}) == ["202510", "202511", "202512"]
    first = planned[0]
    assert first == {
        "data_set": "ds1",
        "org_unit": "O6uvpzGd5pu",
        "org_unit_name": "Bo",
        "period": "202510",
        "start_date": "2025-10-01",
        "end_date": "2025-10-31",
        "key": "dhis2/data_value_sets/dataset=ds1/period=202510/org_unit=O6uvpzGd5pu.json.gz",
    }


@pytest.mark.parametrize("today", [date(2026, 1, 1), date(2026, 3, 31), date(2024, 3, 10)])
def test_chunks_match_the_python_extractor(chunks: Any, today: date) -> None:
    """Same months as the extractor's refresh window, same keys and date ranges."""
    planned = chunks.plan_chunks(DISTRICTS, ["ds1"], 3, today)

    this_month = Month.from_date(today)
    expected = [
        Chunk("ds1", unit["id"], this_month.shift(-n))
        for unit in DISTRICTS["organisationUnits"]
        for n in (3, 2, 1)
    ]
    assert [c["key"] for c in planned] == [c.key for c in expected]
    assert [(c["start_date"], c["end_date"]) for c in planned] == [
        (c.month.start.isoformat(), c.month.end.isoformat()) for c in expected
    ]


def test_february_in_a_leap_year_ends_on_the_29th(chunks: Any) -> None:
    (planned,) = chunks.plan_chunks(
        {"organisationUnits": [{"id": "x"}]}, ["ds"], 1, date(2024, 3, 5)
    )
    assert planned["end_date"] == "2024-02-29"


@pytest.mark.parametrize(
    ("org_units", "months_back"),
    [({"organisationUnits": []}, 3), ({"wrong": 1}, 3), (DISTRICTS, 0)],
)
def test_bad_input_is_refused(chunks: Any, org_units: dict[str, Any], months_back: int) -> None:
    with pytest.raises(ValueError):
        chunks.plan_chunks(org_units, ["ds"], months_back, date(2026, 1, 1))


def test_data_set_list_tolerates_spaces_and_blanks(chunks: Any) -> None:
    assert chunks.parse_data_sets(" a, b ,,c ") == ["a", "b", "c"]


# ------------------------------------------------------------- envelope


def test_envelope_and_bytes_match_the_python_extractor(
    wrap: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"dataSet": "ds1", "dataValues": [{"value": "1"}, {"value": "2"}]}
    chunk = {"data_set": "ds1", "org_unit": "ou1", "period": "202601"}
    when = datetime(2026, 2, 1, 8, 30, tzinfo=UTC)
    monkeypatch.setattr(job, "_now", lambda: when)
    extractor = Extractor(cast(Any, None), cast(Any, None), source_url="https://dhis2.test/")

    from_nifi = wrap.envelope(payload, chunk, "https://dhis2.test/", when)
    from_python = extractor._envelope(payload, **chunk, record_count=2)

    assert from_nifi == from_python
    assert wrap.encode(from_nifi) == encode_json_gz(from_python)  # byte for byte


@pytest.mark.parametrize("payload", [[], {"dataValues": "oops"}])
def test_envelope_refuses_what_is_not_a_data_value_set(wrap: Any, payload: Any) -> None:
    with pytest.raises(ValueError):
        wrap.envelope(
            payload, {"data_set": "a", "org_unit": "b", "period": "c"}, "u", datetime.now(UTC)
        )


def test_an_empty_month_is_still_a_valid_file(wrap: Any) -> None:
    """DHIS2 answers {} when nothing was reported: worth storing, it says 'nothing'."""
    document = wrap.envelope(
        {}, {"data_set": "a", "org_unit": "b", "period": "c"}, "u", datetime.now(UTC)
    )

    assert document["meta"]["record_count"] == 0


# --------------------------------------------- the processors, NiFi stubbed


class FlowFile:
    def __init__(self, content: bytes, attributes: dict[str, str] | None = None) -> None:
        self.content = content
        self.attributes = attributes or {}

    def getContentsAsBytes(self) -> bytes:  # noqa: N802  (NiFi's API name)
        return self.content

    def getAttribute(self, name: str) -> str | None:  # noqa: N802
        return self.attributes.get(name)


class Context:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    def getProperty(self, descriptor: Any) -> Any:  # noqa: N802
        return SimpleNamespace(getValue=lambda: self.values[descriptor.name])


def with_nifi_stubs(module: ModuleType) -> ModuleType:
    """Stand-ins for the nifiapi classes the processors use."""
    module.PropertyDescriptor = SimpleNamespace  # type: ignore[attr-defined]
    module.StandardValidators = SimpleNamespace(  # type: ignore[attr-defined]
        NON_EMPTY_VALIDATOR="non-empty", POSITIVE_INTEGER_VALIDATOR="positive-integer"
    )
    module.FlowFileTransformResult = SimpleNamespace  # type: ignore[attr-defined]
    return module


def test_list_processor_emits_the_plan_as_json(chunks: Any) -> None:
    processor = with_nifi_stubs(chunks).ListDhis2Chunks()

    result = processor.transform(
        Context({"Data Sets": "ds1,ds2", "Months Back": "2"}),
        FlowFile(json.dumps(DISTRICTS).encode()),
    )

    assert result.relationship == "success"
    assert len(json.loads(result.contents)) == 2 * 2 * 2
    assert result.attributes["dhis2.chunks"] == "8"
    assert [p.name for p in processor.getPropertyDescriptors()] == ["Data Sets", "Months Back"]


def test_list_processor_routes_bad_responses_to_failure(chunks: Any) -> None:
    processor = with_nifi_stubs(chunks).ListDhis2Chunks()
    processor.logger = SimpleNamespace(error=lambda message: None)

    result = processor.transform(
        Context({"Data Sets": "ds1", "Months Back": "2"}), FlowFile(b'{"organisationUnits": []}')
    )

    assert result.relationship == "failure"


def test_wrap_processor_writes_a_gzipped_bronze_file(wrap: Any) -> None:
    processor = with_nifi_stubs(wrap).WrapDhis2Envelope()
    body = json.dumps({"dataValues": [{"value": "5"}]}).encode()

    result = processor.transform(
        Context({"Source URL": "https://dhis2.test/"}),
        FlowFile(body, {"data_set": "ds1", "org_unit": "ou1", "period": "202601"}),
    )

    assert result.relationship == "success"
    document = json.loads(gzip.decompress(result.contents))
    assert document["meta"]["period"] == "202601"
    assert document["payload"] == {"dataValues": [{"value": "5"}]}
    assert result.attributes["dhis2.records"] == "1"


def test_wrap_processor_needs_the_chunk_attributes(wrap: Any) -> None:
    processor = with_nifi_stubs(wrap).WrapDhis2Envelope()
    processor.logger = SimpleNamespace(error=lambda message: None)

    result = processor.transform(Context({"Source URL": "u"}), FlowFile(b"{}", {"data_set": "ds1"}))

    assert result.relationship == "failure"
