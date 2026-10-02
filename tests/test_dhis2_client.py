"""DHIS2 client tests against an in-process fake server (no network needed)."""

from collections.abc import Callable
from datetime import date
from typing import Any

import httpx
import pytest

from hmis_dq.dhis2 import DHIS2Client, DHIS2Error

Handler = Callable[[httpx.Request], httpx.Response]


def make_client(handler: Handler, *, max_attempts: int = 3) -> DHIS2Client:
    return DHIS2Client(
        "https://dhis2.test/demo",
        "reader",
        "secret",
        max_attempts=max_attempts,
        max_backoff_seconds=0,  # no real sleeping in tests
        transport=httpx.MockTransport(handler),
    )


def org_unit(uid: str, level: int = 4) -> dict[str, Any]:
    return {"id": uid, "name": f"Facility {uid}", "level": level, "path": f"/root/{uid}"}


def test_requests_go_to_the_api_path_with_basic_auth() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"version": "2.43.1", "systemName": "Demo"})

    with make_client(handler) as client:
        info = client.system_info()

    assert info.version == "2.43.1"
    assert seen[0].url.path == "/demo/api/system/info"
    assert seen[0].headers["Authorization"].startswith("Basic ")


def test_iter_pages_follows_the_pager_until_the_last_page() -> None:
    pages = {
        "1": [org_unit("a"), org_unit("b")],
        "2": [org_unit("c")],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        page = request.url.params["page"]
        return httpx.Response(
            200,
            json={
                "pager": {"page": int(page), "pageCount": 2, "total": 3, "pageSize": 2},
                "organisationUnits": pages[page],
            },
        )

    with make_client(handler) as client:
        units = list(client.organisation_units(level=4))

    assert [u.id for u in units] == ["a", "b", "c"]


def test_organisation_units_sends_level_filter() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["filter"] == "level:eq:2"
        return httpx.Response(200, json={"organisationUnits": [org_unit("d", level=2)]})

    with make_client(handler) as client:
        (district,) = client.organisation_units(level=2)

    assert district.level == 2


def test_data_value_set_parses_camel_case_records() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["startDate"] == "2025-01-01"
        assert request.url.params["children"] == "true"
        return httpx.Response(
            200,
            json={
                "dataSet": "ds1",
                "dataValues": [
                    {
                        "dataElement": "de1",
                        "period": "202501",
                        "orgUnit": "ou1",
                        "categoryOptionCombo": "coc1",
                        "attributeOptionCombo": "aoc1",
                        "value": "13",
                        "lastUpdated": "2010-03-17T00:00:00.000+0000",
                    }
                ],
            },
        )

    with make_client(handler) as client:
        dvs = client.data_value_set("ds1", "ou1", date(2025, 1, 1), date(2025, 1, 31))

    (value,) = dvs.data_values
    assert value.data_element == "de1"
    assert value.value == "13"
    assert value.last_updated is not None and value.last_updated.year == 2010


def test_transient_server_errors_are_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"version": "2.43.1"})

    with make_client(handler, max_attempts=3) as client:
        assert client.system_info().version == "2.43.1"

    assert calls == 3


def test_network_errors_are_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectTimeout("slow server", request=request)
        return httpx.Response(200, json={"version": "2.43.1"})

    with make_client(handler) as client:
        client.system_info()

    assert calls == 2


def test_client_errors_fail_immediately_without_retry() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401)

    with make_client(handler) as client, pytest.raises(DHIS2Error, match="401"):
        client.system_info()

    assert calls == 1


def test_gives_up_after_max_attempts() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(502)

    with make_client(handler, max_attempts=4) as client, pytest.raises(DHIS2Error, match="502"):
        client.system_info()

    assert calls == 4
