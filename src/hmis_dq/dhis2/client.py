"""HTTP client for the DHIS2 Web API.

- One pooled connection with basic auth and timeouts
- Retries with exponential backoff on network errors, 429 and 5xx
  (never on 4xx like 401/404 — retrying those just hides a real problem)
- Pagination for metadata endpoints
- Chunked reads for data values (``dataValueSets`` ignores paging, so the
  caller asks for small slices, e.g. one district x one month)
"""

import logging
from collections.abc import Iterator
from datetime import date
from types import TracebackType
from typing import Any, Self

import httpx
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from hmis_dq.config import Settings
from hmis_dq.dhis2.models import DataValueSet, OrganisationUnit, Pager, SystemInfo

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
DEFAULT_ORG_UNIT_FIELDS = "id,name,level,path,parent[id],openingDate,closedDate,geometry"


class DHIS2Error(RuntimeError):
    """Raised when DHIS2 returns an error we should not retry."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):  # timeouts, connection resets, DNS
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    return False


class DHIS2Client:
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        *,
        timeout: float = 60.0,
        max_attempts: int = 5,
        max_backoff_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url.rstrip("/") + "/api/",
            auth=(username, password),
            timeout=timeout,
            headers={"Accept": "application/json"},
            transport=transport,
        )
        self._max_attempts = max_attempts
        self._max_backoff = max_backoff_seconds

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> Self:
        return cls(
            str(settings.dhis2_base_url),
            settings.dhis2_username,
            settings.dhis2_password.get_secret_value(),
            timeout=settings.dhis2_timeout_seconds,
            **kwargs,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------ core

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET an API path and return the decoded JSON body, retrying transient failures."""
        retrying = Retrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(initial=1, max=self._max_backoff),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        try:
            for attempt in retrying:
                with attempt:
                    response = self._http.get(path, params=params)
                    response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise DHIS2Error(
                f"DHIS2 returned {exc.response.status_code} for {exc.request.url}"
            ) from exc

        body: dict[str, Any] = response.json()
        return body

    def iter_pages(
        self,
        path: str,
        collection: str,
        params: dict[str, Any] | None = None,
        *,
        page_size: int = 500,
    ) -> Iterator[dict[str, Any]]:
        """Yield every item of a paged metadata collection, fetching page by page."""
        page = 1
        while True:
            body = self.get_json(path, {**(params or {}), "page": page, "pageSize": page_size})
            yield from body.get(collection, [])

            pager = Pager.model_validate(body["pager"]) if "pager" in body else None
            if pager is None or pager.page >= pager.page_count:
                return
            page += 1

    # ------------------------------------------------------------- endpoints

    def system_info(self) -> SystemInfo:
        return SystemInfo.model_validate(self.get_json("system/info"))

    def organisation_units(
        self,
        *,
        level: int | None = None,
        fields: str = DEFAULT_ORG_UNIT_FIELDS,
    ) -> Iterator[OrganisationUnit]:
        params: dict[str, Any] = {"fields": fields}
        if level is not None:
            params["filter"] = f"level:eq:{level}"
        for item in self.iter_pages("organisationUnits", "organisationUnits", params):
            yield OrganisationUnit.model_validate(item)

    def data_value_set(
        self,
        data_set: str,
        org_unit: str,
        start: date,
        end: date,
        *,
        children: bool = True,
    ) -> DataValueSet:
        """Fetch data values for one dataset, org unit subtree and date range."""
        body = self.data_value_set_raw(data_set, org_unit, start, end, children=children)
        return DataValueSet.model_validate(body)

    def data_value_set_raw(
        self,
        data_set: str,
        org_unit: str,
        start: date,
        end: date,
        *,
        children: bool = True,
    ) -> dict[str, Any]:
        """Same as :meth:`data_value_set` but returns the JSON exactly as the server sent it."""
        return self.get_json(
            "dataValueSets",
            {
                "dataSet": data_set,
                "orgUnit": org_unit,
                "startDate": start.isoformat(),
                "endDate": end.isoformat(),
                "children": str(children).lower(),
            },
        )
