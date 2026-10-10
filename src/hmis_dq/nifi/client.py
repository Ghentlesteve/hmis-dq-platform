"""A small client for the NiFi 2 REST API: log in, then JSON in and out.

NiFi 2 always runs on HTTPS with a login. The client trades the username and
password for a token once, then sends it with every request.
"""

from types import TracebackType
from typing import Any, Self

import httpx

from hmis_dq.config import Settings

JSON = dict[str, Any]


class NiFiError(RuntimeError):
    """NiFi refused a request; the message carries NiFi's own explanation."""


class NiFiClient:
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        *,
        verify: bool = True,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url.rstrip("/") + "/nifi-api/",
            verify=verify,
            timeout=timeout,
            transport=transport,
        )
        self._username = username
        self._password = password

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> Self:
        if not settings.nifi_username or settings.nifi_password is None:
            raise NiFiError("set HMIS_NIFI_USERNAME and HMIS_NIFI_PASSWORD in .env")
        return cls(
            str(settings.nifi_url),
            settings.nifi_username,
            settings.nifi_password.get_secret_value(),
            verify=settings.nifi_verify_tls,
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

    def login(self) -> None:
        response = self._http.post(
            "access/token", data={"username": self._username, "password": self._password}
        )
        self._check(response, "log in")
        self._http.headers["Authorization"] = f"Bearer {response.text.strip()}"

    def request(self, method: str, path: str, **kwargs: Any) -> JSON:
        if "Authorization" not in self._http.headers:
            self.login()
        response = self._http.request(method, path, **kwargs)
        self._check(response, f"{method} {path}")
        return response.json() if response.content else {}

    def get(self, path: str, **params: Any) -> JSON:
        return self.request("GET", path, params=params or None)

    def post(self, path: str, body: JSON) -> JSON:
        return self.request("POST", path, json=body)

    def put(self, path: str, body: JSON) -> JSON:
        return self.request("PUT", path, json=body)

    def delete(self, path: str, **params: Any) -> JSON:
        return self.request("DELETE", path, params=params or None)

    @staticmethod
    def _check(response: httpx.Response, action: str) -> None:
        if response.is_success:
            return
        # NiFi explains refusals in plain text ("Unable to ... because ...")
        raise NiFiError(f"NiFi refused to {action} ({response.status_code}): {response.text}")
