"""Deploying the flow, against a fake NiFi that follows the real API's rules.

The fake keeps what the deployer builds and validates it the way NiFi does:
known property names only (unless the processor takes dynamic properties), and
every relationship must be connected or auto-terminated before a processor is
valid. That catches the mistakes NiFi would only report after a deploy.
"""

import itertools
import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from hmis_dq import cli
from hmis_dq.config import Settings
from hmis_dq.nifi.client import NiFiClient, NiFiError
from hmis_dq.nifi.deploy import Deployer
from hmis_dq.nifi.flow import FAILED, FLOW_NAME, dhis2_to_bronze

# Processor types the fake knows: properties as (internal name, display name),
# whether dynamic properties are allowed, and the relationships.
INVOKE_HTTP = {
    "properties": [
        ("HTTP Method", "HTTP Method"),
        ("HTTP URL", "HTTP URL"),
        ("Request Username", "Request Username"),
        # as in older NiFi versions: the API name differs from the one on screen
        ("Basic Authentication Password", "Request Password"),
        ("Socket Read Timeout", "Socket Read Timeout"),
    ],
    "dynamic": False,
    "relationships": ["Original", "Response", "Retry", "No Retry", "Failure"],
}
PYTHON_TRANSFORM = ["success", "failure", "original"]
TYPES: dict[str, dict[str, Any]] = {
    "org.apache.nifi.processors.standard.InvokeHTTP": INVOKE_HTTP,
    "ListDhis2Chunks": {
        "properties": [("Data Sets", "Data Sets"), ("Months Back", "Months Back")],
        "dynamic": False,
        "relationships": PYTHON_TRANSFORM,
    },
    "org.apache.nifi.processors.standard.SplitJson": {
        "properties": [("JsonPath Expression", "JsonPath Expression")],
        "dynamic": False,
        "relationships": ["original", "split", "failure"],
    },
    "org.apache.nifi.processors.standard.EvaluateJsonPath": {
        "properties": [("Destination", "Destination")],
        "dynamic": True,
        "relationships": ["matched", "unmatched", "failure"],
    },
    "WrapDhis2Envelope": {
        "properties": [("Source URL", "Source URL")],
        "dynamic": False,
        "relationships": PYTHON_TRANSFORM,
    },
    "org.apache.nifi.processors.aws.s3.PutS3Object": {
        "properties": [
            (name, name)
            for name in (
                "Bucket",
                "Object Key",
                "Region",
                "AWS Credentials Provider Service",
                "Endpoint Override URL",
                "Use Path Style Access",
                "Content Type",
            )
        ],
        "dynamic": False,
        "relationships": ["success", "failure"],
    },
}
SERVICE_TYPE = (
    "org.apache.nifi.processors.aws.credentials.provider.service."
    "AWSCredentialsProviderControllerService"
)
SERVICE_PROPERTIES = [("access-key", "Access Key ID"), ("secret-key", "Secret Access Key")]
BUNDLE = {"group": "org.apache.nifi", "artifact": "nifi-standard-nar", "version": "2.12.0"}


class FakeNiFi:
    def __init__(self) -> None:
        self.ids = (f"id-{n}" for n in itertools.count(1))
        self.token = "jwt-token"
        self.logins = 0
        self.groups: dict[str, dict[str, Any]] = {}
        self.contexts: dict[str, dict[str, Any]] = {}
        self.processors: dict[str, dict[str, Any]] = {}
        self.services: dict[str, dict[str, Any]] = {}
        self.funnels: dict[str, dict[str, Any]] = {}
        self.connections: dict[str, dict[str, Any]] = {}
        self.calls: list[str] = []
        self.missing_types: set[str] = set()

    # -------------------------------------------------------------- routing

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/nifi-api/")
        self.calls.append(f"{request.method} {path}")
        if path == "access/token":
            self.logins += 1
            form = dict(x.split("=") for x in request.content.decode().split("&"))
            if form != {"username": "admin", "password": "a-long-password"}:
                return httpx.Response(401, text="The supplied username and password are not valid.")
            return httpx.Response(201, text=self.token)
        if request.headers.get("Authorization") != f"Bearer {self.token}":
            return httpx.Response(401, text="Authentication required")
        body = json.loads(request.content) if request.content else {}
        result = self.route(request.method, path.split("/"), body, request.url.params)
        if isinstance(result, httpx.Response):
            return result
        return httpx.Response(200, json=result)

    def route(self, method: str, parts: list[str], body: Any, params: Any) -> Any:  # noqa: PLR0911, PLR0912
        match method, parts:
            case "GET", ["flow", "process-groups", "root"]:
                groups = [self._group_entity(g) for g in self.groups.values()]
                return {"processGroupFlow": {"id": "root-id", "flow": {"processGroups": groups}}}
            case "GET", ["flow", "processor-types"]:
                return {
                    "processorTypes": [
                        {"type": t, "bundle": BUNDLE} for t in TYPES if t not in self.missing_types
                    ]
                }
            case "GET", ["flow", "controller-service-types"]:
                return {"controllerServiceTypes": [{"type": SERVICE_TYPE, "bundle": BUNDLE}]}
            case "POST", ["parameter-contexts"]:
                return self._new(self.contexts, body["component"])
            case "GET", ["parameter-contexts", context_id]:
                return self._entity(self.contexts[context_id])
            case "DELETE", ["parameter-contexts", context_id]:
                del self.contexts[context_id]
                return {}
            case "POST", ["process-groups", parent, "process-groups"]:
                return self._new(self.groups, {**body["component"], "parent": parent})
            case "GET", ["process-groups", group_id]:
                return self._group_entity(self.groups[group_id])
            case "DELETE", ["process-groups", group_id]:
                if any(c["queued"] for c in self.connections.values()):
                    return httpx.Response(409, text="Cannot delete: connections have data queued")
                del self.groups[group_id]
                self.processors.clear()
                self.services.clear()
                self.connections.clear()
                self.funnels.clear()
                return {}
            case "POST", ["process-groups", _, "controller-services"]:
                return self._new(self.services, {**body["component"], "state": "DISABLED"})
            case "PUT", ["controller-services", service_id]:
                self.services[service_id].update(body["component"])
                return self._entity(self.services[service_id])
            case "GET", ["controller-services", service_id]:
                return self._entity(self.services[service_id])
            case "POST", ["process-groups", _, "processors"]:
                return self._new(self.processors, {**body["component"], "state": "STOPPED"})
            case "PUT", ["processors", processor_id]:
                self.processors[processor_id]["config"] = body["component"]["config"]
                return self._entity(self.processors[processor_id])
            case "GET", ["processors", processor_id]:
                return self._entity(self.processors[processor_id])
            case "PUT", ["processors", processor_id, "run-status"]:
                processor = self.processors[processor_id]
                processor["history"] = [*processor.get("history", []), body["state"]]
                # running once ends stopped
                processor["state"] = "STOPPED" if body["state"] == "RUN_ONCE" else body["state"]
                return self._entity(processor)
            case "POST", ["process-groups", _, "funnels"]:
                return self._new(self.funnels, body["component"])
            case "POST", ["process-groups", _, "connections"]:
                return self._new(self.connections, {**body["component"], "queued": 0})
            case "PUT", ["flow", "process-groups", group_id]:
                for processor in self.processors.values():
                    processor["state"] = body["state"]
                return body
            case "PUT", ["flow", "process-groups", group_id, "controller-services"]:
                for service in self.services.values():
                    service["state"] = body["state"]
                return body
            case "GET", ["flow", "process-groups", group_id, "controller-services"]:
                return {"controllerServices": [self._entity(s) for s in self.services.values()]}
            case "GET", ["flow", "process-groups", group_id]:
                return {
                    "processGroupFlow": {
                        "id": group_id,
                        "flow": {
                            "processors": [self._entity(p) for p in self.processors.values()],
                            "connections": [self._entity(c) for c in self.connections.values()],
                        },
                    }
                }
            case "POST", ["process-groups", _, "empty-all-connections-requests"]:
                for connection in self.connections.values():
                    connection["queued"] = 0
                return {"dropRequest": {"id": "drop-1", "finished": False}}
            case "GET", ["process-groups", _, "empty-all-connections-requests", _]:
                return {"dropRequest": {"id": "drop-1", "finished": True}}
            case "DELETE", ["process-groups", _, "empty-all-connections-requests", _]:
                return {}
        return httpx.Response(404, text=f"fake NiFi has no {method} {'/'.join(parts)}")

    # ------------------------------------------------------------- entities

    def _new(self, store: dict[str, dict[str, Any]], component: dict[str, Any]) -> dict[str, Any]:
        item_id = next(self.ids)
        store[item_id] = {**component, "id": item_id, "version": 1}
        return self._entity(store[item_id])

    def _group_entity(self, group: dict[str, Any]) -> dict[str, Any]:
        return {"id": group["id"], "revision": {"version": 1}, "component": group}

    def _entity(self, item: dict[str, Any]) -> dict[str, Any]:
        component = dict(item)
        if item["id"] in self.processors:
            kind = TYPES[item["type"]]
            config = item.get("config", {})
            component["config"] = {
                **config,
                "descriptors": {n: {"name": n, "displayName": d} for n, d in kind["properties"]},
            }
            component["validationErrors"] = self._processor_errors(item, kind)
            component["position"] = item["position"]
            return {
                "id": item["id"],
                "revision": {"version": item["version"]},
                "component": component,
                "status": {"aggregateSnapshot": {"flowFilesIn": 4, "flowFilesOut": 4}},
            }
        if item["id"] in self.services:
            component["descriptors"] = {
                n: {"name": n, "displayName": d} for n, d in SERVICE_PROPERTIES
            }
            known = {n for n, _ in SERVICE_PROPERTIES}
            component["validationErrors"] = [
                f"'{n}' is not a supported property"
                for n in item.get("properties", {})
                if n not in known
            ]
        if item["id"] in self.connections:
            return {
                "id": item["id"],
                "component": component,
                "status": {"aggregateSnapshot": {"flowFilesQueued": item["queued"]}},
            }
        return {"id": item["id"], "revision": {"version": item["version"]}, "component": component}

    def _processor_errors(self, item: dict[str, Any], kind: dict[str, Any]) -> list[str]:
        config = item.get("config", {})
        errors = []
        known = {name for name, _ in kind["properties"]}
        for name in config.get("properties", {}):
            if name not in known and not kind["dynamic"]:
                errors.append(f"'{name}' is not a supported property")
        handled = set(config.get("autoTerminatedRelationships", []))
        for connection in self.connections.values():
            if connection["source"]["id"] == item["id"]:
                handled |= set(connection["selectedRelationships"])
        for relationship in kind["relationships"]:
            if relationship not in handled:
                errors.append(f"Relationship '{relationship}' is not connected or auto-terminated")
        return errors


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        dhis2_base_url="https://dhis2.test/demo",
        dhis2_username="reader",
        dhis2_password="dhis2-secret",
        s3_access_key="hmis-local",
        s3_secret_key="lake-secret",
        nifi_username="admin",
        nifi_password="a-long-password",
    )


@pytest.fixture
def fake() -> FakeNiFi:
    return FakeNiFi()


@pytest.fixture
def client(fake: FakeNiFi, settings: Settings) -> Iterator[NiFiClient]:
    with NiFiClient.from_settings(settings, transport=httpx.MockTransport(fake)) as nifi:
        yield nifi


@pytest.fixture
def deployer(client: NiFiClient) -> Deployer:
    return Deployer(client, timeout=1, poll=0)


def by_name(fake: FakeNiFi, name: str) -> dict[str, Any]:
    return next(p for p in fake.processors.values() if p["name"] == name)


# ----------------------------------------------------------------- client


def test_client_logs_in_once_and_sends_the_token(client: NiFiClient, fake: FakeNiFi) -> None:
    client.get("flow/process-groups/root")
    client.get("flow/process-groups/root")

    assert fake.logins == 1


def test_wrong_password_is_reported_with_nifis_reason(fake: FakeNiFi) -> None:
    nifi = NiFiClient("https://nifi.test", "admin", "wrong", transport=httpx.MockTransport(fake))

    with pytest.raises(NiFiError, match="not valid"):
        nifi.login()


def test_missing_credentials_are_explained(settings: Settings) -> None:
    with pytest.raises(NiFiError, match="HMIS_NIFI_USERNAME"):
        NiFiClient.from_settings(settings.model_copy(update={"nifi_password": None}))


# ----------------------------------------------------------------- deploy


def test_the_flow_deploys_valid(deployer: Deployer, fake: FakeNiFi, settings: Settings) -> None:
    flow = dhis2_to_bronze(settings)

    deployment = deployer.deploy(flow)

    assert deployment.problems == {}
    assert len(fake.processors) == len(flow.processors) == 7
    assert len(fake.connections) == len(flow.connections)
    assert len(fake.funnels) == 1
    (group,) = fake.groups.values()
    assert group["name"] == FLOW_NAME
    assert group["parameterContext"]["id"] in fake.contexts


def test_secrets_go_to_nifi_as_sensitive_parameters(
    deployer: Deployer, fake: FakeNiFi, settings: Settings
) -> None:
    deployer.deploy(dhis2_to_bronze(settings))

    (context,) = fake.contexts.values()
    parameters = {p["parameter"]["name"]: p["parameter"] for p in context["parameters"]}
    assert parameters["dhis2.password"]["sensitive"] is True
    assert parameters["s3.secret.key"]["sensitive"] is True
    assert parameters["dhis2.username"]["sensitive"] is False
    assert parameters["dhis2.api"]["value"] == "https://dhis2.test/demo/api"
    # processors only ever see a reference, never the secret itself
    fetch = by_name(fake, "Fetch data values")["config"]["properties"]
    assert fetch["Basic Authentication Password"] == "#{dhis2.password}"
    assert "dhis2-secret" not in json.dumps(list(fake.processors.values()))


def test_screen_names_are_translated_to_api_names(
    deployer: Deployer, fake: FakeNiFi, settings: Settings
) -> None:
    deployer.deploy(dhis2_to_bronze(settings))

    properties = by_name(fake, "List districts")["config"]["properties"]
    assert "Request Password" not in properties
    assert properties["Basic Authentication Password"] == "#{dhis2.password}"
    (service,) = fake.services.values()
    assert service["properties"] == {
        "access-key": "#{s3.access.key}",
        "secret-key": "#{s3.secret.key}",
    }


def test_the_bronze_writer_uses_the_flows_credentials_service(
    deployer: Deployer, fake: FakeNiFi, settings: Settings
) -> None:
    deployer.deploy(dhis2_to_bronze(settings))

    (service_id,) = fake.services
    save = by_name(fake, "Save to bronze")["config"]["properties"]
    assert save["AWS Credentials Provider Service"] == service_id
    assert save["Object Key"] == "${s3.key}"


def test_schedule_retries_and_parallelism(
    deployer: Deployer, fake: FakeNiFi, settings: Settings
) -> None:
    deployer.deploy(dhis2_to_bronze(settings))

    districts = by_name(fake, "List districts")["config"]
    assert districts["schedulingStrategy"] == "CRON_DRIVEN"
    assert districts["schedulingPeriod"] == "0 0 2 * * ?"
    fetch = by_name(fake, "Fetch data values")["config"]
    assert fetch["concurrentlySchedulableTaskCount"] == 3
    assert fetch["retriedRelationships"] == ["Retry"]
    assert fetch["backoffMechanism"] == "PENALIZE_FLOWFILE"


def test_everything_that_can_fail_ends_in_the_failed_funnel(
    deployer: Deployer, fake: FakeNiFi, settings: Settings
) -> None:
    deployer.deploy(dhis2_to_bronze(settings))

    (funnel_id,) = fake.funnels
    parked = {
        (fake.processors[c["source"]["id"]]["name"], r)
        for c in fake.connections.values()
        if c["destination"]["id"] == funnel_id
        for r in c["selectedRelationships"]
    }
    assert ("Fetch data values", "Retry") in parked  # after the last retry
    assert ("Save to bronze", "failure") in parked
    assert all(
        c["name"] == FAILED
        for c in fake.connections.values()
        if c["destination"]["id"] == funnel_id
    )


def test_problems_nifi_reports_are_returned(
    deployer: Deployer, fake: FakeNiFi, settings: Settings
) -> None:
    del TYPES["org.apache.nifi.processors.aws.s3.PutS3Object"]["properties"][
        5
    ]  # no path-style option
    try:
        deployment = deployer.deploy(dhis2_to_bronze(settings))
    finally:
        TYPES["org.apache.nifi.processors.aws.s3.PutS3Object"]["properties"].insert(
            5, ("Use Path Style Access", "Use Path Style Access")
        )

    assert deployment.problems == {
        "Save to bronze": ["'Use Path Style Access' is not a supported property"]
    }


def test_a_missing_python_processor_is_named(
    deployer: Deployer, fake: FakeNiFi, settings: Settings
) -> None:
    fake.missing_types.add("WrapDhis2Envelope")

    with pytest.raises(NiFiError, match=r"'WrapDhis2Envelope'.*python_extensions"):
        deployer.deploy(dhis2_to_bronze(settings))


def test_redeploy_needs_replace_and_then_rebuilds_cleanly(
    deployer: Deployer, fake: FakeNiFi, settings: Settings
) -> None:
    flow = dhis2_to_bronze(settings)
    deployer.deploy(flow)
    next(iter(fake.connections.values()))["queued"] = 5  # data waiting in the old flow

    with pytest.raises(NiFiError, match="--replace"):
        deployer.deploy(flow)
    deployer.deploy(flow, replace=True)

    assert len(fake.groups) == 1
    assert len(fake.contexts) == 1  # the old parameter context went with the old group
    assert len(fake.processors) == 7


# ---------------------------------------------------------------- operate


def test_start_enables_services_then_runs_processors(
    deployer: Deployer, fake: FakeNiFi, settings: Settings
) -> None:
    deployment = deployer.deploy(dhis2_to_bronze(settings))

    deployer.start(deployment)

    assert {s["state"] for s in fake.services.values()} == {"ENABLED"}
    assert {p["state"] for p in fake.processors.values()} == {"RUNNING"}
    enable = fake.calls.index(f"PUT flow/process-groups/{deployment.group_id}/controller-services")
    assert enable < fake.calls.index(f"PUT flow/process-groups/{deployment.group_id}")


def test_run_once_returns_the_processor_to_its_schedule(
    deployer: Deployer, fake: FakeNiFi, settings: Settings
) -> None:
    deployment = deployer.deploy(dhis2_to_bronze(settings))
    deployer.start(deployment)

    deployer.run_once(deployment.group_id, "List districts")

    districts = by_name(fake, "List districts")
    assert districts["history"] == ["STOPPED", "RUN_ONCE", "RUNNING"]
    assert districts["state"] == "RUNNING"


def test_status_lists_steps_in_flow_order_and_counts_failures(
    deployer: Deployer, fake: FakeNiFi, settings: Settings
) -> None:
    deployment = deployer.deploy(dhis2_to_bronze(settings))
    failed_queue = next(c for c in fake.connections.values() if c["name"] == FAILED)
    failed_queue["queued"] = 2

    rows, failed = deployer.status(deployment.group_id)

    assert [r["name"] for r in rows] == [p.name for p in dhis2_to_bronze(settings).processors]
    assert failed == 2


# --------------------------------------------------------------------- CLI


def test_cli_deploy_reports_problems_and_fails(
    monkeypatch: pytest.MonkeyPatch, deployer: Deployer, settings: Settings
) -> None:
    del TYPES["WrapDhis2Envelope"]["properties"][0]  # NiFi wouldn't know "Source URL"
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "_nifi", lambda: deployer)
    try:
        result = CliRunner().invoke(cli.app, ["nifi", "deploy"])
    finally:
        TYPES["WrapDhis2Envelope"]["properties"].insert(0, ("Source URL", "Source URL"))

    assert result.exit_code == 1
    assert "Wrap as bronze file" in result.output
    assert "Source URL" in result.output


def test_cli_deploy_and_start(
    monkeypatch: pytest.MonkeyPatch, deployer: Deployer, fake: FakeNiFi, settings: Settings
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "_nifi", lambda: deployer)

    result = CliRunner().invoke(cli.app, ["nifi", "deploy", "--start"])

    assert result.exit_code == 0, result.output
    assert "Every processor and service is valid" in result.output
    assert {p["state"] for p in fake.processors.values()} == {"RUNNING"}
