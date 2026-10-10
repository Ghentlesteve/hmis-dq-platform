"""Build a Flow (flow.py) in a running NiFi through its REST API.

The whole flow lives in one process group with its own parameter context, so
redeploying means: stop it, empty it, delete it, build it again. Nothing is
clicked by hand, so the flow in NiFi is always the one in this repository.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hmis_dq.nifi.client import JSON, NiFiClient, NiFiError
from hmis_dq.nifi.flow import FAILED, SERVICE, Flow, Processor

NEW = {"version": 0}  # the revision of something that doesn't exist yet
ROOT = "root"


@dataclass
class Deployment:
    group_id: str
    processors: dict[str, str] = field(default_factory=dict)  # flow key -> NiFi id
    services: dict[str, str] = field(default_factory=dict)
    problems: dict[str, list[str]] = field(default_factory=dict)  # name -> NiFi's complaints


def wait_until(check: Callable[[], bool], what: str, timeout: float, poll: float) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        if time.monotonic() > deadline:
            raise NiFiError(f"timed out waiting for {what}")
        time.sleep(poll)


class Deployer:
    def __init__(self, nifi: NiFiClient, *, timeout: float = 60.0, poll: float = 1.0) -> None:
        self.nifi = nifi
        self.timeout = timeout
        self.poll = poll
        self._types: dict[str, dict[str, JSON]] = {}

    # ---------------------------------------------------------------- lookups

    def root_id(self) -> str:
        return str(self.nifi.get(f"flow/process-groups/{ROOT}")["processGroupFlow"]["id"])

    def find_group(self, name: str) -> JSON | None:
        flow = self.nifi.get(f"flow/process-groups/{ROOT}")["processGroupFlow"]["flow"]
        return next((g for g in flow["processGroups"] if g["component"]["name"] == name), None)

    def _type(self, kind: str, name: str) -> JSON:
        """{"type": full class name, "bundle": {...}} for a class name like "InvokeHTTP"."""
        if kind not in self._types:
            listing = self.nifi.get(f"flow/{kind}-types")
            key = "processorTypes" if kind == "processor" else "controllerServiceTypes"
            self._types[kind] = {}
            for item in listing[key]:  # later versions overwrite earlier ones
                short = item["type"].rsplit(".", 1)[-1]
                self._types[kind][short] = {"type": item["type"], "bundle": item["bundle"]}
        try:
            return self._types[kind][name]
        except KeyError:
            hint = " (is it in nifi/python_extensions?)" if kind == "processor" else ""
            raise NiFiError(f"NiFi has no {kind} type {name!r}{hint}") from None

    # ----------------------------------------------------------------- remove

    def remove(self, group: JSON) -> None:
        """Stop, empty and delete a deployed flow, and its parameter context."""
        group_id = group["id"]
        self.nifi.put(f"flow/process-groups/{group_id}", {"id": group_id, "state": "STOPPED"})
        self.nifi.put(
            f"flow/process-groups/{group_id}/controller-services",
            {"id": group_id, "state": "DISABLED"},
        )
        wait_until(
            lambda: self._services_disabled(group_id), "services to stop", self.timeout, self.poll
        )

        drop = self.nifi.post(f"process-groups/{group_id}/empty-all-connections-requests", {})
        drop_url = (
            f"process-groups/{group_id}/empty-all-connections-requests/{drop['dropRequest']['id']}"
        )
        wait_until(
            lambda: bool(self.nifi.get(drop_url)["dropRequest"]["finished"]),
            "queues to empty",
            self.timeout,
            self.poll,
        )
        self.nifi.delete(drop_url)

        current = self.nifi.get(f"process-groups/{group_id}")
        context = (current["component"].get("parameterContext") or {}).get("id")
        self.nifi.delete(f"process-groups/{group_id}", version=current["revision"]["version"])
        if context:
            entity = self.nifi.get(f"parameter-contexts/{context}")
            self.nifi.delete(f"parameter-contexts/{context}", version=entity["revision"]["version"])

    def _services_disabled(self, group_id: str) -> bool:
        services = self.nifi.get(f"flow/process-groups/{group_id}/controller-services")
        return all(s["component"]["state"] == "DISABLED" for s in services["controllerServices"])

    # ----------------------------------------------------------------- deploy

    def deploy(self, flow: Flow, *, replace: bool = False) -> Deployment:
        existing = self.find_group(flow.name)
        if existing is not None:
            if not replace:
                raise NiFiError(f"{flow.name!r} is already deployed; use --replace to rebuild it")
            self.remove(existing)

        context = self.nifi.post(
            "parameter-contexts",
            {
                "revision": NEW,
                "component": {
                    "name": flow.name,
                    "parameters": [
                        {
                            "parameter": {
                                "name": p.name,
                                "value": p.value,
                                "sensitive": p.sensitive,
                                "description": p.description,
                            }
                        }
                        for p in flow.parameters
                    ],
                },
            },
        )
        group = self.nifi.post(
            f"process-groups/{self.root_id()}/process-groups",
            {
                "revision": NEW,
                "component": {
                    "name": flow.name,
                    "position": {"x": 0, "y": 0},
                    "parameterContext": {"id": context["id"]},
                },
            },
        )
        deployment = Deployment(group["id"])

        for service in flow.services:
            created = self.nifi.post(
                f"process-groups/{group['id']}/controller-services",
                {
                    "revision": NEW,
                    "component": {
                        **self._type("controller-service", service.type),
                        "name": service.key,
                    },
                },
            )
            properties = _by_internal_name(service.properties, created["component"]["descriptors"])
            self.nifi.put(
                f"controller-services/{created['id']}",
                {
                    "revision": created["revision"],
                    "component": {"id": created["id"], "properties": properties},
                },
            )
            deployment.services[service.key] = created["id"]

        for processor in flow.processors:
            deployment.processors[processor.key] = self._add_processor(
                group["id"], processor, deployment.services
            )

        failed = self.nifi.post(
            f"process-groups/{group['id']}/funnels",
            {
                "revision": NEW,
                "component": {
                    "position": {"x": flow.failed_position[0], "y": flow.failed_position[1]}
                },
            },
        )
        for connection in flow.connections:
            source = {"id": deployment.processors[connection.source], "type": "PROCESSOR"}
            target = (
                {"id": failed["id"], "type": "FUNNEL"}
                if connection.destination == FAILED
                else {"id": deployment.processors[connection.destination], "type": "PROCESSOR"}
            )
            self.nifi.post(
                f"process-groups/{group['id']}/connections",
                {
                    "revision": NEW,
                    "component": {
                        "name": FAILED if connection.destination == FAILED else "",
                        "source": {**source, "groupId": group["id"]},
                        "destination": {**target, "groupId": group["id"]},
                        "selectedRelationships": list(connection.relationships),
                    },
                },
            )

        deployment.problems = self.validation_problems(deployment)
        return deployment

    def _add_processor(self, group_id: str, spec: Processor, services: dict[str, str]) -> str:
        created = self.nifi.post(
            f"process-groups/{group_id}/processors",
            {
                "revision": NEW,
                "component": {
                    **self._type("processor", spec.type),
                    "name": spec.name,
                    "position": {"x": spec.position[0], "y": spec.position[1]},
                },
            },
        )
        values = {
            name: services[value.removeprefix(SERVICE)] if value.startswith(SERVICE) else value
            for name, value in spec.properties.items()
        }
        config: JSON = {
            "properties": _by_internal_name(values, created["component"]["config"]["descriptors"]),
            "schedulingStrategy": "CRON_DRIVEN" if spec.cron else "TIMER_DRIVEN",
            "schedulingPeriod": spec.schedule,
            "concurrentlySchedulableTaskCount": spec.concurrent_tasks,
            "autoTerminatedRelationships": list(spec.auto_terminate),
        }
        if spec.retry:
            config |= {
                "retriedRelationships": list(spec.retry.relationships),
                "retryCount": spec.retry.attempts,
                "backoffMechanism": "PENALIZE_FLOWFILE",
                "maxBackoffPeriod": spec.retry.max_backoff,
            }
        self.nifi.put(
            f"processors/{created['id']}",
            {"revision": created["revision"], "component": {"id": created["id"], "config": config}},
        )
        return str(created["id"])

    def validation_problems(self, deployment: Deployment) -> dict[str, list[str]]:
        """What NiFi says is wrong with each part, once it has finished checking."""
        paths = [f"processors/{i}" for i in deployment.processors.values()]
        paths += [f"controller-services/{i}" for i in deployment.services.values()]
        problems: dict[str, list[str]] = {}
        for path in paths:
            entity: JSON = {}

            def checked(path: str = path) -> bool:
                nonlocal entity
                entity = self.nifi.get(path)
                return bool(entity["component"].get("validationStatus") != "VALIDATING")

            wait_until(checked, f"NiFi to validate {path}", self.timeout, self.poll)
            errors = entity["component"].get("validationErrors") or []
            if errors:
                problems[entity["component"]["name"]] = list(errors)
        return problems

    # ---------------------------------------------------------------- operate

    def start(self, deployment: Deployment) -> None:
        group_id = deployment.group_id
        self.nifi.put(
            f"flow/process-groups/{group_id}/controller-services",
            {"id": group_id, "state": "ENABLED"},
        )
        self.nifi.put(f"flow/process-groups/{group_id}", {"id": group_id, "state": "RUNNING"})

    def processors(self, group_id: str) -> list[JSON]:
        flow = self.nifi.get(f"flow/process-groups/{group_id}")["processGroupFlow"]["flow"]
        return list(flow["processors"])

    def run_once(self, group_id: str, name: str) -> None:
        """Run one processor now (e.g. the daily one), then return it to its schedule."""
        entity = next(
            (p for p in self.processors(group_id) if p["component"]["name"] == name), None
        )
        if entity is None:
            raise NiFiError(f"no processor named {name!r} in the flow")
        path = f"processors/{entity['id']}"
        was_running = entity["component"]["state"] == "RUNNING"
        if was_running:  # NiFi only runs a stopped processor once
            entity = self.nifi.put(
                f"{path}/run-status", {"revision": entity["revision"], "state": "STOPPED"}
            )
        entity = self.nifi.put(
            f"{path}/run-status", {"revision": entity["revision"], "state": "RUN_ONCE"}
        )

        def finished() -> bool:
            nonlocal entity
            entity = self.nifi.get(path)
            idle = entity["status"]["aggregateSnapshot"].get("activeThreadCount", 0) == 0
            return bool(entity["component"]["state"] == "STOPPED" and idle)

        wait_until(finished, f"{name} to run", self.timeout, self.poll)
        if was_running:
            self.nifi.put(
                f"{path}/run-status", {"revision": entity["revision"], "state": "RUNNING"}
            )

    def status(self, group_id: str) -> tuple[list[JSON], int]:
        """Per processor: name, state and 5-minute counts; plus FlowFiles parked as failed."""
        flow = self.nifi.get(f"flow/process-groups/{group_id}")["processGroupFlow"]["flow"]
        rows = [
            {
                "name": p["component"]["name"],
                "state": p["component"]["state"],
                "in": p["status"]["aggregateSnapshot"].get("flowFilesIn", 0),
                "out": p["status"]["aggregateSnapshot"].get("flowFilesOut", 0),
            }
            for p in sorted(flow["processors"], key=_position)
        ]
        failed = sum(
            c["status"]["aggregateSnapshot"].get("flowFilesQueued", 0)
            for c in flow["connections"]
            if c["component"]["destination"]["type"] == "FUNNEL"
        )
        return rows, failed


def _position(entity: JSON) -> tuple[float, float]:
    """Flow order: down the first column, then up the second (as flow.py lays it out)."""
    x = entity["component"]["position"]["x"]
    y = entity["component"]["position"]["y"]
    return (x, y if x == 0 else -y)


def _by_internal_name(values: dict[str, str], descriptors: dict[str, Any]) -> dict[str, str]:
    """Translate property names as shown in NiFi's UI to the names its API uses.

    A name that isn't one of the component's own properties is passed on as is:
    that's a dynamic property (e.g. EvaluateJsonPath's attribute -> JSON path),
    and NiFi reports it as invalid if the component doesn't accept those.
    """
    by_display = {d.get("displayName", key): key for key, d in descriptors.items()}
    return {by_display.get(name, name): value for name, value in values.items()}
