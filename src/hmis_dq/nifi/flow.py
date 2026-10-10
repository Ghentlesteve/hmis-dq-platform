"""The NiFi flow that refreshes the bronze layer, described as data.

    every day at 02:00
    List districts ──> Plan chunks ──> One chunk per FlowFile ──> Chunk attributes
    (InvokeHTTP)       (Python)        (SplitJson)                (EvaluateJsonPath)
                                                                         │
    Save to bronze <── Wrap as bronze file <── Fetch data values <───────┘
    (PutS3Object)      (Python)                (InvokeHTTP, 3 at a time)

Anything that still fails after NiFi's retries is parked in the "Failed" funnel,
where it waits, visible on the canvas, instead of disappearing.

Properties use the names shown in NiFi's UI; `#{...}` are parameters (set from
.env at deploy time, secrets marked sensitive) and `${...}` are FlowFile
attributes. deploy.py turns this description into NiFi API calls.
"""

from dataclasses import dataclass, field

from hmis_dq.config import Settings
from hmis_dq.extract.catalog import DATASETS, DEFAULT_DATASETS

FLOW_NAME = "DHIS2 to bronze"
FAILED = "Failed"  # the funnel failures are parked in
SERVICE = "@service:"  # a property value naming a controller service of this flow


@dataclass(frozen=True)
class Parameter:
    name: str
    value: str
    sensitive: bool = False
    description: str = ""


@dataclass(frozen=True)
class Service:
    key: str
    type: str  # class name, e.g. "AWSCredentialsProviderControllerService"
    properties: dict[str, str]


@dataclass(frozen=True)
class Retry:
    """NiFi's built-in retry: send FlowFiles back to the same processor, waiting
    longer each time, before they finally take the relationship."""

    relationships: tuple[str, ...]
    attempts: int = 3
    max_backoff: str = "10 mins"


@dataclass(frozen=True)
class Processor:
    key: str
    name: str
    type: str  # class name, e.g. "InvokeHTTP", or a Python processor's name
    position: tuple[int, int]
    properties: dict[str, str] = field(default_factory=dict)
    schedule: str = "0 sec"
    cron: bool = False
    concurrent_tasks: int = 1
    auto_terminate: tuple[str, ...] = ()
    retry: Retry | None = None


@dataclass(frozen=True)
class Connection:
    source: str
    destination: str  # a processor key, or FAILED
    relationships: tuple[str, ...]


@dataclass(frozen=True)
class Flow:
    name: str
    parameters: tuple[Parameter, ...]
    services: tuple[Service, ...]
    processors: tuple[Processor, ...]
    connections: tuple[Connection, ...]
    failed_position: tuple[int, int] = (1000, 600)


def dhis2_to_bronze(settings: Settings) -> Flow:
    api = str(settings.dhis2_base_url).rstrip("/") + "/api"
    secret = settings.s3_secret_key.get_secret_value() if settings.s3_secret_key else ""
    parameters = (
        Parameter("dhis2.api", api, description="DHIS2 Web API base URL"),
        Parameter("dhis2.source.url", str(settings.dhis2_base_url), description="lineage"),
        Parameter("dhis2.username", settings.dhis2_username),
        Parameter("dhis2.password", settings.dhis2_password.get_secret_value(), sensitive=True),
        Parameter("dhis2.data.sets", ",".join(DATASETS[name] for name in DEFAULT_DATASETS)),
        Parameter("refresh.months", str(settings.nifi_refresh_months)),
        Parameter("s3.endpoint", str(settings.nifi_s3_endpoint_url).rstrip("/")),
        Parameter("s3.region", settings.s3_region),
        Parameter("s3.bucket", settings.bronze_bucket),
        Parameter("s3.access.key", settings.s3_access_key or ""),
        Parameter("s3.secret.key", secret, sensitive=True),
    )
    dhis2_login = {"Request Username": "#{dhis2.username}", "Request Password": "#{dhis2.password}"}
    # retried first; what still fails after the last attempt is parked
    invoke_http_ends = ("Retry", "No Retry", "Failure")

    processors = (
        Processor(
            "districts",
            "List districts",
            "InvokeHTTP",
            (0, 0),
            {
                "HTTP Method": "GET",
                "HTTP URL": (
                    "#{dhis2.api}/organisationUnits.json?level=2&fields=id,name&paging=false"
                ),
                **dhis2_login,
            },
            schedule=settings.nifi_schedule,
            cron=True,
            auto_terminate=("Original",),
            retry=Retry(("Retry",)),
        ),
        Processor(
            "plan",
            "Plan chunks",
            "ListDhis2Chunks",
            (0, 200),
            {"Data Sets": "#{dhis2.data.sets}", "Months Back": "#{refresh.months}"},
            auto_terminate=("original",),
        ),
        Processor(
            "split",
            "One chunk per FlowFile",
            "SplitJson",
            (0, 400),
            {"JsonPath Expression": "$.*"},
            auto_terminate=("original",),
        ),
        Processor(
            "attributes",
            "Chunk attributes",
            "EvaluateJsonPath",
            (0, 600),
            {
                "Destination": "flowfile-attribute",
                # dynamic properties: attribute name -> JSON path in the chunk
                **{
                    name: f"$.{name}"
                    for name in ("data_set", "org_unit", "period", "start_date", "end_date")
                },
                "s3.key": "$.key",
            },
        ),
        Processor(
            "fetch",
            "Fetch data values",
            "InvokeHTTP",
            (500, 600),
            {
                "HTTP Method": "GET",
                "HTTP URL": (
                    "#{dhis2.api}/dataValueSets.json?dataSet=${data_set}&orgUnit=${org_unit}"
                    "&startDate=${start_date}&endDate=${end_date}&children=true"
                ),
                **dhis2_login,
                "Socket Read Timeout": "120 secs",
            },
            concurrent_tasks=3,  # the demo server copes with a few at a time
            auto_terminate=("Original",),
            retry=Retry(("Retry",)),
        ),
        Processor(
            "wrap",
            "Wrap as bronze file",
            "WrapDhis2Envelope",
            (500, 400),
            {"Source URL": "#{dhis2.source.url}"},
            auto_terminate=("original",),
        ),
        Processor(
            "save",
            "Save to bronze",
            "PutS3Object",
            (500, 200),
            {
                "Bucket": "#{s3.bucket}",
                "Object Key": "${s3.key}",
                "Region": "#{s3.region}",
                "AWS Credentials Provider Service": f"{SERVICE}lake",
                "Endpoint Override URL": "#{s3.endpoint}",
                "Use Path Style Access": "true",  # the lake has no per-bucket hostnames
                "Content Type": "application/gzip",
            },
            auto_terminate=("success",),
            retry=Retry(("failure",)),
        ),
    )
    connections = (
        Connection("districts", "plan", ("Response",)),
        Connection("plan", "split", ("success",)),
        Connection("split", "attributes", ("split",)),
        Connection("attributes", "fetch", ("matched",)),
        Connection("fetch", "wrap", ("Response",)),
        Connection("wrap", "save", ("success",)),
        Connection("districts", FAILED, invoke_http_ends),
        Connection("plan", FAILED, ("failure",)),
        Connection("split", FAILED, ("failure",)),
        Connection("attributes", FAILED, ("unmatched", "failure")),
        Connection("fetch", FAILED, invoke_http_ends),
        Connection("wrap", FAILED, ("failure",)),
        Connection("save", FAILED, ("failure",)),
    )
    services = (
        Service(
            "lake",
            "AWSCredentialsProviderControllerService",
            {"Access Key ID": "#{s3.access.key}", "Secret Access Key": "#{s3.secret.key}"},
        ),
    )
    return Flow(FLOW_NAME, parameters, services, processors, connections)
