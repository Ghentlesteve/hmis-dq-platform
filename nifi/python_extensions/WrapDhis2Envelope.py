"""NiFi processor: turn a DHIS2 dataValueSets response into a bronze file.

In:  the response body exactly as DHIS2 sent it, with the chunk's attributes
     (data_set, org_unit, period) set by the flow.
Out: gzipped JSON {"meta": {...lineage...}, "payload": <the untouched response>},
     the same envelope and encoding the Python extractor writes, so the Spark
     silver job reads NiFi's files and the extractor's alike.

Self-contained (standard library only): it runs inside NiFi's Python.
tests/test_nifi_processors.py checks it matches hmis_dq.extract.
"""

import gzip
import json
from datetime import UTC, datetime
from typing import Any

try:
    from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult
    from nifiapi.properties import PropertyDescriptor, StandardValidators
except ImportError:  # outside NiFi (unit tests): only envelope()/encode() are used there
    FlowFileTransform = object

SOURCE = "dhis2"
CHUNK_ATTRIBUTES = ("data_set", "org_unit", "period")


def envelope(
    payload: Any, chunk: dict[str, str], source_url: str, extracted_at: datetime
) -> dict[str, Any]:
    """Wrap the untouched response with lineage metadata (as Extractor._envelope)."""
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object from dataValueSets")
    values = payload.get("dataValues", [])
    if not isinstance(values, list):
        raise ValueError("'dataValues' is not a list")
    return {
        "meta": {
            "source": SOURCE,
            "source_url": source_url,
            "extracted_at": extracted_at.isoformat(),
            **{name: chunk[name] for name in CHUNK_ATTRIBUTES},
            "record_count": len(values),
        },
        "payload": payload,
    }


def encode(document: dict[str, Any]) -> bytes:
    """Compact UTF-8 JSON, gzipped with a fixed mtime (as store.encode_json_gz)."""
    raw = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return gzip.compress(raw, mtime=0)


class WrapDhis2Envelope(FlowFileTransform):
    class Java:
        implements = ["org.apache.nifi.python.processor.FlowFileTransform"]

    class ProcessorDetails:
        version = "0.1.0"
        description = (
            "Wraps a DHIS2 dataValueSets response in the bronze envelope "
            "(lineage metadata + untouched payload) and gzips it."
        )
        tags = ["dhis2", "hmis", "bronze"]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.source_url = PropertyDescriptor(
            name="Source URL",
            description="The DHIS2 server the data came from, recorded as lineage.",
            required=True,
            validators=[StandardValidators.NON_EMPTY_VALIDATOR],
        )

    def getPropertyDescriptors(self) -> list[Any]:
        return [self.source_url]

    def transform(self, context: Any, flowfile: Any) -> Any:
        try:
            chunk = {name: flowfile.getAttribute(name) for name in CHUNK_ATTRIBUTES}
            if not all(chunk.values()):
                raise ValueError(f"missing chunk attributes: {chunk}")
            document = envelope(
                json.loads(flowfile.getContentsAsBytes()),
                chunk,
                context.getProperty(self.source_url).getValue(),
                datetime.now(UTC),
            )
        except ValueError as exc:  # json.JSONDecodeError is a ValueError
            self.logger.error(f"Not a valid dataValueSets response: {exc}")
            return FlowFileTransformResult(relationship="failure")
        return FlowFileTransformResult(
            relationship="success",
            contents=encode(document),
            attributes={
                "mime.type": "application/gzip",
                "dhis2.records": str(document["meta"]["record_count"]),
            },
        )
