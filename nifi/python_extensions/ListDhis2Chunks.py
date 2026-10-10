"""NiFi processor: plan the DHIS2 extraction chunks for the latest months.

In:  the DHIS2 response listing districts ({"organisationUnits": [{"id", "name"}, ...]})
Out: a JSON array, one object per dataset x district x month, ready for SplitJson.

Chunks match the Python extractor (hmis_dq.extract.job.Chunk) exactly, so NiFi
and `hmis-dq extract` write the same bronze keys: whichever runs last refreshes
the file. The months are the last N complete ones, the ones DHIS2 users still
correct, which is what a scheduled refresh needs to re-fetch.

Self-contained (standard library only): it runs inside NiFi's Python, where the
hmis_dq package isn't installed. tests/test_nifi_processors.py checks it agrees
with hmis_dq.
"""

import calendar
import json
from datetime import UTC, date, datetime
from typing import Any

try:
    from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult
    from nifiapi.properties import PropertyDescriptor, StandardValidators
except ImportError:  # outside NiFi (unit tests): only plan_chunks() is used there
    FlowFileTransform = object

SOURCE = "dhis2"


def _month(year: int, month: int, shift: int) -> tuple[int, int]:
    index = year * 12 + (month - 1) + shift
    return index // 12, index % 12 + 1


def recent_months(today: date, months_back: int) -> list[tuple[int, int]]:
    """The last ``months_back`` complete months before ``today``, oldest first."""
    if months_back < 1:
        raise ValueError(f"months_back must be at least 1, got {months_back}")
    return [_month(today.year, today.month, -n) for n in range(months_back, 0, -1)]


def plan_chunks(
    org_units: dict[str, Any], data_sets: list[str], months_back: int, today: date
) -> list[dict[str, str]]:
    """One chunk per dataset x district x month, with everything the flow needs."""
    districts = org_units.get("organisationUnits")
    if not isinstance(districts, list) or not districts:
        raise ValueError("expected a non-empty 'organisationUnits' list from DHIS2")
    chunks = []
    for data_set in data_sets:
        for unit in districts:
            for year, month in recent_months(today, months_back):
                period = f"{year}{month:02d}"
                last_day = calendar.monthrange(year, month)[1]
                chunks.append(
                    {
                        "data_set": data_set,
                        "org_unit": unit["id"],
                        "org_unit_name": unit.get("name", ""),
                        "period": period,
                        "start_date": date(year, month, 1).isoformat(),
                        "end_date": date(year, month, last_day).isoformat(),
                        "key": (
                            f"{SOURCE}/data_value_sets/dataset={data_set}"
                            f"/period={period}/org_unit={unit['id']}.json.gz"
                        ),
                    }
                )
    return chunks


def parse_data_sets(text: str) -> list[str]:
    return [uid.strip() for uid in text.split(",") if uid.strip()]


class ListDhis2Chunks(FlowFileTransform):
    class Java:
        implements = ["org.apache.nifi.python.processor.FlowFileTransform"]

    class ProcessorDetails:
        version = "0.1.0"
        description = (
            "Plans DHIS2 dataValueSets requests: one per dataset x district x month "
            "for the last N complete months, from a DHIS2 list of districts."
        )
        tags = ["dhis2", "hmis", "extract"]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.data_sets = PropertyDescriptor(
            name="Data Sets",
            description="Comma-separated DHIS2 dataset UIDs.",
            required=True,
            validators=[StandardValidators.NON_EMPTY_VALIDATOR],
        )
        self.months_back = PropertyDescriptor(
            name="Months Back",
            description="How many complete months to (re-)fetch, ending last month.",
            required=True,
            default_value="3",
            validators=[StandardValidators.POSITIVE_INTEGER_VALIDATOR],
        )

    def getPropertyDescriptors(self) -> list[Any]:
        return [self.data_sets, self.months_back]

    def transform(self, context: Any, flowfile: Any) -> Any:
        try:
            chunks = plan_chunks(
                json.loads(flowfile.getContentsAsBytes()),
                parse_data_sets(context.getProperty(self.data_sets).getValue()),
                int(context.getProperty(self.months_back).getValue()),
                datetime.now(UTC).date(),
            )
        except (ValueError, KeyError, TypeError) as exc:
            self.logger.error(f"Could not plan DHIS2 chunks: {exc}")
            return FlowFileTransformResult(relationship="failure")
        return FlowFileTransformResult(
            relationship="success",
            contents=json.dumps(chunks),
            attributes={"mime.type": "application/json", "dhis2.chunks": str(len(chunks))},
        )
