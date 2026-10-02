"""Extraction job: DHIS2 -> bronze layer.

Data values are pulled in chunks of one dataset x one district x one month.
Small chunks keep each request fast, let several run in parallel, and mean a
failure only costs one chunk, not the whole run.

The job is resumable and safe to re-run: chunks already in the store are skipped,
except the most recent months, which are always re-fetched because facilities
keep submitting and correcting reports for weeks after a period closes.
"""

import logging
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from itertools import product
from typing import Any, Literal

from hmis_dq.dhis2 import DHIS2Client
from hmis_dq.dhis2.models import DataValueSet
from hmis_dq.extract.periods import Month
from hmis_dq.extract.store import RawStore

logger = logging.getLogger(__name__)

SOURCE = "dhis2"

# Metadata needed to interpret the data values (names, hierarchy, disaggregations,
# reporting deadlines for timeliness checks, and geometry for the map).
METADATA_RESOURCES: dict[str, str] = {
    "organisationUnits": (
        "id,name,shortName,level,path,parent[id],openingDate,closedDate,geometry"
    ),
    "organisationUnitLevels": "id,level,name",
    "dataSets": (
        "id,name,periodType,openFuturePeriods,expiryDays,timelyDays,"
        "dataSetElements[dataElement[id]],organisationUnits[id]"
    ),
    "dataElements": "id,name,shortName,valueType,aggregationType,domainType,categoryCombo[id]",
    "categoryCombos": "id,name",
    "categoryOptionCombos": "id,name,categoryCombo[id]",
}

ChunkStatus = Literal["fetched", "skipped", "failed"]


@dataclass(frozen=True)
class Chunk:
    data_set: str
    org_unit: str
    month: Month

    @property
    def key(self) -> str:
        return (
            f"{SOURCE}/data_value_sets/dataset={self.data_set}"
            f"/period={self.month.dhis2_period}/org_unit={self.org_unit}.json.gz"
        )


@dataclass(frozen=True)
class ChunkResult:
    chunk: Chunk
    status: ChunkStatus
    records: int = 0
    bytes_written: int = 0
    error: str | None = None


@dataclass
class RunSummary:
    run_id: str
    source_url: str
    started_at: str
    finished_at: str | None = None
    server_version: str | None = None
    planned: int = 0
    fetched: int = 0
    skipped: int = 0
    failed: int = 0
    records: int = 0
    bytes_written: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed == 0

    def add(self, result: ChunkResult) -> None:
        if result.status == "fetched":
            self.fetched += 1
        elif result.status == "skipped":
            self.skipped += 1
        else:
            self.failed += 1
            self.failures.append({"key": result.chunk.key, "error": result.error or ""})
        self.records += result.records
        self.bytes_written += result.bytes_written

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_chunks(
    data_sets: Iterable[str], org_units: Iterable[str], months: Iterable[Month]
) -> list[Chunk]:
    return [Chunk(ds, ou, m) for ds, ou, m in product(data_sets, org_units, months)]


def _now() -> datetime:
    return datetime.now(UTC)


class Extractor:
    def __init__(
        self,
        client: DHIS2Client,
        store: RawStore,
        *,
        source_url: str,
        workers: int = 4,
        refresh_recent_months: int = 3,
        today: date | None = None,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self.client = client
        self.store = store
        self.source_url = source_url
        self.workers = workers
        self.refresh_recent_months = refresh_recent_months
        self.today = today or _now().date()

    # -------------------------------------------------------------- helpers

    def _envelope(self, payload: Any, **meta: Any) -> dict[str, Any]:
        """Wrap the untouched server response with lineage metadata."""
        return {
            "meta": {
                "source": SOURCE,
                "source_url": self.source_url,
                "extracted_at": _now().isoformat(),
                **meta,
            },
            "payload": payload,
        }

    def _is_recent(self, month: Month) -> bool:
        """True for the last N *complete* months, and the current (partial) month."""
        if self.refresh_recent_months <= 0:
            return False
        cutoff = Month.from_date(self.today).shift(-self.refresh_recent_months)
        return month >= cutoff

    # ------------------------------------------------------------- metadata

    def extract_metadata(self, run_id: str) -> dict[str, int]:
        """Snapshot every metadata resource. Returns item counts per resource."""
        counts: dict[str, int] = {}
        for resource, fields in METADATA_RESOURCES.items():
            items = list(self.client.iter_pages(resource, resource, {"fields": fields}))
            key = f"{SOURCE}/metadata/{resource}/snapshot={run_id}.json.gz"
            self.store.put_json(
                key, self._envelope({resource: items}, resource=resource, record_count=len(items))
            )
            counts[resource] = len(items)
            logger.info("metadata %s: %d items", resource, len(items))
        return counts

    # ---------------------------------------------------------- data values

    def _fetch_chunk(self, chunk: Chunk, *, force: bool) -> ChunkResult:
        if not force and not self._is_recent(chunk.month) and self.store.exists(chunk.key):
            return ChunkResult(chunk, "skipped")
        try:
            body = self.client.data_value_set_raw(
                chunk.data_set, chunk.org_unit, chunk.month.start, chunk.month.end
            )
            records = len(DataValueSet.model_validate(body).data_values)  # schema check
            written = self.store.put_json(
                chunk.key,
                self._envelope(
                    body,
                    data_set=chunk.data_set,
                    org_unit=chunk.org_unit,
                    period=chunk.month.dhis2_period,
                    record_count=records,
                ),
            )
        except Exception as exc:  # one bad chunk must not stop the run
            logger.warning("chunk failed %s: %s", chunk.key, exc)
            return ChunkResult(chunk, "failed", error=f"{type(exc).__name__}: {exc}")
        return ChunkResult(chunk, "fetched", records=records, bytes_written=written)

    def extract_data_values(
        self,
        data_sets: Sequence[str],
        org_units: Sequence[str],
        months: Sequence[Month],
        *,
        run_id: str,
        force: bool = False,
        on_result: Callable[[ChunkResult], None] | None = None,
    ) -> RunSummary:
        chunks = plan_chunks(data_sets, org_units, months)
        summary = RunSummary(
            run_id=run_id,
            source_url=self.source_url,
            started_at=_now().isoformat(),
            planned=len(chunks),
        )
        summary.server_version = self.client.system_info().version

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self._fetch_chunk, c, force=force) for c in chunks]
            for future in as_completed(futures):
                result = future.result()
                summary.add(result)
                if on_result is not None:
                    on_result(result)

        summary.finished_at = _now().isoformat()
        self.store.put_json(f"{SOURCE}/_runs/run={run_id}.json.gz", summary.to_dict())
        return summary


def new_run_id() -> str:
    return _now().strftime("%Y%m%dT%H%M%SZ")
