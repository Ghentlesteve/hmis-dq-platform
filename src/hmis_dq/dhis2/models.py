"""Typed models for the parts of the DHIS2 Web API this project reads.

Validating responses at the boundary means a change in the server's JSON shape
fails loudly here instead of silently corrupting the lake further down.
Unknown fields are ignored, so newer DHIS2 versions adding fields don't break us.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class DHIS2Model(BaseModel):
    """Base model: camelCase JSON <-> snake_case Python, immutable, extra fields ignored."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        frozen=True,
        extra="ignore",
    )


class Ref(DHIS2Model):
    """A reference to another DHIS2 object by its 11-character UID."""

    id: str


class SystemInfo(DHIS2Model):
    version: str
    system_name: str | None = None
    server_date: datetime | None = None


class Pager(DHIS2Model):
    page: int
    page_count: int
    total: int
    page_size: int


class OrganisationUnit(DHIS2Model):
    id: str
    name: str
    level: int
    path: str
    parent: Ref | None = None
    opening_date: datetime | None = None
    closed_date: datetime | None = None
    # GeoJSON geometry (Point for facilities, (Multi)Polygon for admin areas)
    geometry: dict[str, Any] | None = None


class DataValue(DHIS2Model):
    """One reported number: data element x period x org unit x disaggregation."""

    data_element: str
    period: str
    org_unit: str
    category_option_combo: str
    attribute_option_combo: str
    # Kept as text: DHIS2 stores numbers, booleans and free text in the same field.
    # Parsing happens in the silver layer, where bad values become a DQ finding.
    value: str | None = None
    stored_by: str | None = None
    created: datetime | None = None
    last_updated: datetime | None = None
    comment: str | None = None
    followup: bool = False


class DataValueSet(DHIS2Model):
    data_set: str | None = None
    data_values: list[DataValue] = []
