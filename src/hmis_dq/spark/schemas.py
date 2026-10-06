"""Explicit schemas for the bronze JSON files.

Giving Spark the schema instead of letting it infer one means it doesn't scan
every file first, and a field changing shape upstream shows up as nulls we can
count, instead of a silently different column type.

Every bronze file is an envelope: ``{"meta": {...}, "payload": <server response>}``.
"""

from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DataType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)


def _struct(**fields: DataType) -> StructType:
    return StructType([StructField(name, dtype, nullable=True) for name, dtype in fields.items()])


REF = _struct(id=StringType())

META = _struct(
    source=StringType(),
    source_url=StringType(),
    extracted_at=StringType(),
    resource=StringType(),
    data_set=StringType(),
    org_unit=StringType(),
    period=StringType(),
    record_count=LongType(),
)

DATA_VALUE = _struct(
    dataElement=StringType(),
    period=StringType(),
    orgUnit=StringType(),
    categoryOptionCombo=StringType(),
    attributeOptionCombo=StringType(),
    value=StringType(),
    storedBy=StringType(),
    created=StringType(),
    lastUpdated=StringType(),
    comment=StringType(),
    followup=BooleanType(),
)

DATA_VALUE_FILE = _struct(
    meta=META,
    payload=_struct(dataSet=StringType(), dataValues=ArrayType(DATA_VALUE)),
)

# Metadata item schemas, keyed by DHIS2 resource name
METADATA_ITEMS: dict[str, StructType] = {
    "organisationUnits": _struct(
        id=StringType(),
        name=StringType(),
        shortName=StringType(),
        level=IntegerType(),
        path=StringType(),
        parent=REF,
        openingDate=StringType(),
        closedDate=StringType(),
        # GeoJSON whose nesting depends on the type (Point vs MultiPolygon),
        # so it is kept as raw JSON text: Spark returns objects as text for a string field.
        geometry=StringType(),
    ),
    "organisationUnitLevels": _struct(id=StringType(), level=IntegerType(), name=StringType()),
    "dataSets": _struct(
        id=StringType(),
        name=StringType(),
        periodType=StringType(),
        openFuturePeriods=IntegerType(),
        expiryDays=DoubleType(),
        timelyDays=DoubleType(),
        dataSetElements=ArrayType(_struct(dataElement=REF)),
        organisationUnits=ArrayType(REF),
    ),
    "dataElements": _struct(
        id=StringType(),
        name=StringType(),
        shortName=StringType(),
        valueType=StringType(),
        aggregationType=StringType(),
        domainType=StringType(),
        categoryCombo=REF,
    ),
    "categoryCombos": _struct(id=StringType(), name=StringType()),
    "categoryOptionCombos": _struct(id=StringType(), name=StringType(), categoryCombo=REF),
}


def metadata_file_schema(resource: str) -> StructType:
    items = METADATA_ITEMS[resource]
    return _struct(meta=META, payload=_struct(**{resource: ArrayType(items)}))
