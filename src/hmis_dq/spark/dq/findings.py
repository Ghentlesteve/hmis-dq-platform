"""The shared findings format every check writes to.

One row per problem found. Checks only need to supply the columns they know;
the rest are filled with typed nulls, so all findings union into one table.
"""

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

# column -> Spark SQL type, in output order
FINDING_COLUMNS: dict[str, str] = {
    "check": "string",
    "dimension": "string",
    "severity": "string",
    "dataset_id": "string",
    "district_id": "string",
    "district": "string",
    "org_unit_id": "string",
    "facility": "string",
    "data_element_id": "string",
    "data_element": "string",
    "period": "string",
    "value": "double",
    "expected": "double",
    "score": "double",
    "message": "string",
}


def to_findings(
    frame: DataFrame,
    *,
    check: str,
    dimension: str,
    severity: Column | str,
    message: Column,
    value: Column | None = None,
    expected: Column | None = None,
    score: Column | None = None,
) -> DataFrame:
    """Project a frame of flagged rows into the findings format."""
    supplied: dict[str, Column] = {
        "check": F.lit(check),
        "dimension": F.lit(dimension),
        "severity": severity if isinstance(severity, Column) else F.lit(severity),
        "message": message,
    }
    for name, column in (("value", value), ("expected", expected), ("score", score)):
        if column is not None:
            supplied[name] = column

    columns = []
    for name, dtype in FINDING_COLUMNS.items():
        if name in supplied:
            column = supplied[name]
        elif name in frame.columns:
            column = F.col(name)
        else:
            column = F.lit(None)
        columns.append(column.cast(dtype).alias(name))
    return frame.select(*columns)
