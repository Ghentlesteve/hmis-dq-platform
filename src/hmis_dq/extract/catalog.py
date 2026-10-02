"""Friendly names for the DHIS2 datasets this project extracts.

UIDs are from the DHIS2 Sierra Leone demo database. Any other dataset can be
passed by its raw 11-character UID.
"""

import re

DATASETS: dict[str, str] = {
    "child_health": "BfMAe6Itzgt",  # immunisation, vitamin A, nutrition
    "reproductive_health": "QX4ZTUbOt3a",  # ANC visits, deliveries, maternal outcomes
}

DEFAULT_DATASETS = ("child_health", "reproductive_health")

_UID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9]{10}$")


def resolve_data_set(name_or_uid: str) -> str:
    if name_or_uid in DATASETS:
        return DATASETS[name_or_uid]
    if _UID_PATTERN.match(name_or_uid):
        return name_or_uid
    known = ", ".join(sorted(DATASETS))
    raise ValueError(f"unknown dataset {name_or_uid!r}; use one of {known} or a DHIS2 UID")
