import pytest

from hmis_dq.extract.catalog import resolve_data_set


def test_resolves_friendly_names() -> None:
    assert resolve_data_set("child_health") == "BfMAe6Itzgt"


def test_passes_through_raw_uids() -> None:
    assert resolve_data_set("Nyh6laLdBEJ") == "Nyh6laLdBEJ"


def test_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="child_health"):
        resolve_data_set("malaria")
