from datetime import date

import pytest

from hmis_dq.extract.periods import Month, month_range


@pytest.mark.parametrize("text", ["2025-01", "202501", " 2025-01 "])
def test_parse_accepts_iso_and_dhis2_formats(text: str) -> None:
    assert Month.parse(text) == Month(2025, 1)


@pytest.mark.parametrize("text", ["2025-13", "2025-1", "Jan 2025", ""])
def test_parse_rejects_bad_input(text: str) -> None:
    with pytest.raises(ValueError):
        Month.parse(text)


def test_month_boundaries_handle_leap_years() -> None:
    assert Month(2024, 2).end == date(2024, 2, 29)
    assert Month(2025, 2).end == date(2025, 2, 28)
    assert Month(2025, 12).start == date(2025, 12, 1)


def test_shift_crosses_year_boundaries() -> None:
    assert Month(2025, 12).shift(1) == Month(2026, 1)
    assert Month(2025, 1).shift(-1) == Month(2024, 12)
    assert Month(2025, 3).shift(-14) == Month(2024, 1)


def test_month_range_is_inclusive() -> None:
    months = month_range(Month(2024, 11), Month(2025, 2))
    assert [str(m) for m in months] == ["2024-11", "2024-12", "2025-01", "2025-02"]
    assert months[0].dhis2_period == "202411"


def test_month_range_rejects_reversed_bounds() -> None:
    with pytest.raises(ValueError):
        month_range(Month(2025, 2), Month(2025, 1))
