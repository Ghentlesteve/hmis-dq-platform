"""Calendar months, the unit DHIS2 monthly datasets are reported in."""

import calendar
import re
from dataclasses import dataclass
from datetime import date
from typing import Self

_MONTH_PATTERN = re.compile(r"^(\d{4})-?(\d{2})$")
MONTHS_PER_YEAR = 12


@dataclass(frozen=True, order=True)
class Month:
    year: int
    month: int

    def __post_init__(self) -> None:
        if not 1 <= self.month <= MONTHS_PER_YEAR:
            raise ValueError(f"month must be 1-12, got {self.month}")

    @classmethod
    def parse(cls, text: str) -> Self:
        """Parse ``2025-01`` or ``202501`` (DHIS2 period format)."""
        match = _MONTH_PATTERN.match(text.strip())
        if not match:
            raise ValueError(f"expected YYYY-MM or YYYYMM, got {text!r}")
        return cls(int(match[1]), int(match[2]))

    @classmethod
    def from_date(cls, day: date) -> Self:
        return cls(day.year, day.month)

    @property
    def start(self) -> date:
        return date(self.year, self.month, 1)

    @property
    def end(self) -> date:
        return date(self.year, self.month, calendar.monthrange(self.year, self.month)[1])

    @property
    def dhis2_period(self) -> str:
        return f"{self.year}{self.month:02d}"

    def shift(self, months: int) -> "Month":
        index = self.year * MONTHS_PER_YEAR + (self.month - 1) + months
        return Month(index // MONTHS_PER_YEAR, index % MONTHS_PER_YEAR + 1)

    def __str__(self) -> str:
        return f"{self.year}-{self.month:02d}"


def month_range(start: Month, end: Month) -> list[Month]:
    """All months from ``start`` to ``end``, inclusive."""
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    months = [start]
    while months[-1] < end:
        months.append(months[-1].shift(1))
    return months
