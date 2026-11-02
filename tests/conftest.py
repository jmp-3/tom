"""Shared fixtures.

The dataset is loaded once for the whole session: `load_dataset` is cached and
the frames are read-only by contract, so there is nothing to isolate between
tests and no reason to pay the parse cost repeatedly.
"""

from __future__ import annotations

from datetime import date

import pytest

from emporio.data.loader import Dataset, load_dataset

# A fixed date inside the dataset's own range (orders run 2025-10 to 2026-03).
# Every deadline assertion passes this explicitly: `days_since_delivery` defaults
# to `date.today()`, so a suite that read the system date directly would start
# failing on its own as the calendar moved.
REFERENCE_DATE = date(2026, 3, 1)


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    return load_dataset()
