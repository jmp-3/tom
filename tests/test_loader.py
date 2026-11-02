"""Loading and derivation."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from emporio.data.loader import (
    EXPECTED_ROWS,
    PIX_DISCOUNT_PERCENT,
    Dataset,
    DatasetError,
    apply_discount,
    brl,
    csv_path,
    days_since_delivery,
    money,
    normalize_text,
    within_return_window,
)
from tests.conftest import REFERENCE_DATE


@pytest.mark.parametrize("table,expected", sorted(EXPECTED_ROWS.items()))
def test_row_counts_are_exact(dataset: Dataset, table: str, expected: int) -> None:
    """
    None of the CSVs ends in a newline. Pandas handles this correctly, but a
    shell pipeline or a hand-written parser drops the final record silently.
    Asserting the counts turns that into a failing test instead of a product
    that disappears from the catalog unnoticed.
    """
    assert len(getattr(dataset, table)) == expected


@pytest.mark.parametrize("table", sorted(EXPECTED_ROWS))
def test_every_table_resolves_to_a_file(table: str) -> None:
    assert csv_path(table).is_file()


def test_tables_resolve_under_plain_or_prefixed_names(tmp_path) -> None:
    """No filename pattern is hardcoded, so the data works as delivered, with
    whatever prefix it carries, and equally well renamed to plain table names."""
    for name in ("products.csv", "some vendor prefix - orders.csv"):
        (tmp_path / name).touch()
    assert csv_path("products", tmp_path).name == "products.csv"
    assert csv_path("orders", tmp_path).name == "some vendor prefix - orders.csv"


def test_a_plain_name_wins_over_a_prefixed_one(tmp_path) -> None:
    (tmp_path / "products.csv").touch()
    (tmp_path / "old - products.csv").touch()
    assert csv_path("products", tmp_path).name == "products.csv"


def test_orders_does_not_resolve_to_order_items(tmp_path) -> None:
    """Suffix matching, not substring matching: loading the wrong table would
    only be detected much later, as wrong prices, rather than as an error here."""
    (tmp_path / "x - order_items.csv").touch()
    with pytest.raises(DatasetError):
        csv_path("orders", tmp_path)
    assert csv_path("order_items", tmp_path).name == "x - order_items.csv"


def test_ambiguous_and_missing_tables_both_raise(tmp_path) -> None:
    with pytest.raises(DatasetError, match="no CSV found"):
        csv_path("products", tmp_path)
    (tmp_path / "a - products.csv").touch()
    (tmp_path / "b - products.csv").touch()
    with pytest.raises(DatasetError, match="more than one file"):
        csv_path("products", tmp_path)


def test_money_rounds_half_up_not_to_even() -> None:
    # 5% off R$ 599,90 is 569.905. Binary floats and banker's rounding both land
    # on 569.90; a price list lands on 569.91.
    assert money(Decimal("569.905")) == 569.91
    assert apply_discount(599.9, PIX_DISCOUNT_PERCENT) == 569.91


@pytest.mark.parametrize(
    "value,expected",
    [
        (1299, "R$ 1.299,00"),
        (1299.0, "R$ 1.299,00"),
        (599.9, "R$ 599,90"),
        (7999, "R$ 7.999,00"),
        (0, "R$ 0,00"),
        (1234567.89, "R$ 1.234.567,89"),
        (Decimal("439.195"), "R$ 439,20"),
    ],
)
def test_brl_writes_brazilian_currency(value: object, expected: str) -> None:
    """The string the model quotes instead of formatting a float itself.

    `1299.0` and `7999.0` are in the list because those are the two values that
    came back from the model as "R$ 129,90" and "R$ 79,99".
    """
    assert brl(value) == expected  # type: ignore[arg-type]


def test_brl_rounds_the_way_the_price_list_does() -> None:
    # Same half-up rounding as `money`, so the label can never disagree with the
    # number beside it.
    assert brl(2.345) == "R$ 2,35"
    assert brl(money(2.345)) == brl(2.345)


def test_normalize_text_ignores_accents_and_case() -> None:
    assert normalize_text("Violões") == normalize_text("violoes") == "violoes"
    assert normalize_text(None) == ""


def test_specs_are_parsed_into_dicts(dataset: Dataset) -> None:
    assert all(isinstance(specs, dict) for specs in dataset.products["specs"])
    # Every product in the file describes at least its colour.
    assert all(specs for specs in dataset.products["specs"])


def test_every_product_resolves_to_a_category(dataset: Dataset) -> None:
    assert dataset.products["category"].notna().all()


def test_dates_are_parsed(dataset: Dataset) -> None:
    assert pd.api.types.is_datetime64_any_dtype(dataset.orders["order_date"])
    assert pd.api.types.is_datetime64_any_dtype(dataset.orders["estimated_delivery"])
    assert pd.api.types.is_datetime64_any_dtype(dataset.products["created_at"])


def test_cancelled_orders_have_no_delivery_date(dataset: Dataset) -> None:
    cancelled = dataset.orders[dataset.orders["status"] == "cancelled"]
    assert len(cancelled) == 4
    assert cancelled["estimated_delivery"].isna().all()
    assert cancelled["tracking_code"].isna().all()


def test_days_since_delivery_returns_none_instead_of_raising() -> None:
    """Cancelled and pending orders have no delivery date at all.
    The deadline helpers must handle that rather than raise."""
    assert days_since_delivery(None, REFERENCE_DATE) is None
    assert days_since_delivery(pd.NaT, REFERENCE_DATE) is None
    assert within_return_window(pd.NaT, REFERENCE_DATE) is None


def test_days_since_delivery_counts_from_the_estimate() -> None:
    assert days_since_delivery(pd.Timestamp("2026-02-22"), date(2026, 3, 1)) == 7
    assert days_since_delivery(pd.Timestamp("2026-03-10"), date(2026, 3, 1)) == -9


def test_return_window_is_seven_days_inclusive() -> None:
    delivered = pd.Timestamp("2026-02-22")
    assert within_return_window(delivered, date(2026, 2, 22)) is True
    assert within_return_window(delivered, date(2026, 3, 1)) is True  # day 7
    assert within_return_window(delivered, date(2026, 3, 2)) is False  # day 8
    # Estimate still in the future, so the period has not begun.
    assert within_return_window(pd.Timestamp("2026-03-10"), date(2026, 3, 1)) is False


def test_only_active_promotions_are_derived(dataset: Dataset) -> None:
    """Expired promotions should not be counted."""
    promoted = dataset.products[dataset.products["has_promotion"]]
    assert sorted(promoted["product_id"]) == [90, 94, 121, 127]
    assert (promoted["discount_percent"] > 0).all()
