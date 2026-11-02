"""Loads the store CSVs and derives the business fields.

Price, stock and deadlines are computed here and nowhere else. The model reads
the derived columns; it never multiplies out a discount or subtracts two dates.

Money is computed with `Decimal` and rounded half-up.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import pandas as pd

from emporio.config import get_settings

# Asserted by the test suite.
EXPECTED_ROWS: Final[dict[str, int]] = {
    "products": 65,
    "orders": 20,
    "order_items": 22,
    "customers": 50,
    "promotions": 25,
    "categories": 9,
}

PIX_DISCOUNT_PERCENT: Final = Decimal(5)  # policy manual, section 6.2
RETURN_WINDOW_DAYS: Final = 7  # policy manual, section 4.1

_CENT = Decimal("0.01")

# Explicit dtypes: ids must not become floats because of a blank cell, and text
# columns must not be guessed. `specs` arrives as a JSON string and is parsed in
# `_derive_products`; the date columns are listed in _DATE_COLUMNS.
_DTYPES: Final[dict[str, dict[str, str]]] = {
    "products": {
        "product_id": "int64",
        "price_brl": "float64",
        "name": "string",
        "category_id": "int64",
        "description": "string",
        "stock_quantity": "int64",
        "status": "string",
        "specs": "string",
    },
    "orders": {
        "order_id": "int64",
        "customer_id": "int64",
        "status": "string",
        "total_brl": "float64",
        "payment_method": "string",
        "tracking_code": "string",
        "notes": "string",
    },
    "order_items": {"order_id": "int64", "quantity": "int64", "product_id": "int64"},
    "customers": {
        "customer_id": "int64",
        "name": "string",
        "phone": "string",
        "email": "string",
        "city": "string",
    },
    "promotions": {
        "promotion_id": "int64",
        "product_id": "int64",
        "discount_percent": "int64",
        "description": "string",
        "is_active": "int64",
    },
    "categories": {"category_id": "int64", "name": "string", "description": "string"},
}

_DATE_COLUMNS: Final[dict[str, list[str]]] = {
    "products": ["created_at"],
    # Cancelled orders carry neither `estimated_delivery` nor `tracking_code`
    # (order 19, for one), so both columns are nullable: NaT here, None in the
    # payloads, and a None from `days_since_delivery`.
    "orders": ["order_date", "estimated_delivery"],
}


class DatasetError(RuntimeError):
    """A CSV is missing, unreadable, or not shaped the way the loader expects."""


@dataclass(frozen=True)
class Dataset:
    """The whole store, in memory, with the derived columns already in place.

    The frames are shared by every caller and must be treated as read-only:
    `queries.py` filters and copies, never assigns.
    """

    products: pd.DataFrame
    orders: pd.DataFrame
    order_items: pd.DataFrame
    customers: pd.DataFrame
    promotions: pd.DataFrame
    categories: pd.DataFrame


def normalize_text(value: Any) -> str:
    """Casefold and strip accents, for matching customer wording against the data.

    A customer, and a small model writing a tool argument, types "violoes" or
    "Violões" with equal probability, and neither should miss the category.
    """
    if value is None or value is pd.NA:
        return ""
    text = str(value)
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold().strip()


def money(value: Decimal | float | int) -> float:
    """Round to cents, half-up, the way a price list does."""
    return float(Decimal(str(value)).quantize(_CENT, rounding=ROUND_HALF_UP))


def apply_discount(price: Decimal | float, discount_percent: Decimal | float) -> float:
    """Price with `discount_percent` taken off, rounded to cents."""
    factor = (Decimal(100) - Decimal(str(discount_percent))) / Decimal(100)
    return money(Decimal(str(price)) * factor)


def csv_path(table: str, data_dir: Path | None = None) -> Path:
    """Locate a table's CSV, by exact name first and then by suffix.

    Matching on the suffix rather than anywhere in the name is what keeps
    `orders` from also matching `order_items.csv`.
    """
    directory = data_dir if data_dir is not None else get_settings().data_dir
    exact = directory / f"{table}.csv"
    if exact.is_file():
        return exact

    candidates = sorted(p for p in directory.glob(f"*{table}.csv") if p.is_file())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise DatasetError(
            f"no CSV found for table {table!r} in {directory}. Expected "
            f"{table}.csv, or a file whose name ends with it. Check DATA_DIR."
        )
    raise DatasetError(
        f"table {table!r} matches more than one file in {directory}: "
        f"{[p.name for p in candidates]}. Leave exactly one in place."
    )


def days_since_delivery(estimated_delivery: Any, today: date | None = None) -> int | None:
    """Days elapsed since the estimated delivery date, or None when there is none.

    Returns None, and never raises, for the orders that have no delivery date
    at all: everything cancelled, pending or merely confirmed. A negative result
    means the estimate is still in the future.
    """
    if estimated_delivery is None or estimated_delivery is pd.NA or pd.isna(estimated_delivery):
        return None
    reference = today or date.today()
    return (reference - _as_date(estimated_delivery)).days


def within_return_window(estimated_delivery: Any, today: date | None = None) -> bool | None:
    """Whether the 7-day right-of-withdrawal window is still open (section 4.1)."""
    elapsed = days_since_delivery(estimated_delivery, today)
    if elapsed is None:
        return None
    return 0 <= elapsed <= RETURN_WINDOW_DAYS


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def _parse_specs(raw: Any) -> dict[str, Any]:
    """`specs` is a JSON object inside a CSV cell; an unparseable one is empty."""
    if raw is None or raw is pd.NA or pd.isna(raw):
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _read_table(table: str, data_dir: Path | None = None) -> pd.DataFrame:
    path = csv_path(table, data_dir)
    frame = pd.read_csv(
        path,
        dtype=_DTYPES[table],
        parse_dates=_DATE_COLUMNS.get(table),
    )
    missing = set(_DTYPES[table]) - set(frame.columns)
    if missing:
        raise DatasetError(f"{path.name} is missing columns: {sorted(missing)}")
    return frame


def _active_promotions(promotions: pd.DataFrame) -> pd.DataFrame:
    """One row per product with a live promotion.

    Only `is_active == 1` counts. The data happens to hold at most one active
    promotion per product, but nothing in the CSV enforces that, so the largest
    discount wins, not depending on row order.
    """
    active = promotions[promotions["is_active"] == 1]
    return (
        active.sort_values(["product_id", "discount_percent", "promotion_id"])
        .groupby("product_id", as_index=False)
        .last()
        .loc[:, ["product_id", "discount_percent", "description"]]
        .rename(columns={"description": "promotion"})
    )


def _derive_products(
    products: pd.DataFrame, promotions: pd.DataFrame, categories: pd.DataFrame
) -> pd.DataFrame:
    frame = products.merge(_active_promotions(promotions), on="product_id", how="left")
    frame = frame.merge(
        categories.loc[:, ["category_id", "name"]].rename(columns={"name": "category"}),
        on="category_id",
        how="left",
    )

    frame["specs"] = frame["specs"].map(_parse_specs)
    frame["has_promotion"] = frame["discount_percent"].notna()
    frame["discount_percent"] = frame["discount_percent"].astype("Int64")

    # `effective_price` is what the customer pays today; `pix_price` is 5% off
    # the LIST price, deliberately not off `effective_price`: section 6.2 says
    # the PIX discount does not apply on top of an already promotional price. So
    # for a product on promotion the two are alternatives that cannot be
    # combined, and the cheaper of them is the lowest price available.
    frame["effective_price"] = [
        apply_discount(price, discount) if pd.notna(discount) else money(price)
        for price, discount in zip(frame["price_brl"], frame["discount_percent"])
    ]
    frame["pix_price"] = [
        apply_discount(price, PIX_DISCOUNT_PERCENT) for price in frame["price_brl"]
    ]
    frame["best_price"] = [
        min(effective, pix) for effective, pix in zip(frame["effective_price"], frame["pix_price"])
    ]

    frame["in_stock"] = frame["stock_quantity"] > 0
    frame["available"] = (frame["status"] == "active") & frame["in_stock"]

    # Precomputed match targets. `_name_blob` ranks a hit on the product name
    # above one buried in a description; `_search_blob` also carries the category
    # and the spec values, so "violao de nylon" finds what it should.
    frame["_name_blob"] = frame["name"].map(normalize_text)
    frame["_search_blob"] = [
        normalize_text(
            " ".join(
                [str(name), str(description), str(category), *(str(v) for v in specs.values())]
            )
        )
        for name, description, category, specs in zip(
            frame["name"], frame["description"], frame["category"], frame["specs"]
        )
    ]
    return frame


@lru_cache(maxsize=4)
def load_dataset(data_dir: Path | None = None) -> Dataset:
    """Read every CSV and derive the business fields. Cached per data directory."""
    tables = {table: _read_table(table, data_dir) for table in EXPECTED_ROWS}
    return Dataset(
        products=_derive_products(tables["products"], tables["promotions"], tables["categories"]),
        orders=tables["orders"],
        order_items=tables["order_items"],
        customers=tables["customers"],
        promotions=tables["promotions"],
        categories=tables["categories"],
    )
