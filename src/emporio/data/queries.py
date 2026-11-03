"""Pure query functions over the loaded dataset."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd

from emporio.data.loader import (
    Dataset,
    brl,
    days_since_delivery,
    load_dataset,
    normalize_text,
    within_return_window,
)
from emporio.data.models import (
    DEFAULT_ORDER_LIMIT,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SIMILAR_LIMIT,
    MAX_SEARCH_LIMIT,
)

# "This order does not exist" and "this order exists but belongs to someone
# else" must produce byte-identical answers.
ORDER_NOT_FOUND: Final[dict[str, Any]] = {"found": False}

# Customer-facing wording, so Portuguese. The model gets the canonical English
# status *and* the label: it should never have to invent the translation.
STATUS_LABELS: Final[dict[str, str]] = {
    "pending": "aguardando pagamento",
    "confirmed": "pagamento confirmado",
    "shipped": "a caminho",
    "delivered": "entregue",
    "cancelled": "cancelado",
}

PAYMENT_LABELS: Final[dict[str, str]] = {
    "pix": "PIX",
    "boleto": "boleto",
    "debit": "cartão de débito",
    "credit_3x": "cartão de crédito em 3x",
    "credit_6x": "cartão de crédito em 6x",
    "credit_12x": "cartão de crédito em 12x",
}

# A small model asked to filter by status will write the Portuguese word it just
# read in the conversation. Accepting it costs one dict and removes a whole
# class of empty-result confusion.
_STATUS_ALIASES: Final[dict[str, str]] = {
    "aguardando pagamento": "pending",
    "aguardando": "pending",
    "pendente": "pending",
    "confirmado": "confirmed",
    "pagamento confirmado": "confirmed",
    "a caminho": "shipped",
    "enviado": "shipped",
    "despachado": "shipped",
    "entregue": "delivered",
    "entregado": "delivered",
    "cancelado": "cancelled",
    "canceled": "cancelled",
}


class MissingCustomerIdError(ValueError):
    """Raised when an order query arrives without an identified customer.

    The signature could have been `customer_id=None` meaning "no filter", and
    then one omitted argument would make the privacy filter disappear in
    silence. It is mandatory instead, and absence is an error.
    """


def _dataset(dataset: Dataset | None, data_dir: Path | None = None) -> Dataset:
    return dataset if dataset is not None else load_dataset(data_dir)


def _require_customer_id(customer_id: Any) -> int:
    if customer_id is None or (isinstance(customer_id, str) and not customer_id.strip()):
        raise MissingCustomerIdError(
            "customer_id is mandatory: order queries only run for an identified "
            "customer (the session supplies it, never the model)."
        )
    try:
        resolved = int(customer_id)
    except (TypeError, ValueError) as exc:
        raise MissingCustomerIdError(f"invalid customer_id: {customer_id!r}") from exc
    if resolved <= 0:
        raise MissingCustomerIdError(f"invalid customer_id: {customer_id!r}")
    return resolved


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """`to_dict("records")` with the key type narrowed to `str`.

    A DataFrame column can be any hashable, so pandas types the result as
    `dict[Hashable, Any]` while every projection here takes `dict[str, Any]`.
    Every column in this dataset is a string, so the narrowing is correct. It is
    done once here rather than as eleven `type: ignore`s at the call sites.
    """
    return cast("list[dict[str, Any]]", frame.to_dict("records"))


def _opt_str(value: Any) -> str | None:
    """Normalize pandas' several flavours of missing into a plain None."""
    if value is None or value is pd.NA or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _opt_int(value: Any) -> int | None:
    if value is None or value is pd.NA or pd.isna(value):
        return None
    return int(value)


def _iso_date(value: Any) -> str | None:
    if value is None or value is pd.NA or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _availability(row: dict[str, Any]) -> str:
    """One field that answers "can I buy it?", so the model never has to reason
    its way there from a status string and a stock count (section 7.3)."""
    if row["status"] == "discontinued":
        return "fora de linha"
    if row["status"] == "coming_soon":
        return "em breve"
    if not row["in_stock"]:
        return "sem estoque"
    return "disponível"


def _lean_product(row: dict[str, Any]) -> dict[str, Any]:
    """The search projection.

    Every money field carries a `_label` beside it: the number for code to
    compare and the string for the model to quote. Same reasoning as
    `status_label` and `payment_label` -- the agent should never have to
    construct what it says from a raw value -- but here it is not a convenience.
    A model that reformats a float gets the price wrong (see `brl`), and a price
    quoted wrong to a customer is the worst failure this prototype can have.
    """
    return {
        "product_id": int(row["product_id"]),
        "name": row["name"],
        "category": _opt_str(row["category"]),
        "list_price": float(row["price_brl"]),
        "list_price_label": brl(row["price_brl"]),
        "effective_price": float(row["effective_price"]),
        "effective_price_label": brl(row["effective_price"]),
        "discount_percent": _opt_int(row["discount_percent"]),
        "pix_price": float(row["pix_price"]),
        "pix_price_label": brl(row["pix_price"]),
        "in_stock": bool(row["in_stock"]),
        "availability": _availability(row),
    }


def _full_product(row: dict[str, Any]) -> dict[str, Any]:
    """The detail projection: one product, everything about it."""
    return _lean_product(row) | {
        "description": _opt_str(row["description"]),
        "specs": dict(row["specs"]),
        "stock_quantity": int(row["stock_quantity"]),
        "status": row["status"],
        "available": bool(row["available"]),
        "promotion": _opt_str(row["promotion"]),
        # Both prices restated as the cheapest one, so the agent can present
        # the lowest available price without doing the comparison itself.
        # Section 6.2 forbids combining the PIX discount with a promotional
        # price, so this is the minimum of two alternatives, never a product of
        # both.
        "best_price": float(row["best_price"]),
        "best_price_label": brl(row["best_price"]),
    }


def _order_items(dataset: Dataset, order_id: int) -> list[dict[str, Any]]:
    items = dataset.order_items[dataset.order_items["order_id"] == order_id]
    joined = items.merge(
        dataset.products.loc[:, ["product_id", "name"]], on="product_id", how="left"
    )
    # No historical unit price exists in this dataset, only the order total and
    # today's catalog price, which may since have changed. Quantities and names
    # only: the agent cannot state what was paid per item, so it will not guess.
    return [
        {
            "product_id": int(row["product_id"]),
            "name": _opt_str(row["name"]),
            "quantity": int(row["quantity"]),
        }
        for row in _records(joined)
    ]


def _lean_order(dataset: Dataset, row: dict[str, Any]) -> dict[str, Any]:
    """The listing projection.

    It carries the tracking code and the delivery estimate even though they make
    the row larger, because "onde está meu pedido?" is the most common question
    on an authenticated channel and this way it is answerable in one tool call.
    """
    return {
        "order_id": int(row["order_id"]),
        "status": row["status"],
        "status_label": STATUS_LABELS.get(row["status"], row["status"]),
        "order_date": _iso_date(row["order_date"]),
        "total_brl": float(row["total_brl"]),
        "total_brl_label": brl(row["total_brl"]),
        "tracking_code": _opt_str(row["tracking_code"]),
        "estimated_delivery": _iso_date(row["estimated_delivery"]),
        "items": _order_items(dataset, int(row["order_id"])),
    }


def _full_order(dataset: Dataset, row: dict[str, Any], today: date | None) -> dict[str, Any]:
    estimated = row["estimated_delivery"]
    return _lean_order(dataset, row) | {
        "found": True,
        "payment_method": row["payment_method"],
        "payment_label": PAYMENT_LABELS.get(row["payment_method"], row["payment_method"]),
        # Computed here precisely so the model never subtracts two dates.
        "days_since_delivery": days_since_delivery(estimated, today),
        "within_return_window": within_return_window(estimated, today),
        # The cancellation reason is in `notes` and is the one thing the
        # customer needs to know about a cancelled order.
        "notes": _opt_str(row["notes"]),
    }


def search_products(
    *,
    category: str | None = None,
    text: str | None = None,
    price_min: float | None = None,
    price_max: float | None = None,
    only_available: bool = True,
    limit: int = DEFAULT_SEARCH_LIMIT,
    dataset: Dataset | None = None,
) -> list[dict[str, Any]]:
    """Search the catalog. Returns reduced rows, cheapest first.

    `price_min`/`price_max` compare against `effective_price`, the price the
    customer would pay today.
    """
    data = _dataset(dataset)
    frame = data.products

    if only_available:
        frame = frame[frame["available"]]

    if category:
        wanted = normalize_text(category)
        matched = (
            frame["category"].map(normalize_text).map(lambda name: wanted in name or name in wanted)
        )
        # Tolerate a category id arriving as a string, which a small model does
        # when it has just read a numbered list.
        if not matched.any() and wanted.isdigit():
            matched = frame["category_id"] == int(wanted)
        frame = frame[matched]

    if price_min is not None:
        frame = frame[frame["effective_price"] >= float(price_min)]
    if price_max is not None:
        frame = frame[frame["effective_price"] <= float(price_max)]

    frame = frame.assign(_rank=0)
    if text:
        terms = [term for term in normalize_text(text).split() if term]
        if terms:
            in_name = frame["_name_blob"].map(lambda blob: any(t in blob for t in terms))
            in_blob = frame["_search_blob"].map(lambda blob: any(t in blob for t in terms))
            frame = frame[in_blob]
            # A hit on the product name outranks one buried in a description.
            frame = frame.assign(_rank=(~in_name.loc[frame.index]).astype(int))

    # product_id last so equal prices never reorder between runs.
    frame = frame.sort_values(["_rank", "effective_price", "product_id"])
    rows = _records(frame.head(max(1, min(int(limit), MAX_SEARCH_LIMIT))))
    return [_lean_product(row) for row in rows]


def get_product(
    product_id: int | None = None,
    name: str | None = None,
    *,
    dataset: Dataset | None = None,
) -> dict[str, Any] | None:
    """Full detail for one product, specs included. None when nothing matches.

    Either identifier works. For `name`, an exact match wins over a partial one,
    and among partial matches the shortest name wins, so that "Yamaha C40"
    resolves to the C40 and not to a longer name that happens to contain it.
    """
    if (product_id is None) == (name is None):
        raise ValueError("pass exactly one of product_id or name")

    data = _dataset(dataset)

    if product_id is not None:
        matched = data.products[data.products["product_id"] == int(product_id)]
        rows = _records(matched)
        return _full_product(rows[0]) if rows else None

    wanted = normalize_text(name)
    if not wanted:
        return None
    exact = data.products[data.products["_name_blob"] == wanted]
    if not exact.empty:
        return _full_product(_records(exact)[0])
    partial = data.products[data.products["_name_blob"].str.contains(wanted, regex=False, na=False)]
    if partial.empty:
        return None
    best = partial.assign(_length=partial["name"].str.len()).sort_values(["_length", "product_id"])
    return _full_product(_records(best)[0])


def list_categories(*, dataset: Dataset | None = None) -> list[dict[str, Any]]:
    """Categories that actually have products.

    Categories 6, 7 and 8 exist in the CSV with no product behind them, and the
    policy manual advertises nine categories and "over 300 instruments".
    Offering the customer a category the store cannot sell from is worse than a
    shorter list, so the catalog decides this, not the manual.
    """
    data = _dataset(dataset)
    counts = (
        data.products.groupby("category_id")
        .agg(products=("product_id", "size"), available=("available", "sum"))
        .reset_index()
    )
    joined = data.categories.merge(counts, on="category_id", how="inner")
    return [
        {
            "category_id": int(row["category_id"]),
            "name": row["name"],
            "description": _opt_str(row["description"]),
            "products": int(row["products"]),
            "available": int(row["available"]),
        }
        for row in _records(joined.sort_values("category_id"))
    ]


def list_orders(
    customer_id: int,
    *,
    status: str | None = None,
    limit: int = DEFAULT_ORDER_LIMIT,
    dataset: Dataset | None = None,
) -> list[dict[str, Any]]:
    """This customer's orders, most recent first. Empty list when there are none.

    Exists because of the authenticated channel.
    """
    resolved = _require_customer_id(customer_id)
    data = _dataset(dataset)

    frame = data.orders[data.orders["customer_id"] == resolved]
    if status:
        wanted = normalize_text(status)
        frame = frame[frame["status"] == _STATUS_ALIASES.get(wanted, wanted)]

    frame = frame.sort_values(["order_date", "order_id"], ascending=False)
    rows = _records(frame.head(max(1, min(int(limit), MAX_SEARCH_LIMIT))))
    return [_lean_order(data, row) for row in rows]


def get_order(
    order_id: int,
    customer_id: int,
    *,
    today: date | None = None,
    dataset: Dataset | None = None,
) -> dict[str, Any]:
    """One order, scoped to its owner.

    Returns `ORDER_NOT_FOUND` both when the order does not exist and when it
    belongs to another customer. No name, phone or e-mail is ever included, not
    even the requesting customer's own: they already know their phone number,
    and putting it in the payload only increases the risk of exposure across the
    conversation history and the committed transcripts (policy manual, section 9).
    """
    resolved = _require_customer_id(customer_id)
    data = _dataset(dataset)

    try:
        wanted = int(order_id)
    except (TypeError, ValueError):
        return dict(ORDER_NOT_FOUND)

    matched = data.orders[
        (data.orders["order_id"] == wanted) & (data.orders["customer_id"] == resolved)
    ]
    if matched.empty:
        return dict(ORDER_NOT_FOUND)
    return _full_order(data, _records(matched)[0], today)


def similar_products(
    product_id: int,
    *,
    limit: int = DEFAULT_SIMILAR_LIMIT,
    dataset: Dataset | None = None,
) -> list[dict[str, Any]]:
    """Available alternatives to a product, same category, closest price first.

    Section 7.3 requires suggesting equivalents when an item is out of stock or
    discontinued, so the reference product itself need not be available, but
    every suggestion must be.
    """
    data = _dataset(dataset)
    reference = data.products[data.products["product_id"] == int(product_id)]
    if reference.empty:
        return []
    row = _records(reference)[0]

    candidates = data.products[
        (data.products["category_id"] == row["category_id"])
        & (data.products["product_id"] != row["product_id"])
        & data.products["available"]
    ]
    if candidates.empty:
        return []
    ranked = candidates.assign(
        _distance=(candidates["effective_price"] - row["effective_price"]).abs()
    ).sort_values(["_distance", "product_id"])
    rows = _records(ranked.head(max(1, min(int(limit), MAX_SEARCH_LIMIT))))
    return [_lean_product(candidate) for candidate in rows]


def load_customer(customer_id: int, *, dataset: Dataset | None = None) -> dict[str, Any] | None:
    """The name, for the session greeting. Nothing else.

    Session bootstrap, called once by the CLI, and never registered as a tool:
    the model has no way to look a customer up. Phone, e-mail and the registered
    city are in the CSV and are deliberately not projected here -- the city
    because a destination that is assumed is a wrong freight stated with
    confidence (see `agent.context.for_customer`).
    """
    resolved = _require_customer_id(customer_id)
    data = _dataset(dataset)
    matched = data.customers[data.customers["customer_id"] == resolved]
    if matched.empty:
        return None
    row = _records(matched)[0]
    return {"customer_id": resolved, "name": row["name"]}
