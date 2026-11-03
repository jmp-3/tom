"""Query functions: pricing arithmetic, catalog filters, and customer scope."""

from __future__ import annotations

from datetime import date

import pytest

from emporio.data.loader import Dataset, brl
from emporio.data.models import (
    GetOrderInput,
    GetProductInput,
    ListOrdersInput,
    SearchProductsInput,
    tool_schema,
)
from emporio.data.queries import (
    ORDER_NOT_FOUND,
    MissingCustomerIdError,
    get_order,
    get_product,
    list_categories,
    list_orders,
    load_customer,
    search_products,
    similar_products,
)
from tests.conftest import REFERENCE_DATE

# Fixtures drawn from the CSVs, with the arithmetic done by hand so the test
# does not re-implement the code it is checking.
PROMOTED = 121  # Ohana CK-20, R$ 549,00 list, 20% active
PLAIN = 81  # Yamaha C40, R$ 599,90 list, no promotion
OUT_OF_STOCK = 96  # Giannini GF-3D, active but stock 0
DISCOUNTED_INTO_RANGE = 90  # Crafter HT-100, R$ 2.399 list, 18% off -> 1967.18
DISCONTINUED = 113  # Shelby SN-7C, discontinued and stock 0
COMING_SOON = 130  # Kala KA-ABP-CTG, coming_soon with stock 2

# Order 1 belongs to customer 3; customer 7 owns orders 2 and 20.
ORDER_OF_CUSTOMER_3 = 1
OTHER_CUSTOMER = 7
CUSTOMER_WITHOUT_ORDERS = 40  # 32 of the 50 customers are in this position


# --- pricing -----------------------------------------------------------------


def test_effective_price_without_promotion_is_the_list_price() -> None:
    product = get_product(PLAIN)
    assert product is not None
    assert product["list_price"] == 599.9
    assert product["effective_price"] == 599.9
    assert product["discount_percent"] is None
    assert product["promotion"] is None


def test_effective_price_applies_the_active_discount() -> None:
    product = get_product(PROMOTED)
    assert product is not None
    assert product["list_price"] == 549.0
    assert product["discount_percent"] == 20
    assert product["effective_price"] == 439.2


def test_pix_discount_is_not_cumulative_with_a_promotion() -> None:
    """Section 6.2: the 5% PIX discount does not apply on top of a price that is
    already promotional. So PIX is always 5% off the LIST price, here 521.55 and
    not 439.20 * 0.95 = 417.24. The two are alternatives that cannot be
    combined."""
    product = get_product(PROMOTED)
    assert product is not None
    assert product["pix_price"] == 521.55
    assert product["pix_price"] != round(product["effective_price"] * 0.95, 2)
    # The lowest available price is the cheaper of the two, here the promotion.
    assert product["best_price"] == 439.2


def test_pix_discount_on_a_product_without_promotion() -> None:
    product = get_product(PLAIN)
    assert product is not None
    assert product["pix_price"] == 569.91  # 599.90 - 5%, rounded half-up
    assert product["best_price"] == 569.91


# --- availability ------------------------------------------------------------


def test_product_with_zero_stock_is_not_available() -> None:
    product = get_product(OUT_OF_STOCK)
    assert product is not None
    assert product["status"] == "active"
    assert product["stock_quantity"] == 0
    assert product["in_stock"] is False
    assert product["available"] is False
    assert product["availability"] == "sem estoque"


def test_discontinued_product_is_labelled_as_such() -> None:
    product = get_product(DISCONTINUED)
    assert product is not None
    assert product["availability"] == "fora de linha"
    assert product["available"] is False


def test_coming_soon_product_has_stock_but_is_not_for_sale() -> None:
    product = get_product(COMING_SOON)
    assert product is not None
    assert product["stock_quantity"] == 2
    assert product["in_stock"] is True
    assert product["available"] is False
    assert product["availability"] == "em breve"


def test_search_only_ever_returns_available_products() -> None:
    """The invariant, checked without relying on any particular `limit`."""
    for rows in (search_products(limit=20), search_products(category="Violões", limit=20)):
        assert rows
        assert all(row["availability"] == "disponível" for row in rows)


@pytest.mark.parametrize("product_id", [OUT_OF_STOCK, DISCONTINUED, COMING_SOON])
def test_search_hides_unavailable_products_by_default(product_id: int) -> None:
    """Searched by its own name, so that price ranking cannot be what excludes
    it: if the product is missing, the availability filter is the only reason.

    An earlier version of this test asked for the 20 cheapest products and
    asserted the absence of each. That passed for the coming-soon baritone only
    because it lists at R$ 1.749 while the 20th cheapest is R$ 849, so the
    assertion held even with the filter switched off entirely.
    """
    reference = get_product(product_id)
    assert reference is not None
    found = {row["product_id"] for row in search_products(text=reference["name"], limit=20)}
    assert product_id not in found


@pytest.mark.parametrize("product_id", [OUT_OF_STOCK, DISCONTINUED, COMING_SOON])
def test_search_surfaces_unavailable_products_when_asked(product_id: int) -> None:
    """For "do you still sell the X?", where the answer needs the record
    that the default search hides."""
    reference = get_product(product_id)
    assert reference is not None
    found = {
        row["product_id"]
        for row in search_products(text=reference["name"], only_available=False, limit=20)
    }
    assert product_id in found


# --- search ------------------------------------------------------------------


def test_every_money_field_carries_the_string_the_model_should_quote() -> None:
    """The model gets the price wrong when it has to format one (see `brl`).

    So no money field travels to a tool without its label, and the label always
    agrees with the number beside it. A new money field added without one is a
    price the agent can misquote, which is why this walks the projection instead
    of naming fields.
    """
    product = get_product(PROMOTED)
    assert product is not None
    order = get_order(ORDER_OF_CUSTOMER_3, 3, today=REFERENCE_DATE)

    for row in (product, order):
        money_fields = [
            key
            for key, value in row.items()
            if isinstance(value, float) and not key.endswith("_percent")
        ]
        assert money_fields, "the projection should carry prices"
        for key in money_fields:
            assert f"{key}_label" in row, f"{key} has no label"
            assert row[f"{key}_label"] == brl(row[key])


def test_price_range_filters_on_the_price_the_customer_would_pay() -> None:
    rows = search_products(price_min=400, price_max=600, limit=20)
    assert rows
    assert all(400 <= row["effective_price"] <= 600 for row in rows)
    # The promoted ukulele lists at 549.00 and sells at 439.20, so it belongs to
    # the 400-600 band and not to a band chosen by its list price.
    assert PROMOTED in {row["product_id"] for row in rows}
    assert PROMOTED not in {row["product_id"] for row in search_products(price_min=500, limit=20)}


def test_price_ceiling_lets_a_discount_bring_a_product_into_range() -> None:
    """The case that actually distinguishes "effective" from "list".

    The Crafter folk lists at R$ 2.399 and sells at R$ 1.967,18 with its 18%
    off, so a R$ 2.000 ceiling has to include it. A filter on the list price
    would exclude the one product the discount was supposed to make affordable.
    The range is narrow because exactly one product falls inside it, which stops
    `limit` from being what decides the result.
    """
    rows = search_products(price_min=1900, price_max=2000, limit=20)
    assert [row["product_id"] for row in rows] == [DISCOUNTED_INTO_RANGE]
    assert rows[0]["list_price"] == 2399.0
    assert rows[0]["effective_price"] == 1967.18


def test_category_matches_without_accents() -> None:
    accented = search_products(category="Violões", limit=20)
    plain = search_products(category="violoes", limit=20)
    assert accented == plain
    assert accented
    assert all(row["category"] == "Violões" for row in accented)


def test_search_results_are_ordered_and_capped() -> None:
    rows = search_products(category="Violões", limit=5)
    assert len(rows) == 5
    assert [row["effective_price"] for row in rows] == sorted(
        row["effective_price"] for row in rows
    )


def test_search_matches_text_in_name_description_and_specs() -> None:
    by_brand = search_products(text="Yamaha", limit=20)
    assert by_brand
    # "nylon" appears only in the spec values of some products, never the name.
    assert search_products(text="nylon", limit=20)
    # A rival keyboard whose description compares it to a Yamaha is a legitimate
    # hit, but it must not outrank the Yamahas themselves.
    named = ["yamaha" in row["name"].casefold() for row in by_brand]
    assert named[0] is True
    assert named == sorted(named, reverse=True), "name matches must come first"


def test_search_projection_omits_specs_and_description() -> None:
    """The main cost reduction in the project: twelve guitars with full specs
    cost about 1,800 tokens per request against about 500 for this projection."""
    row = search_products(limit=1)[0]
    assert "specs" not in row
    assert "description" not in row
    full = get_product(row["product_id"])
    assert full is not None
    assert isinstance(full["specs"], dict) and full["specs"]
    assert full["description"]


def test_get_product_requires_exactly_one_identifier() -> None:
    with pytest.raises(ValueError):
        get_product()
    with pytest.raises(ValueError):
        get_product(PLAIN, "Yamaha C40 Nylon Natural")


def test_get_product_by_name_is_accent_and_case_insensitive() -> None:
    assert get_product(name="yamaha c40 nylon natural") == get_product(PLAIN)
    # A partial name resolves to the shortest match, not to an arbitrary one.
    partial = get_product(name="Yamaha C40")
    assert partial is not None and partial["product_id"] == PLAIN


def test_get_product_returns_none_when_nothing_matches() -> None:
    assert get_product(9999) is None
    assert get_product(name="Stradivarius") is None


def test_list_categories_only_returns_categories_with_products(
    dataset: Dataset,
) -> None:
    """Categories 6, 7 and 8 exist in the CSV with nothing behind them. The
    policy manual also advertises nine categories and "over 300 instruments";
    the catalog is the source of truth about what can actually be sold."""
    categories = list_categories()
    ids = [row["category_id"] for row in categories]
    assert ids == [1, 2, 3, 4, 5, 9]
    assert 6 not in ids and 7 not in ids and 8 not in ids
    assert sum(row["products"] for row in categories) == len(dataset.products)


def test_similar_products_suggests_available_alternatives() -> None:
    """Section 7.3: when an item is out of stock or discontinued, suggest
    equivalents. The reference product may be unavailable; the suggestions
    may not."""
    for reference in (OUT_OF_STOCK, DISCONTINUED):
        rows = similar_products(reference, limit=3)
        assert len(rows) == 3
        assert all(row["availability"] == "disponível" for row in rows)
        assert reference not in {row["product_id"] for row in rows}

    reference_product = get_product(DISCONTINUED)
    assert reference_product is not None
    assert all(
        row["category"] == reference_product["category"] for row in similar_products(DISCONTINUED)
    )


def test_similar_products_of_an_unknown_product_is_empty() -> None:
    assert similar_products(9999) == []


# --- orders ------------------------------------------------------------------


def test_get_order_returns_the_order_for_its_owner() -> None:
    order = get_order(ORDER_OF_CUSTOMER_3, 3, today=REFERENCE_DATE)
    assert order["found"] is True
    assert order["status"] == "delivered"
    assert order["status_label"] == "entregue"
    assert order["tracking_code"] == "BRAB1234567BR"
    assert order["items"] == [
        {"product_id": 93, "name": "Martin D-28 Dreadnought Natural", "quantity": 1}
    ]


def test_cancelled_order_reports_its_reason_and_no_deadlines() -> None:
    """Order 19 has neither an estimated delivery nor a tracking code, so the
    deadline fields must come back None rather than raise, and `notes` carries
    the one thing the customer needs to know."""
    order = get_order(19, 3, today=REFERENCE_DATE)
    assert order["found"] is True
    assert order["estimated_delivery"] is None
    assert order["days_since_delivery"] is None
    assert order["within_return_window"] is None
    assert order["tracking_code"] is None
    assert order["notes"] == "Pagamento não confirmado dentro do prazo"


def test_return_window_is_computed_against_an_injected_date() -> None:
    """Order 1's estimate is 2025-10-25. The suite fixes the date so the result
    cannot change on its own as the calendar moves. The production default is
    `date.today()`."""
    inside = get_order(ORDER_OF_CUSTOMER_3, 3, today=date(2025, 10, 30))
    assert inside["days_since_delivery"] == 5
    assert inside["within_return_window"] is True

    outside = get_order(ORDER_OF_CUSTOMER_3, 3, today=date(2025, 11, 5))
    assert outside["days_since_delivery"] == 11
    assert outside["within_return_window"] is False


def test_list_orders_returns_most_recent_first() -> None:
    orders = list_orders(OTHER_CUSTOMER)
    assert [row["order_id"] for row in orders] == [20, 2]
    # Enough to answer "onde esta meu pedido?" without a second tool call.
    assert orders[1]["tracking_code"] == "BRCX9876543BR"
    assert orders[1]["estimated_delivery"] == "2025-11-15"
    assert orders[0]["tracking_code"] is None  # cancelled


def test_list_orders_filters_by_status_in_either_language() -> None:
    assert [row["order_id"] for row in list_orders(OTHER_CUSTOMER, status="cancelled")] == [20]
    # A small model will write the Portuguese word it just read in the chat.
    assert [row["order_id"] for row in list_orders(OTHER_CUSTOMER, status="cancelado")] == [20]
    assert list_orders(OTHER_CUSTOMER, status="pending") == []


def test_order_items_carry_no_unit_price() -> None:
    """The dataset holds only the order total and today's catalog price, which
    may have changed since. Leaving unit prices out means the agent cannot state
    what was paid per item, so it will not invent it."""
    order = get_order(2, OTHER_CUSTOMER, today=REFERENCE_DATE)
    assert set(order["items"][0]) == {"product_id", "name", "quantity"}


# --- customer scope and PII --------------------------------------------------


def test_foreign_order_is_indistinguishable_from_a_missing_one() -> None:
    """The negative answer must not reveal which orders exist.

    Order 1 exists and belongs to customer 3; order 999 does not exist at all.
    Asked as customer 7, both must come back identical. Otherwise trying every
    order_id from 1 to 20 reveals which orders exist and which belong to other
    people.
    """
    foreign = get_order(ORDER_OF_CUSTOMER_3, OTHER_CUSTOMER, today=REFERENCE_DATE)
    missing = get_order(999, OTHER_CUSTOMER, today=REFERENCE_DATE)
    assert foreign == missing == ORDER_NOT_FOUND
    assert foreign["found"] is False
    # Same shape for an id that is not a number at all.
    assert get_order("abc", OTHER_CUSTOMER) == ORDER_NOT_FOUND  # type: ignore[arg-type]


@pytest.mark.parametrize("missing", [None, "", "   "])
def test_order_queries_refuse_an_unidentified_customer(missing: object) -> None:
    """`customer_id` has no default on purpose: were it `None` meaning "no
    filter", one omitted argument would make the privacy filter disappear
    silently."""
    with pytest.raises(MissingCustomerIdError):
        get_order(ORDER_OF_CUSTOMER_3, missing)  # type: ignore[arg-type]
    with pytest.raises(MissingCustomerIdError):
        list_orders(missing)  # type: ignore[arg-type]
    with pytest.raises(MissingCustomerIdError):
        load_customer(missing)  # type: ignore[arg-type]


def test_customer_without_orders_gets_an_empty_list_not_an_error() -> None:
    """32 of the 50 customers are in this position, so "you have no orders here
    yet" is a correct and frequent answer, and has to read as an answer."""
    assert list_orders(CUSTOMER_WITHOUT_ORDERS) == []
    assert load_customer(CUSTOMER_WITHOUT_ORDERS) is not None


def test_order_payloads_carry_no_pii(dataset: Dataset) -> None:
    """Not even the requesting customer's own name, phone or e-mail.

    They already know their phone number, and putting it in a tool result only
    increases the risk of exposure across the conversation history and the
    committed transcripts (policy manual, section 9).
    """
    secrets = set(dataset.customers["phone"]) | set(dataset.customers["email"])
    names = set(dataset.customers["name"])

    payloads = [
        repr(get_order(order_id, customer_id, today=REFERENCE_DATE))
        for order_id, customer_id in zip(dataset.orders["order_id"], dataset.orders["customer_id"])
    ]
    payloads += [repr(list_orders(customer_id)) for customer_id in dataset.customers["customer_id"]]
    payloads += [repr(search_products(limit=20)), repr(list_categories())]

    blob = " ".join(payloads)
    assert not [secret for secret in secrets if secret in blob]
    assert not [name for name in names if name in blob]
    assert "phone" not in blob and "email" not in blob


def test_load_customer_projects_only_the_name() -> None:
    """Session bootstrap: a greeting by name, and nothing else. It is never
    registered as a tool, so the model cannot look a customer up at all.

    Not the registered city either: the shipping rules depend on where the order
    is going, which the checkout asks for, and not on where the customer lives.
    """
    customer = load_customer(3)
    assert customer == {"customer_id": 3, "name": "Pedro Henrique Oliveira"}
    assert load_customer(9999) is None


# --- tool schemas ------------------------------------------------------------


def test_order_schemas_do_not_expose_the_customer() -> None:
    """The structural half of the privacy argument.

    The model can ask for an order by number and nothing else. `customer_id` is
    injected by the executor from the session, so it cannot be persuaded to read
    someone else's orders, because there is no field to put the claim in.
    """
    for model in (GetOrderInput, ListOrdersInput):
        properties = tool_schema(model)["properties"]
        assert "customer_id" not in properties
        assert not any("customer" in name for name in properties)


def test_schemas_are_clean_prompt_text() -> None:
    schema = tool_schema(SearchProductsInput)
    assert "title" not in schema
    assert "description" not in schema  # the English docstring must not leak
    assert schema["additionalProperties"] is False
    for name, field in schema["properties"].items():
        assert "title" not in field, name
        # Optional fields are plain types, not unions with null: a 4B model
        # reading `anyOf: [string, null]` is likelier to emit "null" as a string.
        assert "anyOf" not in field, name
        assert field.get("description"), name


def test_schemas_validate_what_the_functions_accept() -> None:
    args = SearchProductsInput(category="Violões", price_max=800).model_dump(exclude_none=True)
    assert search_products(**args)
    with pytest.raises(ValueError):
        SearchProductsInput(limit=99)  # above MAX_SEARCH_LIMIT
    with pytest.raises(ValueError):
        SearchProductsInput(categoria="Violões")  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        GetProductInput()  # neither identifier
