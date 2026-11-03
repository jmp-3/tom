"""The query functions of `emporio.data`, and the policy retriever, as tools.

Nothing here queries anything: every function below forwards to
`emporio.data.queries` -- or, for the policies, to `emporio.rag` -- and returns
what it gets. The wrapper exists for the three things the model should not see.

*   `dataset` and `today`, the keyword-only arguments the test suite injects,
    are not parameters here, so they never reach the schema.
*   `customer_id` is not a parameter either. It arrives through
    `runtime.context` (see `emporio.agent.context`), and LangChain strips the
    `runtime` parameter from the schema, so the model cannot fill it in.
*   Optional arguments get a typed default instead of `| None`. A union with
    null in the schema is measurably likelier to make a small model emit the
    string "null" than to omit the argument (see `test_agent.py` and
    `data/models.py`), so "no filter" is spelled `""` or `0` here and mapped
    back to None on the way in.

Each tool is defined by **its signature and its docstring, never by
`args_schema`**: a tool that declares `args_schema` does not get its
`ToolRuntime` injected, which would break the privacy rule in silence.
`parse_docstring=True` turns the `Args:` section into the argument descriptions.

The docstrings are in English, like the rest of the code; what the customer
reads -- the persona, the answers, the labels inside the payloads -- stays in
Portuguese.
"""

from __future__ import annotations

from typing import Any, Final

from langchain.tools import BaseTool, ToolRuntime, tool

from emporio.agent.context import SessionContext
from emporio.data import queries
from emporio.data.models import (
    DEFAULT_ORDER_LIMIT,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SIMILAR_LIMIT,
    MAX_SEARCH_LIMIT,
)
from emporio.rag import policy_retriever

# Returned when the model asks for a product that does not exist, or asks
# without saying which one. Same shape as `queries.ORDER_NOT_FOUND`, so "not
# found" looks the same to the model whatever it asked about. The `hint` is
# model-facing instruction, hence Portuguese.
NOT_FOUND: Final[dict[str, Any]] = {"found": False}


def _session_customer_id(runtime: ToolRuntime[SessionContext]) -> Any:
    """The customer the session was opened for, or None.

    None reaches `queries`, which raises `MissingCustomerIdError`: the order
    tools are not even offered to the model without an identified customer (see
    `middleware.scope_tools_to_session`), so getting here means the filter
    failed, and the data layer refusing is the second lock.
    """
    return getattr(runtime.context, "customer_id", None)


@tool(parse_docstring=True)
def search_products(
    text: str = "",
    category: str = "",
    price_min: float = 0,
    price_max: float = 0,
    only_available: bool = True,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> dict[str, Any]:
    """Search the store catalog. Use it for any question about what the store sells.

    Returns `products` (cheapest first, prices already formatted), `shown` with
    how many of them came back, and `more`, which is true when the catalog holds
    further matches beyond this page. With `more` true the list is a sample, not
    the whole answer, and must not be presented as everything the store has.

    Args:
        text: The customer's own words, e.g. 'violão nylon iniciante' or 'Yamaha'.
        category: One catalog category: Guitarras, Baixos, Baterias e Percussão,
            Teclados e Pianos, Violões, Ukuleles. Empty searches every category.
        price_min: Lowest price in reais, 0 for no minimum.
        price_max: Highest price in reais, 0 for no maximum. Compares against the
            price the customer would pay today.
        only_available: True returns only what is on sale and in stock. Use false
            only to look up a specific item that may be sold out or discontinued.
        limit: How many products to return at most.
    """
    # Asked for the maximum and sliced here, so that "there are more" is a fact
    # from the data instead of something the model has to infer from a full page.
    # Without it the model reports the page as the catalog: measured on
    # `claude-haiku-4-5`, "8 violões até R$ 1.000" for a filter that matches 12.
    # The edge is a filter with more than `MAX_SEARCH_LIMIT` matches, where
    # `more` is true but the count behind it is itself capped -- which is why
    # this reports a flag and never a total.
    wanted = max(1, min(int(limit), MAX_SEARCH_LIMIT))
    found = queries.search_products(
        text=text or None,
        category=category or None,
        price_min=price_min or None,
        price_max=price_max or None,
        only_available=only_available,
        limit=MAX_SEARCH_LIMIT,
    )
    products = found[:wanted]
    return {"products": products, "shown": len(products), "more": len(found) > len(products)}


@tool(parse_docstring=True)
def get_product(product_id: int = 0, name: str = "") -> dict[str, Any]:
    """Everything about one product: specs, stock, availability and every price.

    Use it when the customer asks about a specific instrument. Pass the id when a
    search already returned one, the name otherwise.

    Args:
        product_id: The product code, 0 when it is unknown.
        name: The product name, whole or in part, when the code is unknown.
    """
    if product_id:
        found = queries.get_product(int(product_id))
    elif name.strip():
        found = queries.get_product(name=name)
    else:
        return NOT_FOUND | {"hint": "Informe product_id ou name."}
    return found if found is not None else dict(NOT_FOUND)


@tool(parse_docstring=True)
def list_categories() -> list[dict[str, Any]]:
    """List the catalog categories that have products, and how many each has.

    Use it when the customer asks what kinds of instrument the store sells, or
    before searching, when it is unclear which category they mean.
    """
    return queries.list_categories()


@tool(parse_docstring=True)
def similar_products(product_id: int, limit: int = DEFAULT_SIMILAR_LIMIT) -> list[dict[str, Any]]:
    """Available alternatives to a product: same category, closest price first.

    Use it whenever a product is sold out or discontinued, to offer an equivalent.

    Args:
        product_id: The code of the product the customer asked about.
        limit: How many alternatives to return at most.
    """
    return queries.similar_products(product_id, limit=limit)


@tool(parse_docstring=True)
def list_orders(
    runtime: ToolRuntime[SessionContext],
    status: str = "",
    limit: int = DEFAULT_ORDER_LIMIT,
) -> list[dict[str, Any]]:
    """This customer's orders, most recent first. Use it for 'quais são meus pedidos?'.

    Only when the customer gave no order number: when they did, use `get_order`
    instead. Returns an empty list when the customer has no order, and only ever
    the orders of the customer this session belongs to.

    Args:
        status: Keep only one situation: pending, confirmed, shipped, delivered
            or cancelled. Empty returns every order.
        limit: How many orders to return at most.
    """
    return queries.list_orders(_session_customer_id(runtime), status=status or None, limit=limit)


@tool(parse_docstring=True)
def get_order(order_id: int, runtime: ToolRuntime[SessionContext]) -> dict[str, Any]:
    """Status, items, payment and delivery window of one of this customer's orders.

    Always use this one, and not `list_orders`, when the customer names an order
    number. `found: false` comes back both when the order does not exist and when
    it is not this customer's, and the two must be answered the same way.

    Args:
        order_id: The order number the customer gave.
    """
    return queries.get_order(order_id, _session_customer_id(runtime))


@tool(parse_docstring=True)
def search_policies(question: str) -> dict[str, Any]:
    """The store's policy manual: returns, exchanges, warranty, shipping, payment, hours.

    Use it for any question about a store rule instead of about a product or an
    order: how many days there are to return or exchange something, what the
    warranty covers, what shipping costs and how long it takes, which payment
    methods and installments exist, opening hours, how promotions combine, what
    happens to personal data.

    Returns the closest excerpts of the manual, each with the `section` that has
    to be cited in the answer. An excerpt that does not answer the question is
    not an answer: say what is missing instead of filling the gap.

    Args:
        question: The customer's question, in their own words.
    """
    # Nothing is caught here. A missing index or a refused embedding call is an
    # exception, which `middleware.tool_error_to_message` turns into a message
    # the model can read -- the same path every other tool takes, and the reason
    # the operator gets the warning at startup instead.
    found = policy_retriever().invoke(question)
    excerpts = [
        {
            "section": document.metadata["section"],
            "title": document.metadata["title"],
            "text": document.page_content,
        }
        for document in found
    ]
    return {"excerpts": excerpts, "shown": len(excerpts)}


# The catalog: public information, offered in every session.
CATALOG_TOOLS: Final[list[BaseTool]] = [
    search_products,
    get_product,
    list_categories,
    similar_products,
]

# The policy manual: also public, and also offered in every session -- an
# anonymous visitor asking about the return window is asking about a published
# store rule, not about anyone's data.
POLICY_TOOLS: Final[list[BaseTool]] = [search_policies]

PUBLIC_TOOLS: Final[list[BaseTool]] = [*CATALOG_TOOLS, *POLICY_TOOLS]

# Personal data of one customer: offered only when the session is identified.
ORDER_TOOLS: Final[list[BaseTool]] = [list_orders, get_order]

ORDER_TOOL_NAMES: Final = frozenset(instrument.name for instrument in ORDER_TOOLS)

# Everything is registered on the agent; the middleware decides, per model call,
# which of them that call is allowed to see.
STORE_TOOLS: Final[list[BaseTool]] = [*PUBLIC_TOOLS, *ORDER_TOOLS]
