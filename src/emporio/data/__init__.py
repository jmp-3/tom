"""Data layer: loading, derived business fields, and the query functions.

No LLM call happens anywhere below this package. Every price, stock count and
deadline the agent states is computed here, deterministically and under test.
"""

from emporio.data.loader import (
    Dataset,
    DatasetError,
    days_since_delivery,
    load_dataset,
    within_return_window,
)
from emporio.data.models import (
    GetOrderInput,
    GetProductInput,
    ListCategoriesInput,
    ListOrdersInput,
    SearchProductsInput,
    SimilarProductsInput,
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

__all__ = [
    "ORDER_NOT_FOUND",
    "Dataset",
    "DatasetError",
    "GetOrderInput",
    "GetProductInput",
    "ListCategoriesInput",
    "ListOrdersInput",
    "MissingCustomerIdError",
    "SearchProductsInput",
    "SimilarProductsInput",
    "days_since_delivery",
    "get_order",
    "get_product",
    "list_categories",
    "list_orders",
    "load_customer",
    "load_dataset",
    "search_products",
    "similar_products",
    "tool_schema",
    "within_return_window",
]
