"""Input schemas for the query functions."""

from __future__ import annotations

from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

OrderStatus = Literal["pending", "confirmed", "shipped", "delivered", "cancelled"]

DEFAULT_SEARCH_LIMIT: Final = 8
MAX_SEARCH_LIMIT: Final = 20
DEFAULT_ORDER_LIMIT: Final = 10
DEFAULT_SIMILAR_LIMIT: Final = 3


class ToolInput(BaseModel):
    """Base for every tool input.

    `extra="forbid"` turns an invented argument into a validation error instead
    of a silently ignored one.
    """

    model_config = ConfigDict(extra="forbid")


class SearchProductsInput(ToolInput):
    category: str | None = Field(
        default=None,
        description=(
            "Categoria do catálogo, pelo nome (ex.: 'Violões', 'Ukuleles', "
            "'Guitarras'). Use a ferramenta de categorias para ver as válidas."
        ),
    )
    text: str | None = Field(
        default=None,
        description=(
            "Palavras do cliente para buscar no nome, na descrição e nas "
            "especificações (ex.: 'nylon iniciante', 'Yamaha')."
        ),
    )
    price_min: float | None = Field(
        default=None,
        ge=0,
        description="Preço mínimo em reais.",
    )
    price_max: float | None = Field(
        default=None,
        ge=0,
        description="Preço máximo em reais, comparado ao preço com desconto já aplicado.",
    )
    only_available: bool = Field(
        default=True,
        description=(
            "Se verdadeiro (padrão), devolve só produtos à venda e com estoque. "
            "Use falso apenas quando o cliente perguntar por um item específico "
            "que pode estar esgotado ou fora de linha."
        ),
    )
    limit: int = Field(
        default=DEFAULT_SEARCH_LIMIT,
        ge=1,
        le=MAX_SEARCH_LIMIT,
        description="Quantos produtos devolver, no máximo.",
    )


class GetProductInput(ToolInput):
    product_id: int | None = Field(default=None, ge=1, description="Código do produto.")
    name: str | None = Field(
        default=None,
        description="Nome do produto, inteiro ou em parte, quando não se tem o código.",
    )

    @model_validator(mode="after")
    def _exactly_one_identifier(self) -> GetProductInput:
        if (self.product_id is None) == (self.name is None):
            raise ValueError("informe product_id ou name, apenas um dos dois")
        return self


class ListCategoriesInput(ToolInput):
    """No arguments. Bedrock still requires a schema, hence the empty model."""


class ListOrdersInput(ToolInput):
    """Note the absence of `customer_id`: the session injects it, not the model."""

    status: OrderStatus | None = Field(
        default=None,
        description=(
            "Filtra por situação do pedido: pending (aguardando pagamento), "
            "confirmed (pagamento confirmado), shipped (a caminho), "
            "delivered (entregue), cancelled (cancelado)."
        ),
    )
    limit: int = Field(
        default=DEFAULT_ORDER_LIMIT,
        ge=1,
        le=MAX_SEARCH_LIMIT,
        description="Quantos pedidos devolver, no máximo, do mais recente para o mais antigo.",
    )


class GetOrderInput(ToolInput):
    """Note the absence of `customer_id`: the session injects it, not the model."""

    order_id: int = Field(ge=1, description="Número do pedido informado pelo cliente.")


class SimilarProductsInput(ToolInput):
    product_id: int = Field(ge=1, description="Código do produto para o qual se quer alternativas.")
    limit: int = Field(
        default=DEFAULT_SIMILAR_LIMIT,
        ge=1,
        le=MAX_SEARCH_LIMIT,
        description="Quantas alternativas devolver, no máximo.",
    )


def _collapse_optional(field: dict[str, Any]) -> dict[str, Any]:
    """Turn `anyOf: [{type: string}, {type: null}]` back into `type: string`.

    Pydantic spells `str | None` as a two-branch union, which is correct JSON
    Schema and bad prompt engineering: a 4B model reading a union of a type and
    null is measurably likelier to emit the string "null" than to omit the
    argument. An optional field is already optional by not being in `required`.
    """
    branches = field.get("anyOf")
    if not branches or len(branches) != 2:
        return field
    concrete = [b for b in branches if b.get("type") != "null"]
    if len(concrete) != 1 or len(branches) - len(concrete) != 1:
        return field
    collapsed = {k: v for k, v in field.items() if k != "anyOf"}
    if collapsed.get("default") is None:
        collapsed.pop("default", None)
    return concrete[0] | collapsed


def tool_schema(model: type[ToolInput]) -> dict[str, Any]:
    """JSON Schema for a tool input, ready for a Bedrock `toolSpec`'s `inputSchema`."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema.pop("description", None)
    properties = schema.setdefault("properties", {})
    for name, field in properties.items():
        field.pop("title", None)
        properties[name] = _collapse_optional(field)
    return schema
