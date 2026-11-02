"""Throwaway tools, used to prove the agent loop before the real ones exist.

Nothing here reads the store data: the point is to test the mechanics of tool
calling -- does the model pick a tool, with valid arguments, and then answer from
what came back -- with no data layer in the picture to blame. The one thing it
does borrow is `brl`, the price formatter, because the toy tool has to have the
same shape as the real ones on the one detail the model gets wrong. This module
goes away once the query functions are wired in.

Note the shape, which is the shape every real tool will have: types in the
signature plus a docstring, never an `args_schema`. A tool that declares
`args_schema` does not get a `ToolRuntime` injected, and `ToolRuntime` is how
`customer_id` will reach the tools without the model ever seeing it.
"""

from __future__ import annotations

from typing import Any, Final

from langchain.tools import BaseTool, tool

from emporio.data.loader import brl

# Fixed answer: the test is the tool call, not the number. A stub that invented
# plausible stock counts would make a wrong answer look right.
STUB_STOCK: Final = 3
STUB_PRICE: Final = 1299.0


# The docstrings below are in Portuguese because the model reads them: with a
# tool defined by its signature, the docstring *is* the tool description, and
# `parse_docstring=True` turns the `Args:` section into the argument
# descriptions. Everything the model reads stays in the customer's language.
@tool(parse_docstring=True)
def consultar_estoque(produto: str) -> dict[str, Any]:
    """Consulta quantas unidades de um produto há no estoque da loja e por qual preço.

    Args:
        produto: Nome do produto como o cliente escreveu, por exemplo
            'violão de nylon' ou 'teclado Yamaha'.
    """
    return {
        "produto": produto,
        "unidades_em_estoque": STUB_STOCK,
        "preco": STUB_PRICE,
        # The string the model is meant to quote. Handed the float alone it
        # answers "R$ 129,90" for 1299.0; the data layer's `brl` is what the
        # real tools use for the same reason.
        "preco_formatado": brl(STUB_PRICE),
    }


@tool(parse_docstring=True)
def falhar_de_proposito(motivo: str) -> str:
    """Ferramenta de teste que sempre falha. Nunca use para atender um cliente.

    Args:
        motivo: Qualquer texto.
    """
    raise RuntimeError(f"falha proposital: {motivo}")


TOY_TOOLS: Final[list[BaseTool]] = [consultar_estoque]
