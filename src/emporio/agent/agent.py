"""The agent loop.

`create_agent` returns a compiled LangGraph graph: it runs the model, routes a
tool call to the tools and feeds the result back, until the model answers
without calling anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel

from emporio import bedrock
from emporio.agent.middleware import record_stop_reason, tool_error_to_message
from emporio.agent.toy_tools import TOY_TOOLS

# Deliberately plain, and in Portuguese because the model reads it. The persona
# will be refined later.
SYSTEM_PROMPT = (
    "Você é o atendente virtual do Empório da Música, uma loja de instrumentos "
    "musicais em Campo Grande/MS. Responda em português do Brasil, de forma "
    "direta.\n\n"
    "Use as ferramentas disponíveis para consultar dados da loja (produtos, "
    "preços, estoque, pedidos) e responda com base no que elas devolverem. "
    "Quando a pergunta não for sobre dados da loja, responda direto, sem chamar "
    "ferramenta nenhuma."
)


def build_agent(
    *,
    model: BaseChatModel | None = None,
    tools: Sequence[BaseTool] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
):
    """Build the agent graph.

    Everything is an argument with a default so that a test or a probe can swap
    the model (the fallback ladder) or the tools (the fault-injection tool)
    without a second `create_agent` call existing anywhere.
    """
    # Order matters: the tool-error middleware is outermost, so an exception
    # inside a tool becomes a message before anything else sees it.
    stack: list[AgentMiddleware[Any, Any, Any]] = [tool_error_to_message, record_stop_reason]
    return create_agent(
        model=model if model is not None else bedrock.chat_model(),
        tools=list(TOY_TOOLS if tools is None else tools),
        system_prompt=system_prompt,
        middleware=stack,
    )
