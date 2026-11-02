"""The agent loop.

`create_agent` returns a compiled LangGraph graph: it runs the model, routes a
tool call to the tools and feeds the result back, until the model answers
without calling anything.

Three features of the framework do the work here that would otherwise be code:
`context_schema` plus `ToolRuntime` keep `customer_id` out of the model's reach,
`wrap_model_call` decides which tools each call is offered, and `system_prompt`
carries the persona.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, dynamic_prompt
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from emporio import bedrock
from emporio.agent.context import SessionContext
from emporio.agent.middleware import (
    record_stop_reason,
    scope_tools_to_session,
    tool_error_to_message,
)
from emporio.agent.prompt import PERSONA, session_block
from emporio.agent.tools import STORE_TOOLS


@dynamic_prompt
def session_persona(request: ModelRequest[SessionContext]) -> str:
    """Complete the persona with the half that depends on the session.

    The fixed persona travels in `system_prompt`, which is the path the framework
    expects; what cannot travel there is the part that depends on the session --
    the customer's first name, and whether order questions are answerable at all.
    This appends it per model call, reading the same `system_prompt` back off the
    request so the persona is never written down twice.
    """
    base = request.system_message.text if request.system_message is not None else PERSONA
    return f"{base}\n\n{session_block(request.runtime.context)}"


def build_agent(
    *,
    model: BaseChatModel | None = None,
    tools: Sequence[BaseTool] | None = None,
    system_prompt: str = PERSONA,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    extra_middleware: Sequence[AgentMiddleware[Any, Any, Any]] = (),
):
    """Build the agent graph.

    Everything is an argument with a default so that a test or a probe can swap
    the model (the fallback ladder), the tools (the fault-injection tool) or the
    checkpointer without a second `create_agent` call existing anywhere.

    `extra_middleware` goes last, which makes it the innermost layer: the CLI's
    `--debug` logger belongs there, after the tool filter has already decided
    what the model is allowed to see.
    """
    # Order matters: the tool-error middleware is outermost, so an exception
    # inside a tool becomes a message before anything else sees it. The tool
    # filter comes before the prompt so that an anonymous session cannot be
    # offered an order tool whatever the prompt ends up saying.
    stack: list[AgentMiddleware[Any, Any, Any]] = [
        tool_error_to_message,
        scope_tools_to_session,
        session_persona,
        record_stop_reason,
        *extra_middleware,
    ]
    return create_agent(
        model=model if model is not None else bedrock.chat_model(),
        tools=list(STORE_TOOLS if tools is None else tools),
        system_prompt=system_prompt,
        middleware=stack,
        # The session's identity, which the tools read and the model cannot.
        context_schema=SessionContext,
        checkpointer=checkpointer,
    )
