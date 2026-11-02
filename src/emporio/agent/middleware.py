"""Middleware for the agent loop.

`record_stop_reason`, because `create_agent` owns the loop and there is no
`while` of ours left to branch on `stopReason`. Bedrock still reports it on
every model call, and its failure values are the measurement that decides
whether the model is good enough.

`scope_tools_to_session`, because the two session modes differ in the *kind* of
data they may touch: public catalog information in both, one customer's personal
data only when the session carries that customer. The filter is applied to the
request, so the order tools are not merely refused in an anonymous session --
they are never offered.

`tool_error_to_message`, because a broken tool must not end a conversation.
`ToolNode` converts argument and validation errors into a message by itself, but
as of langgraph 1.2 its default handler re-raises anything else, so an exception
inside a tool propagates and kills the run. Handling is opt-in by design, and
this is the opt-in.
"""

from __future__ import annotations

import operator
from collections import Counter
from collections.abc import Callable
from typing import Annotated, Any, Final

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    Runtime,
    ToolCallRequest,
    ToolErrorMiddleware,
    after_model,
    wrap_model_call,
)
from langchain.tools import BaseTool

from emporio.agent.context import SessionContext
from emporio.agent.tools import ORDER_TOOL_NAMES

# Bedrock puts the reason in `AIMessage.response_metadata["stopReason"]`.
# These two are the healthy outcomes; everything else is a failure of some kind.
HEALTHY_STOP_REASONS: Final = frozenset({"end_turn", "tool_use", "stop_sequence"})

# The failures worth telling apart. `malformed_tool_use` is the number this
# project cares about: it is a 4B model choosing among six tools in Portuguese.
FAILED_STOP_REASONS: Final = frozenset(
    {
        "malformed_tool_use",
        "malformed_model_output",
        "max_tokens",
        "model_context_window_exceeded",
        "guardrail_intervened",
        "content_filtered",
    }
)


class StopReasonState(AgentState):
    """Agent state plus one `stopReason` per model call, in order.

    State rather than a counter on the side: the list comes back with the run,
    so a caller reads its own numbers and concurrent runs cannot mix.
    """

    stop_reasons: Annotated[list[str], operator.add]


@after_model(state_schema=StopReasonState, name="StopReasonMiddleware")
def record_stop_reason(state: StopReasonState, runtime: Runtime) -> dict[str, Any]:
    """Append the `stopReason` of the model call that just finished."""
    metadata = getattr(state["messages"][-1], "response_metadata", {}) or {}
    return {"stop_reasons": [metadata.get("stopReason") or "unknown"]}


def tally(stop_reasons: list[str]) -> Counter[str]:
    """Count stop reasons by value. The eval reports the failed ones."""
    return Counter(stop_reasons)


def failures(stop_reasons: list[str]) -> Counter[str]:
    """Count only the stop reasons that mean something went wrong."""
    return Counter(r for r in stop_reasons if r not in HEALTHY_STOP_REASONS)


# What the model is told when a tool raises. Portuguese, because the model reads
# it, and the exception type only: the message of a pandas or botocore error
# carries file paths and internals that have no business in a conversation with
# a customer. The customer-facing wording is the model's to choose; this is the
# instruction behind it.
TOOL_ERROR_TEMPLATE = (
    "A consulta falhou por um erro interno do sistema ({tipo}). Não tente a "
    "mesma consulta de novo: diga ao cliente que não foi possível verificar "
    "isso agora e ofereça outro caminho."
)


def _tool_error_content(error: Exception, request: ToolCallRequest) -> str:
    return TOOL_ERROR_TEMPLATE.format(tipo=type(error).__name__)


# Every exception is turned into a message, because the alternative is a
# conversation that ends mid-sentence. The resulting `ToolMessage` carries
# `status="error"`, which is how the eval counts them.
tool_error_to_message = ToolErrorMiddleware(on_error=_tool_error_content)


def _tool_name(instrument: BaseTool | dict[str, Any]) -> str:
    """The name of a bound tool, which may already be a provider-shaped dict."""
    if isinstance(instrument, dict):
        return str(instrument.get("name", ""))
    return instrument.name


def _tool_signature(instrument: BaseTool | dict[str, Any]) -> str:
    """`name(arg, arg)` as the model sees it, for the `--debug` output.

    From `BaseTool.args`, which is the schema actually sent, not the Python
    signature: `runtime` is absent there, and so is `customer_id`. The pitfall
    this prints against is a tool that exposes `customer_id` by accident -- worth
    reading off the request rather than trusting the intention.
    """
    if isinstance(instrument, dict):
        return _tool_name(instrument)
    return f"{instrument.name}({', '.join(instrument.args)})"


@wrap_model_call(name="ToolScopeMiddleware")
def scope_tools_to_session(
    request: ModelRequest[SessionContext],
    handler: Callable[[ModelRequest[SessionContext]], ModelResponse],
) -> ModelCallResult:
    """Keep the order tools out of the request when the session is anonymous.

    Every tool is registered on the agent; this decides which ones *that* call
    is allowed to see. Not sending a tool is the access control: the model has no
    way to call what it was not given, and therefore no way to leak what it
    cannot look up. Stronger than a refusal written in the prompt -- a 4B-class
    model is not an access lock -- and it costs tokens rather than spending them.
    """
    if getattr(request.runtime.context, "customer_id", None) is not None:
        return handler(request)
    public = [t for t in request.tools if _tool_name(t) not in ORDER_TOOL_NAMES]
    return handler(request.override(tools=public))


def log_offered_tools(sink: Callable[[str], None]) -> AgentMiddleware[Any, Any, Any]:
    """Middleware that reports the tools of each model request to `sink`.

    A factory and not a constant because the destination is the caller's: the CLI
    prints to the terminal under `--debug`, a test collects into a list. It runs
    inside `scope_tools_to_session`, so what it reports is what the model was
    really offered, after the filtering.
    """

    @wrap_model_call(name="ToolLogMiddleware")
    def log(
        request: ModelRequest[SessionContext],
        handler: Callable[[ModelRequest[SessionContext]], ModelResponse],
    ) -> ModelCallResult:
        sink(" · ".join(_tool_signature(t) for t in request.tools) or "nenhuma ferramenta")
        return handler(request)

    return log
