"""Middleware for the agent loop.

`record_stop_reason`, because `create_agent` owns the loop and there is no
`while` of ours left to branch on `stopReason`. Bedrock still reports it on
every model call, and its failure values are the measurement that decides 
whether the model is good enough.

`tool_error_to_message`, because a broken tool must not end a conversation.
`ToolNode` converts argument and validation errors into a message by itself, but
as of langgraph 1.2 its default handler re-raises anything else, so an exception
inside a tool propagates and kills the run. Handling is opt-in by design, and
this is the opt-in.
"""

from __future__ import annotations

import operator
from collections import Counter
from typing import Annotated, Any, Final

from langchain.agents.middleware import (
    AgentState,
    Runtime,
    ToolCallRequest,
    ToolErrorMiddleware,
    after_model,
)

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
