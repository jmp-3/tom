"""The agent: the chat model, the tools and the loop that connects them."""

from emporio.agent.agent import build_agent, session_persona
from emporio.agent.context import ANONYMOUS, SessionContext, for_customer
from emporio.agent.middleware import (
    FAILED_STOP_REASONS,
    HEALTHY_STOP_REASONS,
    StopReasonState,
    failures,
    log_offered_tools,
    record_stop_reason,
    scope_tools_to_session,
    tally,
    tool_error_to_message,
)
from emporio.agent.prompt import PERSONA, as_user_prefix, persona, session_block
from emporio.agent.tools import (
    CATALOG_TOOLS,
    ORDER_TOOLS,
    POLICY_TOOLS,
    PUBLIC_TOOLS,
    STORE_TOOLS,
)

__all__ = [
    "ANONYMOUS",
    "CATALOG_TOOLS",
    "FAILED_STOP_REASONS",
    "HEALTHY_STOP_REASONS",
    "ORDER_TOOLS",
    "PERSONA",
    "POLICY_TOOLS",
    "PUBLIC_TOOLS",
    "STORE_TOOLS",
    "SessionContext",
    "StopReasonState",
    "as_user_prefix",
    "build_agent",
    "failures",
    "for_customer",
    "log_offered_tools",
    "persona",
    "record_stop_reason",
    "scope_tools_to_session",
    "session_block",
    "session_persona",
    "tally",
    "tool_error_to_message",
]
