"""The agent: the chat model, the tools and the loop that connects them."""

from emporio.agent.agent import SYSTEM_PROMPT, build_agent
from emporio.agent.middleware import (
    FAILED_STOP_REASONS,
    HEALTHY_STOP_REASONS,
    StopReasonState,
    failures,
    record_stop_reason,
    tally,
    tool_error_to_message,
)

__all__ = [
    "FAILED_STOP_REASONS",
    "HEALTHY_STOP_REASONS",
    "SYSTEM_PROMPT",
    "StopReasonState",
    "build_agent",
    "failures",
    "record_stop_reason",
    "tally",
    "tool_error_to_message",
]
