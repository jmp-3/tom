"""The agent's mechanics: tool shape, tool errors and the stopReason record.

No billed call happens here. What is tested is everything about the loop that
can be checked without the model: that the tools expose to the model exactly
what they should, that a tool raising an exception turns into a message instead
of an exception, and that the stop reasons are recorded and classified. The
model's *judgement* is not testable offline -- that is what the gate in
`scripts/smoke_tool_use.py` is for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from langchain.agents.middleware import AgentState
from langchain.tools import BaseTool, ToolRuntime, tool
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode

from emporio.agent import SYSTEM_PROMPT, build_agent
from emporio.agent.middleware import TOOL_ERROR_TEMPLATE, failures, tally
from emporio.agent.toy_tools import TOY_TOOLS, consultar_estoque, falhar_de_proposito

TOOL_CALL = {
    "name": "falhar_de_proposito",
    "args": {"motivo": "teste"},
    "id": "call_1",
    "type": "tool_call",
}


class ScriptedModel(BaseChatModel):
    """A chat model that replays prepared answers, so the loop runs offline.

    The loop itself -- model, tool, model again -- is worth testing without
    Bedrock in the way: it costs nothing, it cannot flake, and it is the only
    way to assert what happens around a tool that raises without depending on
    the model agreeing to call it.
    """

    script: list[AIMessage] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


@dataclass
class Session:
    """Stand-in for the session context."""

    customer_id: int


def _bound_tools() -> list[BaseTool]:
    """The tools the agent actually binds."""
    return list(TOY_TOOLS)


def _model_facing_schema(instrument: BaseTool) -> dict[str, Any]:
    return convert_to_openai_tool(instrument)["function"]["parameters"]


class TestToolShape:
    def test_docstring_becomes_the_description(self) -> None:
        assert "estoque" in consultar_estoque.description
        schema = _model_facing_schema(consultar_estoque)
        # `parse_docstring=True` is what carries the `Args:` section into the
        # schema; without it the model sees a bare `{"type": "string"}`.
        assert "description" in schema["properties"]["produto"]

    def test_no_optional_argument_reaches_the_model_as_a_null_union(self) -> None:
        """`str | None` in a signature is a malformed tool call waiting to happen.

        Measured, not assumed: on `us.amazon.nova-micro-v1:0`, a tool whose only
        argument was typed `str | None` produced
        `ModelErrorException: Model produced invalid sequence as part of ToolUse`
        in 3 of 5 runs; the same tool with `status: str = ""` ran 5 of 5 clean.
        An argument that may be omitted gets a type and an empty default, and a
        closed set of values gets a `Literal`.
        """
        for instrument in _bound_tools():
            for name, field in _model_facing_schema(instrument)["properties"].items():
                branches = field.get("anyOf", [])
                nullable = [b for b in branches if b.get("type") == "null"]
                assert not nullable, f"{instrument.name}.{name} is a union with null"

    def test_tool_runtime_stays_invisible_to_the_model(self) -> None:
        """
        A tool defined by its signature gets its `ToolRuntime` injected and the
        parameter never appears in the schema. Declaring `args_schema` instead
        would silently skip the injection, which is why no tool in this project
        declares one.
        """

        @tool(parse_docstring=True)
        def meus_pedidos(runtime: ToolRuntime[Session], limite: int = 5) -> dict[str, Any]:
            """Lista os pedidos do cliente.

            Args:
                limite: Quantos pedidos devolver.
            """
            return {"customer_id": runtime.context.customer_id, "limite": limite}

        assert "runtime" not in meus_pedidos.args
        assert set(_model_facing_schema(meus_pedidos)["properties"]) == {"limite"}


class TestToolErrors:
    def test_the_tool_node_alone_lets_the_exception_through(self) -> None:
        """Why the middleware exists, pinned to the version that made it necessary.

        As of langgraph 1.2 the `ToolNode` default handler returns a message for
        an argument error and re-raises everything else. If a later version goes
        back to catching exceptions, this test fails and the middleware in
        `build_agent` can be reconsidered -- which is the point of asserting it.
        """
        builder = StateGraph(AgentState)
        builder.add_node("tools", ToolNode([falhar_de_proposito]))
        builder.add_edge(START, "tools")
        graph = builder.compile()

        with pytest.raises(RuntimeError, match="falha proposital"):
            graph.invoke({"messages": [AIMessage(content="", tool_calls=[TOOL_CALL])]})

    def test_a_raising_tool_becomes_a_message_the_model_can_read(self) -> None:
        """The whole loop, offline: tool call, exception, recovery, answer.

        A broken tool has to come back as something the model can read and
        recover from, never as an exception that drops the conversation. The
        message names the exception type and nothing else, so a traceback with
        file paths in it never reaches a customer.
        """
        model = ScriptedModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[TOOL_CALL],
                    response_metadata={"stopReason": "tool_use"},
                ),
                AIMessage(
                    content="Não consegui verificar isso agora, desculpe.",
                    response_metadata={"stopReason": "end_turn"},
                ),
            ]
        )
        agent = build_agent(model=model, tools=[falhar_de_proposito])

        result = agent.invoke({"messages": [{"role": "user", "content": "quebra, por favor"}]})

        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert [m.status for m in tool_messages] == ["error"]
        assert tool_messages[0].text == TOOL_ERROR_TEMPLATE.format(tipo="RuntimeError")
        assert "falha proposital" not in tool_messages[0].text
        assert result["messages"][-1].text == "Não consegui verificar isso agora, desculpe."

    def test_every_model_call_leaves_its_stop_reason_in_the_state(self) -> None:
        """For later."""
        model = ScriptedModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[TOOL_CALL],
                    response_metadata={"stopReason": "tool_use"},
                ),
                AIMessage(content="pronto", response_metadata={"stopReason": "end_turn"}),
            ]
        )
        agent = build_agent(model=model, tools=[falhar_de_proposito])

        result = agent.invoke({"messages": [{"role": "user", "content": "oi"}]})

        assert result["stop_reasons"] == ["tool_use", "end_turn"]

    def test_a_response_without_metadata_is_recorded_as_unknown(self) -> None:
        model = ScriptedModel(script=[AIMessage(content="olá")])
        agent = build_agent(model=model, tools=[consultar_estoque])

        result = agent.invoke({"messages": [{"role": "user", "content": "oi"}]})

        assert result["stop_reasons"] == ["unknown"]


class TestStopReasons:
    def test_failures_keeps_only_what_went_wrong(self) -> None:
        reasons = ["end_turn", "tool_use", "malformed_tool_use", "max_tokens", "end_turn"]
        assert tally(reasons)["end_turn"] == 2
        assert dict(failures(reasons)) == {"malformed_tool_use": 1, "max_tokens": 1}


class TestAgentGraph:
    def test_the_graph_has_the_loop_and_the_middleware(self) -> None:
        nodes = set(build_agent().get_graph().nodes)
        assert {"model", "tools"} <= nodes
        assert any("StopReason" in node for node in nodes)

    def test_the_system_prompt_names_the_store_and_the_tools(self) -> None:
        # For later use, replacing prompt with the persona;
        # the gate's numbers only mean something next to the prompt that produced
        # them.
        assert "Empório da Música" in SYSTEM_PROMPT
        assert "ferramenta" in SYSTEM_PROMPT
