"""The agent's mechanics: tool shape, session scope, persona, errors, stopReason.

No billed call happens here. What is tested is everything about the loop that
can be checked without the model: that the tools expose to the model exactly
what they should and nothing more, that an anonymous session is never even
offered the order tools, that `customer_id` reaches a tool from the session and
not from the model, that a tool raising an exception turns into a message
instead of an exception, and that the stop reasons are recorded and classified.
The model's *judgement* is not testable offline -- that is what `emporio eval`
is for.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents.middleware import AgentState
from langchain.tools import BaseTool, ToolRuntime, tool
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode

from emporio.agent import (
    ANONYMOUS,
    ORDER_TOOLS,
    PERSONA,
    PUBLIC_TOOLS,
    STORE_TOOLS,
    SessionContext,
    build_agent,
    for_customer,
    log_offered_tools,
    persona,
)
from emporio.agent.middleware import TOOL_ERROR_TEMPLATE, failures, tally
from emporio.agent.prompt import ANONYMOUS_BLOCK
from emporio.agent.toy_tools import consultar_estoque, falhar_de_proposito

TOOL_CALL = {
    "name": "falhar_de_proposito",
    "args": {"motivo": "teste"},
    "id": "call_1",
    "type": "tool_call",
}

# Customer 3 is the demo session: order 1 is his, order 2 is customer 7's.
CUSTOMER = 3
ORDER_OF_CUSTOMER = 1
ORDER_OF_SOMEONE_ELSE = 2


def _order_call(order_id: int) -> dict[str, Any]:
    return {"name": "get_order", "args": {"order_id": order_id}, "id": "c1", "type": "tool_call"}


class ScriptedModel(BaseChatModel):
    """A chat model that replays prepared answers, so the loop runs offline.

    The loop itself -- model, tool, model again -- is worth testing without
    Bedrock in the way: it costs nothing, it cannot flake, and it is the only
    way to assert what happens around a tool that raises without depending on
    the model agreeing to call it.

    It also records what every call was bound to and what it was sent, which is
    how the two requirements of this stage are checked where they are actually
    enforced: in the request that leaves for the model.
    """

    script: list[AIMessage] = []
    bound_tools: list[list[str]] = []
    system_prompts: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:
        self.bound_tools.append([getattr(t, "name", str(t)) for t in tools])
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.system_prompts += [m.text for m in messages if isinstance(m, SystemMessage)]
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


def _say(text: str) -> AIMessage:
    return AIMessage(content=text, response_metadata={"stopReason": "end_turn"})


def _call(tool_call: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[tool_call], response_metadata={"stopReason": "tool_use"}
    )


def _model_facing_schema(instrument: BaseTool) -> dict[str, Any]:
    return convert_to_openai_tool(instrument)["function"]["parameters"]


class TestToolShape:
    def test_every_tool_has_a_description_and_described_arguments(self) -> None:
        for instrument in STORE_TOOLS:
            assert instrument.description
            schema = _model_facing_schema(instrument)
            for name, field in schema["properties"].items():
                # `parse_docstring=True` is what carries the `Args:` section into
                # the schema; without it the model sees a bare `{"type": ...}`.
                assert field.get("description"), f"{instrument.name}.{name} has no description"

    def test_no_optional_argument_reaches_the_model_as_a_null_union(self) -> None:
        """`str | None` in a signature is a malformed tool call waiting to happen.

        Measured, not assumed: on `us.amazon.nova-micro-v1:0`, a tool whose only
        argument was typed `str | None` produced
        `ModelErrorException: Model produced invalid sequence as part of ToolUse`
        in 3 of 5 runs; the same tool with `status: str = ""` ran 5 of 5 clean.
        An argument that may be omitted gets a type and an empty default, which
        is why "no filter" is spelled `""` or `0` in `agent/tools.py`.
        """
        for instrument in STORE_TOOLS:
            for name, field in _model_facing_schema(instrument)["properties"].items():
                branches = field.get("anyOf", [])
                nullable = [b for b in branches if b.get("type") == "null"]
                assert not nullable, f"{instrument.name}.{name} is a union with null"

    def test_no_tool_lets_the_model_choose_a_customer(self) -> None:
        """The privacy requirement, read off the schema instead of the intention.

        `customer_id` is mandatory in `queries` and is injected by the session, so
        it must appear in no schema at all -- not even as an optional argument a
        model could fill in with someone else's number.
        """
        for instrument in STORE_TOOLS:
            assert "customer_id" not in instrument.args
            assert "runtime" not in instrument.args
            assert "dataset" not in instrument.args
            assert "today" not in instrument.args

    def test_tool_runtime_stays_invisible_to_the_model(self) -> None:
        """
        A tool defined by its signature gets its `ToolRuntime` injected and the
        parameter never appears in the schema. Declaring `args_schema` instead
        would silently skip the injection, which is why no tool in this project
        declares one.
        """

        @tool(parse_docstring=True)
        def meus_pedidos(runtime: ToolRuntime[SessionContext], limite: int = 5) -> dict[str, Any]:
            """Lista os pedidos do cliente.

            Args:
                limite: Quantos pedidos devolver.
            """
            return {"customer_id": runtime.context.customer_id, "limite": limite}

        assert "runtime" not in meus_pedidos.args
        assert set(_model_facing_schema(meus_pedidos)["properties"]) == {"limite"}


class TestSessionScope:
    """What the model is offered, per session mode.

    Asserted on `bind_tools`, which is the last thing before the wire: a tool
    that is not in that list cannot be called, cannot be mis-called, and cannot
    leak what it would have returned. The prompt is not part of this control.
    """

    def test_an_anonymous_session_is_not_offered_the_order_tools(self) -> None:
        model = ScriptedModel(script=[_say("Isso fica na área logada do site.")])
        agent = build_agent(model=model)

        agent.invoke(
            {"messages": [{"role": "user", "content": "e o pedido 1?"}]}, context=ANONYMOUS
        )

        # The public half: the catalog and the policy manual, both published
        # information. What is missing is the order tools and only those.
        assert model.bound_tools == [[t.name for t in PUBLIC_TOOLS]]

    def test_an_identified_session_is_offered_every_tool(self) -> None:
        model = ScriptedModel(script=[_say("Deixa eu ver seus pedidos.")])
        agent = build_agent(model=model)

        agent.invoke(
            {"messages": [{"role": "user", "content": "e o pedido 1?"}]},
            context=SessionContext(customer_id=CUSTOMER, first_name="Pedro"),
        )

        assert model.bound_tools == [[t.name for t in STORE_TOOLS]]

    def test_the_debug_log_reports_the_tools_of_the_request(self) -> None:
        """What `--debug` prints is the request's own tool list, filtering included."""
        offered: list[str] = []
        model = ScriptedModel(script=[_say("pronto")])
        agent = build_agent(model=model, extra_middleware=[log_offered_tools(offered.append)])

        agent.invoke({"messages": [{"role": "user", "content": "oi"}]}, context=ANONYMOUS)

        assert len(offered) == 1
        assert "get_order" not in offered[0]
        # The arguments as the model sees them: no `customer_id` anywhere.
        assert "search_products(text, category, price_min, price_max" in offered[0]
        assert "customer_id" not in offered[0]


class TestCustomerScope:
    """`customer_id` comes from the session, through the agent, or not at all.

    Through the agent on purpose: the injection only happens when the tool runs
    inside the graph, so calling the tool function directly would prove nothing.
    """

    def test_an_order_of_this_customer_comes_back(self) -> None:
        model = ScriptedModel(script=[_call(_order_call(ORDER_OF_CUSTOMER)), _say("Entregue!")])
        agent = build_agent(model=model)

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "como está o pedido 1?"}]},
            context=SessionContext(customer_id=CUSTOMER),
        )

        payload = [m for m in result["messages"] if isinstance(m, ToolMessage)][0].text
        assert '"found": true' in payload.lower()
        assert '"status": "delivered"' in payload

    def test_another_customers_order_is_indistinguishable_from_a_missing_one(self) -> None:
        model = ScriptedModel(script=[_call(_order_call(ORDER_OF_SOMEONE_ELSE)), _say("Hmm.")])
        agent = build_agent(model=model)

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "como está o pedido 2?"}]},
            context=SessionContext(customer_id=CUSTOMER),
        )

        payload = [m for m in result["messages"] if isinstance(m, ToolMessage)][0].text
        assert "found" in payload and "false" in payload.lower()
        # Nothing of the real owner's order is in there, not even its total.
        assert "tracking" not in payload

    def test_an_order_tool_reached_without_a_customer_refuses(self) -> None:
        """The second lock, for the day the first one is changed by accident.

        The filter means the model cannot call this in an anonymous session. If it
        somehow does, `queries` raises `MissingCustomerIdError` and the error
        middleware turns it into a message -- no order data either way.
        """
        model = ScriptedModel(script=[_call(_order_call(ORDER_OF_CUSTOMER)), _say("Não deu.")])
        agent = build_agent(model=model, tools=ORDER_TOOLS)

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "pedido 1"}]}, context=ANONYMOUS
        )

        message = [m for m in result["messages"] if isinstance(m, ToolMessage)][0]
        assert message.status == "error"
        assert message.text == TOOL_ERROR_TEMPLATE.format(tipo="MissingCustomerIdError")


class TestPersona:
    def test_the_persona_carries_the_hard_rules(self) -> None:
        """Numbered rules, because the gate's numbers only mean something next to
        the prompt that produced them."""
        assert "Empório da Música" in PERSONA
        assert "ferramenta" in PERSONA
        # The assistant's name, and the reason the style heading is not "Tom:".
        assert "Você é o Tom," in PERSONA
        assert "Tom:" not in PERSONA
        for rule in ("preço", "estoque", "acessórios", "telefone", "percentual"):
            assert rule in PERSONA
        # No instruction about *how* to emit a tool call: that competes with the
        # framework's own mechanism and produces malformed calls.
        for forbidden in ("JSON", "json", "tool_use", "function_call"):
            assert forbidden not in PERSONA

    def test_the_anonymous_session_gets_the_order_refusal_rule(self) -> None:
        text = persona(ANONYMOUS)
        assert "área logada" in text
        assert "Não confirme nem negue" in text

    def test_the_identified_session_gets_the_first_name(self) -> None:
        session = for_customer(CUSTOMER)
        assert session is not None
        text = persona(session)
        assert "Pedro" in text
        assert "Oliveira" not in text  # first name only, nothing else from the row
        assert "área logada" not in text

    def test_an_unknown_customer_has_no_session(self) -> None:
        assert for_customer(9999) is None

    def test_the_model_receives_the_persona_completed_by_the_session(self) -> None:
        """The system block the model actually gets, not the one we meant to send."""
        model = ScriptedModel(script=[_say("Olá, Pedro!")])
        agent = build_agent(model=model)
        session = SessionContext(customer_id=CUSTOMER, first_name="Pedro", city="Campo Grande")

        agent.invoke({"messages": [{"role": "user", "content": "oi"}]}, context=session)

        assert model.system_prompts == [persona(session)]
        assert "Pedro" in model.system_prompts[0]

    def test_the_fallback_prefix_renders_the_same_text(self) -> None:
        """One text source, two destinations: the system block and, for a model
        that ignores it, the first user turn."""
        from emporio.agent import as_user_prefix

        prefixed = as_user_prefix("quanto custa o Yamaha C40?", ANONYMOUS)
        assert prefixed.startswith(PERSONA)
        assert ANONYMOUS_BLOCK in prefixed
        assert prefixed.endswith("quanto custa o Yamaha C40?")


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
            script=[_call(TOOL_CALL), _say("Não consegui verificar isso agora, desculpe.")]
        )
        agent = build_agent(model=model, tools=[falhar_de_proposito])

        result = agent.invoke({"messages": [{"role": "user", "content": "quebra, por favor"}]})

        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert [m.status for m in tool_messages] == ["error"]
        assert tool_messages[0].text == TOOL_ERROR_TEMPLATE.format(tipo="RuntimeError")
        assert "falha proposital" not in tool_messages[0].text
        assert result["messages"][-1].text == "Não consegui verificar isso agora, desculpe."

    def test_every_model_call_leaves_its_stop_reason_in_the_state(self) -> None:
        model = ScriptedModel(script=[_call(TOOL_CALL), _say("pronto")])
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

    def test_the_history_survives_inside_one_thread(self) -> None:
        """The checkpointer, which is why the CLI keeps no message list of its own.

        `InMemorySaver` here, `SqliteSaver` later: one line apart, which is
        the whole reason for using a checkpointer this early.
        """
        from langgraph.checkpoint.memory import InMemorySaver

        model = ScriptedModel(script=[_say("primeira"), _say("segunda")])
        agent = build_agent(model=model, checkpointer=InMemorySaver())
        config: Any = {"configurable": {"thread_id": "t1"}}

        agent.invoke(
            {"messages": [{"role": "user", "content": "oi"}]}, context=ANONYMOUS, config=config
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "e então?"}]},
            context=ANONYMOUS,
            config=config,
        )

        assert [m.text for m in result["messages"] if isinstance(m, AIMessage)] == [
            "primeira",
            "segunda",
        ]
