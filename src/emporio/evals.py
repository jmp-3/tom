"""The model gate: the cases in `evals/casos.yaml`, run against the real agent.

This is the first point in the project where the real conditions exist -- six
typed tools, a persona in Portuguese, answers that depend on numbers coming from
the data -- so it is the first point where "does this model hold up?" can be
answered honestly. The gate (`scripts/smoke_tool_use.py`) proved tool
use with one toy tool, which is a much easier test.

What is scored, and why those three things:

*   **The choice of tool.** Everything downstream is built on it. A wrong choice
    is a wrong answer with a confident tone.
*   **The arguments.** Valid against the tool's own schema, and, where the case
    says so, the right values -- `order_id: 1` when the customer said "pedido 1".
*   **`malformed_tool_use`.** The stop reason that says the model tried to call a
    tool and produced something unparseable. It is invisible in the answer and it
    is the number that decides a model change.

The final answer is checked too, but as information rather than as a gate: a
case declares the fragments that would show the answer came from the tool
result, and a model can be right about the tool and clumsy about the wording.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml
from botocore.exceptions import BotoCoreError, ClientError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, ValidationError

from emporio.agent import (
    ANONYMOUS,
    STORE_TOOLS,
    SessionContext,
    build_agent,
    for_customer,
    log_offered_tools,
)
from emporio.agent.middleware import failures, tally
from emporio.config import PROJECT_ROOT
from emporio.data.loader import normalize_text

DEFAULT_CASES: Final = PROJECT_ROOT / "evals" / "casos.yaml"

# The one stop reason this gate exists to count.
MALFORMED: Final = "malformed_tool_use"

# Spelled out so a typo in the YAML is an error instead of an ignored key.
_CASE_KEYS: Final = frozenset(
    {
        "id",
        "pergunta",
        "cliente_id",
        "ferramentas",
        "ferramentas_aceitas",
        "argumentos",
        "resposta",
        "motivo",
    }
)


@dataclass(frozen=True)
class Case:
    """One case: a question, the session it runs in, and the expected decision."""

    id: str
    question: str
    customer_id: int | None
    tools: tuple[str, ...]
    accepted: tuple[str, ...]
    arguments: dict[str, Any]
    answer: tuple[str, ...]
    reason: str

    @property
    def mode(self) -> str:
        return "anônimo" if self.customer_id is None else f"cliente {self.customer_id}"


def load_cases(path: Path | str = DEFAULT_CASES) -> list[Case]:
    """Read the case file, rejecting anything the runner would silently ignore.

    An unknown key or a tool name that does not exist is a case that would pass
    or fail for the wrong reason, which is worse than a missing case.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    known = {instrument.name for instrument in STORE_TOOLS}
    cases: list[Case] = []
    for entry in raw:
        unknown = sorted(set(entry) - _CASE_KEYS)
        if unknown:
            raise ValueError(f"{entry.get('id', '?')}: chave desconhecida: {unknown}")
        tools = tuple(entry.get("ferramentas") or ())
        accepted = tuple(entry.get("ferramentas_aceitas") or ())
        missing = sorted(name for name in tools + accepted if name not in known)
        if missing:
            raise ValueError(f"{entry['id']}: ferramenta inexistente: {missing}")
        cases.append(
            Case(
                id=entry["id"],
                question=entry["pergunta"],
                customer_id=entry.get("cliente_id"),
                tools=tools,
                accepted=accepted,
                arguments=dict(entry.get("argumentos") or {}),
                answer=tuple(str(item) for item in entry.get("resposta") or ()),
                reason=str(entry.get("motivo", "")).strip(),
            )
        )
    return cases


def _matches(expected: Any, actual: Any) -> bool:
    """Compare one expected argument with what the model actually sent.

    Numbers exactly -- `1000` and `1000.0` are the same answer -- and text by
    content, because "Yamaha C40" and "Yamaha C40 Nylon Natural" are the same
    intention and the gate is about the decision, not about the spelling.
    """
    if isinstance(expected, bool) or isinstance(actual, bool):
        return bool(expected) is bool(actual)
    if isinstance(expected, int | float):
        try:
            return float(expected) == float(actual)
        except (TypeError, ValueError):
            return False
    wanted, got = normalize_text(expected), normalize_text(actual)
    return bool(wanted) and (wanted in got or got in wanted)


@dataclass
class CaseResult:
    """What one run of one case produced."""

    case: Case
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    invalid_args: list[str] = field(default_factory=list)
    offered: list[str] = field(default_factory=list)
    stop_reasons: list[str] = field(default_factory=list)
    tool_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    answer: str = ""
    error: str | None = None

    @property
    def called(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.calls)

    @property
    def chose_right(self) -> bool:
        """The expected calls, in order, plus anything the case tolerates after them.

        Exact sequence by default. A case declares `ferramentas_aceitas` when a
        further call is a defensible drill-down rather than a wrong decision --
        `list_orders` and then `get_order` on the order it just found, for
        instance. It is the decision that is scored, not the round trips.
        """
        if self.error is not None:
            return False
        required = self.case.tools
        if self.called[: len(required)] != required:
            return False
        return all(name in self.case.accepted for name in self.called[len(required) :])

    @property
    def arguments_right(self) -> bool:
        """Every declared argument present with the expected value.

        Looked up in the first call of the expected tool, so a case that declares
        no arguments is right by default.
        """
        if not self.case.arguments:
            return True
        sent = next((args for name, args in self.calls if name in self.case.tools), None)
        if sent is None:
            return False
        return all(_matches(value, sent.get(key)) for key, value in self.case.arguments.items())

    @property
    def answer_right(self) -> bool | None:
        """None when the case declares nothing about the answer."""
        if not self.case.answer:
            return None
        text = normalize_text(self.answer)
        return any(normalize_text(fragment) in text for fragment in self.case.answer)

    @property
    def malformed(self) -> int:
        return sum(1 for reason in self.stop_reasons if reason == MALFORMED)

    @property
    def passed(self) -> bool:
        """The gate's verdict for this case. The answer check is not part of it."""
        return (
            self.error is None
            and self.chose_right
            and self.arguments_right
            and not self.invalid_args
            and not self.malformed
        )


def _session(case: Case) -> SessionContext | None:
    if case.customer_id is None:
        return ANONYMOUS
    return for_customer(case.customer_id)


def _check_arguments(name: str, args: dict[str, Any]) -> list[str]:
    """Validate a call against the tool's own schema, never against a copy of it."""
    instrument = next((t for t in STORE_TOOLS if t.name == name), None)
    if instrument is None:
        return [f"{name}: ferramenta inexistente"]
    schema = instrument.tool_call_schema
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        return []
    try:
        schema.model_validate(args)
    except ValidationError as exc:
        return [f"{name}: {exc.errors()[0]['msg']}"]
    return []


def run_case(agent: Any, case: Case, offered: list[str]) -> CaseResult:
    """Run one case in its own session. `offered` is the debug sink, cleared here."""
    result = CaseResult(case=case)
    session = _session(case)
    if session is None:
        result.error = f"cliente {case.customer_id} não existe na base"
        return result

    offered.clear()
    try:
        state = agent.invoke(
            {"messages": [{"role": "user", "content": case.question}]}, context=session
        )
    except (ClientError, BotoCoreError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    result.offered = list(offered)
    result.stop_reasons = list(state.get("stop_reasons", []))
    for message in state["messages"]:
        if isinstance(message, AIMessage):
            usage: dict[str, Any] = dict(message.usage_metadata or {})
            result.input_tokens += usage.get("input_tokens", 0)
            result.output_tokens += usage.get("output_tokens", 0)
            for call in message.tool_calls:
                result.calls.append((call["name"], dict(call["args"])))
                result.invalid_args += _check_arguments(call["name"], call["args"])
        elif isinstance(message, ToolMessage) and message.status == "error":
            result.tool_errors += 1

    last = state["messages"][-1]
    result.answer = last.text if isinstance(last.text, str) else str(last.content)
    return result


def run_cases(
    cases: Sequence[Case],
    *,
    model: BaseChatModel | None = None,
    on_case: Any = None,
) -> list[CaseResult]:
    """Run every case against one agent, in order.

    One agent for both modes: the session is an argument of `invoke`, which is
    the point of `context_schema` -- the same graph serves an anonymous turn and
    an identified one, and the middleware decides what each is offered.
    """
    offered: list[str] = []
    agent = build_agent(model=model, extra_middleware=[log_offered_tools(offered.append)])
    results: list[CaseResult] = []
    for case in cases:
        if on_case is not None:
            on_case(case)
        results.append(run_case(agent, case, offered))
    return results


@dataclass(frozen=True)
class Summary:
    """The numbers the gate is read from."""

    cases: int
    chose_right: int
    arguments_right: int
    answers_checked: int
    answers_right: int
    invalid_args: list[str]
    malformed: int
    tool_errors: int
    errors: int
    stop_reasons: Counter[str]
    failed_stop_reasons: Counter[str]
    input_tokens: int
    output_tokens: int

    @property
    def passed(self) -> bool:
        return (
            self.errors == 0
            and self.chose_right == self.cases
            and self.malformed == 0
            and (self.arguments_right == self.cases and not self.invalid_args)
        )


def summarize(results: Sequence[CaseResult]) -> Summary:
    reasons = [reason for result in results for reason in result.stop_reasons]
    checked = [result for result in results if result.answer_right is not None]
    return Summary(
        cases=len(results),
        chose_right=sum(1 for r in results if r.chose_right),
        arguments_right=sum(1 for r in results if r.arguments_right),
        answers_checked=len(checked),
        answers_right=sum(1 for r in checked if r.answer_right),
        invalid_args=[problem for r in results for problem in r.invalid_args],
        malformed=sum(r.malformed for r in results),
        tool_errors=sum(r.tool_errors for r in results),
        errors=sum(1 for r in results if r.error),
        stop_reasons=tally(reasons),
        failed_stop_reasons=failures(reasons),
        input_tokens=sum(r.input_tokens for r in results),
        output_tokens=sum(r.output_tokens for r in results),
    )
