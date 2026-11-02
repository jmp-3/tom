#!/usr/bin/env python
"""Tool use smoke test.

uv run python scripts/smoke_tool_use.py
uv run python scripts/smoke_tool_use.py --model amazon.nova-lite-v1:0

"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A script, not an installed entry point: make the package importable when it is
# run straight from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer  # noqa: E402
from botocore.exceptions import BotoCoreError, ClientError  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from pydantic import BaseModel, ValidationError  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.padding import Padding  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

from emporio import bedrock  # noqa: E402
from emporio.agent import build_agent  # noqa: E402
from emporio.agent.middleware import failures, tally  # noqa: E402
from emporio.agent.toy_tools import STUB_PRICE, TOY_TOOLS  # noqa: E402
from emporio.config import get_settings  # noqa: E402

# Explicit width off a terminal: the per-run tables are the evidence this stage
# leaves behind, and at rich's 80-column default for pipes they are unreadable.
console = Console(width=None if sys.stdout.isatty() else 130)

# One question that cannot be answered without the tool, one that is answered by
# knowing where the store is. The same two, repeated, on purpose: the gate
# measures how *consistently* the model decides, not how wide its coverage is.
TOOL_QUESTION = "Vocês têm ukulele em estoque? Quanto custa?"
NO_TOOL_QUESTION = "Bom dia! Vocês ficam em que cidade?"

# The one right answer to TOOL_QUESTION, and the only tool bound in this stage.
TOOL_TO_CALL = "consultar_estoque"

# What a correct answer to TOOL_QUESTION has to repeat back from the tool result.
# The price only, in the spellings a model might choose: the stock count is a
# single digit, and a digit appears in made-up answers too, so matching on it
# would score an invented price as a correct one.
STUB_EVIDENCE = (
    f"{STUB_PRICE:,.2f}".replace(",", "@").replace(".", ",").replace("@", "."),
    f"{STUB_PRICE:.2f}",
    f"{int(STUB_PRICE):,}".replace(",", "."),
    str(int(STUB_PRICE)),
)


@dataclass
class Turn:
    """One conversation turn, reduced to what the gate needs to judge."""

    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    invalid_args: list[str] = field(default_factory=list)
    unknown_tools: list[str] = field(default_factory=list)
    tool_errors: int = 0
    stop_reasons: list[str] = field(default_factory=list)
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    answer: str = ""
    error: str | None = None

    @property
    def model_calls(self) -> int:
        """One stop reason per model call, so the list length is the count."""
        return len(self.stop_reasons)

    @property
    def used_the_result(self) -> bool:
        return any(token in self.answer for token in STUB_EVIDENCE)


def run_turn(agent: Any, question: str) -> Turn:
    turn = Turn()
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    except (ClientError, BotoCoreError) as exc:
        turn.error = f"{type(exc).__name__}: {exc}"
        return turn

    turn.stop_reasons = list(result.get("stop_reasons", []))
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            # RetryAttempts is botocore's own count: a retry does not show up as
            # an extra message, so this is the only place it is visible.
            response_metadata = message.response_metadata.get("ResponseMetadata", {})
            turn.retries += response_metadata.get("RetryAttempts", 0)
            usage: dict[str, Any] = dict(message.usage_metadata or {})
            turn.input_tokens += usage.get("input_tokens", 0)
            turn.output_tokens += usage.get("output_tokens", 0)
            for call in message.tool_calls:
                turn.calls.append((call["name"], call["args"]))
                tool = next((t for t in TOY_TOOLS if t.name == call["name"]), None)
                if tool is None:
                    turn.unknown_tools.append(call["name"])
                    continue
                # The tool's own schema is the judge of "valid arguments", so
                # the gate cannot drift from what the model was offered.
                schema = tool.tool_call_schema
                if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
                    continue
                try:
                    schema.model_validate(call["args"])
                except ValidationError as exc:
                    turn.invalid_args.append(f"{call['name']}: {exc.errors()[0]['msg']}")
        elif isinstance(message, ToolMessage):
            if message.status == "error":
                turn.tool_errors += 1

    last = result["messages"][-1]
    turn.answer = last.text if isinstance(last.text, str) else str(last.content)
    return turn


def _count(problems: list[str]) -> str:
    return str(len(problems)) + (f" — {problems}" if problems else "")


def _counter(counts: Counter[str]) -> str:
    return ", ".join(f"{reason}={n}" for reason, n in counts.items())


def _chose_right(turn: Turn, *, expects_tool: bool) -> bool:
    """Did this turn decide correctly? The same judgement the summary counts."""
    names = [name for name, _ in turn.calls]
    return names == [TOOL_TO_CALL] if expects_tool else names == []


def _render(turns: list[Turn], *, expects_tool: bool) -> None:
    """One block per turn, wrapped to the terminal.

    A table was the obvious shape and the wrong one: eight columns of mostly
    short values squeeze the two that matter -- the arguments and the answer --
    into ellipses, and an answer the reader cannot see is the whole point of
    reading the output. So: one line of verdict per turn, one line of numbers,
    and the answer in full, indented and wrapped.
    """
    for index, turn in enumerate(turns, start=1):
        if turn.error:
            console.print(f"[bold]{index:>2}[/] [red]falhou[/]")
            console.print(Padding(Text(turn.error), (0, 0, 1, 4)))
            continue

        right = _chose_right(turn, expects_tool=expects_tool)
        mark = "[green]ok[/]" if right else "[red]xx[/]"
        if turn.calls:
            calls = "  ".join(f"{name}({args})" for name, args in turn.calls)
        else:
            calls = "[dim]nenhuma ferramenta[/]"
        console.print(f"[bold]{index:>2}[/] {mark} {calls}")

        notes = [
            ", ".join(turn.stop_reasons),
            f"{turn.model_calls} chamada(s) ao modelo",
            f"{turn.retries} retry",
            f"{turn.input_tokens}/{turn.output_tokens} tokens",
        ]
        if turn.tool_errors:
            notes.append(f"[yellow]{turn.tool_errors} erro(s) de ferramenta[/]")
        if expects_tool:
            notes.append(
                "[green]repetiu o preço do stub[/]"
                if turn.used_the_result
                else "[yellow]não repetiu o preço do stub[/]"
            )
        console.print(Padding(f"[dim]{' · '.join(notes)}[/]", (0, 0, 0, 4)))
        console.print(Padding(Text(turn.answer.strip(), style="none"), (0, 0, 1, 4)))


def main(
    runs: int = typer.Option(5, "--runs", "-n", help="Quantas vezes repetir cada pergunta."),
    model: str = typer.Option("", "--model", "-m", help="modelId a testar (padrão: o do .env)."),
    temperature: float = typer.Option(0.2, "--temperature", "-t", help="Temperatura do modelo."),
) -> None:
    """Run the gate and exit with code 1 if the model does not pass it."""
    settings = get_settings()
    model_id = model or settings.model_id
    console.print(
        f"[bold]modelo[/] {model_id}   [bold]região[/] {settings.region}   "
        f"[bold]temperatura[/] {temperature}   [bold]rodadas[/] {runs}×2\n"
    )

    agent = build_agent(model=bedrock.chat_model(model_id=model_id, temperature=temperature))
    with_tool: list[Turn] = []
    without_tool: list[Turn] = []
    blocks = (
        ("A", "precisa de ferramenta", TOOL_QUESTION, with_tool, True),
        ("B", "não precisa de ferramenta", NO_TOOL_QUESTION, without_tool, False),
    )
    for label, caption, question, turns, expects_tool in blocks:
        console.print(f"[bold]{label}.[/] {caption} — {question!r}")
        # Each block is one billed call per turn, two when a tool is involved,
        # so a long run sits silent for a while. The counter is the difference
        # between waiting and wondering whether it hung.
        with console.status(f"rodada 1/{runs}") as status:
            for index in range(1, runs + 1):
                status.update(f"rodada {index}/{runs}")
                turn = run_turn(agent, question)
                turns.append(turn)
                # A Bedrock error is a configuration problem, not a model
                # result: it repeats identically, so the first one ends the
                # block instead of billing the rest.
                if turn.error:
                    break
        _render(turns, expects_tool=expects_tool)

    every_turn = with_tool + without_tool
    errors = [t for t in every_turn if t.error]
    called_right = sum(1 for t in with_tool if not t.error and _chose_right(t, expects_tool=True))
    spurious = sum(
        1 for t in without_tool if not t.error and not _chose_right(t, expects_tool=False)
    )
    invalid = [problem for t in every_turn for problem in t.invalid_args]
    unknown = [name for t in every_turn for name in t.unknown_tools]
    answered_from_result = sum(1 for t in with_tool if not t.error and t.used_the_result)
    stop_reasons = [reason for t in every_turn for reason in t.stop_reasons]
    retries = sum(t.retries for t in every_turn)
    tokens_in = sum(t.input_tokens for t in every_turn)
    tokens_out = sum(t.output_tokens for t in every_turn)

    console.print("\n[bold]Resultado[/]")
    summary = Table(show_header=False, box=None)
    summary.add_row("escolheu a ferramenta (A)", f"{called_right}/{runs}")
    summary.add_row("respondeu com o dado da ferramenta (A)", f"{answered_from_result}/{runs}")
    summary.add_row("chamou ferramenta sem precisar (B)", f"{spurious}/{runs}")
    summary.add_row("argumentos inválidos", _count(invalid))
    summary.add_row("ferramentas inexistentes", _count(unknown))
    summary.add_row("stopReason", _counter(tally(stop_reasons)))
    summary.add_row("stopReason de falha", _counter(failures(stop_reasons)) or "nenhum")
    summary.add_row("retries do botocore", str(retries))
    summary.add_row("tokens", f"{tokens_in} entrada / {tokens_out} saída")
    if errors:
        summary.add_row("[red]erros de chamada[/]", str(len(errors)))
    console.print(summary)

    passed = not errors and called_right == runs and spurious == 0 and not invalid and not unknown
    console.print()
    if passed:
        console.print(f"[green]PASSOU[/] — tool use confiável em {model_id}.")
        return
    console.print(
        f"[red]FALHOU[/] — {model_id} não sustenta o tool use deste projeto.\n"
        "Suba um degrau da escada de fallback do README (trocar BEDROCK_MODEL_ID "
        "e habilitar o model access do novo modelo) antes de seguir para o Estágio 3."
    )
    raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
