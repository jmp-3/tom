"""Agent CLI."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
import yaml
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from emporio import __version__, bedrock
from emporio.agent import ANONYMOUS, SessionContext, build_agent, for_customer, log_offered_tools
from emporio.config import get_settings
from emporio.evals import DEFAULT_CASES, Summary, load_cases, run_cases, summarize
from emporio.rag import (
    CHUNKS_FILE,
    DEFAULT_INDEX_DIR,
    VECTORS_FILE,
    PolicyError,
    build_index,
    check_index,
    policies_path,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Customer service agent for Empório da Música (prototype).",
)
console = Console()

# Hints for the Bedrock errors that actually show up while configuring access.
# Messages shown to the operator stay in Portuguese, like the rest of the CLI output.
_HINTS = {
    # Bedrock raises AccessDeniedException both for an invalid key and for a model
    # that is not enabled; only the message tells the two apart (see _describe_failure).
    "AccessDeniedException": (
        "a credencial é aceita, mas este modelo não está liberado para a conta. "
        "Habilite-o em Bedrock > Model access, na mesma região."
    ),
    "AccessDeniedException/auth": (
        "a API key foi recusada — inválida, expirada ou de outra região. "
        "Gere outra em Bedrock > API keys e atualize o .env."
    ),
    "ValidationException": (
        "o modelId não é aceito nesta região. Confira a lista com `tom models`."
    ),
    "ResourceNotFoundException": (
        "modelo inexistente nesta região. Confira a lista com `tom models`."
    ),
    "ThrottlingException": "limite de requisições atingido; tente de novo em alguns segundos.",
    "UnrecognizedClientException": "credencial inválida ou expirada (confira o .env).",
}


def _describe_failure(exc: Exception) -> str:
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        message = exc.response.get("Error", {}).get("Message", str(exc))
        key = code
        # Bedrock words the same refusal two ways: "Authentication failed" for a
        # key it does not recognize and "Bearer Token has expired" for one that
        # ran out. Both are the credential and neither is the model access, so
        # they must not point the operator at the Model access page.
        if code == "AccessDeniedException" and (
            "Authentication failed" in message or "has expired" in message
        ):
            key = "AccessDeniedException/auth"
        hint = _HINTS.get(key)
        return f"{code}: {message}" + (f"\n  → {hint}" if hint else "")
    if isinstance(exc, NoCredentialsError):
        return "nenhuma credencial encontrada — preencha o .env (veja .env.example)."
    return f"{type(exc).__name__}: {exc}"


@app.command()
def check() -> None:
    """Validate credentials and model access: one conversation call and one embedding call."""
    settings = get_settings()

    table = Table(show_header=False, box=None)
    table.add_row("região", settings.region)
    table.add_row("modelo", settings.model_id)
    table.add_row("embeddings", settings.embed_model_id)
    table.add_row(
        "credencial",
        "API key do Bedrock"
        if settings.has_bearer_token
        else "credenciais IAM"
        if settings.has_iam_keys
        else "[yellow]nenhuma no .env (tentando a cadeia padrão do boto3)[/]",
    )
    console.print(table)
    console.print()

    failures = 0

    console.print(f"[bold]conversa[/] ({settings.model_id})")
    try:
        # The prompts stay in Portuguese on purpose: this is the agent talking to
        # customers of a Brazilian store, so the check exercises real behavior.
        answer = bedrock.text_of(
            bedrock.converse(
                "Responda em uma frase curta: que loja é o Empório da Música?",
                system=(
                    "Você atende clientes de uma loja de instrumentos musicais em Campo Grande/MS."
                ),
                max_tokens=64,
            )
        )
        console.print(f"  [green]ok[/] — resposta: {answer}")
    except (ClientError, BotoCoreError, KeyError) as exc:
        failures += 1
        console.print(f"  [red]falhou[/] — {_describe_failure(exc)}")

    console.print(f"\n[bold]embeddings[/] ({settings.embed_model_id})")
    try:
        vector = bedrock.embed("violão de nylon para iniciante")
        console.print(f"  [green]ok[/] — vetor de {len(vector)} dimensões")
    except (ClientError, BotoCoreError, KeyError) as exc:
        failures += 1
        console.print(f"  [red]falhou[/] — {_describe_failure(exc)}")

    console.print()
    if failures:
        console.print(f"[red]{failures} de 2 verificações falharam.[/]")
        raise typer.Exit(code=1)
    console.print("[green]Acesso ao Bedrock validado.[/]")


@app.command()
def models(
    search: str = typer.Option("", "--search", "-s", help="Filter by part of the modelId."),
) -> None:
    """List the text models available in the configured region."""
    settings = get_settings()
    try:
        response = bedrock.get_client("bedrock").list_foundation_models(byOutputModality="TEXT")
    except (ClientError, BotoCoreError) as exc:
        console.print(f"[red]falhou[/] — {_describe_failure(exc)}")
        raise typer.Exit(code=1)

    table = Table(title=f"Modelos de texto em {settings.region}")
    table.add_column("modelId")
    table.add_column("provedor")
    table.add_column("streaming")
    for model in response["modelSummaries"]:
        model_id = model["modelId"]
        if search and search.lower() not in model_id.lower():
            continue
        table.add_row(
            model_id,
            model.get("providerName", ""),
            "sim" if model.get("responseStreamingSupported") else "não",
        )
    console.print(table)


# --- build-index --------------------------------------------------------------


@app.command("build-index")
def build_index_command(
    index_dir: Path = typer.Option(
        DEFAULT_INDEX_DIR, "--index-dir", help="Onde gravar os dois arquivos do índice."
    ),
) -> None:
    """Build the policy index: chunk the manual, embed the chunks, write `index/`."""
    settings = get_settings()
    try:
        pdf = policies_path()
    except PolicyError as exc:
        console.print(f"[red]falhou[/] — {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold]manual[/] {pdf.name}   [bold]embeddings[/] {settings.embed_model_id}\n")
    try:
        with console.status("extraindo e embeddando…"):
            index = build_index(index_dir=index_dir)
    except (ClientError, BotoCoreError, NoCredentialsError) as exc:
        console.print(f"[red]falhou[/] — {_describe_failure(exc)}")
        # The trap of this step: Titan Embeddings is enabled separately from the
        # chat model, so the account can pass `tom check`'s first half and
        # still be refused here.
        console.print(
            "[dim]o modelo de embeddings é liberado separadamente do modelo de "
            "conversa, na mesma região.[/]"
        )
        raise typer.Exit(code=1) from exc
    except PolicyError as exc:
        console.print(f"[red]falhou[/] — {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(show_header=False, box=None)
    table.add_row("trechos", str(len(index.chunks)))
    table.add_row("seções", ", ".join(chunk.section for chunk in index.chunks))
    table.add_row("dimensão", str(index.dimension))
    table.add_row("hash do PDF", index.pdf_digest)
    table.add_row("arquivos", f"{index_dir / CHUNKS_FILE}, {index_dir / VECTORS_FILE}")
    console.print(table)
    console.print("\n[green]Índice de políticas construído.[/]")


def _warn_about_index() -> None:
    """Say so when the stored index cannot be trusted, and keep going.

    A warning and not an error: the catalog and the order half of the agent work
    without the index, and the operator is the one who can fix it. What must not
    happen is the silent case -- an index built with another embedding model
    still produces similarities, and they rank nonsense.
    """
    check = check_index()
    if check.usable:
        return
    if check.missing:
        console.print(
            "[yellow]sem índice de políticas[/] — perguntas sobre regras da loja vão "
            "falhar. Rode `tom build-index`."
        )
        return
    if check.model_changed:
        console.print(
            "[yellow]o índice de políticas foi construído com outro modelo de "
            "embeddings[/] — as similaridades ficam sem sentido. Rode `tom build-index`."
        )
    if check.pdf_changed:
        console.print(
            "[yellow]o manual de políticas mudou depois do índice[/] — rode `tom build-index`."
        )


# --- chat ---------------------------------------------------------------------

EXIT_COMMANDS = {"/sair", "/quit", "/exit"}


def _session(cliente_id: int | None) -> SessionContext:
    """The identity the login would have provided, or the anonymous session.

    `--cliente-id` stands in for the authenticated channel: the chat is simulated
    as if it lived in the logged-in area of the site, so the identity arrives
    with the session and is never asked for in conversation. It is also the one
    place a wrong id can be caught, which is why it is resolved before the first
    turn rather than on the first order question.
    """
    if cliente_id is None:
        return ANONYMOUS
    session = for_customer(cliente_id)
    if session is None:
        console.print(
            f"[red]cliente {cliente_id} não existe na base.[/] "
            "Os ids vão de 1 a 50 (veja data/*customers.csv)."
        )
        raise typer.Exit(code=1)
    return session


def _tool_calls(messages: list[BaseMessage]) -> list[str]:
    """`name({args})` for every tool call in a turn, for the transcript."""
    return [
        f"{call['name']}({json.dumps(call['args'], ensure_ascii=False)})"
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]


def _render_debug(messages: list[BaseMessage], offered: list[str], reasons: list[str]) -> None:
    """What `--debug` shows: the tools of each request, then the calls and results.

    The offered tools come from the request itself (see
    `middleware.log_offered_tools`), which is the only honest way to check that
    an anonymous session was never given the order tools and that no schema
    carries `customer_id`.
    """
    lines = [f"ferramentas oferecidas ({index}): {tools}" for index, tools in enumerate(offered, 1)]
    for message in messages:
        if isinstance(message, AIMessage):
            lines += [
                f"→ {call['name']}({json.dumps(call['args'], ensure_ascii=False)})"
                for call in message.tool_calls
            ]
        elif isinstance(message, ToolMessage):
            status = "erro" if message.status == "error" else "ok"
            payload = " ".join(message.text.split())
            lines.append(f"← {message.name} [{status}] {payload[:400]}")
    lines.append(f"stopReason: {', '.join(reasons)}")
    for line in lines:
        console.print(Padding(Text(line, style="dim"), (0, 0, 0, 2)))


def _transcript_header(session: SessionContext, model_id: str, region: str) -> list[str]:
    mode = (
        f"cliente {session.customer_id} ({session.first_name})"
        if session.authenticated
        else "anônima (sem `--cliente-id`)"
    )
    return [
        "# Transcript — `tom chat`",
        "",
        f"- modelo: `{model_id}` em {region}",
        f"- sessão: {mode}",
        f"- gerado em: {datetime.now():%Y-%m-%d %H:%M}",
        "",
    ]


@app.command()
def chat(
    cliente_id: int | None = typer.Option(
        None,
        "--cliente-id",
        help="Identidade da sessão, no lugar do login. Sem a flag, a sessão é anônima.",
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Mostra as ferramentas oferecidas, as chamadas e os stopReason."
    ),
    transcript: Path | None = typer.Option(
        None, "--transcript", help="Grava a conversa em markdown neste arquivo."
    ),
) -> None:
    """Talk to the agent. `/sair` ends the session."""
    settings = get_settings()
    session = _session(cliente_id)

    offered: list[str] = []
    agent = build_agent(
        # In memory in this stage; `SqliteSaver` later.
        checkpointer=InMemorySaver(),
        extra_middleware=[log_offered_tools(offered.append)] if debug else [],
    )
    # One thread: the whole CLI session is one conversation.
    config: Any = {"configurable": {"thread_id": "cli"}}

    header = Table(show_header=False, box=None)
    header.add_row("modelo", settings.model_id)
    header.add_row(
        "sessão",
        f"cliente {session.customer_id} — {session.first_name}"
        if session.authenticated
        else "anônima (só informação pública; pedidos ficam na área logada)",
    )
    console.print(header)
    _warn_about_index()
    console.print("[dim]`/sair` para encerrar.[/]")

    lines = _transcript_header(session, settings.model_id, settings.region)
    seen_messages = 0
    seen_reasons = 0

    while True:
        try:
            question = console.input("\n[bold cyan]você[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            break

        offered.clear()
        try:
            with console.status("consultando…"):
                state = agent.invoke(
                    {"messages": [{"role": "user", "content": question}]},
                    context=session,
                    config=config,
                )
        except (ClientError, BotoCoreError) as exc:
            console.print(f"[red]falhou[/] — {_describe_failure(exc)}")
            continue

        messages = list(state["messages"])
        fresh = messages[seen_messages:]
        seen_messages = len(messages)
        # The checkpointer accumulates the stop reasons across the thread, so
        # only the tail belongs to this turn.
        reasons = list(state.get("stop_reasons", []))[seen_reasons:]
        seen_reasons += len(reasons)

        if debug:
            _render_debug(fresh, offered, reasons)

        answer = messages[-1].text
        console.print("[bold green]atendente[/]", end=" ")
        console.print(Text(answer))

        calls = _tool_calls(fresh)
        lines += ["---", "", f"**Cliente:** {question}", ""]
        if calls:
            lines += ["**Ferramentas:** " + ", ".join(f"`{call}`" for call in calls), ""]
        lines += [f"**Atendente:** {answer}", ""]
        if transcript is not None:
            # Written every turn, not at the end: a session ended with Ctrl-C
            # still leaves the transcript it had produced.
            transcript.write_text("\n".join(lines), encoding="utf-8")

    if transcript is not None:
        console.print(f"[dim]transcript em {transcript}[/]")


# --- eval ---------------------------------------------------------------------


def _render_case(result: Any) -> None:
    """One block per case: verdict, calls, numbers, answer in full.

    Same shape as the gate's output and for the same reason: the answer
    is the thing the reader needs to see, and a table squeezes it into ellipses.
    """
    case = result.case
    if result.error:
        console.print(f"[red]xx[/] [bold]{case.id}[/] [dim]({case.mode})[/]")
        console.print(Padding(Text(result.error), (0, 0, 1, 3)))
        return

    mark = "[green]ok[/]" if result.passed else "[red]xx[/]"
    calls = (
        "  ".join(f"{name}({json.dumps(args, ensure_ascii=False)})" for name, args in result.calls)
        or "[dim]nenhuma ferramenta[/]"
    )
    console.print(f"{mark} [bold]{case.id}[/] [dim]({case.mode})[/] {calls}")

    if not result.chose_right:
        expected = ", ".join(case.tools) or "nenhuma ferramenta"
        console.print(Padding(f"[yellow]esperado: {expected}[/]", (0, 0, 0, 3)))
    if not result.arguments_right:
        console.print(Padding(f"[yellow]argumentos esperados: {case.arguments}[/]", (0, 0, 0, 3)))

    notes = [", ".join(result.stop_reasons), f"{result.input_tokens}/{result.output_tokens} tokens"]
    if result.invalid_args:
        notes.append(f"[red]argumentos inválidos: {result.invalid_args}[/]")
    if result.tool_errors:
        notes.append(f"[yellow]{result.tool_errors} erro(s) de ferramenta[/]")
    if result.answer_right is False:
        notes.append(f"[yellow]resposta não cita {list(case.answer)}[/]")
    # The control that matters in an anonymous session: not refused, not sent.
    if case.customer_id is None and any("get_order" in tools for tools in result.offered):
        notes.append("[red]ferramentas de pedido oferecidas numa sessão anônima[/]")
    console.print(Padding(f"[dim]{' · '.join(notes)}[/]", (0, 0, 0, 3)))
    console.print(Padding(Text(result.answer.strip()), (0, 0, 1, 3)))


def _render_summary(summary: Summary) -> None:
    table = Table(show_header=False, box=None)
    table.add_row("escolheu a ferramenta certa", f"{summary.chose_right}/{summary.cases}")
    table.add_row("argumentos esperados", f"{summary.arguments_right}/{summary.cases}")
    table.add_row(
        "resposta citou o dado da ferramenta",
        f"{summary.answers_right}/{summary.answers_checked} [dim](não entra no gate)[/]",
    )
    table.add_row("argumentos inválidos", str(len(summary.invalid_args)))
    table.add_row("malformed_tool_use", str(summary.malformed))
    table.add_row("erros de ferramenta", str(summary.tool_errors))
    table.add_row(
        "stopReason", ", ".join(f"{k}={v}" for k, v in summary.stop_reasons.items()) or "nenhum"
    )
    table.add_row(
        "stopReason de falha",
        ", ".join(f"{k}={v}" for k, v in summary.failed_stop_reasons.items()) or "nenhum",
    )
    table.add_row("tokens", f"{summary.input_tokens} entrada / {summary.output_tokens} saída")
    if summary.errors:
        table.add_row("[red]erros de chamada[/]", str(summary.errors))
    console.print(table)


@app.command("eval")
def run_eval(
    casos: Path = typer.Option(DEFAULT_CASES, "--casos", help="Arquivo YAML com os casos."),
    runs: int = typer.Option(1, "--runs", "-n", help="Quantas vezes rodar a suíte inteira."),
    model: str = typer.Option(
        "", "--model", "-m", help="modelId a testar, sem mexer no .env (padrão: o do .env)."
    ),
) -> None:
    """Run the model gate: the cases in `evals/casos.yaml`, one billed call each."""
    settings = get_settings()
    # `--model` so that trying the next rung of the fallback ladder is one
    # command and not an edit to the `.env` that someone forgets to undo.
    model_id = model or settings.model_id
    console.print(
        f"[bold]modelo[/] {model_id}   [bold]região[/] {settings.region}   [bold]casos[/] {casos}\n"
    )
    try:
        cases = load_cases(casos)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        console.print(f"[red]não foi possível ler os casos[/] — {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1) from exc

    red = False
    for run in range(1, runs + 1):
        if runs > 1:
            console.print(f"[bold]execução {run}/{runs}[/]\n")
        with console.status("rodando…") as status:
            results = run_cases(
                cases,
                model=bedrock.chat_model(model_id=model_id),
                on_case=lambda case: status.update(f"caso {case.id}"),
            )
        for result in results:
            _render_case(result)
        summary = summarize(results)
        _render_summary(summary)
        red = red or not summary.passed
        console.print()

    if red:
        console.print(
            f"[red]GATE VERMELHO[/] — {model_id} não sustenta a tarefa deste estágio.\n"
            "Suba um degrau da escada de fallback (trocar BEDROCK_MODEL_ID e habilitar o "
            "model access) antes de construir RAG e persistência em cima."
        )
        raise typer.Exit(code=1)
    console.print(f"[green]GATE VERDE[/] — {model_id} escolhe a ferramenta certa.")


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", help="Show the version and exit."),
) -> None:
    if version:
        console.print(f"tom {__version__}")
        raise typer.Exit()
