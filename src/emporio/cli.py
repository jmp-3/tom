"""Agent CLI."""

from __future__ import annotations

import typer
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from rich.console import Console
from rich.table import Table

from emporio import __version__, bedrock
from emporio.config import get_settings

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
        "o modelId não é aceito nesta região. Confira a lista com `emporio models`."
    ),
    "ResourceNotFoundException": (
        "modelo inexistente nesta região. Confira a lista com `emporio models`."
    ),
    "ThrottlingException": "limite de requisições atingido; tente de novo em alguns segundos.",
    "UnrecognizedClientException": "credencial inválida ou expirada (confira o .env).",
}


def _describe_failure(exc: Exception) -> str:
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        message = exc.response.get("Error", {}).get("Message", str(exc))
        key = code
        if code == "AccessDeniedException" and "Authentication failed" in message:
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
                system="Você atende clientes de uma loja de instrumentos musicais em Campo Grande/MS.",
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
        response = bedrock.get_client("bedrock").list_foundation_models(
            byOutputModality="TEXT"
        )
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


@app.command()
def chat() -> None:
    """Talk to the agent (not implemented yet)."""
    console.print(
        "[yellow]Ainda não implementado. "
        "Use `emporio check` para validar o acesso ao Bedrock."
    )
    raise typer.Exit(code=1)


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", help="Show the version and exit."),
) -> None:
    if version:
        console.print(f"emporio {__version__}")
        raise typer.Exit()
