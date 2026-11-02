# Empório da Música — customer service agent

Prototype of a customer service agent for the fictional store **Empório da Música**. Command-line interface, in Python, with the
models running on Amazon Bedrock.

## Requirements

- Python 3.11+
- uv (resolves dependencies from `uv.lock`)
- An AWS account with Bedrock access enabled in the configured region

## Installation

```bash
uv sync                      # creates the .venv and installs from the lockfile
cp .env.example .env         # then fill in the variables (see below)
```

## Configuration

All variables live in the `.env` at the project root (not version controlled).
`.env.example` documents each one:

| variable | purpose |
| --- | --- |
| `AWS_REGION` | region where the models are enabled (default `us-east-1`) |
| `AWS_BEARER_TOKEN_BEDROCK` | Bedrock API key — the simplest way to authenticate |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | alternative to the API key, if you use IAM credentials |
| `BEDROCK_MODEL_ID` | conversation model (default `google.gemma-3-4b-it`) |
| `BEDROCK_EMBED_MODEL_ID` | embeddings model (default `amazon.titan-embed-text-v2:0`) |
| `DATA_DIR` | where the store data lives (default `./data`) |

## Usage

```bash
uv run emporio check          # validates credentials and model access
uv run emporio models -s nova # lists the region's text models
uv run emporio chat           # not implemented yet
```

`emporio check` makes exactly two billed calls, one for conversation and one for
embeddings, and exits with code 1 if either fails, explaining the error (model
access not enabled, modelId invalid in the region, expired credential).

## Structure

```
src/emporio/
├── cli.py        # commands (typer + rich)
├── bedrock.py    # single entry point to the models
└── config.py     # configuration read from .env
```
