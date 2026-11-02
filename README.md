# Empório da Música: customer service agent

Prototype of a customer service agent for the fictional store **Empório da Música**, which sells musical
instruments in Campo Grande/MS. Command-line interface, in Python, with the models running on Amazon Bedrock.

Price and stock counts are deterministic, auditable, tested code. The agent's job is to read what the code returns.

**Language convention.** Identifiers, comments, docstrings and this README are in English. Everything the
model or the customer reads is in Portuguese: CLI output, tool and field descriptions, status labels. The
data itself is Portuguese and is never translated.

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
| `AWS_BEARER_TOKEN_BEDROCK` | Bedrock API key, the simplest way to authenticate |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | alternative to the API key, if you use IAM credentials |
| `BEDROCK_MODEL_ID` | conversation model (default `google.gemma-3-4b-it`) |
| `BEDROCK_EMBED_MODEL_ID` | embeddings model (default `amazon.titan-embed-text-v2:0`) |
| `DATA_DIR` | where the store data lives (default `./data`) |

## Usage

```bash
uv run emporio check          # validates credentials and model access
uv run emporio models -s nova # lists the region's text models
uv run emporio chat           # not implemented yet
uv run pytest                 # the data layer, no billed calls
uv run mypy                   # type check
uv run ruff check             # lint
uv run ruff format            # format in place; add --diff to preview first
```

`mypy` and `ruff` are both configured in `pyproject.toml`.

`emporio check` makes exactly two billed calls, one for conversation and one for
embeddings, and exits with code 1 if either fails, explaining the error (model
access not enabled, modelId invalid in the region, expired credential).

## Structure

```
src/emporio/
├── cli.py            # commands (typer + rich)
├── bedrock.py        # single entry point to the models
├── config.py         # configuration read from .env
└── data/
    ├── loader.py     # CSV loading and the derived business fields
    ├── models.py     # pydantic input schemas, which generate the toolSpec
    └── queries.py    # pure query functions (the agent's tools)
tests/
├── conftest.py       # the shared dataset fixture and the fixed reference date
├── test_loader.py    # row counts, parsing, money and date arithmetic
└── test_queries.py   # pricing, catalog filters, customer scope and PII
```

## The data layer

No LLM call happens anywhere inside `emporio.data`. Every price, stock count and deadline the agent
states is computed here, deterministically and under test.

### Query functions

The API surface. These will be the function-calling tools.

### Data scope and PII

The chat is simulated as if it lived inside the logged-in area of the e-commerce site.
`customer_id` is mandatory and has no default; it
is injected by the session and is absent from every schema the model sees; a refused lookup returns the
same answer whether the order is missing or belongs to someone else; no name, phone or e-mail is ever
projected; and there is no `search_customers` or `list_customers` to call.


## Assumptions

- **Authenticated channel.** The chat is simulated as if it lived inside the logged-in area of the
  e-commerce site: the customer's identity arrives ready and is never collected in conversation. In the
  prototype the `--cliente-id` flag of stage 3 replaces the login.
- **The CSV is the source of truth about availability.** The policy PDF advertises "more than 300
  instruments" across 9 categories; the CSV holds 65 products across 6. Categories 6, 7 and 8 have no
  product behind them, so `list_categories` omits them: offering a category the store cannot sell from is
  worse than a shorter list.
- **The reference date is the system date.** Orders run from 2025-10 to 2026-03 and there is no variable to
  fix "today" (deliberate). Since every delivery estimate is already in the past, the 7-day
  right-of-withdrawal window of sec. 4.1 is closed for every order in the dataset, and the day count changes
  with each day that passes. Accepted consequence: the stage 6 transcripts are not reproducible
  byte-for-byte, and the returns demo shows the agent declining while citing the rule. The test suite
  injects the date as a function argument rather than reading the system date, so the suite does not
  change result on its own.
- **`estimated_delivery` is used in place of the receipt date.** sec. 4.1 counts the 7 days from when the customer
  receives the product; the dataset records only the estimate, which is the closest available proxy.
- **No historical unit prices exist.** The dataset has the order total and today's catalog price, which may
  have changed since. Order items therefore carry name and quantity only, so the agent cannot state what
  was paid per item and will not invent it.
- **Price filters compare against `effective_price`**, the price the customer would pay today: "a guitar up
  to 2000" is a statement about the bill, not about the list price.
- **At most one active promotion per product.** True of the data, not enforced by the file, so the largest
  discount wins, a rule that does not depend on row order.
