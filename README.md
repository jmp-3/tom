# Empório da Música: customer service agent

Prototype of a customer service agent for the fictional store **Empório da Música**, which sells musical
instruments in Campo Grande/MS. Command-line interface, in Python, with the models running on Amazon Bedrock.

Price and stock counts are deterministic, auditable, tested code. The agent's job is to read what the code returns.

**Language convention.** Identifiers, comments, docstrings and this README are in English, tool docstrings
included -- they are code, and the model reads them as well in either language. Everything the *customer*
reads is in Portuguese: the persona, the CLI output, the status and price labels inside the tool payloads,
and the instructions the model is given when something fails. The data itself is Portuguese and is never
translated.

## Requirements

- Python 3.11+
- uv (resolves dependencies from `uv.lock`)
- An AWS account with Bedrock access enabled in the configured region

## Installation

```bash
uv sync                      # creates the .venv and installs from the lockfile
cp .env.example .env         # then fill in the variables (see below)
uv run emporio build-index   # builds the policy index from the PDF (see below)
```

## Configuration

All variables live in the `.env` at the project root (not version controlled).
`.env.example` documents each one:

| variable | purpose |
| --- | --- |
| `AWS_REGION` | region where the models are enabled (default `us-east-1`) |
| `AWS_BEARER_TOKEN_BEDROCK` | Bedrock API key, the simplest way to authenticate |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | alternative to the API key, if you use IAM credentials |
| `BEDROCK_MODEL_ID` | conversation model (default `us.anthropic.claude-haiku-4-5-20251001-v1:0`) |
| `BEDROCK_EMBED_MODEL_ID` | embeddings model (default `amazon.titan-embed-text-v2:0`) |
| `DATA_DIR` | where the store data lives (default `./data`) |

## Usage

```bash
uv run emporio check          # validates credentials and model access
uv run emporio models -s nova # lists the region's text models
uv run emporio build-index    # (re)builds the policy index into index/

uv run emporio chat                      # anonymous session: public information only
uv run emporio chat --cliente-id 3       # identified session, as if logged in
uv run emporio chat --cliente-id 3 --debug --transcript conversa.md

uv run emporio eval           # the model gate: 6 cases, 1 billed call each
uv run emporio eval -n 2      # the gate twice, which is the criterion
uv run emporio eval -m us.amazon.nova-lite-v1:0   # try a candidate model, .env untouched

uv run pytest                 # data layer, agent mechanics and the gate's scoring
uv run mypy                   # type check
uv run ruff check             # lint
uv run ruff format            # format in place; add --diff to preview first
```

`emporio chat` keeps the conversation in memory (`InMemorySaver`), prints with `rich`, and ends on
`/sair`. `--debug` prints, per model call, the tools the request carried and their arguments *as the model
sees them*, then every call and tool result and the `stopReason`. `--transcript` writes the conversation to
markdown after every turn, which is where the example conversations come from.

`scripts/smoke_tool_use.py` is the gate test (one toy tool) and is kept runnable for the record;
`emporio eval` is the real one.

`mypy` and `ruff` are both configured in `pyproject.toml`.

`emporio check` makes exactly two billed calls, one for conversation and one for
embeddings, and exits with code 1 if either fails, explaining the error (model
access not enabled, modelId invalid in the region, expired credential). The two
halves fail independently: the embeddings model is enabled separately from the
conversation model in Bedrock > Model access, so an account can pass the first
half and be refused by `build-index`.

`emporio build-index` is a setup step, not a command anyone runs twice: it reads
`data/políticas_da_loja.pdf` and writes `index/`, which is not committed. It
makes one billed embeddings call. `emporio chat` says so on startup when the
index is missing or no longer matches.

## Structure

```
src/emporio/
├── cli.py            # commands (typer + rich): check, models, chat, eval
├── bedrock.py        # single entry point to the models
├── config.py         # configuration read from .env
├── evals.py          # the model gate: loads evals/casos.yaml and scores a run
├── agent/
│   ├── agent.py      # the loop: create_agent(model, tools, system_prompt, context)
│   ├── prompt.py     # the persona, and the session half of it
│   ├── tools.py      # the query functions wrapped as @tool
│   ├── context.py    # SessionContext: who the session belongs to
│   ├── middleware.py # tool scope per session, stopReason, tool errors
│   └── toy_tools.py  # the gate's tool, plus the one that always fails
├── data/
│   ├── loader.py     # CSV loading and the derived business fields
│   ├── models.py     # pydantic input schemas and the shared limit constants
│   └── queries.py    # pure query functions (what the tools wrap)
└── rag/
    ├── chunker.py    # the policy PDF cut into one chunk per numbered section
    ├── index.py      # embeddings, the two index files, and the staleness check
    └── retriever.py  # BaseRetriever over cosine similarity in numpy
index/                # generated by `emporio build-index`, not committed
evals/
└── casos.yaml        # the gate's cases: question, mode, expected decision
scripts/
└── smoke_tool_use.py # the gate, kept for the record
tests/
├── conftest.py       # the shared dataset fixture and the fixed reference date
├── test_loader.py    # row counts, parsing, money and date arithmetic
├── test_queries.py   # pricing, catalog filters, customer scope and PII
├── test_agent.py     # tool shape, session scope, persona, errors, offline loop
├── test_rag.py       # chunking, the index files, the ranking (no Bedrock call)
└── test_evals.py     # the case file, and how a case is scored
```

## The agent

LangChain 1.0 and LangGraph, which since 1.0 are one stack and not two choices. `create_agent` returns a
compiled LangGraph graph that owns the loop -- model, tool, model again, until the model answers without
calling anything -- and accepts the middleware for future functions. `ChatBedrockConverse` puts the
Bedrock Converse API underneath, so changing models stays a matter of changing `BEDROCK_MODEL_ID`.

`emporio.bedrock` keeps both paths on purpose: `chat_model()` for the agent, and the raw `converse()`

### The persona

`agent/prompt.py` holds it, derived from the service guidelines of the policy manual (sec. 7.1). The assistant is called **Tom**. Everything else in it is a prohibition, because the
model never computes a price, never counts stock and never estimates a deadline -- those come from a tool.
The rules are numbered and imperative, one line each, because a small model follows an enumerated list far
better than a paragraph; when it starts ignoring one, the fix is to shorten it, not to argue with it. That is
not a figure of speech: the accessory rule below went from one three-clause rule to two one-clause rules
after the gate caught the model searching the catalog for guitar strings.

Nothing in the persona says *how* to emit a tool call. `create_agent` and the Converse API own that
mechanism, and text competing with it produces malformed calls. The persona says *when* to use a tool.

The persona travels in `system_prompt`; the half that depends on the session -- the customer's first name,
and whether order questions are answerable at all -- is appended per model call by a `@dynamic_prompt`
middleware. One text source, so the fallback of prefixing the persona to the first user turn (for a model
that ignores the system block) is a call site and not a second copy.

### The two session modes

The chat is simulated as if it lived in the logged-in area of the site. `--cliente-id N` stands in for what
the login would provide; it is never asked for in conversation.

| | without `--cliente-id` | with `--cliente-id` |
| --- | --- | --- |
| tools sent to the model | catalog, categories and policies | the same **+** `list_orders` and `get_order` |
| kind of data | public information | personal data of that one customer |
| question about an order | points at the login, without revealing whether the order exists | answered |

Two framework features carry this, and neither is a prompt:

- **`context_schema` + `ToolRuntime`.** An order tool declares `runtime: ToolRuntime` and reads
  `runtime.context.customer_id`. LangChain strips that parameter from the schema, so `customer_id` appears
  in no tool the model is offered and cannot be filled in by it. Tools are therefore defined by their
  signature and docstring, **never** by `args_schema`: a tool that declares one does not get its
  `ToolRuntime` injected, and the privacy rule would break in silence.
- **`wrap_model_call` + `request.override(tools=...)`.** All seven tools are registered on the agent; the
  middleware decides which ones each call is offered. In an anonymous session the order tools are not
  refused, they are not sent -- the model cannot call what it does not have, and `--debug` shows the
  request's own tool list, which is the only honest way to check it. The filter removes a named set rather
  than keeping one, so a new public tool -- `search_policies` was the first -- is public without being
  listed anywhere: the mistake worth preventing is a personal-data tool reaching an anonymous session.

### The model gate and the fallback ladder

`emporio eval` runs the seven cases in `evals/casos.yaml`, one per kind of decision the agent has to make:
price-range search, product detail, one order by number, "what are my orders?" with no number, an *indirect*
question about orders ("am I owing anything?"), a question that needs no tool, and a request for an
accessory the store does not sell. Each case declares the mode it runs in. It scores the choice of tool, the
arguments (valid against the tool's own schema, and the expected values where the case declares them) and
counts `malformed_tool_use` -- the stop reason that means the model tried to call a tool and produced
something unparseable, which is invisible in the answer.

A case may also declare `ferramentas_aceitas`: extra calls that do not count as a miss because they are a
drill-down rather than a different decision -- `list_orders` and then `get_order` on the order it just
found. What is scored is the decision, not the number of round trips.

## The data layer

No LLM call happens anywhere inside `emporio.data`. Every price, stock count and deadline the agent
states is computed here, deterministically and under test.

### Query functions

The API surface. These will be the function-calling tools.

Every money field comes back twice: `effective_price` for code to compare, and `effective_price_label`
(`"R$ 1.299,00"`) as the string the agent quotes. The reason is in the measurement above -- the model
reformats some floats wrongly -- and the naming follows `status_label` and `payment_label`, which already
exist for the same idea: the agent should never have to construct what it says out of a raw value.

### Data scope and PII

The chat is simulated as if it lived inside the logged-in area of the e-commerce site.
`customer_id` is mandatory and has no default; it
is injected by the session and is absent from every schema the model sees; a refused lookup returns the
same answer whether the order is missing or belongs to someone else; no name, phone or e-mail is ever
projected; and there is no `search_customers` or `list_customers` to call.

## The policy manual

The store's rules live in `data/políticas_da_loja.pdf`: eight pages of prose, ten numbered sections, four
tables. `emporio.rag` is what makes them answerable, and `search_policies` is the tool that reaches them.

## Assumptions

- **Authenticated channel.** The chat is simulated as if it lived inside the logged-in area of the
  e-commerce site: the customer's identity arrives ready and is never collected in conversation. In the
  prototype the `--cliente-id` flag replaces the login: it is a stand-in for the `customer_id` a session
  would already hold, not a feature of the product. The reason it is not asked for in the conversation is
  that `customers.csv` has no credential field, so asking for a phone number would identify the customer
  without proving anything -- and the agent is forbidden to ask for one (LGPD, policy manual sec. 9).
- **The CSV is the source of truth about availability.** The policy PDF advertises "more than 300
  instruments" across 9 categories; the CSV holds 65 products across 6. Categories 6, 7 and 8 have no
  product behind them, so `list_categories` omits them: offering a category the store cannot sell from is
  worse than a shorter list.
- **The reference date is the system date.** Orders run from 2025-10 to 2026-03 and there is no variable to
  fix "today" (deliberate). Since every delivery estimate is already in the past, the 7-day
  right-of-withdrawal window of sec. 4.1 is closed for every order in the dataset, and the day count changes
  with each day that passes. Accepted consequence: the transcripts are not reproducible
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
