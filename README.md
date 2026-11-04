# Tom: customer service agent

**Tom** is the virtual attendant of the fictional **Empório da Música**, a store that sells musical
instruments. Tom answers about the catalog, about the orders of the customer the
session belongs to, and about the store's policies -- and nothing from memory: every price, stock count,
deadline and rule stated comes from a typed tool over the store's data, or from a retrieved excerpt of
the policy manual. Command-line interface, in Python, with the models running on Amazon Bedrock.

```bash
uv run tom chat                    # anonymous: catalog and policies
uv run tom chat --cliente-id 3     # identified: that customer's orders as well
```

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
uv run tom build-index   # builds the policy index from the PDF (see below)
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
uv run tom check          # validates credentials and model access
uv run tom models -s nova # lists the region's text models
uv run tom build-index    # (re)builds the policy index into index/

uv run tom chat                      # anonymous session: public information only
uv run tom chat --cliente-id 3       # identified session, as if logged in
uv run tom chat --cliente-id 3 --debug --transcript conversa.md

uv run tom eval           # the model gate: 6 cases, 1 billed call each
uv run tom eval -n 2      # the gate twice, which is the criterion
uv run tom eval -m us.amazon.nova-lite-v1:0   # try a candidate model, .env untouched

uv run pytest                 # data layer, agent mechanics and the gate's scoring
uv run mypy                   # type check
uv run ruff check             # lint
uv run ruff format            # format in place; add --diff to preview first
```

`tom chat` keeps the conversation in memory (`InMemorySaver`), prints with `rich`, and ends on
`/sair`. `--debug` prints, per model call, the tools the request carried and their arguments *as the model
sees them*, then every call and tool result and the `stopReason`. `--transcript` writes the conversation to
markdown after every turn, which is where the sessions in `example-chats/` come from.

`scripts/smoke_tool_use.py` is the gate test (one toy tool) and is kept runnable for the record;
`tom eval` is the real one.

`mypy` and `ruff` are both configured in `pyproject.toml`.

`tom check` makes exactly two billed calls, one for conversation and one for
embeddings, and exits with code 1 if either fails, explaining the error (model
access not enabled, modelId invalid in the region, expired credential). The two
halves fail independently: the embeddings model is enabled separately from the
conversation model in Bedrock > Model access, so an account can pass the first
half and be refused by `build-index`.

`tom build-index` is a setup step, not a command anyone runs twice: it reads
`data/políticas_da_loja.pdf` and writes `index/`, which is not committed. It
makes one billed embeddings call. `tom chat` says so on startup when the
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
index/                # generated by `tom build-index`, not committed
evals/
└── casos.yaml        # the gate's cases: question, mode, expected decision
example-chats/        # recorded sessions, written by `tom chat --transcript`
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

`tom eval` runs the seven cases in `evals/casos.yaml`, one per kind of decision the agent has to make:
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

## Design decisions

The agent runs on LangChain 1.0 and LangGraph instead of on a loop written by hand. `create_agent` owns the
model-tool-model cycle and exposes it through middleware, and middleware is where the three behaviours
specific to this project live: which tools a session may use, the session half of the system prompt, and the
conversion of a tool exception into a message. `emporio.bedrock` keeps a raw `converse()` path as well. That
path is what made the choice of model trustworthy, because a model that never called a tool could be
retested with no framework in between.

`ChatBedrockConverse` puts the Bedrock Converse API underneath, so the model is a configuration variable.
The project started on a 4B model and ended on `claude-haiku-4-5`, and the two runs differ by one line of
`.env`. `tom eval` scores a candidate on seven cases, so a change of model follows a measurement.

Questions about the catalog and the orders go to typed tools, and questions about the policies go to semantic
retrieval. The catalog and the orders are structured, 65 products and 20 orders over a dozen columns, which a
deterministic filter covers completely while every number it returns stays auditable; no model call happens
inside `emporio.data`. The policies are prose, and "me arrependi da compra, posso devolver?" has no column to
filter on. Each mechanism is wrong for the other kind of data: a price ranked by cosine similarity cannot be
audited, and a policy flattened into a table loses the conditions it depends on.

Retrieval uses numpy and no vector database. The manual is 25 chunks, so a search is one dot product against
a 25 by 1024 matrix. FAISS or Chroma would add a dependency, a storage format, and a second place for the
index to go stale, in exchange for a search that is already instantaneous. `rag/retriever.py` subclasses
`BaseRetriever`, which keeps the decision reversible: if the manual grows to a thousand pages, that one class
changes and `search_policies` does not.

The index is a build step and not a committed file. `tom build-index` derives `index/` from the PDF in about a
second, and the two files record the embedding model, the dimension, and a digest of the PDF, which `tom chat`
compares against the environment on startup. Vectors produced by one embedding model and ranked against a
query embedded by another still give a valid cosine and a meaningless order, so without that check the agent
keeps answering fluently and cites the wrong section.

The framework enforces the two session modes, and the prompt does not. The chat is simulated inside the
logged-in area of the site, so the customer's identity arrives with the session: `customers.csv` holds no
credential, and asking for a phone number would identify a customer without proving anything. `customer_id`
reaches the tools through `ToolRuntime` and appears in no schema the model receives, and in an anonymous
session the order tools are absent from the request rather than refused when called. A prompt can be argued
out of a rule, and a tool that was never offered cannot be called.

The persona states prohibitions and says nothing about how to call a tool. The model never computes a price,
counts stock, or estimates a deadline, because each of those comes from a tool. The call format belongs to
`create_agent` and the Converse API, and text that competes with that mechanism produces malformed calls.

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

## AI usage

Claude was used throughout this project: discussing the design and planning the stages, writing the code,
this README and the docstrings, generating the test suite and the eval cases, and reviewing the diffs. The
decisions above are mine, and everything committed was read, run and tested before it was kept.
