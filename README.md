# agentic-analyst

Ask a plain-English question about a CSV and get a reviewed analyst's report back. A
LangGraph pipeline of four agents interprets the question, computes the statistics in
pandas, draws the charts, and writes the findings — then stops and waits for me to
approve, revise, or reject before the report is committed anywhere.

```
$ uv run python run.py "Which contract type has the highest churn rate?"

Monthly    42.71% churn  (n=3,875)
One year   11.27% churn  (n=1,473)
Two year    2.83% churn  (n=1,695)
```

<!-- Demo gif goes here. The shot worth capturing is the kill-and-resume:
     run to the review gate, kill the process, resume from the checkpoint in a new one.
     ![demo](docs/demo.gif) -->

## Why it's built this way

Four agents is not the interesting part. The interesting part is four constraints I set
before writing any code, each of which shaped the architecture.

### No raw data ever reaches an LLM

The obvious way to build "chat with your data" is to paste rows into the prompt. That
breaks immediately: you burn the context window on any real dataset, you pay per row on
every question, and you're asking a language model to do arithmetic.

So every number here is computed in pandas. The models decide *what* to compute and
interpret what comes back — they never see a row.

I enforced this structurally rather than by prompt discipline. The DataFrame is never in
graph state at all; state carries a `dataset_id` string and the frame lives in an
in-process registry that nodes fetch from. A model cannot leak data that was never put
in front of it, and the checkpointer isn't serialising 7,043 rows on every step. What
the agents *do* receive is a schema card — column names, dtypes, null counts,
cardinality, and distinct labels only for low-cardinality categoricals. Columns above 12
distinct values are classed as identifiers and contribute a count and nothing else, so
no customer ID is ever in a prompt. On top of that, `enforce_budget()` gates every
model-bound payload and **raises** past the cap rather than truncating — a guardrail
that quietly trims is one you discover in the bill.

The payoff is measurable: duplicating the dataset from 7,043 to 14,086 rows changes the
prompt by **0.02%** (4,397 → 4,398 characters). Token cost is flat in dataset size.
`verify.py` proves it rather than asserting it.

### The dirty data is handled on purpose

`TotalCharges` in this dataset is stored as text, and 11 rows contain a single space.
They are not `NaN` — `isna()` reports zero nulls across the entire frame, so a naive
missing-value check finds nothing at all. That's the trap, and a plain
`to_numeric(errors="coerce").dropna()` would silently delete those rows and skew every
downstream statistic.

I profiled them instead of guessing. All 11 have `Tenure = 0`: brand-new customers who
haven't been billed a cycle yet. So `0.0` is the *accurate* value, not an imputation.
The cleaning step coerces them, flags them with `is_new_customer`, and records all 11
customer IDs in the report that ships with the analysis. Row count in equals row count
out, asserted in code — nothing is ever dropped. The `Tenure == 0` assumption is
re-checked on every run and downgraded to a loud warning if it stops holding, rather
than trusting a profile I did once.

One more thing I found by profiling: this is a modified Telco export, not the canonical
Kaggle file. Columns are TitleCase (`Gender`, `Tenure`), `Contract` uses `Monthly`
rather than `Month-to-month`, and `PaymentMethod` collapses the two check types into
`Manual`. Code copied from a standard Telco tutorial breaks on it. So there are no
hardcoded column names in the analysis path, and the loader asserts the expected schema
at load time and fails loudly rather than producing quietly wrong numbers.

### The human gate is a real interrupt, not an `input()`

`human_review` calls LangGraph's `interrupt()` and the graph genuinely halts, with state
persisted to SQLite keyed by `thread_id`.

I used `SqliteSaver` rather than `InMemorySaver` deliberately. In-memory would technically
satisfy "uses `interrupt()`", but the pause would only survive inside one process, which
makes it a glorified function call. With SQLite I can kill the process at the review gate
and resume in a brand new interpreter — which is the difference between demonstrating the
pattern and just naming it.

A useful side effect: the two front ends share one checkpoint database, so a run started
in the browser can be resumed from the terminal.

### Model output never becomes executable code

The other obvious approach is to have the LLM write pandas code and `exec()` it. I
rejected that outright — it's arbitrary code execution driven by model output, and it
fails in ways you can't catch.

Instead the agents emit Pydantic objects (`QueryPlan`, `AnalysisPlan`, `VizPlan`) with
strict JSON-schema decoding enforced server-side, and a whitelist of six pandas
operations executes them: `value_counts`, `group_agg`, `describe`, `correlate`,
`detect_outliers`, `crosstab_rate`. An operation or column outside the whitelist is
rejected before touching pandas. The worst a bad plan can do is name a column that
doesn't exist and get a clean error back.

The same principle covers charts: the viz agent emits a declarative `ChartSpec` and
matplotlib renders it. The model describes the chart; it never draws it.

### The sink is pluggable, and ships on local storage

Report output sits behind a `ReportSink` ABC with two implementations.

`LocalReportSink` is the default and always works: on approval, `report.md` and the
chart PNGs land in a timestamped folder under `outputs/`. A fresh clone runs end to end
with nothing but an OpenAI key.

`DriveMCPSink` is implemented and stays in the codebase as the optional path. I'll be
straight about why it isn't the default: both Google Drive MCP servers I evaluated turned
out to be read-only. `@modelcontextprotocol/server-gdrive` was archived in January 2025
and exposes only search and read; the alternative I tried was read-only in practice too.
Neither has a create or upload tool, so neither can fulfil the sink's write contract.

Rather than hardcode a tool name, the sink discovers the server's tools at runtime and
matches a create/upload tool by pattern. If it can't find one — or the server isn't
configured, or the connection fails — it logs the reason and falls back to local, and
the run still succeeds. That's the value of the abstraction: pointing this at a
write-capable server is a `.env` change, not a code change.

## Architecture

```
START
  │
  ▼
load_and_clean ── loads the CSV, cleans it, registers the frame in-process.
  │               State gets a dataset_id string — never the DataFrame.
  ▼
query ─────────── reads the question + schema card, emits a structured QueryPlan.
  │               The whitelist executes it. The agent computes nothing itself.
  ▼
analyze ───────── picks 0-4 further operations for depth, given the results so far.
  │               Re-applies the same filters so both passes describe the same
  │               population.
  ▼
visualize ─────── emits ChartSpec objects; matplotlib/seaborn renders them
  │               to outputs/chart_NN_*.png.
  ▼
narrate ───────── writes plain-English findings from the computed numbers only.
  │               Failed operations are shown to it on purpose — a report that
  │               hides what didn't compute is worse than one that mentions it.
  ▼
assemble_report ─ deterministic markdown assembly, no model involved.
  │
  ▼
human_review ──── interrupt() — THE GRAPH HALTS, STATE PERSISTS TO SQLITE
  │
  ├── approve ──→ commit ──→ outputs/ (or Drive over MCP, if configured) ──→ END
  ├── edit ─────→ back to narrate with my feedback (max 2 revisions)
  └── reject ───→ END, nothing written
```

### The agents

| Agent | Model | Job |
|---|---|---|
| Query | `gpt-5.4-mini` — structured, temp 0 | Turns the question + schema into a plan of filters and operations |
| Analysis | `gpt-5.4-mini` — structured, temp 0 | Picks further computations for statistical depth |
| Viz | `gpt-5.4-mini` — structured, temp 0 | Chooses charts declaratively |
| Narrator | `openai/gpt-oss-120b` on Groq — prose | Writes the findings from the computed numbers |

Every model is overridable per agent via `.env` using `provider:model` syntax, and they
are instantiated directly (`ChatOpenAI` / `ChatGroq`) rather than through
`init_chat_model` — the per-provider kwargs differ enough that the indirection costs
more than it saves.

The narrator is on a different provider deliberately. gpt-oss models tend to leak
reasoning fragments into tool-call slots, so I keep them away from anything the graph
has to parse. The narrator emits pure prose — no tool calls, no schema — which means
that failure mode has nowhere to land, and I get Groq's speed where it's safe. If
`GROQ_API_KEY` isn't set, the narrator falls back to the OpenAI model and everything
still runs.

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd agentic-analyst

uv venv
uv pip install -r requirements.txt

cp .env.example .env    # then add your keys
```

Only `OPENAI_API_KEY` is required. Groq, LangSmith, and Drive are all optional and
degrade cleanly.

## Running it

### Web UI

```bash
uv run streamlit run app.py
```

Pick a question, watch the agents run, and land on the review gate: charts, the draft
report, and **Approve / Request changes / Reject**. The sidebar shows which model each
agent is using, whether tracing is on, where the report will be committed, and the
current thread id.

### CLI

The same graph from a terminal. This is the scriptable path, and the clearest
demonstration that the pause is real.

```bash
# Ask a question. Runs to the review gate, then prompts.
uv run python run.py "Which contract type has the highest churn rate?"

# Stop at the gate without prompting; prints a thread id to resume with.
uv run python run.py "What drives churn among fiber optic customers?" --no-interaction

# Resume a paused run — in a completely new process.
uv run python run.py --resume run-8c1f6618 --decision approve

# Ask for changes instead.
uv run python run.py --resume run-8c1f6618 --decision edit \
  --feedback "Cut it to two paragraphs and lead with the dollar difference."

# Point it at a different CSV.
uv run python run.py "What's the average order value by region?" --dataset data/other.csv
```

The kill-and-resume is the demo I care about: start a run, let it reach the review gate,
kill the terminal, then run the `--resume` command in a fresh shell. It reloads from the
SQLite checkpoint and commits. That only works because the pause is a genuine
`interrupt()` with state living outside the process.

### Verifying the claims

```bash
uv run python verify.py
```

16 checks, no model calls, so it's free and fast. It confirms the cleaning behaviour, the
schema-drift guard, that no customer ID reaches a prompt, that prompt size is flat in
dataset size, and that the budget guard raises rather than truncates.

## Observability

Set `LANGCHAIN_API_KEY` for per-agent traces in LangSmith — token usage, latency, and
inputs/outputs for every node.

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=agentic-analyst
```

Built-in auto-instrumentation only; every graph node shows up as a span on its own.
Tracing is entirely optional — leave the key blank and the pipeline runs normally with
tracing explicitly disabled, so a stale env var can't break a fresh clone.

## Tech stack

**LangGraph** for orchestration, the `interrupt()` gate, and SQLite checkpointing ·
**LangChain** for model interfaces · **MCP** via `langchain-mcp-adapters` for the
optional Drive sink · **pandas** for all computation · **matplotlib** / **seaborn** for
charts · **Streamlit** for the web UI · **OpenAI** (`gpt-5.4-mini`) and **Groq**
(`gpt-oss-120b`) for the agents · **LangSmith** for optional tracing · **uv** for
environment and dependency management.

## Layout

```
app.py                        Streamlit front end
run.py                        CLI entrypoint / resume path
verify.py                     proves the claims above
src/agentic_analyst/
  config.py                   env, per-agent model registry, optional tracing
  models.py                   ChatOpenAI / ChatGroq construction
  state.py                    graph state + the Pydantic schemas agents emit
  graph.py                    StateGraph wiring, interrupt(), SQLite checkpointer
  hitl.py                     console review UI
  report.py                   markdown assembly
  data/
    source.py                 DataSource ABC  ← the SQL swap point
    csv_source.py             CSV implementation
    cleaning.py               declarative cleaning rules
    registry.py               keeps the DataFrame out of graph state
    summaries.py              schema cards + the payload budget
  tools/pandas_ops.py         the six-operation whitelist
  agents/                     query, analysis, viz, narrator
  sinks/                      base ABC, local (default), drive_mcp (optional)
```

## What's next

SQL input. `DataSource` is already an ABC with `CsvSource` as one implementation, and
nothing downstream knows where the rows came from — so a database source is a new
subclass rather than a rewrite. After that, streaming per-node progress to the UI
instead of a single spinner.

---

**Shashwat Goswami** · [wshashwatgoswami@gmail.com](mailto:wshashwatgoswami@gmail.com)
