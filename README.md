# atomir

[![PyPI](https://img.shields.io/pypi/v/atomir)](https://pypi.org/project/atomir/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/atomir?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/atomir)
[![Python](https://img.shields.io/pypi/pyversions/atomir)](https://pypi.org/project/atomir/)

Atomic memory infrastructure for agents. **Memory is atomic on both ends:**
atomic facts on write (extract → reconcile), atomic sub-question decomposition on
read (decompose → retrieve per sub-question → union).

## The thesis

Most memory systems store raw text blobs and retrieve with a single fuzzy
similarity search. atomir does the opposite at both ends:

- **Write** — a message is split into small, self-contained facts, and each is
  *reconciled* into memory (ADD new, UPDATE a changed value keeping history,
  DELETE what's no longer true, NOOP duplicates). A similarity gate biases toward
  ADD so distinct facts never over-merge.
- **Read** — a question is decomposed into atomic sub-questions (only when it
  helps), each retrieved independently, then results are unioned. This surfaces
  facts a single whole-question embedding misses.

## Vendor-neutral by construction

The LLM, the embedder, and the vector store are each an **interface** chosen at
runtime by config (`{provider, config}` blocks). The engine imports only the
interfaces — never a provider SDK or vendor name. Swapping providers is one
config change; adding a new one is a single class plus a registry line.
Defaults use `fake` backends, so everything runs with **no external keys**.

| Slot | Built-in providers |
|---|---|
| LLM | `fake`, `groq`, `openai`, `anthropic`, `ollama` |
| Embedder | `fake`, `jina`, `voyage`, `openai`, `ollama` |
| Store | `json`, `qdrant` |

Each provider is selected by `LLM_BACKEND` / `EMBED_BACKEND` / `STORE_BACKEND`
with its key in `LLM_API_KEY` / `EMBED_API_KEY` (Ollama and the fakes need none).
`LLM_BASE_URL` / `EMBED_BASE_URL` can point at self-hosted or proxy endpoints.

## Install

```bash
pip install -e .                 # core (offline: fake LLM + fake embedder + JSON store)
pip install -e ".[qdrant]"       # add the Qdrant backend
pip install -e ".[api]"          # add the FastAPI server
pip install -e ".[all]"          # everything
```

`groq` and `jina` need no extra — they call their HTTP APIs over the standard
library.

## Quickstart — embedded, no Docker

Runs fully offline with the default `fake` backends:

```python
from atomir.assembly import build_memory_service

mem = build_memory_service()                      # backends chosen by .env
mem.add("user123", "I'm vegetarian and my manager is Dana Lopez.")
mem.add("user123", "I'm working on Project Atlas.")

hits = mem.search("user123", "who should I email about my project?")
print(hits["subquestions"])                       # the sub-questions it asked
for r in hits["results"]:
    print(r["text"], round(r["score"], 3))

mem.get_all("user123")
mem.delete("user123", fact_id)
mem.reset("user123")
```

To use real providers, copy `.env.example` to `.env` and set the keys/backends.

## Production — Docker Compose (API + Qdrant server)

```bash
cp .env.example .env             # optional: add real keys; without it, LLM/embedder run fake
docker compose up --build        # brings up the API and a Qdrant server
```

The API points at the Qdrant service via `STORE_URL=http://qdrant:6333`. Then:

```bash
curl -XPOST localhost:8000/memories -H 'content-type: application/json' \
  -d '{"user_id":"u1","text":"My manager is Dana."}'
curl -XPOST localhost:8000/search -H 'content-type: application/json' \
  -d '{"user_id":"u1","query":"who is my manager?"}'
```

## HTTP endpoints

| Method | Path | Body / query | Returns |
|---|---|---|---|
| POST | `/memories` | `{user_id, text}` | `{operations, facts}` |
| POST | `/search` | `{user_id, query, k?, decompose?}` | `{subquestions, results}` |
| POST | `/answer` | `{user_id, query, k?, decompose?}` | `{answer, subquestions, results}` |
| GET | `/memories` | `?user_id=` | list of facts |
| DELETE | `/memories/{id}` | `?user_id=` | `{deleted, id}` (404 if absent) |
| DELETE | `/memories` | `?user_id=` | `{reset}` |
| GET | `/health` | — | `{status, store, llm, embedder}` |

`MemoryClient(base_url)` (in `atomir.client`) wraps these with the same method
names and return shapes.

## Using atomir with agents & frameworks

atomir is the *memory*, not the model. Your app (or agent) does the reasoning;
atomir supplies the relevant facts. The universal pattern is **recall before,
remember after**:

1. **Recall** relevant facts for the incoming message → inject into the prompt.
2. Your LLM/agent produces the answer.
3. **Remember** the turn → atomir extracts atomic facts and reconciles them.

Runnable examples live in [`examples/`](examples/).

### Memory topology — choose the `user_id`

`user_id` is an opaque namespace, so *how you pick it* defines your memory
topology (use `scope()` from the LangGraph helper to build these consistently):

| Namespace | Meaning |
|---|---|
| `"user:123"` | **Shared** — the whole agent crew sees one memory of the user (default) |
| `"user:123#agent:researcher"` | **Agent-private** — an agent's own scratchpad |
| `"acme|user:123"` | **Multi-tenant** — tenant + user hierarchy |

### In-process vs. as a service

- **In-process:** `build_memory_service()` — direct calls, fastest.
- **As a service:** run `uvicorn atomir.api:app`, then give every agent a
  `MemoryClient(url)`. Best for distributed multi-agent systems — all agents
  share one memory; the per-user lock keeps concurrent writes safe.

### LangChain — `pip install "atomir[langchain]"`

`AtomirRetriever` is a real `BaseRetriever` (drops into any chain); `AtomirMemory`
is a recall/remember helper. Both accept a `MemoryService` or a `MemoryClient`.

```python
from atomir.assembly import build_memory_service
from atomir.integrations.langchain import AtomirMemory

mem = AtomirMemory(build_memory_service(), user_id="user:123")
mem.remember("I'm vegetarian and my manager is Dana.")
context   = mem.recall("who is my manager?")   # formatted string for a prompt
retriever = mem.as_retriever()                 # a real LangChain BaseRetriever
```

### LangGraph & multi-agent — `pip install "atomir[langgraph]"`

Ready-made `recall`/`remember` nodes (plain `state → state` callables — the
integration itself needs no framework) plus `scope()` for namespacing:

```python
from langgraph.graph import StateGraph, START, END
from atomir.assembly import build_memory_service
from atomir.integrations.langgraph import recall_node, remember_node, scope

mem = build_memory_service()
g = StateGraph(dict)
g.add_node("recall", recall_node(mem))       # -> state["memories"]
g.add_node("agent", my_agent)                # reads state["memories"]
g.add_node("remember", remember_node(mem))   # stores state["input"]
g.add_edge(START, "recall"); g.add_edge("recall", "agent")
g.add_edge("agent", "remember"); g.add_edge("remember", END)
app = g.compile()
```

In a multi-agent graph, agents **coordinate through shared memory**: the
researcher writes findings, the writer reads them — persisting across runs, not
just within the graph's state.

**Write policy:** don't let every agent `remember()` every message — memory fills
with noise. Store confirmed/durable findings only, often from a single
consolidation node. Concurrent writes to one namespace are serialized by the
per-user lock, so they're safe.

## Configuration

All config is read from the environment (see `.env.example`): `LLM_BACKEND`,
`LLM_API_KEY`, `MODEL`, `EMBED_BACKEND`, `EMBED_API_KEY`, `EMBED_DIM`,
`RECONCILE_MIN_SIM`, `STORE_BACKEND`, `COLLECTION`, `STORE_URL`, `STORE_PATH`.

## Composed answers

`search` returns facts (+ sub-questions); `answer` additionally composes a
grounded final sentence from them (LLM told to use only the retrieved facts):

```python
mem.answer("user123", "who should I email about my project?")
# -> {"answer": "...", "subquestions": [...], "results": [...]}
```

## Robustness

Provider calls retry transient failures (HTTP 429 rate-limits and 5xx) with
backoff, honoring `Retry-After`. The JSON store writes atomically (temp file →
`fsync` → `os.replace`), so a crash mid-write can't corrupt the file.

## Known limitations

- **`RECONCILE_MIN_SIM` is embedder-dependent.** The default `0.5` is tuned for
  Jina via `eval/tune.py` (it sits between measured unrelated ~0.45 and
  same-attribute ~0.60 similarity). Switching embedders? Re-run `eval/tune.py`.
- **JSON store is single-process.** Writes are now atomic (no corruption), but
  it holds no cross-process lock and rewrites the whole file per save — great for
  dev and small deployments; use Qdrant at scale.
- **No multi-fact transactions.** Each write is atomic and per-user serialized
  (Step 9), and a partial `add` is self-healing on retry (reconcile NOOPs facts
  already stored). Full all-or-nothing rollback across an `add` is deferred
  (DECISION #5) — open an issue if you need it.
