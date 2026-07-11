# atomir

[![PyPI](https://img.shields.io/pypi/v/atomir)](https://pypi.org/project/atomir/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/atomir?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/atomir)
[![Python](https://img.shields.io/pypi/pyversions/atomir)](https://pypi.org/project/atomir/)

Atomic memory for LLM agents — **atomic on both ends**: facts are extracted and
reconciled on write; questions are decomposed into sub-questions on read.

## Why

Most memory systems store text blobs and retrieve with one fuzzy search. atomir doesn't:

- **Write** — split a message into atomic facts, then reconcile each (ADD /
  UPDATE-with-history / DELETE / NOOP). A similarity gate stops distinct facts over-merging.
- **Read** — decompose a question into sub-questions (only when useful), retrieve
  each, union the results. Surfaces facts a single-blob search misses.

Vendor-neutral: LLM, embedder, and store are interfaces chosen by config.
Defaults are `fake`, so it runs with no keys.

## Install

```bash
pip install atomir                          # core, offline-capable
pip install "atomir[qdrant,api]"            # Qdrant backend + HTTP API
pip install "atomir[langchain,langgraph]"   # framework integrations
```

## Quickstart

```python
from atomir.assembly import build_memory_service

mem = build_memory_service()          # backends from .env; defaults to fake (no keys)
mem.add("user123", "I'm vegetarian and my manager is Dana.")

hits = mem.search("user123", "who should I email about my project?")
print(hits["subquestions"], [r["text"] for r in hits["results"]])

mem.answer("user123", "who is my manager?")   # composed answer + the facts used
mem.get_all("user123"); mem.delete("user123", fact_id); mem.reset("user123")
```

Real providers: copy `.env.example` → `.env`, then set backends + keys.

## Providers

| Slot | Options | Config |
|---|---|---|
| LLM | `fake` `groq` `openai` `anthropic` `ollama` | `LLM_BACKEND`, `LLM_API_KEY`, `MODEL` |
| Embedder | `fake` `jina` `voyage` `openai` `ollama` | `EMBED_BACKEND`, `EMBED_API_KEY`, `EMBED_DIM` |
| Store | `json` `qdrant` | `STORE_BACKEND`, `STORE_URL` / `STORE_PATH` |

Adding a provider is one class + one registry line. `LLM_BASE_URL` /
`EMBED_BASE_URL` target self-hosted or proxy endpoints.

## Agent frameworks

atomir is the memory, not the model: **recall before, remember after**. Scope
memory by `user_id` — `"user:1"` (shared), `"user:1#agent:x"` (agent-private),
`"acme|user:1"` (multi-tenant).

**LangChain** — `AtomirRetriever` is a real `BaseRetriever`:

```python
from atomir.integrations.langchain import AtomirMemory
mem = AtomirMemory(build_memory_service(), user_id="user:1")
retriever = mem.as_retriever()
```

**LangGraph** — drop-in nodes for multi-agent graphs:

```python
from atomir.integrations.langgraph import recall_node, remember_node
g.add_node("recall", recall_node(mem))       # -> state["memories"]
g.add_node("remember", remember_node(mem))   # stores state["input"]
```

Agents coordinate through shared memory (persists across runs). Store durable
findings only. Runnable examples: [`examples/`](examples/).

## HTTP API

Run `uvicorn atomir.api:app` (or `docker compose up`). `MemoryClient(url)` wraps
these with identical shapes.

| Method | Path | Returns |
|---|---|---|
| POST | `/memories` `{user_id, text}` | `{operations, facts}` |
| POST | `/search` `{user_id, query, k?, decompose?}` | `{subquestions, results}` |
| POST | `/answer` `{user_id, query, ...}` | `{answer, subquestions, results}` |
| GET | `/memories?user_id=` | facts |
| DELETE | `/memories/{id}?user_id=` | `{deleted, id}` |
| DELETE | `/memories?user_id=` | `{reset}` |
| GET | `/health` | `{status, store, llm, embedder}` |

## Limitations

- `RECONCILE_MIN_SIM` (default `0.5`) is embedder-dependent — re-tune with
  `eval/tune.py` when you switch embedders.
- JSON store: atomic writes, but single-process and rewrites the whole file —
  dev / small scale only; use Qdrant otherwise.
- No multi-fact transactions; a partial `add` self-heals on retry (writes are
  per-user serialized).

## License

MIT
