# atomir

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
interfaces — never a provider SDK or vendor name. Swapping Groq↔OpenAI,
Jina↔Voyage, or Qdrant↔pgvector is one config change plus one small class.
Defaults use `fake` backends, so everything runs with **no external keys**.

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
| GET | `/memories` | `?user_id=` | list of facts |
| DELETE | `/memories/{id}` | `?user_id=` | `{deleted, id}` (404 if absent) |
| DELETE | `/memories` | `?user_id=` | `{reset}` |
| GET | `/health` | — | `{status, store, llm, embedder}` |

`MemoryClient(base_url)` (in `atomir.client`) wraps these with the same method
names and return shapes.

## Configuration

All config is read from the environment (see `.env.example`): `LLM_BACKEND`,
`LLM_API_KEY`, `MODEL`, `EMBED_BACKEND`, `EMBED_API_KEY`, `EMBED_DIM`,
`RECONCILE_MIN_SIM`, `STORE_BACKEND`, `COLLECTION`, `STORE_URL`, `STORE_PATH`.

## Known limitations

- **Reconciler threshold is untuned.** `RECONCILE_MIN_SIM` defaults to `0.6`; on
  Jina, real "same-attribute" pairs measured ~0.6, so it sits right on the edge.
  It should be tuned per embedder with the eval harness, not trusted as-is.
- **The JSON backend is NOT crash-safe.** It rewrites the whole file without
  atomic replace/fsync — dev and tests only. Use Qdrant for durable storage.
- **No transactions.** Writes are serialized per user with a simple lock
  (Step 9); full transactional rollback is deferred (DECISION #5).
- **Read returns facts, not a composed answer.** `search` returns the relevant
  facts and sub-questions; turning them into a final sentence is the caller's
  LLM's job.
