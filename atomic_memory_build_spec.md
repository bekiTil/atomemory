# Build Spec — Atomic Memory Infrastructure (for agents)

You are building a memory-as-infrastructure service (in the same category as
mem0 / LangMem), but with a specific thesis: **memory is atomic on both ends.**
Write side stores small discrete facts; read side answers by decomposing a
question into small sub-questions, retrieving per sub-question, and composing
the results. Build **inside-out**: engine first, then storage, then
multi-tenancy, then API, then packaging. Do not let an inner layer import an
outer one.

## Global conventions
- Language: Python 3.11+. Style: small modules, one responsibility each.
- **Vendor-neutral architecture (non-negotiable).** This is a memory layer, not a
  bundle of vendors. Three external dependencies — the LLM, the embedding model,
  and the vector store — are each defined as an INTERFACE and chosen at runtime by
  config. No specific provider is privileged in the design, the docs, or the
  engine. The product must be equally usable by someone on OpenAI, Anthropic,
  Ollama/local, Cohere, Gemini, Bedrock, pgvector, Chroma, Pinecone, etc. Breadth
  of audience is a primary goal.
- **Interfaces, not vendors, are the contract:**
  - `LLM` interface — `chat_json`, `chat_text` (Step 5.5).
  - `Embedder` interface — `embed_passage`, `embed_query` (Step 5.5). Supports both
    asymmetric providers (different passage/query embeddings) and symmetric ones
    (same call for both); neither is assumed.
  - `MemoryStore` interface — the storage contract (Step 2), with multiple backends.
- Concrete providers are PLUGINS selected by config: `{llm, embedder, vector_store}`,
  each a `{provider, config}` block (mem0-style). To get the project running, pick
  any LLM + any embedder + any vector store you have keys for; the build will wire
  up one working combination as a starting example, but treat that choice as an
  example, not a recommendation or a dependency.
- **Engine isolation (enforced):** engine modules (extractor, reconciler,
  atomic_read, facade) import ONLY the `LLM`, `Embedder`, and `MemoryStore`
  interfaces — never any provider SDK or vendor name. A specific provider's name
  or import appearing anywhere in the engine is a layering violation to be fixed.
- Config via environment + `.env` (`python-dotenv`). Never hardcode keys. Each
  provider reads its own keys/model/dimension from its config block; `EMBED_DIM`
  travels with the chosen embedder.
- After every step: code must import and run; add a tiny runnable check.
- Provide a `fake` LLM and `fake` embedder (deterministic canned output / local
  vectors) so the system runs and tests with NO external keys — independent of
  which real providers a user later plugs in.
- Layering rule (enforce): inner layers never import outer ones. The engine knows
  nothing about concrete providers, storage backends, the API, or HTTP — only the
  interfaces.

---

## STEP 0 — Project skeleton
**Goal:** an installable, runnable empty project.
**Build:**
- Package folder `memserve/` with `__init__.py`.
- `config.py`: a `Settings` object exposing three swappable provider slots —
  `llm`, `embedder`, `vector_store` — each a `{provider, config}` block (mem0-style).
  Provider-specific values (API keys, model names, `EMBED_DIM`, `COLLECTION`,
  connection URLs/paths) live inside the relevant provider's config block and are
  read from env. The config must not privilege any vendor: whichever providers the
  user configures are the ones used. For first-run convenience, allow ONE
  configured combination to be picked up from env, but that is an example a user
  fills in — not a hardcoded vendor.
- `requirements.txt` (core only — every provider SDK is an optional extra added in
  Step 12), `.env.example` (showing the `{llm, embedder, vector_store}` block shape
  with placeholders, not a single vendor), `README.md` (stub).
**Decisions baked in:** config is centralized and vendor-neutral; providers and
backends are selected by config blocks, not code edits; no provider is privileged.
**Done when:** `python -c "from memserve.config import settings; print(settings)"` prints config.

---

## STEP 1 — First embedder implementation (`memserve/embeddings/`)
**Goal:** a working text→vector implementation to develop against. The named
provider here is ONE interchangeable example — you may pick a different embedder
(OpenAI, Cohere, a local model) and the rest of the build is unchanged. It gets
formalized behind the `Embedder` interface in Step 5.5.
**Build:**
- A concrete embedder exposing `embed_passage(text) -> list[float]` and
  `embed_query(text) -> list[float]`. (Example: Jina `jina-embeddings-v3` with its
  asymmetric `retrieval.passage` / `retrieval.query` tasks. A symmetric provider
  would implement both methods as the same call — keep both methods regardless.)
- A `fake` embedder: deterministic unit vector from a hash of the text, dim =
  the configured embedding dimension. In fake mode, passage and query of the SAME
  text must return the SAME vector (so exact-text search matches in tests).
- Handle empty strings; raise a clear error if a real provider is selected but its
  key is missing.
**Decisions baked in:** asymmetric retrieval is respected; offline testability is free.
**Done when:** in fake mode, `embed_passage("x")==embed_query("x")` and length==EMBED_DIM.

---

## STEP 2 — Storage interface (`memserve/store_base.py`)
**Goal:** the single contract the engine depends on. Backend-agnostic.
**Build:** an abstract base class `MemoryStore` with EXACTLY these methods, all
**scoped by `user_id`**:
- `add(user_id, text, embedding, metadata=None) -> dict`  (returns the stored fact incl. `id`)
- `search(user_id, query_embedding, k=5) -> list[dict]`   (each has `id`, `score`, `text`, ...)
- `get(user_id, fact_id) -> dict | None`
- `update(user_id, fact_id, new_text, new_embedding) -> dict | None`  (push old text into `history`)
- `delete(user_id, fact_id) -> bool`
- `all(user_id, limit=1000) -> list[dict]`
- `clear(user_id) -> bool`
**Fact schema:** `id, user_id, text, embedding(optional in returns), created_at,
updated_at, history (list of superseded texts), metadata`.
**Decisions baked in (DECISION #2):** this minimal method set is the whole contract.
Multi-tenancy is in the interface from day one (`user_id` everywhere), not bolted on later.
**Done when:** the ABC imports and lists its abstract methods.

---

## STEP 3 — Reference backend: JSON store (`memserve/stores/json_store.py`)
**Goal:** simplest possible implementation of `MemoryStore`, for fast local dev.
**Build:** `JsonMemoryStore(MemoryStore)` backed by one JSON file; in-process
numpy cosine for `search`; `user_id`-scoped via a filter on a flat list.
**Decisions baked in:** prove the interface works before adding a real DB.
This backend is demo-grade (linear scan) and will be swapped — that's expected.
**Done when:** a script can add 2 facts for user A + 1 for user B, and
`search`/`all` for A never returns B's facts (isolation), and `update` records
the old text in `history`, and `delete`/`clear` work.

---

## STEP 4 — A production vector-store backend (`memserve/stores/qdrant_store.py`)
**Goal:** same `MemoryStore` interface, backed by a real vector DB. The named DB
here is ONE interchangeable example — pgvector, Chroma, Pinecone, Weaviate, etc.
would each be another `MemoryStore` implementation; the engine never knows which.
**Build:** a `MemoryStore` implementation using a real vector DB (example shown
with `qdrant-client`):
- Accept an injected `QdrantClient` (so tests can pass `QdrantClient(":memory:")`).
- A factory in `config.py`: if `QDRANT_URL` set → server client; else local
  `QdrantClient(path=QDRANT_PATH)` (embedded, zero-ops).
- Create the collection if missing (size=`EMBED_DIM`, COSINE).
- **DECISION #3 — isolation by FILTER:** single collection; store `user_id` in
  the payload; every `search`/`all`/`clear` filters on `user_id`. (Do NOT use a
  collection-per-user.)
- Point ids = `str(uuid4())`. `update` = re-upsert same id with new vector +
  payload (push old text to `history`). `clear` = delete by `user_id` filter.
- Use the current query API (`query_points`) for search; `scroll` for `all`.
**Done when:** the exact same isolation/update/delete checks from STEP 3 pass
against `QdrantMemoryStore` with an in-memory client. The engine code from later
steps must run unchanged on either backend.

---

## STEP 5 — First LLM implementation (`memserve/llm/`)
**Goal:** a robust LLM call layer for extraction, reconciliation, decomposition,
and judging. The named provider here is ONE interchangeable example — OpenAI,
Anthropic, Ollama/local, etc. would each be another implementation behind the
`LLM` interface formalized in Step 5.5. Pick whichever you have a key for.
**Build (example shown with a Groq/llama-style provider):**
- `chat_json(system, user) -> dict` using the provider's JSON mode + a lenient
  parser (strip code fences, extract first `{...}`) because some open models are
  less strict than others at returning clean JSON.
- `chat_text(system, user) -> str`.
- `judge(rubric, content) -> (bool, reason)` for evals.
- A `fake` LLM (canned deterministic output) so the engine and evals run with no
  external key.
**Done when:** `chat_json` returns a dict even when the model wraps JSON in prose;
the `fake` LLM lets the pipeline run with no key set.

---

## STEP 5.5 — Provider factories (LLM + Embedder) — DECISION #6
**Goal:** make providers swappable so the product is accessible to ANYONE, not
only Groq + Jina users — without touching the engine when a new provider is added.
This mirrors mem0's factory-behind-config pattern (mem0 supports 16+ LLMs and
10+ embedders this exact way).
**Build:**
- `memserve/providers/llm_base.py`: abstract `LLM` with `chat_json(system, user)`
  and `chat_text(system, user)`. (Engine may ONLY call these.)
- `memserve/providers/embedder_base.py`: abstract `Embedder` with
  `embed_passage(text)` and `embed_query(text)`. Keep BOTH methods even though
  some providers are symmetric — a symmetric provider implements both as the same
  call; Jina's passage/query asymmetry must not leak into the interface.
- First implementations (wrap whatever concrete LLM and embedder you built in
  Steps 5 and 1 — these are EXAMPLES, not the canonical pair): e.g. a
  `GroqLLM`/`OpenAILLM`-style class implementing `LLM`, and a
  `JinaEmbedder`/`OpenAIEmbedder`-style class implementing `Embedder`. Also
  register the `fake` LLM and `fake` embedder as providers.
- `memserve/providers/factory.py`: `LLMFactory.create(cfg)` and
  `EmbedderFactory.create(cfg)`, each holding a `provider -> class` map and
  instantiating from a config block shaped like mem0's:
  `{"provider": "groq", "config": {...}}` / `{"provider": "jina", "config": {...}}`.
  Unknown provider name → clear error listing valid names.
- Config: three independent swappable slots — `llm`, `embedder`, `vector_store` —
  each a `{provider, config}` block. `EMBED_DIM` is owned by the embedder config
  and travels with the embedder choice (Jina v3 = 1024, OpenAI 3-small = 1536).
**Decisions baked in (DECISION #6):**
- Factory-behind-config for `llm` and `embedder`; Groq + Jina are the FIRST
  implementations, not the only ones. Adding OpenAI/Ollama/Cohere later = write
  one class + register it, with ZERO engine changes.
- `embed_passage`/`embed_query` stay separate in the interface (honest across
  symmetric and asymmetric providers).
- Dimension is config-owned. **Hard rule: you cannot mix embedders in one
  collection** — switching embedders means a new collection / re-embed. Document
  it; do not let it become a silent shape-mismatch crash.
- Explicitly NOT doing now: a generic LiteLLM "any provider" layer. Hand-write two
  clean classes for two providers. Revisit LiteLLM only if you later need many
  providers without writing many classes (it would become one more `LLM` impl).
- Optional dependency groups belong in Step 12 packaging (`atomemory[groq]`,
  `atomemory[jina]`, `atomemory[openai]`, ...) so each provider SDK installs only
  if wanted — this factory is what makes that possible.
**Layering note:** after this step, the engine (extractor, reconciler,
atomic_read, facade) imports the `LLM` / `Embedder` interfaces ONLY — never
`groq` or the Jina code directly. If any engine module imports a provider SDK,
that's a layering violation; fix it here.
**Done when:** swapping the `llm`/`embedder` provider strings in config selects a
different class via the factory; the engine code (Steps 6–8) runs unchanged; an
unknown provider name raises a clear, listed error.

---

## STEP 6 — Atomic WRITE: extractor + reconciler
**Goal:** turn raw text into atomic facts and reconcile each into the store.
**Build:**
- `memserve/extractor.py`: `extract_facts(text, role="user") -> list[str]`.
  Prompt: pull durable, atomic, self-contained facts in third person; drop
  chit-chat; return `{"facts":[...]}`. (Works for user messages AND
  agent-confirmed findings.)
- `memserve/reconciler.py`: `reconcile(store, llm, embedder, user_id, candidate_text) -> dict`.
  Steps: `embedder.embed_query(candidate)` → `store.search(user_id, ...)` for
  nearest facts → `llm.chat_json(...)` decides one of **ADD / UPDATE / DELETE /
  NOOP** → apply via the store (ADD/UPDATE store `embedder.embed_passage(text)`).
  Return the decision. Use the Step 5.5 `LLM`/`Embedder` interfaces — do NOT
  import `groq` or the Jina code here.
  - **DECISION #1a (write bias):** prefer ADD; only UPDATE/NOOP when the nearest
    fact is the SAME attribute. Add a similarity gate: if top neighbor score <
    `RECONCILE_MIN_SIM` (env, e.g. 0.6), skip the LLM and ADD directly. (This is
    the fix for the over-merge failure mode — distinct facts must not collapse.)
**Done when:** ingesting "I'm vegetarian. I'm allergic to peanuts." yields TWO
facts; "I work at Acme" then "I moved to Globex" yields ONE fact (Globex) with
Acme in its `history`.

---

## STEP 7 — Atomic READ: decomposition + retrieval + compose (`memserve/atomic_read.py`)
**Goal:** the signature feature — answer by querying memory atomically.
**Build:**
- `plan(query) -> {"decompose": bool, "subquestions": [...]}` via one LLM call.
  - **DECISION #1b (when to decompose):** the planner returns `decompose=false`
    with `subquestions=[query]` for simple single-fact questions, and
    `decompose=true` with atomic wh-sub-questions for multi-part / multi-hop ones.
    Do NOT always decompose.
- `atomic_search(store, user_id, query, k=6, decompose=True) -> dict`:
  - get sub-questions from `plan` (or `[query]` if `decompose=False`),
  - embed EACH sub-question as a QUERY, `store.search` per sub-question,
  - **dedupe** retrieved facts by `id`, keep best score, sort, take top-k,
  - return `{"subquestions": [...], "results": [...]}` (expose the sub-questions —
    it's the differentiator and aids debugging).
- Independent sub-questions MAY be retrieved concurrently (note where).
**Done when:** a multi-hop query ("who should I email about my project?") returns
sub-questions like ["who is my manager?"] and surfaces the manager fact that a
single blob embedding misses.

---

## STEP 8 — Engine facade (`memserve/memory.py`)
**Goal:** one clean object the API (and SDK users) call. Backend-injected.
**Build:** `class MemoryService` constructed with three injected dependencies —
a `MemoryStore`, an `LLM`, and an `Embedder` (all built by their factories from
config; the facade does not import any concrete provider or backend):
- `add(user_id, text) -> {"operations":[...], "facts":[...]}`  (extract → reconcile each)
- `search(user_id, query, k=6, decompose=True) -> {subquestions, results}`
- `get_all(user_id) -> list`
- `delete(user_id, fact_id) -> bool`
- `reset(user_id) -> bool`
**Decisions baked in:** the facade is the stable seam; swapping JSON↔Qdrant,
Groq↔OpenAI, or Jina↔OpenAI-embeddings is just which instances you inject. The
engine never imports a concrete backend or provider.
**Done when:** the same script runs identically with a `JsonMemoryStore` or a
`QdrantMemoryStore` injected, and with different `llm`/`embedder` providers
selected purely by config.

---

## STEP 9 — Concurrency & durability
**Goal:** safe concurrent writes per user; no corruption on crash.
**Build:**
- A per-`user_id` async lock (or keyed lock) so a user's reconcile
  (read-modify-write) is serialized. Different users run in parallel.
- With Qdrant, durability comes from the DB. Document that the JSON backend is
  NOT crash-safe (dev only).
**DECISION #5:** ship the simple per-user lock now; full transactions are deferred
and explicitly noted as a known limitation.
**Done when:** two concurrent `add` calls for the same user don't interleave into
a corrupt/duplicated state; calls for different users are not blocked by each other.

---

## STEP 10 — API surface (`memserve/api.py`, FastAPI)
**Goal:** HTTP service over the engine.
**Build endpoints (all take/scope `user_id`):**
- `POST /memories`  body `{user_id, text}` → runs `add`, returns operations + facts.
- `POST /search`    body `{user_id, query, k?, decompose?}` → returns
  `{subquestions, results}`. **DECISION #4:** expose `subquestions` in the response.
- `GET /memories?user_id=` → all facts.
- `DELETE /memories/{id}?user_id=` → delete one.
- `DELETE /memories?user_id=` → reset a user.
- `GET /health` → ok + backend name.
- Run sync engine work in a threadpool; apply the per-user lock from STEP 9.
**Done when:** `uvicorn memserve.api:app` serves; a curl `POST /memories` then
`POST /search` round-trips and search shows the sub-questions used.

---

## STEP 11 — Client SDK (`memserve/client.py`)
**Goal:** make adoption one import.
**Build:** `MemoryClient(base_url)` with `add/search/get_all/delete/reset` wrapping
the REST calls. Keep return shapes identical to the API.
**Done when:** a 5-line script using only `MemoryClient` can add and search.

---

## STEP 12 — Packaging & ops
**Goal:** deployable, not just runnable-on-my-machine.
**Build:**
- `Dockerfile` for the API.
- `docker-compose.yml` running the API + a Qdrant server (API points at it via
  `QDRANT_URL`).
- `.env.example` with every var.
- `README.md`: quickstart (embedded mode, no Docker), prod mode (compose),
  endpoint reference, the atomic-memory thesis, and the KNOWN LIMITATIONS
  (reconciler tuning, JSON backend not crash-safe, transactions deferred).
**Done when:** `docker compose up` brings up API+Qdrant and the curl round-trip works.

---

## CROSS-CUTTING — Evaluation harness (`eval/`)
Build alongside, not last. Wraps every layer.
- `eval/eval_set.json`: cases covering ADD, NOOP(dedup), UPDATE(employer+role),
  multi-fact retention, DELETE, distractor, multi-hop, absence/no-hallucination,
  constraint-in-recommendation.
- `eval/run_eval.py`: for each case, ingest history into a FRESH user_id, then
  check the resulting ACTIVE facts (substring + LLM `judge`); with `--answers`,
  also generate + grade answers. **Add a latency column** (p50/p95 per op) so
  every change shows accuracy AND milliseconds together.
- Use this tiny eval as the daily driver; wire LoCoMo / LongMemEval later as the
  occasional like-for-like benchmark (same model + embeddings across all systems,
  always report latency next to accuracy).
**Standing rule:** no step is "done" until the eval still passes (or its
regressions are understood and intended).

---

## Build order (summary)
0 skeleton → 1 embeddings → 2 interface → 3 JSON backend → 4 Qdrant backend →
5 llm → 5.5 provider factories → 6 atomic write → 7 atomic read → 8 facade →
9 concurrency → 10 API → 11 SDK → 12 packaging. Eval harness runs throughout.

## The decisions that are MINE (the human's), already encoded above
1. When to decompose on read (planner gate; not always) — STEP 7.
   And write bias toward ADD + similarity gate — STEP 6.
2. The storage interface's minimal method set — STEP 2.
3. Multi-tenancy by FILTER, not collection-per-user — STEP 4.
4. The search API exposes the sub-questions it used — STEP 10.
5. Provider factories: LLM/Embedder/vector-store are interfaces chosen by config;
   the providers you start with are interchangeable examples, not privileged
   defaults; dimension config-owned; no embedder mixing per collection; no LiteLLM
   yet — STEP 5.5.
If you (Claude Code) want to deviate from any of these, STOP and ask first.
