# Claude Code Prompts — Atomic Memory Infrastructure (paste one at a time)

How to use this: keep `atomic_memory_build_spec.md` in the repo. Fill in the
three blanks in the KICKOFF below, paste it ONCE, then paste Step prompts in
order. After each step, read what it built, run the done-check yourself, and only
then paste the next one. The goal is to learn, so every prompt asks Claude Code to
explain the concept — don't skip those explanations.

DECISIONS ALREADY MADE (filled into the kickoff below):
- Package name: `atomir` (confirm free on PyPI before Step 13; rename = find-replace).
- First LLM: Groq (`llama-3.3-70b-versatile`).
- First embedder: Jina (`jina-embeddings-v3`) — behind the interface; first planned
  swap is Voyage, decided by the eval.
- Vector store: Qdrant (embedded/local mode).
Before you start: put your GROQ_API_KEY and JINA_API_KEY in `.env` (or rely on the
`fake` LLM/embedder to build with NO keys at first and add real ones later).

---

## KICKOFF (fill the 3 blanks, paste once, before Step 0)

> You are my pair-builder and teacher. We are building a memory-as-infrastructure
> product (category: mem0 / LangMem) with a specific thesis: memory is ATOMIC on
> both ends — atomic facts on write, atomic sub-question decomposition on read.
> The full plan is in `atomic_memory_build_spec.md`. Read it now, fully.
>
> PROJECT SETTINGS (use these throughout):
> - Package name: `atomir`  (use this everywhere instead of "atomemory"; I'll
>   confirm it's free on PyPI before the Step 13 publish — if taken, we rename
>   with a find-replace.)
> - First LLM provider to implement (an example behind the LLM interface, NOT a
>   privileged default): Groq, model `llama-3.3-70b-versatile`.
> - First embedder provider to implement (an example behind the Embedder interface,
>   NOT a privileged default): Jina, model `jina-embeddings-v3` (asymmetric
>   passage/query, dim 1024). NOTE: my first planned experiment after it works is
>   to swap the embedder to Voyage via config and compare on the eval — so keep
>   this strictly behind the Embedder interface.
> - Vector store to start with: Qdrant in embedded/local mode (swappable later).
> - Keys are in `.env`. Also wire the `fake` LLM and `fake` embedder so the system
>   runs with no external keys.
>
> Rules for our whole collaboration:
> 1. Build ONE step at a time, in the spec's order (0, 1, 2, 3, 4, 5, 5.5, 6 …).
>    Never jump ahead. After each step, STOP and wait for me.
> 2. Keep code small and auditable — no giant dumps I can't review.
> 3. Teaching mode: before coding each step, explain in 3–5 sentences what
>    system-design / packaging concept this step teaches and WHY it's built this
>    way. After coding, show me the done-check passing.
> 4. VENDOR-NEUTRAL is non-negotiable: the engine (extractor, reconciler,
>    atomic_read, facade) imports ONLY the LLM / Embedder / MemoryStore interfaces,
>    never a provider SDK or vendor name. Treat any provider name inside the engine
>    as a bug.
> 5. Respect the layering rule: inner layers never import outer ones.
> 6. The "human decisions" listed in the spec are mine — if you want to deviate
>    from any, STOP and ask me first.
>
> Confirm you've read the spec, restate the thesis and my 3 project settings in one
> line each, then WAIT — don't build yet.

---

## STEP 0 — Project skeleton

> Implement Step 0 only: the package skeleton. Create the `atomemory/` package,
> `config.py` (Settings dataclass reading env: GROQ_API_KEY, JINA_API_KEY, MODEL,
> EMBED_DIM=1024, EMBED_BACKEND=jina|fake, COLLECTION, QDRANT_URL, QDRANT_PATH),
> plus `requirements.txt`, `.env.example`, stub `README.md`, and a proper
> `pyproject.toml` so this is a real installable package.
> Teach me: what makes a folder a "Python package," and why config belongs in env
> (the 12-factor idea), not in code.
> Done-check: `python -c "from atomemory.config import settings; print(settings)"`.
> Then stop and wait.

## STEP 1 — Embeddings (Jina passage/query + fake mode)

> Implement Step 1 only: `atomemory/embeddings.py` with `embed_passage(text)` and
> `embed_query(text)` using Jina jina-embeddings-v3 (retrieval.passage /
> retrieval.query). Add EMBED_BACKEND=fake mode: deterministic unit vector from a
> hash of the text, dim=EMBED_DIM, where passage and query of the SAME text return
> the SAME vector so tests work offline.
> Teach me: why Jina uses different embeddings for stored text vs queries
> (asymmetric retrieval), and why an offline fake mode is good infra hygiene.
> Done-check: in fake mode, embed_passage("x")==embed_query("x") and len==EMBED_DIM.
> Then stop and wait.

## STEP 2 — Storage interface (the contract)

> Implement Step 2 only: `atomemory/store_base.py`, an abstract base class
> `MemoryStore` with exactly: add, search, get, update, delete, all, clear — every
> method scoped by user_id. Define the fact schema (id, user_id, text, embedding,
> created_at, updated_at, history, metadata). No implementation yet.
> Teach me: why defining an interface BEFORE any database is the move that lets us
> swap backends later, and why user_id lives in the contract from day one (not bolted on).
> Done-check: the ABC imports and lists its abstract methods.
> Then stop and wait.

## STEP 3 — JSON backend (reference implementation)

> Implement Step 3 only: `atomemory/stores/json_store.py`, `JsonMemoryStore`
> implementing MemoryStore, backed by one JSON file, numpy cosine for search,
> user_id-filtered. This is intentionally demo-grade (linear scan) — we'll swap it.
> Teach me: why we build the simplest backend first to validate the interface, and
> what "tenant isolation by filter" means in practice.
> Done-check: a script adds 2 facts for user A + 1 for user B; search/all for A
> never returns B's facts; update records the old text in history; delete/clear work.
> Then stop and wait.

## STEP 4 — Qdrant backend (same interface, real vector DB)

> Implement Step 4 only: `atomemory/stores/qdrant_store.py`, `QdrantMemoryStore`
> implementing the SAME interface with qdrant-client. Inject the QdrantClient
> (tests pass QdrantClient(":memory:")). Add a factory in config: QDRANT_URL →
> server, else local embedded path. Isolation by FILTER (user_id in payload, single
> collection) — NOT collection-per-user. ids = str(uuid4); update = re-upsert same
> id (push old text to history); use query_points for search, scroll for all.
> Teach me: what a vector DB does that my JSON file can't (indexing/HNSW, scale,
> concurrency), and embedded vs server mode.
> Done-check: the exact STEP 3 isolation/update/delete checks pass against Qdrant
> with an in-memory client.
> Then stop and wait.

## STEP 5 — LLM helper (Groq)

> Implement Step 5 only: `atomemory/llm.py` with chat_json (Groq JSON mode + a
> lenient parser that strips ```json fences and extracts the first {...}),
> chat_text, and judge(rubric, content)->(bool, reason).
> Teach me: why we centralize all LLM calls behind one module, and why llama needs
> defensive JSON parsing.
> Done-check: chat_json returns a dict even when the model wraps JSON in prose.
> Then stop and wait.

## STEP 5.5 — Provider factories (LLM + Embedder)

> Implement Step 5.5 only: make LLM and embeddings swappable so anyone can use
> this, not just Groq+Jina users. Create `providers/llm_base.py` (abstract `LLM`:
> chat_json, chat_text), `providers/embedder_base.py` (abstract `Embedder`:
> embed_passage, embed_query — keep both even for symmetric providers), first
> implementations `GroqLLM` and `JinaEmbedder` (wrapping Steps 5 and 1), and
> `providers/factory.py` with `LLMFactory.create(cfg)` / `EmbedderFactory.create(cfg)`
> selecting a class from a {provider, config} block (mem0-style), erroring clearly
> on unknown providers. EMBED_DIM is owned by the embedder config. Do NOT add other
> providers yet and do NOT use LiteLLM.
> Teach me: the factory-behind-config pattern (how mem0 supports 16+ LLMs this
> way), and why keeping provider SDKs out of the engine prevents vendor lock-in.
> Done-check: switching the llm/embedder provider string in config selects a
> different class; an unknown provider name raises a clear listed error; no engine
> file imports `groq` or the Jina code directly.
> Then stop and wait.

## STEP 6 — Atomic WRITE (extractor + reconciler)

> Implement Step 6 only. `atomemory/extractor.py`: extract_facts(text, role) ->
> atomic third-person facts as {"facts":[...]}. `atomemory/reconciler.py`:
> atomemory/reconciler.py: reconcile(store, llm, embedder, user_id,
> candidate_text) -> embedder.embed_query(candidate), store.search nearest,
> llm.chat_json decides ADD/UPDATE/DELETE/NOOP, apply (ADD/UPDATE store
> embedder.embed_passage). Use the Step 5.5 LLM/Embedder interfaces — do NOT
> import groq or the Jina code here. DECISION #1a: bias toward ADD; add a
> similarity gate — if top neighbor score < RECONCILE_MIN_SIM (env, default 0.6),
> skip the LLM and ADD.
> Teach me: why over-merging distinct facts is the classic failure, and how the
> similarity gate prevents it.
> Done-check: "I'm vegetarian. I'm allergic to peanuts." -> TWO facts;
> "I work at Acme" then "I moved to Globex" -> ONE fact (Globex), Acme in history.
> Then stop and wait.

## STEP 7 — Atomic READ (decompose → retrieve → compose)

> Implement Step 7 only: `atomemory/atomic_read.py`. plan(query) -> {"decompose":
> bool, "subquestions":[...]} via one LLM call — DECISION #1b: simple questions
> return decompose=false, subquestions=[query]; multi-part/multi-hop return atomic
> wh-sub-questions. Do NOT always decompose. atomic_search(store, user_id, query,
> k, decompose) -> embed each sub-question as QUERY, search per sub-question, dedupe
> by id (keep best score), top-k, return {"subquestions":[...], "results":[...]}.
> Teach me: why decomposition helps multi-hop retrieval but costs latency, and why
> a planner gate (decompose only when needed) is the right design.
> Done-check: "who should I email about my project?" yields a sub-question like
> "who is my manager?" and surfaces the manager fact a blob query misses.
> Then stop and wait.

## STEP 8 — Engine facade

> Implement Step 8 only: `atomemory/memory.py`, class MemoryService constructed
> with a MemoryStore instance: add(user_id, text) (extract→reconcile each, returns
> {operations, facts}), search(user_id, query, k, decompose), get_all, delete,
> reset. The engine must NEVER import a concrete backend.
> Teach me: what a "facade" is and why it's the stable seam that keeps the API and
> SDK simple.
> Done-check: the same script runs identically with JsonMemoryStore OR
> QdrantMemoryStore injected.
> Then stop and wait.

## EVAL (build now, re-run after every later step)

> Implement the eval harness: `eval/eval_set.json` (cases: ADD, dedup/NOOP,
> UPDATE employer+role, multi-fact retention, DELETE, distractor, multi-hop,
> absence, constraint) and `eval/run_eval.py` that ingests each case into a fresh
> user_id, checks ACTIVE facts (substring + judge), and prints a score. Add a
> LATENCY column (p50/p95 per op) so accuracy and milliseconds show together.
> Use EMBED_BACKEND=fake-free real run, but allow running without --answers.
> Teach me: why a fast internal eval is my daily driver and public benchmarks are
> the occasional dyno test.
> Done-check: `python eval/run_eval.py` prints per-case PASS/FAIL + score + latency.
> Then stop and wait. (Re-run this after Steps 9–12.)

## STEP 9 — Concurrency & durability

> Implement Step 9 only: a per-user_id async (keyed) lock so a single user's
> reconcile (read-modify-write) is serialized, while different users run in
> parallel. Document that the JSON backend is NOT crash-safe (dev only); Qdrant
> gives durability. DECISION #5: simple lock now, transactions deferred + noted.
> Teach me: what a race condition is in a read-modify-write, and why stateless API
> servers + state-in-the-DB is what lets infra scale horizontally.
> Done-check: two concurrent add() for the same user don't corrupt/duplicate;
> different users aren't blocked by each other.
> Then stop and wait.

## STEP 10 — REST API (FastAPI)

> Implement Step 10 only: `atomemory/api.py` (FastAPI). Endpoints under /v1:
> POST /v1/memories {user_id, text}; POST /v1/search {user_id, query, k?,
> decompose?} returning {subquestions, results} (DECISION #4: expose
> subquestions); GET /v1/memories?user_id=; DELETE /v1/memories/{id}?user_id=;
> DELETE /v1/memories?user_id=; GET /health. Run sync engine work in a threadpool;
> apply the per-user lock. Consistent JSON error envelope; proper status codes.
> Teach me: why versioning the API (/v1) from day one matters, why FastAPI's
> auto-generated OpenAPI docs are what make us SDK-agnostic, and the authn vs authz
> distinction (we'll add real API keys later).
> Done-check: `uvicorn atomemory.api:app` serves; curl POST /v1/memories then POST
> /v1/search round-trips and search shows subquestions. Open /docs and see the API.
> Then stop and wait.

## STEP 11 — Python SDK

> Implement Step 11 only: `atomemory/client.py`, class MemoryClient(base_url,
> api_key) with add/search/get_all/delete/reset wrapping the REST calls, returning
> the same shapes as the API. Make results accessible as objects (e.g.
> result.subquestions, hit.text).
> Teach me: why the SDK must be a THIN wrapper with no business logic (so a Go/JS
> user hitting the raw API gets identical behavior) — logic lives in the service.
> Done-check: a 5-line script using only MemoryClient adds and searches against the
> running API.
> Then stop and wait.

## STEP 12 — Packaging & ops (Docker + compose)

> Implement Step 12 only: a Dockerfile for the API; docker-compose.yml running the
> API + a Qdrant server (API uses QDRANT_URL); complete .env.example; and a real
> README with quickstart (embedded mode, no Docker), prod mode (compose), the /v1
> endpoint reference, the atomic-memory thesis, and KNOWN LIMITATIONS (reconciler
> tuning, JSON backend not crash-safe, transactions deferred).
> Teach me: why Docker/compose makes this "deployable, not just runs-on-my-machine,"
> and the 12-factor config idea in practice.
> Done-check: `docker compose up` brings up API + Qdrant; the curl round-trip works
> against the containerized service.
> Then stop and wait.

## STEP 13 — Publish & consume (the finish line)

> Implement Step 13 only: make the package installable and prove the end-user
> experience. Finalize pyproject.toml (name, version, deps, entry points), build a
> wheel, and write `examples/usage.py` containing the THREE end-user faces:
> (1) pip-install SDK usage (MemoryClient: add + search, printing subquestions),
> (2) a raw curl example for /v1/memories and /v1/search,
> (3) a 5-line "agent turn" that recalls before answering and writes after.
> Document the publish flow to TestPyPI then PyPI (`pip install atomemory`).
> Teach me: how a wheel is built, what TestPyPI is for, and what versioning
> (semver) commitment I'm making to users once I publish.
> Done-check (this is the WHOLE-PRODUCT acceptance test): from a clean venv,
> install the package, run examples/usage.py against the running service, and all
> three faces work end to end.
> Then stop — we're done; summarize what we built and what I should harden next.

---

## After it's working — the "harden it" backlog (not day-one)
Real API-key auth + per-tenant rate limiting; Postgres for tenants/keys/usage/
audit logs; observability (structured logs, metrics, request tracing across
decompose→retrieve→compose); caching (embeddings + atomic sub-answers);
parallel sub-question retrieval; contract tests; CI/CD; then LoCoMo/LongMemEval
run like-for-like (same model + embeddings across all systems, latency reported).
Let real usage pull these out of you — don't build them speculatively.
