"""Atomic READ: answer a question by querying memory atomically.

The read-side twin of the write engine. Instead of embedding a whole question
as one blob, a planner may decompose it into atomic sub-questions; each is
retrieved independently and the results are unioned (deduped by fact id, best
score kept). Vendor-neutral: imports ONLY the injected interfaces.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from atomir.providers.embedder_base import Embedder
from atomir.providers.llm_base import LLM
from atomir.ranking import BM25, rrf
from atomir.store_base import MemoryStore

_PLAN_SYSTEM = (
    "You are a query planner for a personal-memory system. Decide whether a "
    "question must be broken into atomic sub-questions to answer it.\n"
    "- SIMPLE single-fact question -> decompose=false and subquestions=[the "
    "original question verbatim].\n"
    "- MULTI-HOP question (the answer depends on an intermediate fact, e.g. a "
    "person, project, or relationship you must resolve first) -> decompose=true "
    "with atomic wh- sub-questions, INCLUDING the intermediate facts.\n\n"
    "Examples:\n"
    'Q: "where does the user live?"\n'
    '{"decompose": false, "subquestions": ["where does the user live?"]}\n'
    'Q: "what gift should I buy my sister?"\n'
    '{"decompose": true, "subquestions": ["who is the user\'s sister?", '
    '"what does the user\'s sister like?"]}\n\n'
    "Respond ONLY with JSON: "
    '{"decompose": <true|false>, "subquestions": ["...", "..."]}'
)


def plan(llm: LLM, query: str) -> dict:
    """Decide whether/how to decompose `query`. Always returns >=1 subquestion."""
    result = llm.chat_json(_PLAN_SYSTEM, query)
    decompose = bool(result.get("decompose", False))
    subs = [
        s.strip()
        for s in (result.get("subquestions") or [])
        if isinstance(s, str) and s.strip()
    ]
    if not subs:  # planner gave nothing usable -> fall back to the raw query
        subs, decompose = [query], False
    return {"decompose": decompose, "subquestions": subs}


def atomic_search(
    store: MemoryStore,
    llm: LLM,
    embedder: Embedder,
    user_id: str,
    query: str,
    k: int = 6,
    decompose: bool = True,
    hybrid: bool = True,
    corpus_limit: int = 5000,
    planner_llm: LLM | None = None,
) -> dict:
    """Retrieve facts for `query`, decomposing into sub-questions when useful.

    With `hybrid` (default), each sub-question contributes TWO rankings — dense
    (embedding similarity) and lexical (BM25 over the user's facts) — and all of
    them are fused with Reciprocal Rank Fusion. Semantic search alone buries
    facts that hinge on rare terms (names, project titles); BM25 recovers them.

    Returns {"subquestions": [...], "results": [...]} — sub-questions are
    exposed deliberately (the differentiator, and a debugging aid). With hybrid,
    `score` is the fused RRF score (rank-based), not a cosine.
    """
    # The planner is a small structured task and can run on a faster model than
    # the main LLM (see `planner_llm`); it dominates read latency otherwise.
    planner = planner_llm or llm

    # Embed the raw query CONCURRENTLY with the planner call — they don't depend
    # on each other. If the planner declines to decompose (subquestions == [query]),
    # its embedding is already in hand and we skip a whole round-trip. If it does
    # decompose, we simply drop it (one cheap embed call, no added latency).
    prefetched: dict[str, list[float]] = {}
    if decompose:
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_plan = ex.submit(plan, planner, query)
            fut_emb = ex.submit(embedder.embed_query, query)
            subquestions = fut_plan.result()["subquestions"]
            try:
                prefetched[query] = fut_emb.result()
            except Exception:  # speculative work must never fail the search
                pass
    else:
        subquestions = [query]

    # Pool per ranking: wider than k so fusion has candidates to promote.
    pool = max(k * 5, 50)
    by_id: dict[str, dict] = {}

    # Lexical index over the user's facts — built ONCE, scored per sub-question.
    bm25 = None
    corpus_ids: list[str] = []
    if hybrid:
        corpus = store.all(user_id, limit=corpus_limit)
        if corpus:
            corpus_ids = [f["id"] for f in corpus]
            bm25 = BM25([f["text"] for f in corpus])
            by_id.update({f["id"]: f for f in corpus})

    # Sub-question retrievals are independent blocking network calls -> run them
    # concurrently. Output stays deterministic (fusion is order-independent).
    def _dense(sq: str) -> list[dict]:
        vec = prefetched.get(sq) or embedder.embed_query(sq)
        return store.search(user_id, vec, k=pool)

    if len(subquestions) == 1:
        dense_hits = [_dense(subquestions[0])]
    else:
        with ThreadPoolExecutor(max_workers=min(len(subquestions), 8)) as ex:
            dense_hits = list(ex.map(_dense, subquestions))

    rankings: list[list[str]] = []
    for sq, hits in zip(subquestions, dense_hits):
        for h in hits:
            by_id.setdefault(h["id"], h)
        rankings.append([h["id"] for h in hits])  # dense ranking
        if bm25 is not None:
            scored = [(i, s) for i, s in enumerate(bm25.scores(sq)) if s > 0]
            scored.sort(key=lambda t: t[1], reverse=True)
            rankings.append([corpus_ids[i] for i, _ in scored[:pool]])  # lexical

    if not hybrid:  # dense-only: rank by raw similarity, as before
        best: dict[str, dict] = {}
        for hits in dense_hits:
            for h in hits:
                if h["id"] not in best or h["score"] > best[h["id"]]["score"]:
                    best[h["id"]] = h
        results = sorted(best.values(), key=lambda h: h["score"], reverse=True)[:k]
        return {"subquestions": subquestions, "results": results}

    fused = rrf(rankings)
    top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    results = [{**by_id[fid], "score": score} for fid, score in top if fid in by_id]
    return {"subquestions": subquestions, "results": results}
