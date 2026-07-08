"""Atomic READ: answer a question by querying memory atomically.

The read-side twin of the write engine. Instead of embedding a whole question
as one blob, a planner may decompose it into atomic sub-questions; each is
retrieved independently and the results are unioned (deduped by fact id, best
score kept). Vendor-neutral: imports ONLY the injected interfaces.
"""

from __future__ import annotations

from atomir.providers.embedder_base import Embedder
from atomir.providers.llm_base import LLM
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
) -> dict:
    """Retrieve facts for `query`, decomposing into sub-questions when useful.

    Returns {"subquestions": [...], "results": [...]} — sub-questions are
    exposed deliberately (the differentiator, and a debugging aid).
    """
    subquestions = plan(llm, query)["subquestions"] if decompose else [query]

    # Each sub-question is embedded as a QUERY (here the input really is a
    # question) and retrieved independently. These retrievals are mutually
    # independent -> safe to run concurrently (deferred to Step 9); sequential
    # here for determinism.
    best_by_id: dict[str, dict] = {}
    for sq in subquestions:
        for hit in store.search(user_id, embedder.embed_query(sq), k=k):
            hid = hit["id"]
            if hid not in best_by_id or hit["score"] > best_by_id[hid]["score"]:
                best_by_id[hid] = hit

    results = sorted(best_by_id.values(), key=lambda h: h["score"], reverse=True)[:k]
    return {"subquestions": subquestions, "results": results}
