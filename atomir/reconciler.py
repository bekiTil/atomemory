"""Atomic WRITE, stage 2: reconcile one candidate fact into the store.

Vendor-neutral engine code: depends ONLY on the injected `MemoryStore`, `LLM`,
and `Embedder` interfaces — no provider SDK and no vendor name anywhere.

Flow: embed the candidate as a PASSAGE -> find nearest existing facts ->
DECISION #1a similarity gate (weak match => ADD directly, skip the LLM) ->
otherwise let the LLM choose ADD / UPDATE / DELETE / NOOP -> apply via the store.

Note: reconciliation compares a new FACT against stored FACTS — a symmetric,
passage-to-passage comparison — so the candidate is embedded as a PASSAGE, not a
query. (embed_query is tuned for questions; on an asymmetric embedder,
using it here collapses same-attribute similarity to noise. The READ path uses
embed_query, because there the input really is a question.)
"""

from __future__ import annotations

from atomir.providers.embedder_base import Embedder
from atomir.providers.llm_base import LLM
from atomir.store_base import MemoryStore

_RECONCILE_SYSTEM = (
    "You maintain a memory of ATOMIC facts about a user. Given a NEW candidate "
    "fact and the user's NEAREST existing facts, choose exactly ONE action:\n"
    "- ADD: the candidate is new information or a DIFFERENT attribute.\n"
    "- UPDATE: the candidate is the SAME attribute as an existing fact but with "
    "a new value -> supply that fact's id.\n"
    "- DELETE: the candidate states an existing fact is no longer true -> supply "
    "that fact's id.\n"
    "- NOOP: the candidate is already captured and adds nothing.\n"
    "Bias STRONGLY toward ADD. Only UPDATE/DELETE/NOOP when a nearest fact is "
    "clearly the same attribute. Respond ONLY with JSON: "
    '{"decision":"ADD|UPDATE|DELETE|NOOP","target_id":"<id or null>","reason":"..."}'
)


def _decide(llm: LLM, candidate: str, neighbors: list[dict]) -> dict:
    listing = "\n".join(
        f"- id={n['id']} (score={n.get('score', 0):.3f}): {n['text']}" for n in neighbors
    )
    user = f"CANDIDATE FACT:\n{candidate}\n\nNEAREST EXISTING FACTS:\n{listing}"
    return llm.chat_json(_RECONCILE_SYSTEM, user)


def reconcile(
    store: MemoryStore,
    llm: LLM,
    embedder: Embedder,
    user_id: str,
    candidate_text: str,
    *,
    k: int = 5,
    min_sim: float = 0.6,
) -> dict:
    """Reconcile one candidate fact into the store. Returns the decision taken."""
    candidate_text = (candidate_text or "").strip()
    if not candidate_text:
        return {"decision": "NOOP", "reason": "empty candidate", "fact": None}

    # Fact-to-fact comparison is symmetric -> embed the candidate as a PASSAGE.
    neighbors = store.search(user_id, embedder.embed_passage(candidate_text), k=k)

    # DECISION #1a: no neighbor, or the nearest is too weak -> ADD, skip the LLM.
    if not neighbors or neighbors[0].get("score", 0.0) < min_sim:
        fact = store.add(user_id, candidate_text, embedder.embed_passage(candidate_text))
        return {"decision": "ADD", "reason": "similarity gate: no close fact", "fact": fact}

    decision = _decide(llm, candidate_text, neighbors)
    action = str(decision.get("decision", "ADD")).upper()
    target_id = decision.get("target_id")
    reason = decision.get("reason", "")

    if action == "UPDATE" and target_id:
        fact = store.update(
            user_id, target_id, candidate_text, embedder.embed_passage(candidate_text)
        )
        if fact is None:  # LLM named a bad id -> safe fallback to ADD
            fact = store.add(user_id, candidate_text, embedder.embed_passage(candidate_text))
            return {"decision": "ADD", "reason": "update target missing; added", "fact": fact}
        return {"decision": "UPDATE", "target_id": target_id, "reason": reason, "fact": fact}

    if action == "DELETE" and target_id:
        deleted = store.delete(user_id, target_id)
        return {"decision": "DELETE", "target_id": target_id, "reason": reason, "deleted": deleted}

    if action == "NOOP":
        return {"decision": "NOOP", "reason": reason, "fact": None}

    # Default / explicit ADD.
    fact = store.add(user_id, candidate_text, embedder.embed_passage(candidate_text))
    return {"decision": "ADD", "reason": reason or "add", "fact": fact}
