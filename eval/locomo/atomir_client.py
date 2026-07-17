"""Drop-in `Mem0Client` replacement backed by atomir, for mem0's
memory-benchmarks framework (github.com/mem0ai/memory-benchmarks).

Wire it in with a one-line import swap in `benchmarks/locomo/run.py`:

    # from benchmarks.common.mem0_client import Mem0Client, format_search_results
    from benchmarks.common.atomir_client import Mem0Client, format_search_results

atomir is configured from env. For a fair, free run matching configs/ollama.yaml
(same models on both sides), set before running:

    LLM_BACKEND=ollama    MODEL=llama3.1
    EMBED_BACKEND=ollama  EMBED_DIM=768        # nomic-embed-text
    STORE_BACKEND=qdrant                        # in-memory per user, handled here

The harness's `add`/`search` are async; atomir is sync, so calls are offloaded
with asyncio.to_thread. One in-memory store per user_id gives clean isolation.

KNOWN GAP: atomir has no temporal reasoning, so LOCOMO's "temporal" category may
score lower — report per-category so this is visible.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from qdrant_client import QdrantClient

from atomir.config import settings
from atomir.memory import MemoryService
from atomir.providers import EmbedderFactory, LLMFactory
from atomir.stores.qdrant_store import QdrantMemoryStore


def format_search_results(search_results: Any) -> tuple[list[dict], dict | None]:
    """Normalize results into (formatted, query_debug) — same contract as mem0's."""
    if not search_results:
        return [], None
    query_debug = None
    if isinstance(search_results, dict):
        query_debug = search_results.get("query_debug")
        search_results = search_results.get("results", [])
    ordered = sorted(search_results, key=lambda x: x.get("score", 0), reverse=True)
    formatted = []
    for r in ordered:
        entry: dict[str, Any] = {
            "memory": r.get("memory", ""),
            "score": r.get("score", 0),
            "id": r.get("id", ""),
        }
        for k in ("created_at", "updated_at", "score_debug"):
            if r.get(k):
                entry[k] = r[k]
        formatted.append(entry)
    return formatted, query_debug


def _messages_to_text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    parts = []
    for m in messages:
        if isinstance(m, dict):
            parts.append(f"{m.get('role', 'user')}: {m.get('content', '')}")
        else:
            parts.append(str(m))
    return "\n".join(parts)


class Mem0Client:
    """atomir-backed, drop-in replacement for the benchmark's Mem0Client."""

    def __init__(self, mode: Any = None, host: Any = None, api_key: Any = None,
                 rpm: Any = None, **_: Any) -> None:
        # Benchmark args are accepted and ignored; atomir is configured via env.
        self._llm = LLMFactory.create(settings.llm)
        self._emb = EmbedderFactory.create(settings.embedder)
        self._svcs: dict[str, MemoryService] = {}

    def _svc(self, user_id: str) -> MemoryService:
        if user_id not in self._svcs:
            store = QdrantMemoryStore(QdrantClient(":memory:"), "bench", settings.embed_dim)
            self._svcs[user_id] = MemoryService(
                store, self._llm, self._emb, reconcile_min_sim=settings.reconcile_min_sim
            )
        return self._svcs[user_id]

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> "Mem0Client":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        pass

    async def add(self, messages: Any, user_id: str, timestamp: Any = None, **_: Any):
        text = _messages_to_text(messages)
        try:
            return await asyncio.to_thread(self._svc(user_id).add, user_id, text)
        except Exception as e:  # benchmark resilience: skip one chunk, keep the run
            sys.stderr.write(f"[atomir_client] add skipped ({type(e).__name__}): {str(e)[:80]}\n")
            return {"results": []}

    async def search(self, query: str, user_id: str, top_k: int = 10,
                     rerank: bool = False, score_debug: bool = False, **_: Any) -> list[dict]:
        res = await asyncio.to_thread(self._svc(user_id).search, user_id, query, top_k)
        return [
            {"memory": r["text"], "score": r.get("score", 0), "id": r.get("id", "")}
            for r in res.get("results", [])
        ]
