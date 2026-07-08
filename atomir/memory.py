"""Engine facade: one clean object the API and SDK call.

`MemoryService` is injected with a `MemoryStore`, an `LLM`, and an `Embedder`
(built elsewhere by their factories). It orchestrates the write and read engines
but imports NO concrete provider or backend — which store, model, or embedder is
used is purely a matter of which instances you inject.
"""

from __future__ import annotations

from atomir.atomic_read import atomic_search
from atomir.extractor import extract_facts
from atomir.providers.embedder_base import Embedder
from atomir.providers.llm_base import LLM
from atomir.reconciler import reconcile
from atomir.store_base import MemoryStore


class MemoryService:
    """Stable seam over the atomic write/read engines. Backend-injected."""

    def __init__(
        self,
        store: MemoryStore,
        llm: LLM,
        embedder: Embedder,
        *,
        reconcile_min_sim: float = 0.6,
    ) -> None:
        self.store = store
        self.llm = llm
        self.embedder = embedder
        self.reconcile_min_sim = reconcile_min_sim

    def add(self, user_id: str, text: str) -> dict:
        """Extract atomic facts from `text` and reconcile each into memory."""
        operations: list[dict] = []
        facts: list[dict] = []
        for candidate in extract_facts(self.llm, text):
            decision = reconcile(
                self.store,
                self.llm,
                self.embedder,
                user_id,
                candidate,
                min_sim=self.reconcile_min_sim,
            )
            operations.append(decision)
            if decision.get("fact"):
                facts.append(decision["fact"])
        return {"operations": operations, "facts": facts}

    def search(
        self, user_id: str, query: str, k: int = 6, decompose: bool = True
    ) -> dict:
        """Answer a query atomically: {subquestions, results}."""
        return atomic_search(
            self.store, self.llm, self.embedder, user_id, query, k=k, decompose=decompose
        )

    def get_all(self, user_id: str) -> list[dict]:
        return self.store.all(user_id)

    def delete(self, user_id: str, fact_id: str) -> bool:
        return self.store.delete(user_id, fact_id)

    def reset(self, user_id: str) -> bool:
        return self.store.clear(user_id)
