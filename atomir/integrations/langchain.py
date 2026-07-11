"""LangChain integration for atomir.

Two pieces:
- `AtomirRetriever` — a `BaseRetriever` backed by atomir's atomic search, so
  atomir drops into any LangChain chain / RAG pipeline as a retriever.
- `AtomirMemory` — a tiny helper to `recall(query)` a formatted context string
  and `remember(text)` a turn, scoped to one user_id.

Both work with either an in-process `MemoryService` (from
`atomir.assembly.build_memory_service`) or a remote `MemoryClient` — both expose
`.search()` and `.add()`. Install with `pip install atomir[langchain]`.

Example:
    from atomir.assembly import build_memory_service
    from atomir.integrations.langchain import AtomirMemory

    mem = AtomirMemory(build_memory_service(), user_id="user:123")
    mem.remember("I'm vegetarian and my manager is Dana.")
    context = mem.recall("who is my manager?")     # formatted string for a prompt
    retriever = mem.as_retriever()                 # or use as a LangChain retriever
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from atomir.integrations import format_memories

__all__ = ["AtomirRetriever", "AtomirMemory", "format_memories"]


class AtomirRetriever(BaseRetriever):
    """A LangChain retriever backed by atomir's decompose-and-retrieve search.

    `memory` is any object with a mem0-style `.search(user_id, query, k,
    decompose)` — a `MemoryService` or a `MemoryClient`.
    """

    memory: Any
    user_id: str
    k: int = 6
    decompose: bool = True

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        res = self.memory.search(self.user_id, query, k=self.k, decompose=self.decompose)
        return [
            Document(
                page_content=r["text"],
                metadata={"id": r.get("id"), "score": r.get("score")},
            )
            for r in res.get("results", [])
        ]


class AtomirMemory:
    """Scoped helper around a MemoryService/MemoryClient for one user_id."""

    def __init__(self, memory: Any, user_id: str, k: int = 6, decompose: bool = True) -> None:
        self.memory = memory
        self.user_id = user_id
        self.k = k
        self.decompose = decompose

    def recall(self, query: str) -> str:
        """Return relevant memories as a formatted string, ready for a prompt."""
        res = self.memory.search(self.user_id, query, k=self.k, decompose=self.decompose)
        return format_memories(res.get("results", []))

    def remember(self, text: str) -> dict:
        """Extract atomic facts from `text` and reconcile them into memory."""
        return self.memory.add(self.user_id, text)

    def as_retriever(self) -> AtomirRetriever:
        return AtomirRetriever(
            memory=self.memory, user_id=self.user_id, k=self.k, decompose=self.decompose
        )
