"""Assembly: build a `MemoryService` from config.

This is the OUTERMOST wiring layer — the one place allowed to name concrete
backends. It selects the LLM and embedder through their factories and the store
through a small backend switch, then injects all three into the (vendor-neutral)
`MemoryService`. Concrete backends are imported lazily so building a fake/JSON
service never requires an unused provider SDK.
"""

from __future__ import annotations

from atomir.config import Settings
from atomir.config import make_qdrant_client
from atomir.config import settings as default_settings
from atomir.memory import MemoryService
from atomir.providers import EmbedderFactory, LLMFactory
from atomir.store_base import MemoryStore


def _build_store(s: Settings) -> MemoryStore:
    provider = s.store_backend
    if provider == "qdrant":
        from atomir.stores.qdrant_store import QdrantMemoryStore

        return QdrantMemoryStore(make_qdrant_client(), s.collection, s.embed_dim)
    if provider == "json":
        from atomir.stores.json_store import JsonMemoryStore

        path = s.store_path if s.store_path.endswith(".json") else "./atomir_store.json"
        return JsonMemoryStore(path=path)
    raise ValueError(
        f"Unknown store backend {provider!r}. Valid backends: ['qdrant', 'json']"
    )


def build_memory_service(s: Settings = default_settings) -> MemoryService:
    """Construct a MemoryService with providers/backend selected purely by config."""
    llm = LLMFactory.create(s.llm)
    embedder = EmbedderFactory.create(s.embedder)
    store = _build_store(s)
    return MemoryService(store, llm, embedder, reconcile_min_sim=s.reconcile_min_sim,
                         hybrid_search=s.hybrid_search)
