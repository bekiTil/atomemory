"""The `Embedder` interface — the ONLY embedding surface the engine may call.

BOTH `embed_passage` and `embed_query` are kept even though some providers are
symmetric: a symmetric provider implements both as the same call, and an
asymmetric one (Jina) uses different task hints. The asymmetry must never leak
into the interface, and symmetry must never be forced onto asymmetric providers.

HARD RULE — you cannot mix embedders within one collection. The vector
dimension (and the geometry of the space) is owned by the embedder choice, so
switching embedders means a NEW collection / a full re-embed. Do not point two
different embedders at the same collection; that is a config error, not a
runtime fallback.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    @abstractmethod
    def embed_passage(self, text: str) -> list[float]:
        """Embed a stored fact/passage."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a query."""
