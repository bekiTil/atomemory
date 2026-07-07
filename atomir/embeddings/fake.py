"""Fake embedder: deterministic, offline, no API key.

Maps text -> a unit vector derived from a hash of the text. Both
`embed_passage` and `embed_query` call the SAME underlying function, so the
same text always yields the same vector — which lets exact-text lookups match
in tests. This is the offline stand-in that lets the whole pipeline run with no
keys, no network, and no cost.

Pure stdlib on purpose: the offline path should depend on nothing external.
"""

from __future__ import annotations

import hashlib
import math
import random

from atomir.providers.embedder_base import Embedder


class FakeEmbedder(Embedder):
    """Deterministic hash-based embedder for offline development and tests."""

    def __init__(self, embed_dim: int = 1024) -> None:
        self.embed_dim = embed_dim

    @classmethod
    def from_config(cls, config: dict) -> "FakeEmbedder":
        return cls(embed_dim=config.get("embed_dim", 1024))

    def _embed(self, text: str) -> list[float]:
        # Empty / whitespace-only text has no content to embed -> zero vector.
        if not text or not text.strip():
            return [0.0] * self.embed_dim
        # Seed a PRNG deterministically from the text's hash, draw a Gaussian
        # vector, and normalize to unit length (a point on the unit sphere).
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest(), "big")
        rng = random.Random(seed)
        vec = [rng.gauss(0.0, 1.0) for _ in range(self.embed_dim)]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_passage(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_query(self, text: str) -> list[float]:
        # Same call as passage: fake mode is symmetric so exact text matches.
        return self._embed(text)
