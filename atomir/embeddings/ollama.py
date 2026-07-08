"""Ollama embedder — local models, no API key.

Talks to a local Ollama server (default http://localhost:11434). Symmetric.
The embedding dimension depends on the chosen model (e.g. nomic-embed-text is
768) — set EMBED_DIM to match. NOT live-tested here — needs a running Ollama.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from atomir._http import request_bytes
from atomir.providers.embedder_base import Embedder

_DEFAULT_BASE = "http://localhost:11434"


class OllamaEmbedder(Embedder):
    def __init__(self, embed_dim: int = 768, model: str = "nomic-embed-text",
                 base_url: str = "") -> None:
        self.embed_dim = embed_dim
        self.model = model
        self.url = (base_url.rstrip("/") if base_url else _DEFAULT_BASE) + "/api/embeddings"

    @classmethod
    def from_config(cls, config: dict) -> "OllamaEmbedder":
        return cls(
            embed_dim=config.get("embed_dim", 768),
            model=config.get("model", "nomic-embed-text"),
            base_url=config.get("base_url", ""),
        )

    def _embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * self.embed_dim
        req = urllib.request.Request(
            self.url,
            data=json.dumps({"model": self.model, "prompt": text}).encode("utf-8"),
            method="POST", headers={"Content-Type": "application/json"},
        )
        try:
            body = json.loads(request_bytes(req).decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Ollama embeddings failed: {e.code} {e.reason}") from e
        return body["embedding"]

    def embed_passage(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
