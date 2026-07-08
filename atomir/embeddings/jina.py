"""Jina embedder: ONE concrete example behind the (future) Embedder interface.

`jina-embeddings-v3` is asymmetric — passages and queries are embedded with
different task hints (`retrieval.passage` vs `retrieval.query`), which improves
retrieval quality. A symmetric provider would point both methods at the same
call; we keep both methods regardless so every embedder has the same shape.

Uses stdlib `urllib` (no new dependency) so simply importing this module never
requires a provider SDK. The network is only touched when a method is called.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from atomir.providers.embedder_base import Embedder

_JINA_URL = "https://api.jina.ai/v1/embeddings"
# A named User-Agent avoids Cloudflare's default-urllib-UA block (see groq.py).
_USER_AGENT = "atomir/0.1"


class JinaEmbedder(Embedder):
    """Calls the Jina embeddings API. Requires an API key."""

    def __init__(
        self,
        api_key: str,
        embed_dim: int = 1024,
        model: str = "jina-embeddings-v3",
    ) -> None:
        # Clear failure if a real provider is selected without its key.
        if not api_key:
            raise ValueError(
                "JinaEmbedder requires an API key. Set EMBED_API_KEY, or use "
                "EMBED_BACKEND=fake to run offline with no key."
            )
        self.api_key = api_key
        self.embed_dim = embed_dim
        self.model = model

    @classmethod
    def from_config(cls, config: dict) -> "JinaEmbedder":
        return cls(
            api_key=config.get("api_key", ""),
            embed_dim=config.get("embed_dim", 1024),
            model=config.get("model", "jina-embeddings-v3"),
        )

    def _embed(self, text: str, task: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * self.embed_dim
        payload = json.dumps(
            {
                "model": self.model,
                "task": task,
                "dimensions": self.embed_dim,
                "input": [text],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            _JINA_URL,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": _USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(req) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Jina embedding request failed: {e.code} {e.reason}"
            ) from e
        return body["data"][0]["embedding"]

    def embed_passage(self, text: str) -> list[float]:
        return self._embed(text, "retrieval.passage")

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, "retrieval.query")
