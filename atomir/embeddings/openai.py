"""OpenAI embedder — symmetric (same call for passage and query).

`text-embedding-3-small` is 1536-dim and supports a `dimensions` param. Stdlib
urllib + shared retry; no SDK. NOT live-tested here — verify with a real key.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from atomir._http import request_bytes
from atomir.providers.embedder_base import Embedder

_DEFAULT_BASE = "https://api.openai.com/v1"
_USER_AGENT = "atomir/0.2"


class OpenAIEmbedder(Embedder):
    def __init__(self, api_key: str, embed_dim: int = 1536,
                 model: str = "text-embedding-3-small", base_url: str = "") -> None:
        if not api_key:
            raise ValueError(
                "OpenAIEmbedder requires an API key. Set EMBED_API_KEY, or use "
                "EMBED_BACKEND=fake to run offline."
            )
        self.api_key = api_key
        self.embed_dim = embed_dim
        self.model = model
        self.url = (base_url.rstrip("/") if base_url else _DEFAULT_BASE) + "/embeddings"

    @classmethod
    def from_config(cls, config: dict) -> "OpenAIEmbedder":
        return cls(
            api_key=config.get("api_key", ""),
            embed_dim=config.get("embed_dim", 1536),
            model=config.get("model", "text-embedding-3-small"),
            base_url=config.get("base_url", ""),
        )

    def _embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * self.embed_dim
        payload: dict = {"input": text, "model": self.model}
        if self.embed_dim:  # 3-* models honor an explicit dimensions request
            payload["dimensions"] = self.embed_dim
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": _USER_AGENT,
            },
        )
        try:
            body = json.loads(request_bytes(req).decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"OpenAI embeddings failed: {e.code} {e.reason}") from e
        return body["data"][0]["embedding"]

    def embed_passage(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)  # symmetric: same call as passage
