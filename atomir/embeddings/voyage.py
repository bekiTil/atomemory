"""Voyage embedder — another example behind the Embedder interface.

Voyage is asymmetric like Jina: passages use input_type "document", queries use
"query". Stdlib urllib + shared retry; no SDK. NOT live-tested in this repo —
verify against a real Voyage key.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from atomir._http import request_bytes
from atomir.providers.embedder_base import Embedder

_VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
_USER_AGENT = "atomir/0.2"


class VoyageEmbedder(Embedder):
    def __init__(self, api_key: str, embed_dim: int = 1024, model: str = "voyage-3",
                 base_url: str = "") -> None:
        if not api_key:
            raise ValueError(
                "VoyageEmbedder requires an API key. Set EMBED_API_KEY, or use "
                "EMBED_BACKEND=fake to run offline."
            )
        self.api_key = api_key
        self.embed_dim = embed_dim
        self.model = model
        self.url = base_url.rstrip("/") if base_url else _VOYAGE_URL

    @classmethod
    def from_config(cls, config: dict) -> "VoyageEmbedder":
        return cls(
            api_key=config.get("api_key", ""),
            embed_dim=config.get("embed_dim", 1024),
            model=config.get("model", "voyage-3"),
            base_url=config.get("base_url", ""),
        )

    def _embed(self, text: str, input_type: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * self.embed_dim
        data = json.dumps(
            {"input": [text], "model": self.model, "input_type": input_type}
        ).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=data, method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": _USER_AGENT,
            },
        )
        try:
            body = json.loads(request_bytes(req).decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Voyage request failed: {e.code} {e.reason}") from e
        return body["data"][0]["embedding"]

    def embed_passage(self, text: str) -> list[float]:
        return self._embed(text, "document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, "query")
