"""Client SDK: adoption in one import.

`MemoryClient` mirrors the HTTP API (and the `MemoryService` facade) method for
method, with IDENTICAL return shapes — so callers can move between the library,
the service, and this SDK without changing code. Built on stdlib urllib, so the
SDK adds no dependency.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from urllib.parse import urlencode


class MemoryClient:
    """Thin REST wrapper around an atomir API server."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, *, body: dict | None = None, params: dict | None = None):
        url = self.base_url + path
        if params:
            url += "?" + urlencode(params)
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(
                f"atomir API {method} {path} failed: {e.code} {e.reason} — {detail}"
            ) from e

    def add(self, user_id: str, text: str) -> dict:
        return self._request("POST", "/memories", body={"user_id": user_id, "text": text})

    def search(self, user_id: str, query: str, k: int = 6, decompose: bool = True) -> dict:
        return self._request(
            "POST", "/search",
            body={"user_id": user_id, "query": query, "k": k, "decompose": decompose},
        )

    def get_all(self, user_id: str) -> list[dict]:
        return self._request("GET", "/memories", params={"user_id": user_id})

    def delete(self, user_id: str, fact_id: str) -> dict:
        return self._request("DELETE", f"/memories/{fact_id}", params={"user_id": user_id})

    def reset(self, user_id: str) -> dict:
        return self._request("DELETE", "/memories", params={"user_id": user_id})

    def health(self) -> dict:
        return self._request("GET", "/health")
