"""Tiny stdlib HTTP helper with retry/backoff for transient failures.

Provider calls (LLM, embedder) share this so a rate-limit (429) or a transient
5xx doesn't fail a whole request. Honors a numeric `Retry-After` header when
present, else exponential backoff. Non-transient errors (e.g. 401/403/400) raise
immediately.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request

_RETRYABLE = {429, 500, 502, 503, 504}


def request_bytes(req: urllib.request.Request, *, timeout: float = 60.0, attempts: int = 5) -> bytes:
    """POST/GET the request and return the response body, retrying transient errors."""
    for i in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code in _RETRYABLE and i < attempts - 1:
                ra = e.headers.get("Retry-After")
                delay = float(ra) if (ra and ra.replace(".", "", 1).isdigit()) else min(2 ** i, 8)
                time.sleep(delay)
                continue
            raise
    raise RuntimeError("unreachable")  # pragma: no cover
