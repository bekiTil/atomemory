"""Fake LLM: deterministic, offline, no API key.

The third leg of the offline tripod (with the fake embedder and JSON store).
Its default `chat_json` performs a naive "atomic extraction" — splitting the
user text into sentence-like fragments under a `facts` key — so the write
pipeline produces something real in fake mode. Tests can inject `canned` output
to pin an exact response for any task.
"""

from __future__ import annotations

import re

from atomir.providers.llm_base import LLM


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


class FakeLLM(LLM):
    """Deterministic stand-in LLM. Runs with no key, no network."""

    def __init__(self, canned: dict | str | None = None) -> None:
        # canned dict -> returned by chat_json; canned str -> returned by chat_text.
        self.canned = canned

    @classmethod
    def from_config(cls, config: dict) -> "FakeLLM":
        return cls()

    def chat_json(self, system: str, user: str) -> dict:
        if isinstance(self.canned, dict):
            return dict(self.canned)
        # Default: naive atomic extraction so the write pipeline works offline.
        return {"facts": _split_sentences(user)}

    def chat_text(self, system: str, user: str) -> str:
        if isinstance(self.canned, str):
            return self.canned
        return user.strip()

    def judge(self, rubric: str, content: str) -> tuple[bool, str]:
        # Deterministic accept, so eval harness runs offline.
        return True, "fake judge: accepted deterministically"
