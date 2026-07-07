"""The `LLM` interface — the ONLY LLM surface the engine may call.

Deliberately just two methods. `judge` (used by the eval harness) is NOT here:
it is not part of the engine's contract, so it stays a concrete-class extra.
Any provider (Groq, OpenAI, Anthropic, Ollama, fake) implements this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLM(ABC):
    @abstractmethod
    def chat_json(self, system: str, user: str) -> dict:
        """Return a dict for a structured task (extract / reconcile / decompose)."""

    @abstractmethod
    def chat_text(self, system: str, user: str) -> str:
        """Return freeform text."""
