"""Ollama LLM — local models, no API key.

Talks to a local Ollama server (default http://localhost:11434) via /api/chat.
Uses Ollama's `format: json` for structured output, then the lenient parser.
NOT live-tested here — needs a running Ollama with the model pulled.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from atomir._http import request_bytes
from atomir.llm.parsing import extract_json, judge_with

_DEFAULT_BASE = "http://localhost:11434"


class OllamaLLM:
    def __init__(self, model: str = "llama3.1", base_url: str = "",
                 temperature: float | None = None) -> None:
        self.model = model
        self.temperature = temperature
        self.url = (base_url.rstrip("/") if base_url else _DEFAULT_BASE) + "/api/chat"

    @classmethod
    def from_config(cls, config: dict) -> "OllamaLLM":
        return cls(
            model=config.get("model", "llama3.1"),
            base_url=config.get("base_url", ""),
            temperature=config.get("temperature"),
        )

    def _chat(self, system: str, user: str, json_mode: bool) -> str:
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        if self.temperature is not None:
            body["options"] = {"temperature": self.temperature}
        if json_mode:
            body["format"] = "json"
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            payload = json.loads(request_bytes(req).decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"Ollama request failed: {e.code} {e.reason} — {detail}") from e
        return payload["message"]["content"]

    def chat_text(self, system: str, user: str) -> str:
        return self._chat(system, user, json_mode=False)

    def chat_json(self, system: str, user: str) -> dict:
        return extract_json(self._chat(system, user, json_mode=True))

    def judge(self, rubric: str, content: str) -> tuple[bool, str]:
        return judge_with(self.chat_json, rubric, content)
