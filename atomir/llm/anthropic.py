"""Anthropic (Claude) LLM.

Different shape from the OpenAI-compatible providers: `x-api-key` header, an
`anthropic-version`, a top-level `system`, and `max_tokens` required; the reply
is in `content[0].text`. No native JSON mode, so `chat_json` asks for JSON and
runs the lenient parser. NOT live-tested here — verify with a real key.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from atomir._http import request_bytes
from atomir.llm.parsing import extract_json, judge_with

_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_VERSION = "2023-06-01"
_USER_AGENT = "atomir/0.2"


class AnthropicLLM:
    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001",
                 base_url: str = "", max_tokens: int = 1024,
                 temperature: float | None = None) -> None:
        if not api_key:
            raise ValueError(
                "AnthropicLLM requires an API key. Set LLM_API_KEY, or use "
                "LLM_BACKEND=fake to run offline."
            )
        # Anthropic accepts 0.0-1.0 (OpenAI allows 0-2). Fail here rather than
        # letting the user discover it as a 400 mid-run.
        if temperature is not None and not 0.0 <= temperature <= 1.0:
            raise ValueError(
                f"AnthropicLLM temperature must be between 0.0 and 1.0 "
                f"(got {temperature}). OpenAI's 0-2 range does not apply here."
            )
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.url = base_url.rstrip("/") if base_url else _ANTHROPIC_URL

    @classmethod
    def from_config(cls, config: dict) -> "AnthropicLLM":
        return cls(
            api_key=config.get("api_key", ""),
            model=config.get("model", "claude-haiku-4-5-20251001"),
            base_url=config.get("base_url", ""),
            temperature=config.get("temperature"),
        )

    def _chat(self, system: str, user: str) -> str:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": _VERSION,
                "User-Agent": _USER_AGENT,
            },
        )
        try:
            payload = json.loads(request_bytes(req).decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"Anthropic request failed: {e.code} {e.reason} — {detail}") from e
        return payload["content"][0]["text"]

    def chat_text(self, system: str, user: str) -> str:
        return self._chat(system, user)

    def chat_json(self, system: str, user: str) -> dict:
        # No native JSON mode: instruct + lenient-parse.
        return extract_json(self._chat(system + "\nRespond ONLY with valid JSON.", user))

    def judge(self, rubric: str, content: str) -> tuple[bool, str]:
        return judge_with(self.chat_json, rubric, content)
