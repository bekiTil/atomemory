"""Groq LLM: ONE concrete example behind the (future) LLM interface.

Groq exposes an OpenAI-compatible chat-completions API, so this class is a thin
wrapper: build messages, optionally request JSON mode, POST via stdlib urllib
(no SDK dependency), and run the response through the lenient parser for
`chat_json`. OpenAI/Anthropic/Ollama would each be another such class.
"""

from __future__ import annotations

import json
import warnings
import urllib.error
import urllib.request

from atomir._http import request_bytes
from atomir.llm.parsing import extract_json
from atomir.providers.llm_base import LLM

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Groq sits behind Cloudflare, which blocks the default urllib User-Agent
# (403 / "error code: 1010"). A named UA avoids that.
_USER_AGENT = "atomir/0.1"


class GroqLLM(LLM):
    """Calls the Groq chat API. Requires an API key."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile",
                 temperature: float | None = None) -> None:
        if not api_key:
            raise ValueError(
                "GroqLLM requires an API key. Set LLM_API_KEY, or use "
                "LLM_BACKEND=fake to run offline with no key."
            )
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    @classmethod
    def from_config(cls, config: dict) -> "GroqLLM":
        return cls(
            api_key=config.get("api_key", ""),
            model=config.get("model", "llama-3.3-70b-versatile"),
            temperature=config.get("temperature"),
        )

    def _chat(self, system: str, user: str, json_mode: bool) -> str:
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            _GROQ_URL,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": _USER_AGENT,
            },
        )
        try:
            payload = json.loads(request_bytes(req).decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            # Some models (e.g. OpenAI reasoning models) reject `temperature`.
            # React to what the API actually says rather than guessing from
            # model names, and retry ONCE without it. Never silent: warn.
            if e.code == 400 and "temperature" in detail.lower() and "temperature" in body:
                warnings.warn(
                    f"Groq rejected temperature={body['temperature']} for model "
                    f"{self.model!r}; retrying without it.",
                    RuntimeWarning, stacklevel=2,
                )
                body.pop("temperature")
                retry = urllib.request.Request(
                    req.full_url, data=json.dumps(body).encode("utf-8"),
                    method="POST", headers=dict(req.headers),
                )
                payload = json.loads(request_bytes(retry).decode("utf-8"))
                return payload["choices"][0]["message"]["content"]
            raise RuntimeError(f"Groq request failed: {e.code} {e.reason} — {detail}") from e
        return payload["choices"][0]["message"]["content"]

    def chat_text(self, system: str, user: str) -> str:
        return self._chat(system, user, json_mode=False)

    def chat_json(self, system: str, user: str) -> dict:
        return extract_json(self._chat(system, user, json_mode=True))

    def judge(self, rubric: str, content: str) -> tuple[bool, str]:
        system = (
            "You are a strict evaluator. Given a rubric and some content, decide "
            "if the content satisfies the rubric. Respond ONLY with JSON: "
            '{"pass": <true|false>, "reason": "<short explanation>"}.'
        )
        user = f"RUBRIC:\n{rubric}\n\nCONTENT:\n{content}"
        result = self.chat_json(system, user)
        return bool(result.get("pass")), str(result.get("reason", ""))
