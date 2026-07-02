"""Groq LLM: ONE concrete example behind the (future) LLM interface.

Groq exposes an OpenAI-compatible chat-completions API, so this class is a thin
wrapper: build messages, optionally request JSON mode, POST via stdlib urllib
(no SDK dependency), and run the response through the lenient parser for
`chat_json`. OpenAI/Anthropic/Ollama would each be another such class.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from atomir.llm.parsing import extract_json

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqLLM:
    """Calls the Groq chat API. Requires an API key."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        if not api_key:
            raise ValueError(
                "GroqLLM requires an API key. Set LLM_API_KEY, or use "
                "LLM_BACKEND=fake to run offline with no key."
            )
        self.api_key = api_key
        self.model = model

    def _chat(self, system: str, user: str, json_mode: bool) -> str:
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            _GROQ_URL,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Groq request failed: {e.code} {e.reason}") from e
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
