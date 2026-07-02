"""Lenient JSON extraction from LLM text.

Even with a provider's JSON mode, some models wrap the object in prose or
```code fences```. These helpers turn "usually valid JSON" into "always a dict"
so callers of `chat_json` never have to handle raw model text.
"""

from __future__ import annotations

import json
import re


def _first_json_object(text: str) -> str | None:
    """Return the first balanced {...} substring, ignoring braces inside strings."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def extract_json(text: str) -> dict:
    """Parse a dict out of possibly-messy model output, or raise ValueError."""
    s = text.strip()
    # Strip a leading/trailing ```json ... ``` fence if present.
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    # Fast path: the whole thing is already a JSON object.
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Fallback: pull the first balanced object out of surrounding prose.
    candidate = _first_json_object(s)
    if candidate is not None:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON from model output: {text[:200]!r}")
