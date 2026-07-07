"""Atomic WRITE, stage 1: raw text -> a list of atomic facts.

Vendor-neutral engine code: it depends ONLY on the injected `LLM` interface,
never on a concrete provider. Works for user messages and for agent-confirmed
findings alike (pass role="assistant").
"""

from __future__ import annotations

from atomir.providers.llm_base import LLM

_EXTRACT_SYSTEM = (
    "You extract durable, ATOMIC, self-contained facts from a message.\n"
    "Rules:\n"
    "- Write each fact in the THIRD person (e.g. 'The user is vegetarian').\n"
    "- ATOMIC: exactly one fact per item; split compound statements apart.\n"
    "- SELF-CONTAINED: understandable without the original message.\n"
    "- DURABLE: skip greetings, chit-chat, and transient remarks.\n"
    'Return ONLY JSON of the form {"facts": ["...", "..."]}. '
    "Use an empty list if nothing durable is present."
)


def extract_facts(llm: LLM, text: str, role: str = "user") -> list[str]:
    """Return atomic facts extracted from `text`. Empty list if none/blank."""
    text = (text or "").strip()
    if not text:
        return []
    # All framing lives in the system prompt; the user content is the raw
    # message only (keeps the content clean for any LLM to reason over).
    system = f"{_EXTRACT_SYSTEM}\n(The message was written by the {role}.)"
    result = llm.chat_json(system, text)
    facts = result.get("facts", [])
    return [f.strip() for f in facts if isinstance(f, str) and f.strip()]
