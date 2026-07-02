"""LLM implementations.

The formal `LLM` interface is introduced in Step 5.5; for now these are
concrete, duck-typed classes exposing `chat_json`, `chat_text`, and `judge`.
"""

from atomir.llm.fake import FakeLLM
from atomir.llm.groq import GroqLLM
from atomir.llm.parsing import extract_json

__all__ = ["FakeLLM", "GroqLLM", "extract_json"]
