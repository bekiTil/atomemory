"""Provider factories: turn a `{provider, config}` block into an implementation.

This is the seam that makes the product accessible to ANYONE, not just Groq +
Jina users. The engine calls `LLMFactory.create(settings.llm)` /
`EmbedderFactory.create(settings.embedder)` and gets back an `LLM` / `Embedder`
without ever naming a vendor. Adding a new provider = write one class with a
`from_config` classmethod, then add one line to the registry here. Zero engine
changes (DECISION #6).
"""

from __future__ import annotations

from atomir.embeddings import FakeEmbedder, JinaEmbedder
from atomir.llm import FakeLLM, GroqLLM
from atomir.providers.embedder_base import Embedder
from atomir.providers.llm_base import LLM


class LLMFactory:
    # provider name -> class. Groq is the FIRST impl, not the only one.
    _REGISTRY: dict[str, type[LLM]] = {
        "fake": FakeLLM,
        "groq": GroqLLM,
    }

    @classmethod
    def create(cls, cfg: dict) -> LLM:
        provider = cfg.get("provider")
        if provider not in cls._REGISTRY:
            raise ValueError(
                f"Unknown LLM provider {provider!r}. "
                f"Valid providers: {sorted(cls._REGISTRY)}"
            )
        return cls._REGISTRY[provider].from_config(cfg.get("config", {}))


class EmbedderFactory:
    _REGISTRY: dict[str, type[Embedder]] = {
        "fake": FakeEmbedder,
        "jina": JinaEmbedder,
    }

    @classmethod
    def create(cls, cfg: dict) -> Embedder:
        provider = cfg.get("provider")
        if provider not in cls._REGISTRY:
            raise ValueError(
                f"Unknown embedder provider {provider!r}. "
                f"Valid providers: {sorted(cls._REGISTRY)}"
            )
        return cls._REGISTRY[provider].from_config(cfg.get("config", {}))
