"""Provider factories: turn a `{provider, config}` block into an implementation.

This is the seam that makes the product accessible to ANYONE, not just Groq +
Jina users. The engine calls `LLMFactory.create(settings.llm)` /
`EmbedderFactory.create(settings.embedder)` and gets back an `LLM` / `Embedder`
without ever naming a vendor. Adding a new provider = write one class with a
`from_config` classmethod, then add one line to the registry here. Zero engine
changes (DECISION #6).
"""

from __future__ import annotations

from atomir.providers.embedder_base import Embedder
from atomir.providers.llm_base import LLM

# The concrete provider classes are imported LAZILY inside the registry builders
# below. That keeps this module's top level free of `atomir.llm` /
# `atomir.embeddings`, avoiding a circular import when those packages are
# imported before `atomir.providers` is fully initialized.


class LLMFactory:
    @staticmethod
    def _registry() -> dict[str, type[LLM]]:
        from atomir.llm import FakeLLM, GroqLLM  # Groq is the FIRST impl, not the only one

        return {"fake": FakeLLM, "groq": GroqLLM}

    @classmethod
    def create(cls, cfg: dict) -> LLM:
        registry = cls._registry()
        provider = cfg.get("provider")
        if provider not in registry:
            raise ValueError(
                f"Unknown LLM provider {provider!r}. "
                f"Valid providers: {sorted(registry)}"
            )
        return registry[provider].from_config(cfg.get("config", {}))


class EmbedderFactory:
    @staticmethod
    def _registry() -> dict[str, type[Embedder]]:
        from atomir.embeddings import FakeEmbedder, JinaEmbedder

        return {"fake": FakeEmbedder, "jina": JinaEmbedder}

    @classmethod
    def create(cls, cfg: dict) -> Embedder:
        registry = cls._registry()
        provider = cfg.get("provider")
        if provider not in registry:
            raise ValueError(
                f"Unknown embedder provider {provider!r}. "
                f"Valid providers: {sorted(registry)}"
            )
        return registry[provider].from_config(cfg.get("config", {}))
