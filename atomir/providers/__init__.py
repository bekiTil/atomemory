"""Provider interfaces + factories — the vendor-neutral seam.

The engine imports ONLY from here (`LLM`, `Embedder`, and the factories),
never a concrete provider module or vendor SDK.
"""

from atomir.providers.embedder_base import Embedder
from atomir.providers.factory import EmbedderFactory, LLMFactory
from atomir.providers.llm_base import LLM

__all__ = ["LLM", "Embedder", "LLMFactory", "EmbedderFactory"]
