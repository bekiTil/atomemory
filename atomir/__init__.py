"""atomir — atomic memory infrastructure for agents.

Memory is atomic on both ends: atomic facts on write, atomic sub-question
decomposition on read. This package is built inside-out (engine → storage →
multi-tenancy → API → packaging) and is vendor-neutral: the LLM, embedder, and
vector store are each an interface chosen at runtime by config.
"""

__version__ = "0.0.0"
