"""atomir — atomic memory infrastructure for agents.

Memory is atomic on both ends: atomic facts on write, atomic sub-question
decomposition on read. This package is built inside-out (engine → storage →
multi-tenancy → API → packaging) and is vendor-neutral: the LLM, embedder, and
vector store are each an interface chosen at runtime by config.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:  # single source of truth is pyproject.toml, so this can't go stale
    __version__ = _pkg_version("atomir")
except PackageNotFoundError:  # running from a source checkout, not installed
    __version__ = "0.0.0+source"
