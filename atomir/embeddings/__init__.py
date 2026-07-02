"""Embedder implementations.

The formal `Embedder` interface is introduced in Step 5.5; for now these are
concrete, duck-typed classes that all expose the same two methods:
`embed_passage(text) -> list[float]` and `embed_query(text) -> list[float]`.
"""

from atomir.embeddings.fake import FakeEmbedder
from atomir.embeddings.jina import JinaEmbedder

__all__ = ["FakeEmbedder", "JinaEmbedder"]
