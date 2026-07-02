"""Concrete `MemoryStore` backends.

Each module here implements the Step 2 contract for a different storage engine;
the engine code never imports these directly — it depends only on `MemoryStore`.
"""

from atomir.stores.json_store import JsonMemoryStore

__all__ = ["JsonMemoryStore"]
