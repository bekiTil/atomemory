"""Per-key locking so writes to one key serialize while other keys run free.

Used to serialize a single user's reconcile (a read-modify-write: search ->
decide -> apply). Two concurrent writes for the SAME user must not both read the
old state and both ADD; different users never block each other.
"""

from __future__ import annotations

import threading


class KeyedLock:
    """Hands out one reentrant-free lock per key, created on first use."""

    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def get(self, key: str) -> threading.Lock:
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock
