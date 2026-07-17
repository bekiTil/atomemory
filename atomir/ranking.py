"""Lexical (BM25) scoring + Reciprocal Rank Fusion.

Pure functions over text — no provider, no store, no dependency beyond stdlib.

Why: semantic similarity alone buries facts that match a query's rare terms
(names, project titles, numbers). BM25 catches those; RRF fuses the two rankings
without needing their scores to be on a comparable scale — it only uses RANK,
which is exactly what makes it robust across mismatched scorers.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 over a fixed corpus. Build once, score many queries."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = [tokenize(d) for d in docs]
        self.n = len(self.corpus)
        self.avgdl = (sum(len(d) for d in self.corpus) / self.n) if self.n else 1.0
        self.tf = [Counter(d) for d in self.corpus]
        self.df: Counter = Counter()
        for d in self.corpus:
            self.df.update(set(d))

    def scores(self, query: str) -> list[float]:
        if not self.n:
            return []
        q = tokenize(query)
        out = []
        for tf, doc in zip(self.tf, self.corpus):
            dl = len(doc)
            s = 0.0
            for t in q:
                f = tf.get(t)
                if not f:
                    continue
                idf = math.log(1 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion over several id-rankings (each best-first).

    score(id) = sum over rankings of 1 / (k + rank). Uses rank only, so scorers
    on different scales (cosine vs BM25) fuse safely.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, _id in enumerate(ranking, start=1):
            fused[_id] = fused.get(_id, 0.0) + 1.0 / (k + rank)
    return fused
