"""Okapi BM25, implemented directly so every score is inspectable."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from plumbline.text import tokenize


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(texts)
        lengths = np.zeros(self.n, dtype=np.float32)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, text in enumerate(texts):
            toks = tokenize(text)
            lengths[i] = len(toks)
            for term, tf in Counter(toks).items():
                postings[term].append((i, tf))
        self.avgdl = float(lengths.mean()) if self.n else 0.0
        self._norm = self.k1 * (1 - self.b + self.b * lengths / max(self.avgdl, 1e-9))
        self.idf: dict[str, float] = {}
        self._docs: dict[str, np.ndarray] = {}
        self._tfs: dict[str, np.ndarray] = {}
        for term, plist in postings.items():
            df = len(plist)
            self.idf[term] = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            self._docs[term] = np.fromiter((d for d, _ in plist), dtype=np.int32, count=df)
            self._tfs[term] = np.fromiter((t for _, t in plist), dtype=np.float32, count=df)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for term in set(tokenize(query)):
            docs = self._docs.get(term)
            if docs is None:
                continue
            tf = self._tfs[term]
            out[docs] += self.idf[term] * tf * (self.k1 + 1) / (tf + self._norm[docs])
        return out

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        return top_k(self.scores(query), k)


def top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Indices of the k best strictly-positive scores, best first."""
    if scores.size == 0:
        return []
    k = min(k, scores.size)
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]
