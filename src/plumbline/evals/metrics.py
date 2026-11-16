"""Ranking metrics. `ranked` is a list of chunk ids, best first; `relevant` a set of ids."""

from __future__ import annotations

import math


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """1 if any relevant passage is in the top k (answers here need one good passage)."""
    return float(any(r in relevant for r in ranked[:k]))


def reciprocal_rank(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    for i, r in enumerate(ranked[:k], start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(ranked[:k], start=1) if r in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(values, p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))] if values else 0.0
