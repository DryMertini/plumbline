"""Reciprocal rank fusion - combines rankings without having to calibrate their scores."""

from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: dict[str, list[int]], k: int = 60, weights: dict[str, float] | None = None
) -> list[tuple[int, float, dict[str, int]]]:
    """Fuse named rankings (lists of item ids, best first).

    Returns (item, fused score, {ranking name: 1-based rank}) sorted best first.
    """
    weights = weights or {}
    fused: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, items in rankings.items():
        w = weights.get(name, 1.0)
        for rank, item in enumerate(items, start=1):
            fused[item] = fused.get(item, 0.0) + w / (k + rank)
            ranks.setdefault(item, {})[name] = rank
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(item, score, ranks[item]) for item, score in ordered]
