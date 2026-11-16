"""Second-stage reranking over the fused candidates.

First-stage retrievers score a passage against a bag of query terms. The reranker looks
at how the query actually sits inside the passage - how much of the query's *informative*
content is covered, whether its terms appear close together and in order, whether the
paper title or section agrees - and blends that with the fused first-stage score.

It is a transparent feature model rather than a neural cross-encoder so it runs offline;
`Reranker` is the seam for swapping in a cross-encoder or a rerank API.
"""

from __future__ import annotations

from typing import Protocol

from plumbline.text import tokenize
from plumbline.types import ScoredChunk

# Tuned by grid search on the dev half of the golden set only (see evals/README notes in the
# main README). The lesson of that search: the fused first-stage rank is by far the strongest
# signal, and lexical features are tie-breakers - bigram and heading matches earned no weight.
WEIGHTS = {"first_stage": 0.7, "proximity": 0.2, "coverage": 0.1, "bigram": 0.0, "heading": 0.0}


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]: ...


def _min_window(positions: list[list[int]]) -> int:
    """Smallest token window containing one occurrence of every matched query term."""
    events = sorted((p, i) for i, plist in enumerate(positions) for p in plist)
    need, have, best, left = len(positions), {}, 10**9, 0
    for right in range(len(events)):
        have[events[right][1]] = have.get(events[right][1], 0) + 1
        while len(have) == need:
            best = min(best, events[right][0] - events[left][0] + 1)
            term = events[left][1]
            have[term] -= 1
            if not have[term]:
                del have[term]
            left += 1
    return best


class FeatureReranker:
    def __init__(self, idf: dict[str, float]):
        self.idf = idf
        self._default_idf = max(idf.values(), default=1.0)

    def features(self, query: str, sc: ScoredChunk) -> dict[str, float]:
        q = tokenize(query)
        q_terms = list(dict.fromkeys(q))
        body = tokenize(sc.chunk.text)
        positions: dict[str, list[int]] = {}
        for i, tok in enumerate(body):
            if tok in q_terms:
                positions.setdefault(tok, []).append(i)
        weight = {t: self.idf.get(t, self._default_idf) for t in q_terms}
        total = sum(weight.values()) or 1.0
        coverage = sum(weight[t] for t in positions) / total
        q_bigrams = set(zip(q, q[1:], strict=False))
        body_bigrams = set(zip(body, body[1:], strict=False))
        bigram = len(q_bigrams & body_bigrams) / len(q_bigrams) if q_bigrams else 0.0
        proximity = 0.0
        if len(positions) >= 2:
            window = _min_window(list(positions.values()))
            proximity = min(1.0, len(positions) / window) * (len(positions) / len(q_terms))
        head = set(tokenize(f"{sc.chunk.title} {sc.chunk.section}"))
        heading = sum(weight[t] for t in q_terms if t in head) / total
        return {"coverage": coverage, "bigram": bigram, "proximity": proximity, "heading": heading}

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        top = max(c.score for c in candidates) or 1.0
        out = []
        for c in candidates:
            f = self.features(query, c)
            f["first_stage"] = c.score / top
            score = sum(WEIGHTS[name] * f[name] for name in WEIGHTS)
            out.append(ScoredChunk(c.chunk, score, c.ranks, f))
        out.sort(key=lambda c: (-c.score, c.chunk.chunk_id))
        return out
