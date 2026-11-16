"""The retrieval facade: BM25 + dense candidates -> RRF fusion -> rerank -> top-k."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from plumbline.config import Settings
from plumbline.retrieve.bm25 import BM25Index, top_k
from plumbline.retrieve.dense import DenseIndex, LSAEmbedder
from plumbline.retrieve.fusion import reciprocal_rank_fusion
from plumbline.retrieve.rerank import FeatureReranker
from plumbline.store import Store
from plumbline.text import tokenize
from plumbline.types import Chunk, ScoredChunk

MODES = ("bm25", "dense", "hybrid", "hybrid+rerank")


@dataclass
class RetrievalResult:
    hits: list[ScoredChunk]
    mode: str
    timings_ms: dict[str, float] = field(default_factory=dict)
    candidates: int = 0

    @property
    def confidence(self) -> float:
        """How much of the query's informative content the best passage covers (0..1)."""
        return max((h.features.get("coverage", 0.0) for h in self.hits), default=0.0)


def build_indexes(chunks: list[Chunk], settings: Settings) -> dict:
    """Fit and persist the dense index. (BM25 is rebuilt in memory at load - it is fast.)"""
    if not chunks:
        settings.index_path.unlink(missing_ok=True)
        return {"chunks": 0}
    t0 = time.time()
    embedder, matrix = LSAEmbedder.fit([c.index_text for c in chunks], dims=settings.lsa_dims)
    embedder.save(settings.index_path, matrix, [c.chunk_id for c in chunks])
    return {
        "chunks": len(chunks), "dense_dims": int(matrix.shape[1]),
        "dense_vocab": len(embedder.vocab), "embedder": embedder.name,
        "build_seconds": round(time.time() - t0, 2),
    }


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder: LSAEmbedder, matrix: np.ndarray,
                 settings: Settings):
        self.chunks = chunks
        self.settings = settings
        self.bm25 = BM25Index([c.index_text for c in chunks])
        self.dense = DenseIndex(embedder, matrix)
        self.reranker = FeatureReranker(self.bm25.idf)
        self.by_id = {c.chunk_id: c for c in chunks}

    @classmethod
    def load(cls, settings: Settings) -> Retriever:
        store = Store(settings.db_path)
        chunks = store.chunks()
        store.close()
        if not chunks or not settings.index_path.exists():
            raise RuntimeError("No index found. Run `plumbline fetch` then `plumbline ingest`.")
        embedder, matrix, ids = LSAEmbedder.load(settings.index_path)
        if ids != [c.chunk_id for c in chunks]:
            raise RuntimeError("Dense index is out of sync with the chunk store - re-run ingest.")
        return cls(chunks, embedder, matrix, settings)

    def search(self, query: str, *, mode: str = "hybrid+rerank", k: int | None = None) -> RetrievalResult:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        k = k or self.settings.top_k
        n_cand = self.settings.candidates_per_retriever
        timings: dict[str, float] = {}
        rankings: dict[str, list[int]] = {}
        raw: dict[str, dict[int, float]] = {}

        if mode != "dense":
            t0 = time.perf_counter()
            hits = self.bm25.search(query, n_cand)
            timings["bm25"] = (time.perf_counter() - t0) * 1000
            rankings["bm25"], raw["bm25"] = [i for i, _ in hits], dict(hits)
        if mode != "bm25":
            t0 = time.perf_counter()
            hits = top_k(self.dense.scores(query), n_cand)
            timings["dense"] = (time.perf_counter() - t0) * 1000
            rankings["dense"], raw["dense"] = [i for i, _ in hits], dict(hits)

        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(rankings, k=self.settings.rrf_k)
        timings["fuse"] = (time.perf_counter() - t0) * 1000
        candidates = [
            ScoredChunk(self.chunks[i], score, ranks,
                        {name: raw[name][i] for name in ranks})
            for i, score, ranks in fused
        ]
        if mode == "hybrid+rerank":
            t0 = time.perf_counter()
            candidates = self.reranker.rerank(query, candidates)
            timings["rerank"] = (time.perf_counter() - t0) * 1000
        else:
            for c in candidates[:k]:
                c.features.update(self.reranker.features(query, c))
        return RetrievalResult(candidates[:k], mode, {n: round(v, 2) for n, v in timings.items()},
                               len(candidates))

    def query_terms(self, query: str) -> list[tuple[str, float]]:
        """Query terms with their IDF - used by the UI to explain a retrieval."""
        return [(t, round(self.bm25.idf.get(t, 0.0), 2)) for t in dict.fromkeys(tokenize(query))]

    def term_weights(self, query: str) -> dict[str, float]:
        """Corpus IDF of each query term; a term the corpus has never seen gets the maximum."""
        ceiling = max(self.bm25.idf.values(), default=1.0)
        return {t: self.bm25.idf.get(t, ceiling) for t in dict.fromkeys(tokenize(query))}
