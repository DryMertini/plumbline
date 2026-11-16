"""Dense retrieval that runs anywhere: latent semantic analysis.

The default embedder is fitted on the corpus itself - TF-IDF followed by a truncated SVD -
so the demo needs no model download, GPU or API key. It captures co-occurrence semantics
('late interaction' lands near 'ColBERT' even without a shared word), which is exactly the
kind of match BM25 cannot make, and it is deterministic, which keeps CI evals stable.

`Embedder` is the seam: anything with `fit`/`embed_documents`/`embed_query` that returns
L2-normalised vectors (a sentence-transformer, an embeddings API) drops in.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from plumbline.text import tokenize


class Embedder(Protocol):
    name: str

    def embed_query(self, text: str) -> np.ndarray: ...


def _terms(text: str) -> list[str]:
    toks = tokenize(text)
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]


def _l2(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.maximum(norms, 1e-9)


class LSAEmbedder:
    name = "lsa"

    def __init__(self, vocab: dict[str, int], idf: np.ndarray, components: np.ndarray):
        self.vocab, self.idf, self.components = vocab, idf, components  # components: |V| x d

    @classmethod
    def fit(cls, texts: list[str], dims: int = 256, min_df: int = 2) -> tuple[LSAEmbedder, np.ndarray]:
        """Fit on a corpus. Returns the embedder and the document matrix (n x d, normalised)."""
        docs = [Counter(_terms(t)) for t in texts]
        df: Counter = Counter()
        for d in docs:
            df.update(d.keys())
        vocab = {t: i for i, t in enumerate(sorted(t for t, n in df.items() if n >= min_df))}
        n = len(texts)
        idf = np.zeros(len(vocab), dtype=np.float32)
        for t, i in vocab.items():
            idf[i] = np.log((1 + n) / (1 + df[t])) + 1
        rows, cols, vals = [], [], []
        for r, d in enumerate(docs):
            for t, tf in d.items():
                c = vocab.get(t)
                if c is not None:
                    rows.append(r)
                    cols.append(c)
                    vals.append((1 + np.log(tf)) * idf[c])
        x = csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)), dtype=np.float32)
        norms = np.sqrt(x.multiply(x).sum(axis=1)).A1
        x = csr_matrix(x.multiply(1 / np.maximum(norms, 1e-9)[:, None]))
        k = max(2, min(dims, min(x.shape) - 1))
        # deterministic start vector -> identical index on every machine and run
        v0 = np.random.default_rng(0).standard_normal(min(x.shape)).astype(np.float32)
        u, s, vt = svds(x, k=k, v0=v0)
        order = np.argsort(-s)
        u, s, vt = u[:, order], s[order], vt[order]
        signs = np.sign(vt[np.arange(k), np.abs(vt).argmax(axis=1)])  # fix SVD sign ambiguity
        vt, u = vt * signs[:, None], u * signs[None, :]
        embedder = cls(vocab, idf, vt.T.astype(np.float32))
        return embedder, _l2((u * s).astype(np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for t, tf in Counter(_terms(text)).items():
            i = self.vocab.get(t)
            if i is not None:
                vec[i] = (1 + np.log(tf)) * self.idf[i]
        norm = np.linalg.norm(vec)
        if norm == 0:
            return np.zeros(self.components.shape[1], dtype=np.float32)
        return _l2((vec / norm) @ self.components)

    def save(self, path: Path, doc_matrix: np.ndarray, chunk_ids: list[str]) -> None:
        terms = sorted(self.vocab, key=self.vocab.get)
        np.savez_compressed(
            path, terms=np.array(terms), idf=self.idf, components=self.components,
            docs=doc_matrix, chunk_ids=np.array(chunk_ids),
        )

    @classmethod
    def load(cls, path: Path) -> tuple[LSAEmbedder, np.ndarray, list[str]]:
        z = np.load(path, allow_pickle=False)
        vocab = {t: i for i, t in enumerate(z["terms"].tolist())}
        return cls(vocab, z["idf"], z["components"]), z["docs"], z["chunk_ids"].tolist()


class DenseIndex:
    def __init__(self, embedder: Embedder, matrix: np.ndarray):
        self.embedder, self.matrix = embedder, matrix

    def scores(self, query: str) -> np.ndarray:
        return self.matrix @ self.embedder.embed_query(query)
