from __future__ import annotations

import numpy as np
import pytest

from plumbline.config import Settings
from plumbline.retrieve.dense import LSAEmbedder
from plumbline.retrieve.retriever import Retriever
from plumbline.types import Chunk

DOCS = {
    ("alpha", "The Alpha Retriever"): [
        "The Alpha retriever encodes every passage into a dense vector with a small encoder. "
        "It was trained for 12 hours on 4 GPUs and reaches 71.5 recall on the Quartz benchmark.",
        "Alpha uses in-batch negatives during training. Each batch holds 128 question and passage pairs.",
        "At query time Alpha searches the index with maximum inner product search over all vectors.",
    ],
    ("beta", "Beta: Low-Rank Tuning"): [
        "Beta adapts a frozen language model by learning low-rank update matrices. "
        "It reduces the number of trainable parameters by 500 times compared with full fine-tuning.",
        "Beta adds no inference latency because the learned matrices are merged into the weights.",
        "The rank used in the Beta experiments is 8 and the optimizer is Adam with warmup.",
    ],
    ("gamma", "Gamma Evaluation Suite"): [
        "The Gamma suite scores a question answering system on faithfulness and answer relevance.",
        "Gamma contains 300 questions written by annotators and 40 questions that have no answer.",
        "A system should refuse the unanswerable questions in Gamma instead of guessing.",
    ],
}


def make_chunks() -> list[Chunk]:
    chunks = []
    for (short, title), texts in DOCS.items():
        for i, text in enumerate(texts):
            chunks.append(Chunk(f"{short}:{i:04d}", short, short, title, f"{i + 1} Section", "text",
                                i + 1, i + 1, text, len(text.split()), i))
    return chunks


@pytest.fixture(scope="session")
def chunks() -> list[Chunk]:
    return make_chunks()


@pytest.fixture(scope="session")
def retriever(chunks) -> Retriever:
    embedder, matrix = LSAEmbedder.fit([c.index_text for c in chunks], dims=6, min_df=1)
    assert np.isfinite(matrix).all()
    return Retriever(chunks, embedder, matrix, Settings(top_k=4))
