"""Core records passed between stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    chunk_id: str  # "<short>:<ordinal>", stable across re-ingests of the same file
    doc_id: str
    short: str
    title: str
    section: str
    kind: str  # text | table | caption
    page_start: int
    page_end: int
    text: str
    n_tokens: int = 0
    ordinal: int = 0

    @property
    def index_text(self) -> str:
        """What gets indexed: the passage plus its provenance, so 'BERT' or 'ablation'
        can match even when the passage itself never repeats the paper or section name."""
        return f"{self.title}. {self.section}. {self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # retriever name -> 1-based rank
    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            **self.chunk.to_dict(),
            "score": round(self.score, 5),
            "ranks": self.ranks,
            "features": {k: round(v, 4) for k, v in self.features.items()},
        }
