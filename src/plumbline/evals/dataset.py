"""The golden set and its relevance judgements."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from plumbline.types import Chunk

DEFAULT_PATH = Path("evals/golden.yaml")


@dataclass
class GoldenItem:
    id: str
    question: str
    type: str
    docs: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    answer: list[str] = field(default_factory=list)
    relevant: set[str] = field(default_factory=set)  # chunk ids, filled by `judge`

    @property
    def split(self) -> str:
        """Deterministic dev/test split. Tunable parameters (reranker weights, abstention
        thresholds) are chosen on dev; test is only ever reported."""
        return "dev" if int(hashlib.sha1(self.id.encode()).hexdigest(), 16) % 2 == 0 else "test"

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"

    def is_correct(self, text: str) -> bool:
        return bool(text) and all(re.search(p, text) for p in self.answer)


def load_golden(path: Path = DEFAULT_PATH) -> list[GoldenItem]:
    data = yaml.safe_load(path.read_text())
    return [GoldenItem(**item) for item in data["items"]]


def judge(items: list[GoldenItem], chunks: list[Chunk]) -> list[str]:
    """Attach relevant chunk ids to each item. Returns problems (items with no evidence)."""
    problems = []
    for item in items:
        if not item.answerable:
            continue
        patterns = [re.compile(p) for p in item.evidence]
        item.relevant = {
            c.chunk_id for c in chunks
            if c.short in item.docs and any(p.search(c.text) for p in patterns)
        }
        if not item.relevant:
            problems.append(f"{item.id}: no chunk in {item.docs} matches {item.evidence}")
    return problems


if __name__ == "__main__":  # pragma: no cover
    from plumbline.config import get_settings
    from plumbline.store import Store

    store = Store(get_settings().db_path)
    golden = load_golden()
    issues = judge(golden, store.chunks())
    for item in golden:
        if item.answerable:
            print(f"{item.id:24} {len(item.relevant):3} relevant chunk(s)")
    print("\n".join(issues) or "all answerable items have evidence")
    raise SystemExit(1 if issues else 0)
