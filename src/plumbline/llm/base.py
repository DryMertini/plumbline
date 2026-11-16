"""The provider seam. Everything above this file is provider-agnostic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Passage:
    id: str
    title: str
    section: str
    pages: str
    kind: str
    text: str
    source: str = ""  # the paper's short name, e.g. "colbert"


@dataclass
class LLMRequest:
    question: str
    passages: list[Passage]
    system: str
    prompt: str
    # set on a repair attempt: the previous raw output and what was wrong with it
    previous_output: str | None = None
    repair_feedback: str | None = None
    # corpus-level IDF of the question's terms; used by the offline reader
    term_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: dict = field(default_factory=dict)


class LLMProvider(Protocol):
    name: str
    model: str

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return the model's raw text. Raise an `LLMError` subclass on failure."""
        ...
