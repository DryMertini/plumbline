"""Grounding checks: is each claim actually supported by the passages it cites?

Three independent, deterministic tests per claim:

1. quote      - the supporting quote must occur in a cited passage (verbatim after
                normalisation, or a near-verbatim contiguous match to tolerate PDF noise)
2. support    - the claim's content words must be covered by the cited passages; a cheap
                lexical stand-in for entailment that catches off-topic citations
3. numbers    - every number in the claim must appear in a cited passage (or the question);
                invented figures are the most damaging hallucination in technical Q&A

These are lexical, not semantic: they catch fabrication and mis-citation reliably, but a
fluent paraphrase that subtly inverts a passage's meaning can pass. An NLI model or an
LLM judge is the next layer up, and plugs in at the same place.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from plumbline.text import numbers, tokenize, words

QUOTE_MIN_TOKENS = 4
QUOTE_FUZZY_RATIO = 0.85
SUPPORT_MIN_COVERAGE = 0.6


def quote_match(quote: str, passage: str) -> float:
    """Fraction of the quote found as one contiguous run in the passage (1.0 = verbatim)."""
    q, p = words(quote), words(passage)
    if not q or not p:
        return 0.0
    if f" {' '.join(q)} " in f" {' '.join(p)} ":
        return 1.0
    m = SequenceMatcher(None, q, p, autojunk=False).find_longest_match(0, len(q), 0, len(p))
    return m.size / len(q)


def support_coverage(claim: str, passages: list[str]) -> float:
    """Share of the claim's content words that occur in the cited passages."""
    want = set(tokenize(claim))
    if not want:
        return 1.0
    have: set[str] = set()
    for p in passages:
        have.update(tokenize(p))
    return len(want & have) / len(want)


def unsupported_numbers(claim: str, passages: list[str], question: str = "") -> set[str]:
    allowed = numbers(question)
    for p in passages:
        allowed |= numbers(p)
    # single digits are usually enumeration ("2 reasons"), not data
    return {n for n in numbers(claim) if n not in allowed and len(n.replace(".", "")) > 1}
