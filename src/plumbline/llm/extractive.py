"""An offline, deterministic reader - the provider that is always available.

It does not generate text. It selects the sentences (or table rows) in the retrieved
passages that best cover the question and returns them as cited, quoted claims in the same
JSON contract a hosted model must satisfy. That makes it three things at once:

* the zero-setup default, so the whole system runs without an API key;
* the last link of the fallback chain - when every hosted model is down the user still
  gets grounded evidence instead of an error page;
* a stable baseline for evals: any hosted model has to beat it to earn its cost.

Knowing when not to answer matters as much as answering. The reader abstains when no
sentence covers enough of the question, and - the "anchor" rule - when a named thing in the
question (a capitalised name or a number: "Australia", "GPT-4", "WMT 2014") is missing from
the passage it would quote: matching the word "capital" is not evidence about "the capital
of Australia", and a BLEU score from one paper is not an answer about another model.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter

from plumbline.llm.base import LLMRequest, LLMResponse
from plumbline.text import numbers, sentences, tokenize

# Tuned on the dev half of the golden set (see `GoldenItem.split`).
MAX_CLAIMS = 3
MIN_COVERAGE = 0.3
KEEP_RATIO = 0.6
BIGRAM_BONUS = 0.08
RANK_PENALTY = 0.0
NUMBER_BONUS = 0.15
CAPTION_PENALTY = 0.1
_NUMERIC_Q = re.compile(
    r"\b(how (many|much|long|large|big|fast)|what (fraction|percentage|share|factor|margin)|"
    r"by (how much|what)|score)\b", re.I)


class ExtractiveLLM:
    def __init__(self, name: str = "extractive"):
        self.name = name
        self.model = "extractive-reader-v1"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(json.dumps(self.read(request)), self.name, self.model, {"calls": 1})

    def read(self, request: LLMRequest) -> dict:
        q_tokens = tokenize(request.question)
        q_terms = set(q_tokens)
        q_bigrams = set(zip(q_tokens, q_tokens[1:], strict=False))
        units: list[tuple[str, str, list[str], int]] = []  # (passage id, text, tokens, rank)
        context: dict[str, set[str]] = {}  # passage id -> every token it (or its heading) has
        for rank, p in enumerate(request.passages):
            if p.kind == "table":
                caption, *rows = p.text.split("\n")
                parts = [caption, *rows]
            else:
                parts = sentences(p.text)
            context[p.id] = set(tokenize(f"{p.source} {p.title} {p.section} {p.text}"))
            for part in parts:
                toks = tokenize(part)
                if len(toks) >= 4:
                    units.append((p.id, part, toks, rank))
        if not units or not q_terms:
            return _abstain()

        if request.term_weights:
            weight = {t: request.term_weights.get(t, 1.0) for t in q_terms}
        else:  # no corpus statistics supplied: fall back to IDF over the passages themselves
            df = Counter(t for _, _, toks, _ in units for t in set(toks))
            weight = {t: math.log(1 + len(units) / (1 + df.get(t, 0))) for t in q_terms}
        total = sum(weight.values()) or 1.0

        anchors = _anchors(request.question)
        eligible = {pid for pid, toks in context.items() if anchors <= toks}
        if not eligible:
            return _abstain(confidence=0.0)

        wants_number = bool(_NUMERIC_Q.search(request.question))
        kinds = {p.id: p.kind for p in request.passages}
        scored = []
        for pid, text, toks, rank in units:
            if pid not in eligible:
                continue
            present = q_terms & set(toks)
            coverage = sum(weight[t] for t in present) / total
            bigrams = len(q_bigrams & set(zip(toks, toks[1:], strict=False)))
            score = coverage + BIGRAM_BONUS * min(bigrams, 3) - RANK_PENALTY * rank
            if wants_number and numbers(text):
                score += NUMBER_BONUS  # "how many / what fraction" is answered by a figure
            if kinds[pid] == "caption":
                score -= CAPTION_PENALTY  # captions name a topic; they rarely state the answer
            scored.append((score, coverage, pid, text, toks))
        if not scored:
            return _abstain()
        scored.sort(key=lambda s: (-s[0], s[2], s[3]))

        best_cov = max(s[1] for s in scored[:MAX_CLAIMS])
        if best_cov < MIN_COVERAGE:
            return _abstain(confidence=round(best_cov, 3))

        claims: list[dict] = []
        for score, _coverage, pid, text, toks in scored:
            if len(claims) >= MAX_CLAIMS or score < KEEP_RATIO * scored[0][0]:
                break
            if any(_jaccard(toks, c["_toks"]) > 0.6 for c in claims):
                continue
            claims.append({"text": text, "citations": [pid], "quote": text, "_toks": toks})
        for c in claims:
            del c["_toks"]
        return {
            "abstained": False,
            "answer": " ".join(c["text"] for c in claims),
            "claims": claims,
            "confidence": round(min(1.0, best_cov), 3),
        }


_RAW_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")


def _anchors(question: str) -> set[str]:
    """Index tokens of the named things in a question: capitalised words (not the first
    word) and anything containing a digit."""
    out: set[str] = set()
    for i, raw in enumerate(_RAW_TOKEN.findall(question)):
        if (i > 0 and any(c.isupper() for c in raw)) or any(c.isdigit() for c in raw):
            out.update(tokenize(raw))
    return out


def _abstain(confidence: float = 0.0) -> dict:
    return {"abstained": True, "answer": "", "claims": [], "confidence": confidence}


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)
