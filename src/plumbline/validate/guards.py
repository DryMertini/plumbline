"""Input-side guards: the question, and - more importantly - the retrieved text.

Retrieved passages are untrusted input that ends up inside the prompt. A document that
says "ignore previous instructions" is an indirect prompt injection; the cheapest defence
is to never show it to the model. Flagged passages are dropped from the context and
reported in the trace.
"""

from __future__ import annotations

import re

from plumbline.llm.base import Passage

MAX_QUESTION_CHARS = 1000

_INJECTION = [
    (r"\b(ignore|disregard|forget)\b[^.\n]{0,40}\b(previous|prior|above|preceding|earlier)\b"
     r"[^.\n]{0,30}\b(instructions?|prompts?|context|messages?|rules)\b", "override-instructions"),
    (r"\byou are now\b[^.\n]{0,60}\b(assistant|ai|model|dan|mode)\b", "role-reassignment"),
    (r"\b(reveal|print|repeat|output|show)\b[^.\n]{0,30}\b(system prompt|hidden instructions|"
     r"your instructions)\b", "prompt-exfiltration"),
    (r"<\|?(system|im_start|im_end)\|?>", "control-tokens"),
    (r"\bnew instructions?\s*:", "override-instructions"),
    (r"\bdo not (tell|inform|mention to) the user\b", "concealment"),
]
_PATTERNS = [(re.compile(p, re.I), label) for p, label in _INJECTION]


def scan_text(text: str) -> list[dict]:
    hits = []
    for pattern, label in _PATTERNS:
        m = pattern.search(text)
        if m:
            lo, hi = max(0, m.start() - 30), min(len(text), m.end() + 30)
            hits.append({"rule": label, "excerpt": text[lo:hi].strip()})
    return hits


def check_question(question: str) -> list[str]:
    problems = []
    if not question or not question.strip():
        problems.append("the question is empty")
    elif len(question) > MAX_QUESTION_CHARS:
        problems.append(f"the question is longer than {MAX_QUESTION_CHARS} characters")
    return problems


def screen_passages(passages: list[Passage]) -> tuple[list[Passage], list[dict]]:
    """Split passages into (safe to show the model, flagged)."""
    clean, flagged = [], []
    for p in passages:
        hits = scan_text(p.text)
        if hits:
            flagged.append({"id": p.id, **hits[0]})
        else:
            clean.append(p)
    return clean, flagged
