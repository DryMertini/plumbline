"""Small, dependency-free text utilities shared by ingestion, retrieval and validation."""

from __future__ import annotations

import re
import unicodedata

_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*%?")

STOPWORDS = frozenset(
    """a an and are as at be been being but by can could did do does for from had has have how
    i if in into is it its may might more most no not of on or our should so such than that the
    their them then there these they this those to was we were what when where which while who
    whom why will with would you your also both each other some any all via per use used using
    paper authors author work propose proposed show shows shown many much get got according""".split()
)

_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "’": "'", "‘": "'",
              "“": '"', "”": '"', "–": "-", "—": "-", "−": "-", "­": "", " ": " "}


def normalize(text: str) -> str:
    """Unicode-normalise, expand ligatures, drop control chars, collapse whitespace."""
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch == "\n" or unicodedata.category(ch)[0] != "C")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def stem(word: str) -> str:
    """A deliberately tiny suffix stripper (plural / -ing / -ed / -ly).

    Not Porter: it only needs to make 'retrievers' ~ 'retriever' and 'encoding' ~ 'encode'
    collide often enough for lexical matching, while staying predictable and fast.
    """
    w = word
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 4 and w.endswith("sses"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    if len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    elif len(w) > 5 and w.endswith("ly"):
        w = w[:-2]
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


def words(text: str) -> list[str]:
    """Lower-cased word tokens, no filtering."""
    return _WORD.findall(text.lower())


def tokenize(text: str) -> list[str]:
    """Index/query tokens: lower-cased, stopwords removed, lightly stemmed."""
    return [stem(w) for w in words(text) if w not in STOPWORDS and len(w) > 1]


def count_tokens(text: str) -> int:
    """Cheap token estimate (whitespace words * 1.3), good enough for chunk budgeting."""
    return int(len(text.split()) * 1.3) + 1


def numbers(text: str) -> set[str]:
    """Numeric literals in a text, normalised ('28.4', '65%' -> '65', '1,000' -> '1000')."""
    out = set()
    for m in _NUM.findall(text):
        n = m.rstrip("%").replace(",", "")
        if n.endswith("."):
            n = n[:-1]
        out.add(n)
    return out


_ABBREV = re.compile(
    r"(?:\bet al|\be\.g|\bi\.e|\bFigs?|\bEqs?|\bSec|\bTab|\bvs|\bcf|\bresp|\bapprox|\bNo|\bDr|\bMr|\bMs|\bProf)\.$"
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9(\[\"'])")


def sentences(text: str) -> list[str]:
    """Sentence split that survives 'et al.', 'Fig. 3', 'e.g.' and decimals."""
    parts: list[str] = []
    for para in re.split(r"\n{2,}", text):
        para = " ".join(para.split())
        if not para:
            continue
        buf = ""
        for piece in _SENT_SPLIT.split(para):
            buf = f"{buf} {piece}".strip() if buf else piece.strip()
            if not _ABBREV.search(buf):
                parts.append(buf)
                buf = ""
        if buf:
            parts.append(buf)
    return [p for p in parts if p]
