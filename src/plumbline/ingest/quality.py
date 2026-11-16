"""Stage 3 - quality gates. Bad data is cheaper to stop here than to debug in answers.

Document gates decide whether a file is trustworthy at all (is it the paper the manifest
says it is? does it contain a real text layer?). Chunk gates remove passages that would
only pollute retrieval: fragments, symbol soup from equations, exact and near duplicates.
Everything rejected is counted and reported - nothing disappears silently.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

from plumbline.ingest.parse import ParsedDoc
from plumbline.text import tokenize, words
from plumbline.types import Chunk

MIN_CHARS_PER_PAGE = 400
MIN_TITLE_OVERLAP = 0.5
MIN_TEXT_TOKENS = 25
MIN_ALPHA_RATIO = 0.55
NEAR_DUP_HAMMING = 3


@dataclass
class DocVerdict:
    status: str = "ok"  # ok | warning | quarantined
    issues: list[str] = field(default_factory=list)


def check_document(doc: ParsedDoc, expected_title: str) -> DocVerdict:
    verdict = DocVerdict()
    chars_per_page = doc.stats.get("chars", 0) / max(doc.n_pages, 1)
    if chars_per_page < MIN_CHARS_PER_PAGE:
        verdict.status = "quarantined"
        verdict.issues.append(
            f"only {chars_per_page:.0f} extractable chars/page - likely scanned or image-only"
        )
        return verdict
    want, got = set(tokenize(expected_title)), set(tokenize(doc.detected_title))
    overlap = len(want & got) / max(len(want), 1)
    if overlap < MIN_TITLE_OVERLAP:
        verdict.status = "quarantined"
        verdict.issues.append(
            f"title on page 1 ({doc.detected_title[:60]!r}) does not match the manifest "
            f"(overlap {overlap:.0%}) - wrong file?"
        )
        return verdict
    if doc.stats.get("headings", 0) < 3:
        verdict.status = "warning"
        verdict.issues.append("fewer than 3 section headings detected - structure may be lost")
    return verdict


def _alpha_ratio(text: str) -> float:
    visible = [c for c in text if not c.isspace()]
    return sum(c.isalpha() for c in visible) / max(len(visible), 1)


def simhash(text: str, bits: int = 64) -> int:
    """64-bit SimHash over word 3-shingles: near-identical passages get near-identical hashes."""
    toks = words(text)
    shingles = [" ".join(toks[i : i + 3]) for i in range(max(len(toks) - 2, 1))]
    acc = np.zeros(bits, dtype=np.int32)
    for sh in shingles:
        h = int.from_bytes(hashlib.blake2b(sh.encode(), digest_size=8).digest(), "big")
        acc += np.array([1 if (h >> i) & 1 else -1 for i in range(bits)], dtype=np.int32)
    return int(sum(1 << i for i in range(bits) if acc[i] > 0))


def filter_chunks(chunks: list[Chunk]) -> tuple[list[Chunk], dict]:
    """Apply chunk gates across the whole corpus. Returns (kept, rejection counts)."""
    rejected = {"too_short": 0, "symbol_soup": 0, "exact_duplicate": 0, "near_duplicate": 0}
    seen_exact: set[str] = set()
    stage1: list[Chunk] = []
    for ch in chunks:
        if ch.kind == "text" and ch.n_tokens < MIN_TEXT_TOKENS:
            rejected["too_short"] += 1
            continue
        if ch.kind == "text" and _alpha_ratio(ch.text) < MIN_ALPHA_RATIO:
            rejected["symbol_soup"] += 1
            continue
        key = hashlib.sha1(re.sub(r"\W+", "", ch.text.lower()).encode()).hexdigest()
        if key in seen_exact:
            rejected["exact_duplicate"] += 1
            continue
        seen_exact.add(key)
        stage1.append(ch)

    kept: list[Chunk] = []
    hashes: list[int] = []
    for ch in stage1:
        h = simhash(ch.text)
        # overlap between neighbouring chunks is intentional, so only non-adjacent
        # passages count as duplicates
        dup = any(
            (h ^ other).bit_count() <= NEAR_DUP_HAMMING
            and not (prev.doc_id == ch.doc_id and abs(prev.ordinal - ch.ordinal) <= 1)
            for other, prev in zip(hashes, kept, strict=True)
        )
        if dup:
            rejected["near_duplicate"] += 1
            continue
        kept.append(ch)
        hashes.append(h)
    return kept, rejected
