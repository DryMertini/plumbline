"""Stage 2 - structure-aware chunking.

Chunks respect the document's own structure instead of a blind character window:
sections are never mixed, sentences are never cut, tables stay whole (or are split on
row boundaries with the caption repeated), and bibliography / acknowledgement sections
are excluded because they attract lexical matches without ever answering a question.
"""

from __future__ import annotations

import re

from plumbline.ingest.parse import Element, ParsedDoc
from plumbline.text import count_tokens, sentences
from plumbline.types import Chunk

SKIP_SECTION = re.compile(r"^(references|bibliography|acknowledge?ments?)$", re.I)


def _section_label(stack: list[tuple[int, str, str]]) -> str:
    return " > ".join(f"{num} {title}".strip() for _, num, title in stack) or "Front matter"


def chunk_document(
    doc: ParsedDoc,
    *,
    doc_id: str,
    short: str,
    title: str,
    target_tokens: int = 200,
    max_tokens: int = 300,
    overlap_sentences: int = 1,
) -> tuple[list[Chunk], dict]:
    chunks: list[Chunk] = []
    stats = {"skipped_sections": 0, "skipped_elements": 0}
    stack: list[tuple[int, str, str]] = []
    skipping = False
    buf: list[tuple[str, int]] = []  # (sentence, page)

    def emit(kind: str, text: str, p0: int, p1: int) -> None:
        ordinal = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{short}:{ordinal:04d}",
                doc_id=doc_id,
                short=short,
                title=title,
                section=_section_label(stack),
                kind=kind,
                page_start=p0,
                page_end=p1,
                text=text,
                n_tokens=count_tokens(text),
                ordinal=ordinal,
            )
        )

    def flush(final: bool = True) -> None:
        nonlocal buf
        if not buf:
            return
        emit("text", " ".join(s for s, _ in buf), buf[0][1], buf[-1][1])
        buf = [] if final else buf[-overlap_sentences:] if overlap_sentences else []

    def add_sentences(el: Element) -> None:
        nonlocal buf
        for sent in sentences(el.text):
            size = count_tokens(" ".join(s for s, _ in buf))
            if buf and size + count_tokens(sent) > max_tokens:
                flush(final=False)
            buf.append((sent, el.page))
            if count_tokens(" ".join(s for s, _ in buf)) >= target_tokens:
                flush(final=False)
                # the carried-over overlap alone is not a chunk; mark it as such
                buf = [(s, p) for s, p in buf]

    def drop_pure_overlap() -> None:
        """At a section boundary, a buffer holding only carried-over overlap is discarded."""
        nonlocal buf
        if chunks and buf and " ".join(s for s, _ in buf) in chunks[-1].text:
            buf = []

    for el in doc.elements:
        if el.kind == "heading":
            drop_pure_overlap()
            flush()
            while stack and stack[-1][0] >= el.level:
                stack.pop()
            stack.append((el.level, el.number, el.text))
            skipping = bool(SKIP_SECTION.match(stack[0][2].strip()))
            if skipping and el.level == 1:
                stats["skipped_sections"] += 1
            continue
        if skipping:
            stats["skipped_elements"] += 1
            continue
        if el.kind == "table":
            _emit_table(el, emit, max_tokens)
        elif el.kind == "caption":
            if len(el.text.split()) >= 12:
                emit("caption", el.text, el.page, el.page)
        else:
            add_sentences(el)
    drop_pure_overlap()
    flush()
    return chunks, stats


def _emit_table(el: Element, emit, max_tokens: int) -> None:
    caption, *rows = el.text.split("\n")
    budget = max(max_tokens * 2, 200)  # tables are dense; allow them to run longer
    part: list[str] = []
    for row in rows:
        if part and count_tokens(caption + " " + " ".join(part + [row])) > budget:
            emit("table", caption + "\n" + "\n".join(part), el.page, el.page)
            part = []
        part.append(row)
    if part:
        emit("table", caption + "\n" + "\n".join(part), el.page, el.page)
