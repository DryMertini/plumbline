"""Stage 1 - turn a research PDF into ordered, typed elements.

Research PDFs are hostile input: two-column layouts, running headers, rotated arXiv
stamps, footnotes, figure labels, hyphenated line breaks and borderless ("booktabs")
tables that generic table finders miss. This module handles each of those explicitly:

* reading order   - per-page column detection (full-width bands, then left/right columns)
* boilerplate     - rotated text, page numbers and headers/footers repeated across pages
* headings        - numbered / well-known section titles detected from font + pattern
* tables          - anchored on "Table N:" captions, grown over adjacent non-prose blocks,
                    rows rebuilt from word geometry
* prose           - de-hyphenation and paragraph stitching across columns and pages
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import pymupdf

from plumbline.text import STOPWORDS, normalize, words

HEADING_NUM = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+([A-Z][^\n]{1,90})$")
HEADING_APP = re.compile(r"^([A-Z](?:\.\d{1,2}){0,2})\.?\s+([A-Z][^\n]{2,90})$")
HEADING_NAMED = re.compile(
    r"^(abstract|introduction|related work|background|conclusions?|discussion|references|"
    r"bibliography|acknowledge?ments?|appendix|appendices|limitations|ethics statement|"
    r"broader impact|experiments|results|method|methods|approach|supplementary material)$",
    re.I,
)
CAPTION = re.compile(r"^(Table|Figure|Fig\.)\s+(\d+)\s*[:.]", re.I)


@dataclass
class Element:
    kind: str  # heading | para | table | caption
    text: str
    page: int  # 1-based
    level: int = 0  # heading depth
    number: str = ""  # heading number, e.g. "3.2"


@dataclass
class ParsedDoc:
    path: Path
    n_pages: int
    detected_title: str
    elements: list[Element]
    stats: dict = field(default_factory=dict)


@dataclass
class _Block:
    text: str
    lines: list[str]
    bbox: tuple[float, float, float, float]
    size: float
    bold: bool
    page: int
    kind: str = "para"

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]


def _is_bold(span: dict) -> bool:
    font = span.get("font", "").lower()
    return bool(span.get("flags", 0) & 16) or any(
        k in font for k in ("bold", "medi", "-bd", "black", "heavy", "semibold", "cmbx", "tb")
    )


_ARXIV_STAMP = re.compile(r"^arXiv:\d{4}\.\d{4,5}")
_HYPH_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)+|[A-Za-z]{3,}")


def _join_lines(lines: list[str], vocab: Counter | None = None) -> str:
    """Join visual lines into one paragraph, repairing hyphenated line breaks.

    'representa-' + 'tion' must lose its hyphen, 'pre-' + 'trained' must keep it. The
    document's own vocabulary decides: whichever spelling the paper uses elsewhere wins.
    """
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
        elif out.endswith("-") and line[:1].islower():
            head = re.split(r"[^A-Za-z-]", out[:-1])[-1].lower()
            tail = re.split(r"[^A-Za-z]", line)[0].lower()
            keep = False
            if vocab is not None and head and tail:
                keep = vocab[f"{head}-{tail}"] > vocab[f"{head}{tail}"]
            out = (out if keep else out[:-1]) + line
        else:
            out += " " + line
    return out


def _styles(page: pymupdf.Page) -> list[tuple[tuple, float, bool, bool]]:
    """(bbox, char-weighted font size, starts bold, fully rotated) per styled block."""
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0 or not b.get("lines"):
            continue
        sizes, first_bold, rotated = [], None, True
        for line in b["lines"]:
            dx, dy = line.get("dir", (1.0, 0.0))
            if abs(dx - 1.0) <= 0.01 and abs(dy) <= 0.01:
                rotated = False
            for sp in line["spans"]:
                if sp["text"].strip():
                    sizes.append((sp["size"], len(sp["text"])))
                    if first_bold is None:
                        first_bold = _is_bold(sp)
        if not sizes:
            continue
        total = sum(n for _, n in sizes)
        out.append((tuple(b["bbox"]), sum(z * n for z, n in sizes) / total, bool(first_bold), rotated))
    return out


def _overlap(a: tuple, b: tuple) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _extract_blocks(page: pymupdf.Page, page_no: int) -> list[_Block]:
    """Text blocks of a page.

    Text comes from MuPDF's plain "blocks" extraction, which is complete. Font size and
    weight come from the "dict" extraction, matched spatially - on some PDFs (subset CID
    fonts) "dict" silently drops most lines, so it is never trusted for the text itself.
    """
    styles = _styles(page)
    blocks: list[_Block] = []
    for x0, y0, x1, y1, text, _no, kind in page.get_text("blocks"):
        if kind != 0 or not text.strip():
            continue
        bbox = (x0, y0, x1, y1)
        match = max(styles, key=lambda st: _overlap(bbox, st[0]), default=None)
        size, bold, rotated = -1.0, False, False
        if match and _overlap(bbox, match[0]) > 0:
            _, size, bold, rotated = match
        lines = [normalize(ln) for ln in text.split("\n") if ln.strip()]
        if rotated or not lines or _ARXIV_STAMP.match(lines[0]):
            continue
        blocks.append(_Block(text="", lines=lines, bbox=bbox, size=size, bold=bold, page=page_no))
    return blocks


def _finalise_text(doc_blocks: list[list[_Block]], body: float) -> None:
    vocab: Counter = Counter()
    for blocks in doc_blocks:
        for b in blocks:
            for line in b.lines:
                vocab.update(w.lower() for w in _HYPH_WORD.findall(line))
    for blocks in doc_blocks:
        for b in blocks:
            b.text = normalize(_join_lines(b.lines, vocab))
            if b.size < 0:
                b.size = body


def _body_font_size(doc_blocks: list[list[_Block]]) -> float:
    counts: Counter = Counter()
    for blocks in doc_blocks:
        for b in blocks:
            if b.size > 0:
                counts[round(b.size * 2) / 2] += sum(len(x) for x in b.lines)
    return counts.most_common(1)[0][0] if counts else 10.0


def _boilerplate_keys(doc_blocks: list[list[_Block]], heights: list[float]) -> set[str]:
    """Normalised texts that recur in page margins (running headers / footers)."""
    seen: Counter = Counter()
    for blocks, h in zip(doc_blocks, heights, strict=True):
        keys = set()
        for b in blocks:
            if b.y1 < h * 0.09 or b.y0 > h * 0.91:
                keys.add(re.sub(r"\d+", "#", " ".join(b.lines).lower()).strip())
        seen.update(keys)
    need = max(3, int(len(doc_blocks) * 0.4))
    return {k for k, n in seen.items() if n >= need and len(k) < 160}


def _reading_order(blocks: list[_Block], width: float) -> list[_Block]:
    """Order blocks the way a person reads a (possibly two-column) page."""
    mid = width / 2
    full, left, right = [], [], []
    for b in blocks:
        # anything straddling the gutter (titles, centred author lines, wide tables) is a
        # band that interrupts both columns, however narrow it is
        if b.x0 < mid - 12 and b.x1 > mid + 12:
            full.append(b)
        elif (b.x0 + b.x1) / 2 < mid:
            left.append(b)
        else:
            right.append(b)
    if not right or len(left) + len(right) < 3 or not _has_gutter(left, right, mid):
        return sorted(blocks, key=lambda b: (round(b.y0 / 4), b.x0))
    ordered: list[_Block] = []
    bands = sorted(full, key=lambda b: b.y0)
    top = float("-inf")
    for band in [*bands, None]:
        bottom = band.y0 if band else float("inf")
        for col in (left, right):
            ordered.extend(
                sorted((b for b in col if top <= b.y0 < bottom), key=lambda b: (b.y0, b.x0))
            )
        if band:
            ordered.append(band)
            top = band.y0
    return ordered


def _has_gutter(left: list[_Block], right: list[_Block], mid: float) -> bool:
    wide_left = [b for b in left if b.x1 - b.x0 > 120]
    wide_right = [b for b in right if b.x1 - b.x0 > 120]
    if not wide_left or not wide_right:
        return False
    return median(b.x1 for b in wide_left) <= mid + 6 and median(b.x0 for b in wide_right) >= mid - 6


def _prose_score(text: str) -> tuple[float, float]:
    """(stopword ratio, numeric-token ratio) - prose is high/low, table cells low/high."""
    toks = text.split()
    if not toks:
        return 0.0, 0.0
    ws = words(text)
    stop = sum(1 for w in ws if w in STOPWORDS) / max(len(ws), 1)
    numeric = sum(1 for t in toks if any(c.isdigit() for c in t)) / len(toks)
    return stop, numeric


def _is_prose(b: _Block) -> bool:
    stop, numeric = _prose_score(b.text)
    return len(b.text.split()) >= 10 and stop >= 0.17 and numeric <= 0.3


def _table_region(caption: _Block, blocks: list[_Block], used: set[int]) -> list[_Block]:
    """Grow a table body from its caption over adjacent, horizontally-overlapping non-prose."""

    def overlaps(b: _Block) -> bool:
        shared = min(b.x1, caption.x1 + 30) - max(b.x0, caption.x0 - 30)
        return shared > 0.6 * (b.x1 - b.x0)

    best: list[_Block] = []
    for direction in (1, -1):
        edge_lo, edge_hi = caption.y0, caption.y1
        grown: list[_Block] = []
        candidates = sorted(
            (b for b in blocks if id(b) not in used and b is not caption and overlaps(b)),
            key=lambda b: b.y0 if direction == 1 else -b.y1,
        )
        for b in candidates:
            gap = b.y0 - edge_hi if direction == 1 else edge_lo - b.y1
            if gap < -4:
                continue
            if gap > 26 or _is_prose(b) or CAPTION.match(b.text):
                break
            grown.append(b)
            edge_lo, edge_hi = min(edge_lo, b.y0), max(edge_hi, b.y1)
        if len(grown) > len(best):
            best = grown
    return best


def _rebuild_rows(page: pymupdf.Page, region: list[_Block]) -> list[str]:
    """Rebuild table rows from word geometry: cluster by baseline, split cells on x-gaps."""
    rect = pymupdf.Rect(
        min(b.x0 for b in region) - 2,
        min(b.y0 for b in region) - 2,
        max(b.x1 for b in region) + 2,
        max(b.y1 for b in region) + 2,
    )
    ws = sorted(page.get_text("words", clip=rect), key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows: list[list[tuple]] = []
    for w in ws:
        yc = (w[1] + w[3]) / 2
        if rows and abs(yc - rows[-1][0][-1]) <= 3.2:
            rows[-1].append((*w[:5], rows[-1][0][-1]))
        else:
            rows.append([(*w[:5], yc)])
    out = []
    for row in rows:
        row.sort(key=lambda w: w[0])
        cells, cur = [], [row[0][4]]
        for prev, nxt in zip(row, row[1:], strict=False):
            if nxt[0] - prev[2] > 7.5:
                cells.append(" ".join(cur))
                cur = []
            cur.append(nxt[4])
        cells.append(" ".join(cur))
        line = normalize(" | ".join(cells))
        if line:
            out.append(line)
    return out


def _heading(b: _Block, body: float) -> tuple[int, str, str] | None:
    """Return (level, number, title) when the block is a section heading."""
    text = b.text.strip()
    if len(b.lines) > 2 or len(text) > 100 or len(text.split()) > 14:
        return None
    letters = [c for c in text if c.isalpha()]
    all_caps = len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.85
    emphasised = b.bold or b.size >= body + 0.8 or all_caps
    if not emphasised or text.endswith((".", ",", ";", ":")):
        return None
    if all_caps:
        text = " ".join(w if w.isdigit() or "." in w else w.capitalize() for w in text.split())
    if HEADING_NAMED.match(text):
        return 1, "", text
    m = HEADING_NUM.match(text)
    if m and int(m.group(1).split(".")[0]) <= 20:
        return m.group(1).count(".") + 1, m.group(1), m.group(2).strip()
    m = HEADING_APP.match(text)
    if m and (b.size >= body + 0.8 or "." in m.group(1)):
        return m.group(1).count(".") + 1, m.group(1), m.group(2).strip()
    return None


def parse_pdf(path: Path) -> ParsedDoc:
    doc = pymupdf.open(path)
    try:
        pages = list(doc)
        doc_blocks = [_extract_blocks(p, i + 1) for i, p in enumerate(pages)]
        heights = [p.rect.height for p in pages]
        body = _body_font_size(doc_blocks)
        _finalise_text(doc_blocks, body)
        boiler = _boilerplate_keys(doc_blocks, heights)
        stats = Counter()
        elements: list[Element] = []

        title_block = max(
            (b for b in (doc_blocks[0] if doc_blocks else []) if b.y0 < heights[0] * 0.5),
            key=lambda b: (round(b.size), -b.y0),
            default=None,
        )
        detected_title = title_block.text if title_block else ""

        for page, blocks, height in zip(pages, doc_blocks, heights, strict=True):
            width = page.rect.width
            kept: list[_Block] = []
            for b in blocks:
                key = re.sub(r"\d+", "#", b.text.lower()).strip()
                in_margin = b.y1 < height * 0.09 or b.y0 > height * 0.91
                if in_margin and (key in boiler or re.fullmatch(r"[\d\s#ivx.\-]{1,8}", key)):
                    stats["boilerplate_blocks"] += 1
                    continue
                kept.append(b)

            used: set[int] = set()
            tables: dict[int, tuple[str, list[str]]] = {}
            for b in kept:
                m = CAPTION.match(b.text)
                if not m:
                    continue
                b.kind = "caption"
                if m.group(1).lower() != "table":
                    continue
                region = _table_region(b, kept, used)
                if not region:
                    continue
                rows = _rebuild_rows(page, region)
                if len(rows) >= 2:
                    used.update(id(r) for r in region)
                    tables[id(b)] = (b.text, rows)

            for b in _reading_order([b for b in kept if id(b) not in used], width):
                if id(b) in tables:
                    caption, rows = tables[id(b)]
                    elements.append(Element("table", caption + "\n" + "\n".join(rows), b.page))
                    stats["tables"] += 1
                    continue
                if b.kind == "caption":
                    elements.append(Element("caption", b.text, b.page))
                    stats["captions"] += 1
                    continue
                head = _heading(b, body)
                if head:
                    level, number, title = head
                    elements.append(Element("heading", title, b.page, level, number))
                    stats["headings"] += 1
                    continue
                if b.size < body - 1.3:
                    # footnotes, figure labels, tiny table debris
                    stats["small_print_blocks"] += 1
                    continue
                stop, numeric = _prose_score(b.text)
                if len(b.text.split()) < 4 or (stop < 0.08 and len(b.text.split()) < 40):
                    stats["fragment_blocks"] += 1
                    continue
                elements.append(Element("para", b.text, b.page))

        first_head = next((i for i, e in enumerate(elements) if e.kind == "heading"), None)
        if first_head is not None and elements[first_head].page == 1:
            stats["frontmatter_blocks"] = first_head
            elements = elements[first_head:]
        elements = _stitch(elements)
        stats["pages"] = len(pages)
        stats["paragraphs"] = sum(1 for e in elements if e.kind == "para")
        stats["chars"] = sum(len(e.text) for e in elements)
        return ParsedDoc(path, len(pages), detected_title, elements, dict(stats))
    finally:
        doc.close()


def _stitch(elements: list[Element]) -> list[Element]:
    """Re-join paragraphs that were split by a column or page break.

    A paragraph that does not end a sentence, followed (possibly after floats that
    interrupted it) by one starting in lower case, is one paragraph.
    """
    out: list[Element] = []
    open_idx: int | None = None  # index in `out` of a paragraph awaiting its continuation
    for el in elements:
        if el.kind == "para":
            if open_idx is not None and el.text[:1].islower():
                prev = out[open_idx]
                joiner = "" if prev.text.endswith("-") else " "
                base = prev.text[:-1] if prev.text.endswith("-") else prev.text
                prev.text = base + joiner + el.text
                if prev.text.rstrip().endswith((".", "!", "?", ":")):
                    open_idx = None
                continue
            out.append(el)
            open_idx = None if el.text.rstrip().endswith((".", "!", "?", ":", ")")) else len(out) - 1
        else:
            out.append(el)
            if el.kind == "heading":
                open_idx = None
    return out
