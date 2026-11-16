from pathlib import Path

import pymupdf

from plumbline.ingest.chunk import chunk_document
from plumbline.ingest.parse import Element, ParsedDoc, _join_lines, parse_pdf
from plumbline.ingest.quality import check_document, filter_chunks, simhash
from plumbline.text import normalize, numbers, sentences, tokenize
from plumbline.types import Chunk


def test_sentences_survive_abbreviations_and_decimals():
    out = sentences("Lewis et al. (2020) show this in Fig. 3. It gets 28.4 BLEU, e.g. on WMT. Done?")
    assert out == ["Lewis et al. (2020) show this in Fig. 3.", "It gets 28.4 BLEU, e.g. on WMT.", "Done?"]


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("The retrievers are encoding passages") == ["retriever", "encod", "passag"]


def test_numbers_are_normalised():
    assert numbers("28.4 BLEU, 65% and 1,000 steps.") == {"28.4", "65", "1000"}


def test_normalize_expands_ligatures():
    assert normalize("eﬃcient  ﬁne-tuning now") == "efficient fine-tuning now"


def test_dehyphenation_uses_document_vocabulary():
    from collections import Counter

    vocab = Counter({"pre-trained": 5, "representation": 4})
    assert _join_lines(["a new representa-", "tion model"], vocab) == "a new representation model"
    assert _join_lines(["uses pre-", "trained weights"], vocab) == "uses pre-trained weights"


def _make_pdf(path: Path) -> None:
    doc = pymupdf.open()
    body = ("Dense retrieval maps questions and passages into one vector space so that "
            "relevant passages land close to the question. ") * 6
    for page_no in range(3):
        page = doc.new_page()
        page.insert_text((72, 40), "Proceedings of the Test Workshop", fontsize=8)
        y = 90
        if page_no == 0:
            page.insert_text((72, y), "A Study of Dense Retrieval", fontsize=18)
            y += 40
            page.insert_text((72, y), "Abstract", fontsize=12, fontname="hebo")
            y += 24
        page.insert_text((72, y), f"{page_no + 1} Section Number {page_no + 1}", fontsize=12,
                         fontname="hebo")
        page.insert_textbox(pymupdf.Rect(72, y + 14, 520, y + 260), body, fontsize=10)
        page.insert_text((300, 800), str(page_no + 1), fontsize=8)
    page = doc.new_page()
    page.insert_text((72, 90), "References", fontsize=12, fontname="hebo")
    page.insert_textbox(pymupdf.Rect(72, 110, 520, 300),
                        "Someone, A. 2020. A paper that must not be indexed because it is a "
                        "bibliography entry and not content of the study itself.", fontsize=10)
    doc.save(path)


def test_parse_chunk_and_gate_a_generated_pdf(tmp_path):
    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    parsed = parse_pdf(pdf)
    headings = [e.text for e in parsed.elements if e.kind == "heading"]
    assert "Abstract" in headings and "References" in headings
    assert parsed.detected_title == "A Study of Dense Retrieval"
    assert parsed.stats["boilerplate_blocks"] >= 3  # the running header was removed
    assert check_document(parsed, "A Study of Dense Retrieval").status == "ok"
    assert check_document(parsed, "Quantum Chromodynamics on a Lattice").status == "quarantined"

    chunks, stats = chunk_document(parsed, doc_id="x", short="x", title="A Study of Dense Retrieval")
    assert chunks and stats["skipped_sections"] == 1
    assert not any("bibliography entry" in c.text for c in chunks)
    assert all("Proceedings of the Test Workshop" not in c.text for c in chunks)
    assert all(c.section != "Front matter" for c in chunks)


def test_scanned_document_is_quarantined():
    empty = ParsedDoc(Path("x.pdf"), 10, "A Title", [Element("para", "tiny", 1)], {"chars": 40})
    verdict = check_document(empty, "A Title")
    assert verdict.status == "quarantined" and "scanned" in verdict.issues[0]


def _chunk(i: int, text: str, doc: str = "d") -> Chunk:
    return Chunk(f"{doc}:{i:04d}", doc, doc, "T", "S", "text", 1, 1, text, int(len(text.split()) * 1.3), i)


def test_quality_gates_drop_fragments_soup_and_duplicates():
    good = ("Dense retrieval maps questions and passages into a shared vector space where "
            "nearest neighbour search finds the passages most relevant to a question. The "
            "encoders are trained with contrastive learning so that a question lands close to "
            "the passages that answer it and far from the ones that do not. At query time the "
            "index returns the top passages in a few milliseconds even for millions of vectors, "
            "which is why the approach scales to open-domain question answering quickly.")
    other = ("Low-rank adaptation freezes the pretrained weights and trains two small matrices "
             "whose product approximates the update that full fine-tuning would have learned.")
    soup = "= ( 1 2 ) + [ 3 4 ] / { 5 6 } * 7 8 9 0 - = + ( ) [ ] 1 2 3 4 5 6 7 8 9 0 = = + + ( 4 ) [ 2 ] 7 7"
    chunks = [_chunk(0, good, "a"), _chunk(1, "Too short."), _chunk(2, soup),
              _chunk(3, good.upper(), "b"), _chunk(4, other, "c"),
              _chunk(9, good.replace("quickly", "fast"), "e")]
    kept, rejected = filter_chunks(chunks)
    assert [c.chunk_id for c in kept] == ["a:0000", "c:0004"]
    assert rejected == {"too_short": 1, "symbol_soup": 1, "exact_duplicate": 1, "near_duplicate": 1}


def test_simhash_is_stable_and_local():
    a = simhash("the quick brown fox jumps over the lazy dog near the river bank today")
    b = simhash("the quick brown fox jumps over the lazy dog near the river bank now")
    c = simhash("low rank adaptation freezes pretrained weights and trains small matrices")
    assert a == simhash("the quick brown fox jumps over the lazy dog near the river bank today")
    assert (a ^ b).bit_count() < (a ^ c).bit_count()
