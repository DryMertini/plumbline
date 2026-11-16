"""The ingestion pipeline: parse -> quality-check -> chunk -> filter -> store -> index.

Incremental by content hash: a document whose SHA-256 and pipeline version are unchanged
is not re-parsed (its parsed chunks are cached next to the raw file). The retrieval
indexes are rebuilt at the end of every run, which takes a couple of seconds at this scale
and guarantees they can never drift from the chunk table.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict

from plumbline.config import Settings
from plumbline.ingest.chunk import chunk_document
from plumbline.ingest.fetch import PaperRef, load_manifest, sha256_file
from plumbline.ingest.parse import parse_pdf
from plumbline.ingest.quality import check_document, filter_chunks
from plumbline.store import Store
from plumbline.types import Chunk

PIPELINE_VERSION = "2026.11-2"  # bump to force a re-parse after parser/chunker changes


def run_ingest(
    settings: Settings,
    *,
    force: bool = False,
    on_event: Callable[[str, PaperRef, str], None] | None = None,
) -> dict:
    from plumbline.retrieve.retriever import build_indexes

    emit = on_event or (lambda *_: None)
    store = Store(settings.db_path)
    cache_dir = settings.data_dir / "parsed"
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    all_chunks: list[Chunk] = []
    report = {"documents": [], "pipeline_version": PIPELINE_VERSION}

    for paper in load_manifest(settings.manifest):
        pdf = settings.raw_dir / f"{paper.id}.pdf"
        entry = {"doc_id": paper.id, "short": paper.short, "title": paper.title}
        if not pdf.exists():
            entry.update(status="missing", issues=["file not downloaded - run `plumbline fetch`"])
            report["documents"].append(entry)
            emit("missing", paper, "")
            continue
        digest = sha256_file(pdf)
        cache = cache_dir / f"{paper.id}.json"
        known = store.document(paper.id)
        fresh = (
            not force and known and cache.exists()
            and known["sha256"] == digest and known["pipeline_version"] == PIPELINE_VERSION
        )
        if fresh:
            cached = json.loads(cache.read_text())
            chunks = [Chunk(**c) for c in cached["chunks"]]
            entry.update(status=known["status"], issues=known["issues"], stats=known["stats"],
                         cached=True)
            emit("unchanged", paper, f"{len(chunks)} chunks")
        else:
            t0 = time.time()
            parsed = parse_pdf(pdf)
            verdict = check_document(parsed, paper.title)
            chunks, chunk_stats = ([], {})
            if verdict.status != "quarantined":
                chunks, chunk_stats = chunk_document(
                    parsed, doc_id=paper.id, short=paper.short, title=paper.title,
                    target_tokens=settings.chunk_target_tokens,
                    max_tokens=settings.chunk_max_tokens,
                    overlap_sentences=settings.chunk_overlap_sentences,
                )
            stats = {**parsed.stats, **chunk_stats, "raw_chunks": len(chunks),
                     "parse_seconds": round(time.time() - t0, 2)}
            store.upsert_document(
                doc_id=paper.id, short=paper.short, title=paper.title, year=paper.year,
                url=paper.abs_url, sha256=digest, pipeline_version=PIPELINE_VERSION,
                n_pages=parsed.n_pages, status=verdict.status, issues=verdict.issues, stats=stats,
            )
            cache.write_text(json.dumps({"chunks": [asdict(c) for c in chunks]}))
            entry.update(status=verdict.status, issues=verdict.issues, stats=stats, cached=False)
            emit(verdict.status, paper, f"{len(chunks)} chunks, {stats['parse_seconds']}s")
        all_chunks.extend(chunks)
        report["documents"].append(entry)

    kept, rejected = filter_chunks(all_chunks)
    store.replace_all_chunks(kept)
    index_stats = build_indexes(kept, settings)
    kinds: dict[str, int] = {}
    for c in kept:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1
    report.update(
        chunks_before_gates=len(all_chunks),
        chunks_indexed=len(kept),
        rejected=rejected,
        chunk_kinds=kinds,
        index=index_stats,
        seconds=round(time.time() - started, 2),
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    store.set_meta("ingest_report", report)
    store.close()
    return report
