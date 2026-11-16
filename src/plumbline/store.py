"""SQLite-backed document and chunk store. One file, zero services."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from plumbline.types import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY, short TEXT, title TEXT, year INTEGER, url TEXT,
    sha256 TEXT, pipeline_version TEXT, n_pages INTEGER, status TEXT,
    issues TEXT, stats TEXT, ingested_at TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY, doc_id TEXT, short TEXT, title TEXT, section TEXT, kind TEXT,
    page_start INTEGER, page_end INTEGER, text TEXT, n_tokens INTEGER, ordinal INTEGER
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

_COLS = "chunk_id, doc_id, short, title, section, kind, page_start, page_end, text, n_tokens, ordinal"


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # -- documents ---------------------------------------------------------------
    def document(self, doc_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return self._doc(row) if row else None

    def documents(self) -> list[dict]:
        rows = self.db.execute("SELECT * FROM documents ORDER BY year, short").fetchall()
        return [self._doc(r) for r in rows]

    @staticmethod
    def _doc(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["issues"] = json.loads(d["issues"] or "[]")
        d["stats"] = json.loads(d["stats"] or "{}")
        return d

    def upsert_document(self, *, doc_id: str, short: str, title: str, year: int, url: str,
                        sha256: str, pipeline_version: str, n_pages: int, status: str,
                        issues: list[str], stats: dict) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (doc_id, short, title, year, url, sha256, pipeline_version, n_pages, status,
             json.dumps(issues), json.dumps(stats), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        )
        self.db.commit()

    # -- chunks ------------------------------------------------------------------
    def replace_all_chunks(self, chunks: list[Chunk]) -> None:
        with self.db:
            self.db.execute("DELETE FROM chunks")
            self.db.executemany(
                f"INSERT INTO chunks ({_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [(c.chunk_id, c.doc_id, c.short, c.title, c.section, c.kind, c.page_start,
                  c.page_end, c.text, c.n_tokens, c.ordinal) for c in chunks],
            )

    def chunks(self) -> list[Chunk]:
        rows = self.db.execute(f"SELECT {_COLS} FROM chunks ORDER BY doc_id, ordinal").fetchall()
        return [Chunk(**dict(r)) for r in rows]

    def chunk_counts(self) -> dict[str, int]:
        rows = self.db.execute("SELECT doc_id, COUNT(*) n FROM chunks GROUP BY doc_id").fetchall()
        return {r["doc_id"]: r["n"] for r in rows}

    # -- meta --------------------------------------------------------------------
    def set_meta(self, key: str, value) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))
        self.db.commit()

    def get_meta(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default
