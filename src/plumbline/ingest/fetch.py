"""Stage 0 - acquire the corpus.

Downloads the PDFs listed in the manifest from arxiv.org. The step is polite (one request
at a time, a pause between requests), resumable (already-valid files are skipped) and
verifiable (every file is magic-byte checked and its SHA-256 recorded in a fetch log).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

USER_AGENT = "plumbline-corpus-fetch/0.1 (research corpus downloader)"
PAUSE_S = 3.0


@dataclass(frozen=True)
class PaperRef:
    id: str
    short: str
    title: str
    year: int

    @property
    def url(self) -> str:
        return f"https://arxiv.org/pdf/{self.id}"

    @property
    def abs_url(self) -> str:
        return f"https://arxiv.org/abs/{self.id}"


def load_manifest(path: Path) -> list[PaperRef]:
    data = yaml.safe_load(path.read_text())
    return [PaperRef(str(p["id"]), p["short"], p["title"], int(p["year"])) for p in data["papers"]]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _is_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def fetch_corpus(
    papers: list[PaperRef],
    raw_dir: Path,
    *,
    client: httpx.Client | None = None,
    pause_s: float = PAUSE_S,
    on_event: Callable[[str, PaperRef, str], None] | None = None,
) -> dict[str, dict]:
    """Download every paper not already present. Returns the fetch log (id -> record)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = raw_dir / "fetch_log.json"
    log: dict[str, dict] = json.loads(log_path.read_text()) if log_path.exists() else {}
    emit = on_event or (lambda *_: None)
    own_client = client is None
    client = client or httpx.Client(
        headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=60
    )
    try:
        downloaded_any = False
        for paper in papers:
            dest = raw_dir / f"{paper.id}.pdf"
            if _is_pdf(dest):
                log.setdefault(paper.id, {})["sha256"] = sha256_file(dest)
                emit("cached", paper, "")
                continue
            if downloaded_any:
                time.sleep(pause_s)
            downloaded_any = True
            try:
                resp = client.get(paper.url)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                log[paper.id] = {"error": str(exc)}
                emit("failed", paper, str(exc))
                continue
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(resp.content)
            if not _is_pdf(tmp):
                tmp.unlink(missing_ok=True)
                log[paper.id] = {"error": "response was not a PDF"}
                emit("failed", paper, "response was not a PDF")
                continue
            tmp.replace(dest)
            log[paper.id] = {
                "sha256": sha256_file(dest),
                "bytes": dest.stat().st_size,
                "url": paper.url,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            emit("downloaded", paper, f"{dest.stat().st_size / 1e6:.1f} MB")
    finally:
        if own_client:
            client.close()
    log_path.write_text(json.dumps(log, indent=2, sort_keys=True))
    return log
