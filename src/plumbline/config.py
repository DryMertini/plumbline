"""Runtime configuration. Everything is overridable with PLUMBLINE_* env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"PLUMBLINE_{name}", default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data")))
    manifest: Path = field(default_factory=lambda: Path(_env("MANIFEST", "corpus/manifest.yaml")))

    # chunking
    chunk_target_tokens: int = 200
    chunk_max_tokens: int = 300
    chunk_overlap_sentences: int = 1

    # retrieval
    lsa_dims: int = 256
    candidates_per_retriever: int = 40
    rrf_k: int = 60
    top_k: int = 6

    # generation / resilience
    provider_chain: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            p.strip() for p in _env("PROVIDERS", "extractive").split(",") if p.strip()
        )
    )
    request_deadline_s: float = float(_env("DEADLINE_S", "20"))
    attempt_timeout_s: float = float(_env("ATTEMPT_TIMEOUT_S", "8"))
    max_attempts: int = int(_env("MAX_ATTEMPTS", "3"))
    max_repairs: int = int(_env("MAX_REPAIRS", "1"))

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "plumbline.sqlite"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "dense_index.npz"


def get_settings() -> Settings:
    return Settings()
