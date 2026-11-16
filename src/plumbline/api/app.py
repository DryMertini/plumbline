"""HTTP API + dashboard.

POST /api/ask streams newline-delimited JSON: one line per pipeline event as it happens
(stage timings, attempts, backoffs, breaker trips), then a final `result` line. The
dashboard renders that stream live, so you can watch a request survive a failure.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from plumbline import __version__
from plumbline.config import get_settings
from plumbline.llm.chaos import PROFILES, ChaosConfig
from plumbline.pipeline import AskOptions, RagPipeline
from plumbline.retrieve.retriever import MODES
from plumbline.store import Store

STATIC = Path(__file__).parent / "static"
state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        state["pipeline"] = RagPipeline.load()
        state["error"] = None
    except RuntimeError as exc:
        state["pipeline"], state["error"] = None, str(exc)
    yield


app = FastAPI(title="plumbline", version=__version__, lifespan=lifespan)


def pipeline() -> RagPipeline:
    if state.get("pipeline") is None:
        raise HTTPException(503, state.get("error") or "index not loaded")
    return state["pipeline"]


class AskBody(BaseModel):
    question: str = Field(max_length=2000)
    mode: str = "hybrid+rerank"
    resilience: bool = True
    validation: bool = True


class ChaosBody(BaseModel):
    profile: str | None = None
    timeout: float | None = None
    rate_limit: float | None = None
    server_error: float | None = None
    malformed_json: float | None = None
    schema_violation: float | None = None
    bad_citation: float | None = None
    ungrounded_claim: float | None = None
    latency_ms: int | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": state.get("pipeline") is not None, "version": __version__,
            "error": state.get("error")}


@app.post("/api/ask")
async def ask(body: AskBody) -> StreamingResponse:
    pipe = pipeline()
    if body.mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    queue: asyncio.Queue = asyncio.Queue()
    options = AskOptions(mode=body.mode, resilience=body.resilience, validation=body.validation)

    async def run() -> None:
        try:
            result = await pipe.answer(body.question, options, on_event=queue.put_nowait)
            result.pop("events", None)
            queue.put_nowait({"type": "result", "result": result})
        except Exception as exc:  # the stream must always terminate
            queue.put_nowait({"type": "fatal", "detail": f"{type(exc).__name__}: {exc}"})
        queue.put_nowait(None)

    async def stream():
        task = asyncio.create_task(run())
        try:
            while (event := await queue.get()) is not None:
                yield json.dumps(event) + "\n"
        finally:
            task.cancel()

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/api/status")
def status() -> dict:
    return {**pipeline().status(), "profiles": list(PROFILES)}


@app.post("/api/chaos")
def set_chaos(body: ChaosBody) -> dict:
    pipe = pipeline()
    values = body.model_dump(exclude_none=True)
    profile = values.pop("profile", None)
    if profile is not None:
        if profile not in PROFILES:
            raise HTTPException(422, f"unknown profile {profile!r}")
        keep = {"latency_ms": pipe.chaos.latency_ms, "seed": pipe.chaos.seed}
        pipe.chaos.update({**ChaosConfig().to_dict(), **keep, **PROFILES[profile]})
    pipe.chaos.update(values)
    return pipe.status()


@app.post("/api/reset")
def reset() -> dict:
    pipe = pipeline()
    pipe.reset()
    return pipe.status()


@app.get("/api/corpus")
def corpus() -> dict:
    settings = get_settings()
    store = Store(settings.db_path)
    try:
        counts = store.chunk_counts()
        docs = [{**d, "chunks": counts.get(d["doc_id"], 0)} for d in store.documents()]
        return {"documents": docs, "report": store.get_meta("ingest_report", {})}
    finally:
        store.close()


@app.get("/api/chunk/{chunk_id}")
def chunk(chunk_id: str) -> dict:
    found = pipeline().retriever.by_id.get(chunk_id)
    if not found:
        raise HTTPException(404, "no such chunk")
    return found.to_dict()


@app.get("/api/eval")
def eval_report() -> dict:
    for path in (Path("reports/latest/report.json"), Path("reports/baseline.json")):
        if path.exists():
            return {"source": str(path), "report": json.loads(path.read_text())}
    raise HTTPException(404, "no eval report yet - run `plumbline eval`")


@app.get("/api/examples")
def examples() -> dict:
    return {"questions": [
        "What BLEU score did the big Transformer reach on WMT 2014 English-to-German?",
        "How long does BERT take versus SBERT to find the most similar pair among 10,000 sentences?",
        "Which operator does ColBERT use to score a query against a document?",
        "By what factor does LoRA cut trainable parameters compared with fine-tuning GPT-3?",
        "Which model does BEIR identify as a robust baseline for zero-shot retrieval?",
        "What is the capital of Australia?",
        "What BLEU score does ColBERT get on WMT 2014 English-to-German?",
    ]}
