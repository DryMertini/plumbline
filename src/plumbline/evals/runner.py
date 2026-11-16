"""Three evaluations, one report.

1. retrieval  - does the right passage come back? An ablation over bm25 / dense / hybrid /
                hybrid+rerank, so every stage has to justify its place in the pipeline.
2. answers    - is the final answer correct, cited to the right paper, and does the system
                refuse the questions it cannot answer?
3. resilience - the same questions under injected faults, with the resilience and
                validation layers switched on and off. This is the number that shows what
                those layers are worth.

Everything is deterministic (seeded faults, offline models), so the report is reproducible
and can gate CI.
"""

from __future__ import annotations

import time

from plumbline.config import Settings, get_settings
from plumbline.evals.dataset import GoldenItem, judge, load_golden
from plumbline.evals.metrics import mean, ndcg_at_k, percentile, recall_at_k, reciprocal_rank
from plumbline.llm.chaos import FABRICATED_CLAIM, PROFILES, ChaosConfig
from plumbline.pipeline import AskOptions, RagPipeline
from plumbline.retrieve.retriever import MODES, Retriever

CHAOS_PROFILES = ("flaky-network", "bad-outputs", "everything", "outage")


def eval_retrieval(retriever: Retriever, items: list[GoldenItem]) -> dict:
    out = {}
    answerable = [i for i in items if i.answerable]
    names = ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10")
    for mode in MODES:
        rows = []
        for item in answerable:
            ranked = [h.chunk.chunk_id for h in retriever.search(item.question, mode=mode, k=10).hits]
            rows.append({
                "id": item.id, "split": item.split,
                "recall@1": recall_at_k(ranked, item.relevant, 1),
                "recall@3": recall_at_k(ranked, item.relevant, 3),
                "recall@5": recall_at_k(ranked, item.relevant, 5),
                "recall@10": recall_at_k(ranked, item.relevant, 10),
                "mrr@10": reciprocal_rank(ranked, item.relevant, 10),
                "ndcg@10": ndcg_at_k(ranked, item.relevant, 10),
            })

        def agg(subset: list[dict]) -> dict:
            return {m: round(mean(r[m] for r in subset), 4) for m in names}

        out[mode] = {
            "metrics": agg(rows),
            "dev": agg([r for r in rows if r["split"] == "dev"]),
            "test": agg([r for r in rows if r["split"] == "test"]),
            "misses@5": [r["id"] for r in rows if not r["recall@5"]],
        }
    return out


def _score(item: GoldenItem, result: dict) -> dict:
    answered = result["status"] in ("answered", "degraded")
    cites = [c["short"] for claim in result["claims"] for c in claim["citations"]]
    uncited = any(not claim["citations"] for claim in result["claims"])
    poisoned = answered and (FABRICATED_CLAIM["text"] in result["answer"] or uncited)
    return {
        "id": item.id, "type": item.type, "split": item.split, "status": result["status"],
        "question": item.question,
        "answered": answered,
        "correct": answered and item.answerable and item.is_correct(result["answer"]),
        "abstained": result["status"] == "abstained",
        "failed": result["status"] == "failed",
        "cited_gold": bool(cites) and all(c in item.docs for c in cites),
        "citation_precision": mean(c in item.docs for c in cites) if cites else None,
        "poisoned": poisoned,
        "served_by": result["served_by"], "retries": result["retries"],
        "fallback": result["fallback_used"], "attempts": len(result["attempts"]),
        "ms": result["trace"]["total_ms"], "answer": result["answer"][:300],
    }


def _answer_metrics(rows: list[dict]) -> dict:
    yes = [r for r in rows if r["type"] != "unanswerable"]
    no = [r for r in rows if r["type"] == "unanswerable"]
    precisions = [r["citation_precision"] for r in yes if r["citation_precision"] is not None]
    return {
        "n": len(rows),
        "answer_accuracy": round(mean(r["correct"] for r in yes), 4),
        "wrong_answer_rate": round(mean(r["answered"] and not r["correct"] for r in yes), 4),
        "false_abstain_rate": round(mean(r["abstained"] for r in yes), 4),
        "citation_precision": round(mean(precisions), 4),
        "refusal_accuracy": round(mean(r["abstained"] for r in no), 4) if no else None,
        "failure_rate": round(mean(r["failed"] for r in rows), 4),
        "p50_ms": round(percentile([r["ms"] for r in rows], 0.5), 1),
        "p95_ms": round(percentile([r["ms"] for r in rows], 0.95), 1),
    }


async def eval_answers(pipeline: RagPipeline, items: list[GoldenItem],
                       options: AskOptions | None = None) -> dict:
    rows = [_score(item, await pipeline.answer(item.question, options)) for item in items]
    return {
        "metrics": _answer_metrics(rows),
        "dev": _answer_metrics([r for r in rows if r["split"] == "dev"]),
        "test": _answer_metrics([r for r in rows if r["split"] == "test"]),
        "rows": rows,
    }


async def _no_sleep(_seconds: float) -> None:
    return None


async def eval_resilience(retriever: Retriever, items: list[GoldenItem], settings: Settings,
                          rounds: int = 3) -> dict:
    """Run the answerable questions under each fault profile, protected and unprotected.

    Backoff sleeps are skipped and the attempt timeout is shortened so the suite finishes in
    seconds; the logic exercised (which attempt fails, what gets retried, what gets blocked)
    is identical to production.
    """
    answerable = [i for i in items if i.answerable]
    variants = {
        "naive": AskOptions(resilience=False, validation=False),
        "validation only": AskOptions(resilience=False, validation=True),
        "resilient": AskOptions(resilience=True, validation=True),
    }
    out = {}
    for profile in CHAOS_PROFILES:
        out[profile] = {}
        for name, options in variants.items():
            rows = []
            for round_no in range(rounds):
                ticks = [0.0]  # virtual time: each request is one second of traffic
                pipeline = RagPipeline(retriever, settings, seed=round_no,
                                       attempt_timeout_s=0.05, sleep=_no_sleep,
                                       clock=lambda t=ticks: t[0])
                pipeline.chaos.update({**ChaosConfig().to_dict(), **PROFILES[profile],
                                       "seed": 100 + round_no})
                pipeline.primary.reseed()
                for item in answerable:
                    ticks[0] += 1.0
                    rows.append(_score(item, await pipeline.answer(item.question, options)))
            out[profile][name] = {
                "requests": len(rows),
                "availability": round(mean(r["answered"] or r["abstained"] for r in rows), 4),
                "correct": round(mean(r["correct"] for r in rows), 4),
                "poisoned_served": round(mean(r["poisoned"] for r in rows), 4),
                "fallback_rate": round(mean(bool(r["fallback"]) for r in rows), 4),
                "mean_attempts": round(mean(r["attempts"] for r in rows), 2),
            }
    return out


async def run_all(settings: Settings | None = None, *, resilience_rounds: int = 3) -> dict:
    settings = settings or get_settings()
    retriever = Retriever.load(settings)
    items = load_golden()
    problems = judge(items, retriever.chunks)
    if problems:
        raise RuntimeError("golden set is out of sync with the index:\n" + "\n".join(problems))
    started = time.time()
    pipeline = RagPipeline(retriever, settings, seed=0)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus": {"chunks": len(retriever.chunks),
                   "papers": len({c.doc_id for c in retriever.chunks})},
        "dataset": {"items": len(items), "answerable": sum(i.answerable for i in items),
                    "unanswerable": sum(not i.answerable for i in items)},
        "providers": [p.name for p in pipeline.chain],
        "model": pipeline.chain[0].model,
        "retrieval": eval_retrieval(retriever, items),
        "answers": await eval_answers(pipeline, items),
        "resilience": await eval_resilience(retriever, items, settings, resilience_rounds),
    }
    report["seconds"] = round(time.time() - started, 1)
    return report
