"""The request path: guard -> retrieve -> screen -> gate -> generate (resiliently) -> validate.

`RagPipeline.answer` never raises for an operational failure. Every outcome - answered,
degraded, abstained, failed - is a structured result with a full trace, because in
production "what happened to this request?" is the question you will be asked most.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import ValidationError

from plumbline.config import Settings, get_settings
from plumbline.llm.base import LLMProvider, Passage
from plumbline.llm.chaos import ChaosConfig, ChaosLLM
from plumbline.llm.extractive import ExtractiveLLM
from plumbline.llm.prompts import build_request
from plumbline.resilience.breaker import CircuitBreaker
from plumbline.resilience.errors import LLMError
from plumbline.resilience.fallback import Attempt, GenerationOutcome, ResilientGenerator
from plumbline.resilience.retry import RetryPolicy
from plumbline.retrieve.retriever import Retriever
from plumbline.tracing import Metrics, Trace
from plumbline.types import ScoredChunk
from plumbline.validate.guards import check_question, screen_passages
from plumbline.validate.parse import ParseFailure, extract_json
from plumbline.validate.schema import Answer
from plumbline.validate.validator import ValidationReport, validate_output

RETRIEVAL_GATE = 0.3  # below this query coverage, do not spend a model call at all


@dataclass
class AskOptions:
    mode: str = "hybrid+rerank"
    k: int | None = None
    resilience: bool = True  # False = one provider, one attempt (the naive baseline)
    validation: bool = True  # False = trust whatever parses (the naive baseline)


def make_provider(name: str, settings: Settings) -> LLMProvider:
    if name == "extractive":
        return ExtractiveLLM("sim-hosted")
    if name == "anthropic":
        from plumbline.llm.anthropic_llm import AnthropicLLM

        return AnthropicLLM(timeout_s=settings.attempt_timeout_s)
    if name in ("openai", "openai-compat"):
        from plumbline.llm.openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(timeout_s=settings.attempt_timeout_s)
    raise ValueError(f"unknown provider {name!r} (expected extractive, anthropic or openai)")


def to_passage(sc: ScoredChunk) -> Passage:
    c = sc.chunk
    pages = str(c.page_start) if c.page_start == c.page_end else f"{c.page_start}-{c.page_end}"
    return Passage(c.chunk_id, c.title, c.section, pages, c.kind, c.text, c.short)


class RagPipeline:
    def __init__(self, retriever: Retriever, settings: Settings | None = None, *,
                 providers: list[LLMProvider] | None = None, seed: int | None = None,
                 attempt_timeout_s: float | None = None, sleep=asyncio.sleep,
                 clock=time.monotonic):
        self.settings = settings or get_settings()
        self.retriever = retriever
        self.metrics = Metrics()
        self.chaos = ChaosConfig()
        if providers is None:
            providers = [make_provider(n, self.settings) for n in self.settings.provider_chain]
        # Faults are injected into the primary only; the last link of the chain is always
        # the offline reader, so there is always something left to fall back to.
        self.simulated = isinstance(providers[0], ExtractiveLLM)
        # The simulated primary answers in milliseconds, so a hung call is obvious far
        # sooner than with a real hosted model; a short timeout keeps chaos demos snappy.
        self.attempt_timeout_s = attempt_timeout_s or (
            min(self.settings.attempt_timeout_s, 2.0) if self.simulated
            else self.settings.attempt_timeout_s)
        self.primary = ChaosLLM(providers[0], self.chaos,
                                timeout_hang_s=self.attempt_timeout_s + 5)
        self.chain: list[LLMProvider] = [self.primary, *providers[1:],
                                         ExtractiveLLM("local-extractive")]
        self.breakers = {p.name: CircuitBreaker(p.name, clock=clock) for p in self.chain}
        self.generator = ResilientGenerator(
            self.chain,
            policy=RetryPolicy(max_attempts=self.settings.max_attempts),
            attempt_timeout_s=self.attempt_timeout_s,
            deadline_s=self.settings.request_deadline_s,
            max_repairs=self.settings.max_repairs,
            breakers=self.breakers,
            rng=random.Random(seed),
            sleep=sleep,
        )

    @classmethod
    def load(cls, settings: Settings | None = None, **kwargs) -> RagPipeline:
        settings = settings or get_settings()
        return cls(Retriever.load(settings), settings, **kwargs)

    # -- operations --------------------------------------------------------------
    def status(self) -> dict:
        return {
            "chain": [{"name": p.name, "model": p.model} for p in self.chain],
            "breakers": [b.snapshot() for b in self.breakers.values()],
            "chaos": {**self.chaos.to_dict(), "injected": dict(self.primary.injected)},
            "metrics": self.metrics.snapshot(),
        }

    def reset(self) -> None:
        for b in self.breakers.values():
            b.reset()
        self.metrics.reset()
        self.primary.injected.clear()
        self.primary.reseed()

    # -- the request path --------------------------------------------------------
    async def answer(self, question: str, options: AskOptions | None = None,
                     on_event: Callable[[dict], None] | None = None) -> dict:
        options = options or AskOptions()
        trace = Trace(on_event)
        result: dict = {
            "id": trace.id, "question": question, "status": "failed", "answer": "",
            "claims": [], "confidence": 0.0, "served_by": None, "model": None,
            "fallback_used": False, "retries": 0, "attempts": [], "validation": None,
            "retrieval": None, "reason": "",
        }

        problems = check_question(question)
        if problems:
            result["reason"] = "; ".join(problems)
            return self._finish(result, trace)

        with trace.span("retrieve") as span:
            found = self.retriever.search(question, mode=options.mode, k=options.k)
            span.update(mode=found.mode, candidates=found.candidates, **found.timings_ms)

        with trace.span("screen_context") as span:
            passages, flagged = screen_passages([to_passage(h) for h in found.hits])
            span.update(flagged=len(flagged))
        result["retrieval"] = {
            "mode": found.mode, "confidence": round(found.confidence, 3),
            "timings_ms": found.timings_ms, "hits": [h.to_dict() for h in found.hits],
            "flagged": flagged, "query_terms": self.retriever.query_terms(question),
        }

        if not passages or found.confidence < RETRIEVAL_GATE:
            trace.emit({"type": "retrieval_gate", "confidence": round(found.confidence, 3),
                        "threshold": RETRIEVAL_GATE})
            result.update(status="abstained", reason=(
                "Nothing in the library covers this question well enough to answer it "
                f"(best passage covers {found.confidence:.0%} of the question)."))
            return self._finish(result, trace)

        request = build_request(question, passages, self.retriever.term_weights(question))

        def validate(raw: str) -> ValidationReport:
            if options.validation:
                return validate_output(raw, passages, question)
            return _trusting_validate(raw)

        with trace.span("generate") as span:
            if options.resilience:
                outcome = await self.generator.generate(request, validate, trace.emit)
            else:
                outcome = await self._single_shot(request, validate, trace)
            span.update(attempts=len(outcome.attempts), provider=outcome.provider,
                        status="ok" if outcome.ok else "error")

        result["attempts"] = [a.to_dict() for a in outcome.attempts]
        result["retries"] = outcome.retries
        result["validation"] = outcome.report.to_dict() if outcome.report else None
        if not outcome.ok:
            result["reason"] = outcome.reason
            return self._finish(result, trace)

        answer = outcome.report.answer
        result.update(served_by=outcome.provider, model=outcome.response.model,
                      fallback_used=outcome.fallback_used, confidence=round(answer.confidence, 3))
        if answer.abstained:
            result.update(status="abstained",
                          reason="The retrieved passages do not contain the answer.")
            return self._finish(result, trace)

        by_id = {h.chunk.chunk_id: h.chunk for h in found.hits}
        result["answer"] = answer.answer
        result["claims"] = [
            {"text": c.text, "quote": c.quote,
             "citations": [{"id": cid, "short": by_id[cid].short, "title": by_id[cid].title,
                            "section": by_id[cid].section, "page": by_id[cid].page_start}
                           for cid in c.citations if cid in by_id]}
            for c in answer.claims
        ]
        degraded = outcome.report.status == "degraded" or outcome.fallback_used
        result["status"] = "degraded" if degraded else "answered"
        if outcome.report.status == "degraded":
            dropped = outcome.report.claims_in - outcome.report.claims_kept
            result["reason"] = f"{dropped} unsupported claim(s) were removed before serving."
        elif outcome.fallback_used:
            result["reason"] = f"The primary model failed; served by {outcome.provider}."
        return self._finish(result, trace)

    async def _single_shot(self, request, validate, trace: Trace) -> GenerationOutcome:
        """The naive baseline: first provider, one attempt, no retry, no fallback."""
        provider = self.chain[0]
        outcome = GenerationOutcome(ok=False)
        t0 = trace.now_ms()
        try:
            response = await asyncio.wait_for(provider.complete(request),
                                              timeout=self.attempt_timeout_s)
        except (TimeoutError, LLMError) as exc:
            kind = getattr(exc, "kind", "timeout")
            outcome.attempts.append(Attempt(provider.name, 1, "error", kind, trace.now_ms() - t0))
            outcome.reason = f"{provider.name} failed: {kind}"
            trace.emit({"type": "attempt_error", "provider": provider.name, "attempt": 1,
                        "kind": kind, "retryable": False, "detail": str(exc)})
            return outcome
        report = validate(response.text)
        outcome.report = report
        verdict = "ok" if report.ok else "invalid"
        outcome.attempts.append(Attempt(provider.name, 1, verdict, report.status,
                                        trace.now_ms() - t0))
        if report.ok:
            outcome.ok, outcome.response, outcome.provider = True, response, provider.name
        else:
            outcome.reason = "the model reply failed validation"
        return outcome

    def _finish(self, result: dict, trace: Trace) -> dict:
        result["trace"] = trace.to_dict()
        result["events"] = trace.events
        trace.emit({"type": "done", "status": result["status"]})
        self.metrics.record(result)
        return result


def _trusting_validate(raw: str) -> ValidationReport:
    """Validation switched off: accept anything that parses. Used only as an eval baseline."""
    try:
        answer = Answer.model_validate(extract_json(raw))
    except (ParseFailure, ValidationError) as exc:
        report = ValidationReport(status="rejected")
        report.checks.append({"name": "json", "passed": False, "detail": str(exc)[:120]})
        return report
    return ValidationReport(status="pass", answer=answer, claims_in=len(answer.claims),
                            claims_kept=len(answer.claims))
