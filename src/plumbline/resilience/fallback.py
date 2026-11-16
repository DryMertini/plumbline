"""The resilient generation loop: timeouts, retries, repair, circuit breakers, fallback.

For each provider in the chain, in order:

    circuit open?  ---------------------------->  skip to the next provider
    call with a per-attempt timeout
      transport error, retryable    ->  back off (full jitter), retry up to max_attempts
      transport error, permanent    ->  next provider
      reply fails validation        ->  one repair turn (the verifier's findings are sent
                                        back to the model), then next provider
      reply validates               ->  done

All of it happens inside one request deadline, so a user never waits on a retry storm.
If the whole chain fails the caller gets a structured `failed` outcome - never an exception.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace

from plumbline.llm.base import LLMProvider, LLMRequest, LLMResponse
from plumbline.resilience.breaker import CircuitBreaker
from plumbline.resilience.errors import LLMError, LLMTimeout, ProviderUnavailable
from plumbline.resilience.retry import RetryPolicy
from plumbline.validate.validator import ValidationReport

Emit = Callable[[dict], None]


@dataclass
class Attempt:
    provider: str
    number: int
    outcome: str  # ok | error | invalid | skipped
    detail: str = ""
    latency_ms: float = 0.0
    repair: bool = False

    def to_dict(self) -> dict:
        return {"provider": self.provider, "number": self.number, "outcome": self.outcome,
                "detail": self.detail, "latency_ms": round(self.latency_ms, 1),
                "repair": self.repair}


@dataclass
class GenerationOutcome:
    ok: bool
    report: ValidationReport | None = None
    response: LLMResponse | None = None
    provider: str | None = None
    attempts: list[Attempt] = field(default_factory=list)
    reason: str = ""

    @property
    def retries(self) -> int:
        return sum(1 for a in self.attempts if a.outcome in ("error", "invalid"))

    @property
    def fallback_used(self) -> bool:
        tried = list(dict.fromkeys(a.provider for a in self.attempts))
        return self.ok and len(tried) > 1


class ResilientGenerator:
    def __init__(
        self,
        providers: list[LLMProvider],
        *,
        policy: RetryPolicy | None = None,
        attempt_timeout_s: float = 8.0,
        deadline_s: float = 20.0,
        max_repairs: int = 1,
        breakers: dict[str, CircuitBreaker] | None = None,
        rng: random.Random | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.providers = providers
        self.policy = policy or RetryPolicy()
        self.attempt_timeout_s, self.deadline_s = attempt_timeout_s, deadline_s
        self.max_repairs = max_repairs
        self.breakers = breakers if breakers is not None else {}
        for p in providers:
            self.breakers.setdefault(p.name, CircuitBreaker(p.name))
        self._rng = rng or random.Random()
        self._sleep, self._clock = sleep, clock

    async def generate(
        self,
        request: LLMRequest,
        validate: Callable[[str], ValidationReport],
        emit: Emit | None = None,
    ) -> GenerationOutcome:
        emit = emit or (lambda _e: None)
        started = self._clock()
        outcome = GenerationOutcome(ok=False)

        def remaining() -> float:
            return self.deadline_s - (self._clock() - started)

        for provider in self.providers:
            breaker = self.breakers[provider.name]
            if not breaker.allow():
                outcome.attempts.append(Attempt(provider.name, 0, "skipped", "circuit open"))
                emit({"type": "provider_skipped", "provider": provider.name,
                      "reason": "circuit open"})
                continue

            req, repairs, number = request, 0, 0
            while number < self.policy.max_attempts + repairs:
                if remaining() <= 0.05:
                    outcome.reason = "request deadline exceeded"
                    emit({"type": "deadline", "provider": provider.name})
                    return outcome
                number += 1
                is_repair = req.repair_feedback is not None
                emit({"type": "attempt_start", "provider": provider.name, "attempt": number,
                      "repair": is_repair})
                t0 = self._clock()
                error: LLMError | None = None
                response: LLMResponse | None = None
                try:
                    response = await asyncio.wait_for(
                        provider.complete(req), timeout=min(self.attempt_timeout_s, remaining())
                    )
                except TimeoutError:
                    error = LLMTimeout(f"no reply within {self.attempt_timeout_s:g}s")
                except LLMError as exc:
                    error = exc
                latency = (self._clock() - t0) * 1000

                if error is not None:
                    outcome.attempts.append(
                        Attempt(provider.name, number, "error", f"{error.kind}: {error}", latency,
                                is_repair))
                    emit({"type": "attempt_error", "provider": provider.name, "attempt": number,
                          "kind": error.kind, "retryable": error.retryable,
                          "detail": str(error), "latency_ms": round(latency, 1)})
                    if isinstance(error, ProviderUnavailable):
                        break  # misconfiguration is not an outage: do not trip the breaker
                    breaker.record_failure()
                    if breaker.state == "open":
                        emit({"type": "breaker_opened", "provider": provider.name})
                        break
                    if not error.retryable or number >= self.policy.max_attempts + repairs:
                        break
                    wait = self.policy.delay(number, self._rng, error.retry_after)
                    if wait >= remaining():
                        break
                    emit({"type": "backoff", "provider": provider.name, "wait_ms": round(wait * 1000)})
                    await self._sleep(wait)
                    continue

                report = validate(response.text)
                if report.ok:
                    breaker.record_success()
                    outcome.attempts.append(
                        Attempt(provider.name, number, "ok", report.status, latency, is_repair))
                    emit({"type": "attempt_ok", "provider": provider.name, "attempt": number,
                          "validation": report.status, "latency_ms": round(latency, 1)})
                    outcome.ok, outcome.report = True, report
                    outcome.response, outcome.provider = response, provider.name
                    return outcome

                breaker.record_failure()
                codes = ", ".join(dict.fromkeys(i.code for i in report.issues))
                outcome.attempts.append(
                    Attempt(provider.name, number, "invalid", codes, latency, is_repair))
                emit({"type": "attempt_invalid", "provider": provider.name, "attempt": number,
                      "issues": [i.to_dict() for i in report.issues],
                      "latency_ms": round(latency, 1)})
                outcome.report = report
                if breaker.state == "open":
                    emit({"type": "breaker_opened", "provider": provider.name})
                    break
                if repairs >= self.max_repairs:
                    break
                repairs += 1
                req = replace(request, previous_output=response.text,
                              repair_feedback=report.feedback())
                emit({"type": "repair", "provider": provider.name})

            emit({"type": "provider_exhausted", "provider": provider.name})

        if not outcome.reason:
            outcome.reason = ("request deadline exceeded" if remaining() <= 0.05
                              else "every provider in the chain failed")
        return outcome
