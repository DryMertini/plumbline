import json
import random

import pytest

from plumbline.llm.base import LLMRequest, LLMResponse, Passage
from plumbline.llm.chaos import ChaosConfig, ChaosLLM
from plumbline.llm.extractive import ExtractiveLLM
from plumbline.llm.prompts import build_request
from plumbline.resilience.breaker import CircuitBreaker
from plumbline.resilience.errors import AuthError, ProviderUnavailable, RateLimited, ServerError
from plumbline.resilience.fallback import ResilientGenerator
from plumbline.resilience.retry import RetryPolicy
from plumbline.validate.validator import validate_output

PASSAGES = [Passage("p1", "Paper", "Results", "3", "text",
                    "The Alpha model reaches 71.5 recall on the Quartz benchmark after training.")]
REQUEST = build_request("What recall does the Alpha model reach on the Quartz benchmark?", PASSAGES)
GOOD = json.dumps({"abstained": False, "answer": "71.5 recall.", "confidence": 0.9, "claims": [
    {"text": "The Alpha model reaches 71.5 recall on the Quartz benchmark.", "citations": ["p1"],
     "quote": "reaches 71.5 recall on the Quartz benchmark"}]})


class Scripted:
    """A provider that plays back a script of replies and exceptions."""

    def __init__(self, name, script):
        self.name, self.model, self.script, self.requests = name, "scripted", list(script), []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, Exception):
            raise step
        return LLMResponse(step, self.name, self.model)


def validate(raw):
    return validate_output(raw, PASSAGES, REQUEST.question)


def generator(providers, **kw):
    waits = []

    async def sleep(seconds):
        waits.append(seconds)

    gen = ResilientGenerator(providers, rng=random.Random(0), sleep=sleep, **kw)
    return gen, waits


# ---- retry policy ----------------------------------------------------------------
def test_backoff_is_jittered_capped_and_honours_retry_after():
    policy, rng = RetryPolicy(base_delay_s=0.5, max_delay_s=4.0), random.Random(1)
    assert all(0 <= policy.delay(1, rng) <= 0.5 for _ in range(50))
    assert all(0 <= policy.delay(10, rng) <= 4.0 for _ in range(50))
    assert policy.delay(1, rng, retry_after=2.0) >= 2.0
    assert policy.delay(1, rng, retry_after=60.0) == 4.0  # never wait longer than the cap


# ---- circuit breaker -------------------------------------------------------------
def test_breaker_opens_then_probes_then_closes():
    now = [0.0]
    b = CircuitBreaker("x", window=4, min_calls=4, failure_rate=0.5, cooldown_s=10, clock=lambda: now[0])
    for ok in (True, False, True):
        b.record_success() if ok else b.record_failure()
    assert b.state == "closed"
    b.record_failure()  # 2 of the last 4 failed
    assert b.state == "open" and not b.allow()
    now[0] = 10.0
    assert b.state == "half_open" and b.allow() and not b.allow()  # exactly one probe
    b.record_failure()
    assert b.state == "open"
    now[0] = 20.0
    assert b.allow()
    b.record_success()
    assert b.state == "closed" and b.allow() and b.times_opened == 2


# ---- the generation loop ---------------------------------------------------------
async def test_retries_transient_errors_then_succeeds():
    primary = Scripted("primary", [ServerError("503"), RateLimited("429", retry_after=0.3), GOOD])
    gen, waits = generator([primary])
    out = await gen.generate(REQUEST, validate)
    assert out.ok and out.provider == "primary" and out.retries == 2 and not out.fallback_used
    assert len(waits) == 2 and waits[1] >= 0.3


async def test_permanent_error_falls_straight_through_to_the_next_provider():
    primary, backup = Scripted("primary", [AuthError("bad key")]), Scripted("backup", [GOOD])
    gen, waits = generator([primary, backup])
    out = await gen.generate(REQUEST, validate)
    assert out.ok and out.provider == "backup" and out.fallback_used
    assert len(primary.requests) == 1 and waits == []


async def test_invalid_reply_gets_one_repair_turn_with_the_validator_feedback():
    primary = Scripted("primary", ['{"abstained": false, "answer": "x"', GOOD])
    gen, _ = generator([primary])
    out = await gen.generate(REQUEST, validate)
    assert out.ok and [a.outcome for a in out.attempts] == ["invalid", "ok"]
    repair = primary.requests[1]
    assert repair.previous_output.startswith('{"abstained"') and "JSON" in repair.repair_feedback


async def test_persistently_bad_output_is_never_served():
    lie = json.dumps({"abstained": False, "answer": "99.9", "confidence": 1, "claims": [
        {"text": "It reaches 99.9 recall.", "citations": ["p1"], "quote": "reaches 99.9 recall"}]})
    primary, backup = Scripted("primary", [lie]), Scripted("backup", [GOOD])
    gen, _ = generator([primary, backup], max_repairs=1)
    out = await gen.generate(REQUEST, validate)
    assert out.provider == "backup" and "99.9" not in out.report.answer.answer
    assert len(primary.requests) == 2  # first try + one repair, then give up on it


async def test_open_breaker_skips_the_provider_entirely():
    primary, backup = Scripted("primary", [ServerError("down")]), Scripted("backup", [GOOD])
    gen, _ = generator([primary, backup], policy=RetryPolicy(max_attempts=2))
    for _ in range(3):
        assert (await gen.generate(REQUEST, validate)).provider == "backup"
    calls = len(primary.requests)
    events = []
    out = await gen.generate(REQUEST, validate, events.append)
    assert len(primary.requests) == calls and out.attempts[0].outcome == "skipped"
    assert events[0]["type"] == "provider_skipped"


async def test_a_hung_call_times_out():
    class Hung:
        name, model = "hung", "m"

        async def complete(self, request):
            import asyncio

            await asyncio.sleep(5)

    gen, _ = generator([Hung(), Scripted("backup", [GOOD])], attempt_timeout_s=0.02,
                       policy=RetryPolicy(max_attempts=1))
    out = await gen.generate(REQUEST, validate)
    assert out.provider == "backup" and out.attempts[0].detail.startswith("timeout")


async def test_unconfigured_provider_does_not_trip_its_breaker():
    primary, backup = Scripted("primary", [ProviderUnavailable("no key")]), Scripted("backup", [GOOD])
    gen, _ = generator([primary, backup])
    for _ in range(6):
        await gen.generate(REQUEST, validate)
    assert gen.breakers["primary"].state == "closed"


async def test_total_failure_is_a_result_not_an_exception():
    gen, _ = generator([Scripted("a", [ServerError("x")]), Scripted("b", ["not json"])])
    out = await gen.generate(REQUEST, validate)
    assert not out.ok and out.reason and out.provider is None


async def test_deadline_bounds_the_whole_request():
    now = [0.0]

    async def sleep(seconds):
        now[0] += seconds

    class Slow(Scripted):
        async def complete(self, request):
            now[0] += 4.0
            return await super().complete(request)

    gen = ResilientGenerator([Slow("slow", [ServerError("x")])], deadline_s=5.0,
                             policy=RetryPolicy(max_attempts=5), rng=random.Random(0),
                             sleep=sleep, clock=lambda: now[0])
    out = await gen.generate(REQUEST, validate)
    assert not out.ok and out.reason == "request deadline exceeded" and len(out.attempts) == 2


# ---- chaos and the offline reader ------------------------------------------------
async def test_chaos_is_reproducible_for_a_seed():
    async def run():
        chaos = ChaosLLM(ExtractiveLLM(), ChaosConfig(server_error=0.4, bad_citation=0.4, seed=3))
        seen = []
        for _ in range(20):
            try:
                seen.append((await chaos.complete(REQUEST)).text)
            except ServerError:
                seen.append("503")
        return seen

    first = await run()
    assert first == await run() and "503" in first and any("ghost:0042" in s for s in first)


async def test_validator_blocks_every_content_fault():
    for fault in ("malformed_json", "schema_violation", "bad_citation", "ungrounded_claim"):
        chaos = ChaosLLM(ExtractiveLLM(), ChaosConfig(**{fault: 1.0}))
        report = validate((await chaos.complete(REQUEST)).text)
        served = report.answer.answer if report.answer else ""
        assert report.status != "pass", fault
        assert "99.7%" not in served and "ghost" not in json.dumps(report.to_dict()["checks"])


def test_chaos_config_clamps_probabilities():
    cfg = ChaosConfig()
    cfg.update({"timeout": 7, "rate_limit": -1, "latency_ms": "250"})
    assert (cfg.timeout, cfg.rate_limit, cfg.latency_ms) == (1.0, 0.0, 250) and cfg.active


def test_reader_answers_abstains_and_respects_anchors():
    reader = ExtractiveLLM()
    answer = reader.read(REQUEST)
    assert not answer["abstained"] and "71.5" in answer["answer"] and answer["claims"][0]["citations"] == ["p1"]
    other_model = build_request("What recall does the Omega model reach on the Quartz benchmark?", PASSAGES)
    assert reader.read(other_model)["abstained"]  # 'Omega' is named but appears nowhere
    assert reader.read(build_request("Who painted the ceiling of the chapel?", PASSAGES))["abstained"]


@pytest.mark.parametrize("fault", ["timeout", "rate_limit", "server_error"])
async def test_transport_faults_raise_typed_errors(fault):
    chaos = ChaosLLM(ExtractiveLLM(), ChaosConfig(**{fault: 1.0}), timeout_hang_s=0.01)
    with pytest.raises(Exception) as err:
        await chaos.complete(REQUEST)
    assert err.value.retryable and chaos.injected == {fault: 1}
