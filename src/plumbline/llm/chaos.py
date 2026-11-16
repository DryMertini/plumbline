"""Fault injection. You cannot claim a system handles failure you have never caused.

`ChaosLLM` wraps any provider and, with configurable probabilities, makes it fail the way
hosted models really fail - in transport (timeouts, 429s, 5xx) and, more insidiously, in
content (truncated JSON, schema drift, citations to passages that were never retrieved,
confident statements nothing supports). The RNG is seeded, so a chaos run is reproducible.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import asdict, dataclass, fields

from plumbline.llm.base import LLMProvider, LLMRequest, LLMResponse
from plumbline.resilience.errors import LLMTimeout, RateLimited, ServerError

TRANSPORT_FAULTS = ("timeout", "rate_limit", "server_error")
CONTENT_FAULTS = ("malformed_json", "schema_violation", "bad_citation", "ungrounded_claim")
FABRICATED_CLAIM = {
    "text": "The method reaches 99.7% accuracy on every benchmark, a 41.3 point improvement.",
    "quote": "reaches 99.7% accuracy on every benchmark, a 41.3 point improvement",
}


@dataclass
class ChaosConfig:
    timeout: float = 0.0
    rate_limit: float = 0.0
    server_error: float = 0.0
    malformed_json: float = 0.0
    schema_violation: float = 0.0
    bad_citation: float = 0.0
    ungrounded_claim: float = 0.0
    latency_ms: int = 0  # simulated network + generation time for successful calls
    seed: int = 7

    def to_dict(self) -> dict:
        return asdict(self)

    def update(self, values: dict) -> None:
        for f in fields(self):
            if f.name in values and values[f.name] is not None:
                current = getattr(self, f.name)
                value = type(current)(values[f.name])
                if isinstance(current, float):
                    value = min(max(value, 0.0), 1.0)
                setattr(self, f.name, value)

    @property
    def active(self) -> bool:
        return any(getattr(self, n) > 0 for n in TRANSPORT_FAULTS + CONTENT_FAULTS)


PROFILES: dict[str, dict] = {
    "off": {},
    "flaky-network": {"timeout": 0.2, "rate_limit": 0.15, "server_error": 0.15},
    "bad-outputs": {"malformed_json": 0.2, "schema_violation": 0.15, "bad_citation": 0.2,
                    "ungrounded_claim": 0.2},
    "everything": {"timeout": 0.15, "rate_limit": 0.1, "server_error": 0.1, "malformed_json": 0.15,
                   "schema_violation": 0.1, "bad_citation": 0.15, "ungrounded_claim": 0.15},
    "outage": {"server_error": 1.0},
}


class ChaosLLM:
    def __init__(self, inner: LLMProvider, config: ChaosConfig | None = None, *,
                 name: str | None = None, timeout_hang_s: float = 30.0):
        self.inner = inner
        self.config = config or ChaosConfig()
        self.name = name or inner.name
        self.model = inner.model
        self._hang_s = timeout_hang_s
        self._rng = random.Random(self.config.seed)
        self.injected: dict[str, int] = {}

    def reseed(self, seed: int | None = None) -> None:
        if seed is not None:
            self.config.seed = seed
        self._rng = random.Random(self.config.seed)

    def _roll(self, fault: str) -> bool:
        hit = self._rng.random() < getattr(self.config, fault)
        if hit:
            self.injected[fault] = self.injected.get(fault, 0) + 1
        return hit

    async def complete(self, request: LLMRequest) -> LLMResponse:
        cfg = self.config
        if self._roll("timeout"):
            await asyncio.sleep(self._hang_s)  # the caller's own timeout is what must fire
            raise LLMTimeout("injected: upstream never answered")
        if self._roll("rate_limit"):
            raise RateLimited("injected: HTTP 429", retry_after=0.3)
        if self._roll("server_error"):
            raise ServerError("injected: HTTP 503")
        if cfg.latency_ms:
            await asyncio.sleep(cfg.latency_ms / 1000 * self._rng.uniform(0.6, 1.4))
        response = await self.inner.complete(request)
        response.provider = self.name

        if self._roll("malformed_json"):
            response.text = response.text[: max(8, int(len(response.text) * 0.6))]
            return response
        try:
            data = json.loads(response.text)
        except ValueError:
            return response
        if not isinstance(data, dict):
            return response
        if self._roll("schema_violation"):
            data.pop("claims", None)
            data["sources"] = "see above"
        claims = data.get("claims")
        if isinstance(claims, list) and claims:
            if self._roll("bad_citation"):
                claims[0]["citations"] = ["ghost:0042"]
            if self._roll("ungrounded_claim"):
                cited = claims[0].get("citations") or ["ghost:0042"]
                claims.append({**FABRICATED_CLAIM, "citations": list(cited)})
                data["answer"] = f"{data.get('answer', '')} {FABRICATED_CLAIM['text']}".strip()
        response.text = json.dumps(data)
        return response
