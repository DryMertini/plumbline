"""Retry policy: capped exponential backoff with full jitter, honouring Retry-After."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.25
    max_delay_s: float = 4.0

    def delay(self, attempt: int, rng: random.Random, retry_after: float | None = None) -> float:
        """Seconds to wait after failed attempt number `attempt` (1-based).

        Full jitter (uniform in [0, cap]) spreads retries out so that many clients failing
        together do not come back together. A server-provided Retry-After is a floor.
        """
        cap = min(self.max_delay_s, self.base_delay_s * (2 ** (attempt - 1)))
        wait = rng.uniform(0, cap)
        if retry_after is not None:
            wait = max(wait, min(retry_after, self.max_delay_s))
        return wait
