"""Circuit breaker: stop sending traffic to a provider that is clearly down.

closed    -> requests flow; outcomes are recorded in a rolling window
open      -> requests are rejected immediately until the cooldown passes
half_open -> one probe is let through; success closes the circuit, failure re-opens it
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        *,
        window: int = 10,
        min_calls: int = 6,
        failure_rate: float = 0.6,
        cooldown_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.name = name
        self.window, self.min_calls = window, min_calls
        self.failure_rate, self.cooldown_s = failure_rate, cooldown_s
        self._clock = clock
        self._outcomes: deque[bool] = deque(maxlen=window)
        self._state = "closed"
        self._opened_at = 0.0
        self._probing = False
        self.times_opened = 0

    @property
    def state(self) -> str:
        if self._state == "open" and self._clock() - self._opened_at >= self.cooldown_s:
            self._state = "half_open"
            self._probing = False
        return self._state

    def allow(self) -> bool:
        state = self.state
        if state == "closed":
            return True
        if state == "half_open" and not self._probing:
            self._probing = True
            return True
        return False

    def record_success(self) -> None:
        if self._state in ("half_open", "open"):
            self._outcomes.clear()
        self._state, self._probing = "closed", False
        self._outcomes.append(True)

    def record_failure(self) -> None:
        if self._state == "half_open":
            self._trip()
            return
        self._outcomes.append(False)
        failures = self._outcomes.count(False)
        if len(self._outcomes) >= self.min_calls and failures / len(self._outcomes) >= self.failure_rate:
            self._trip()

    def _trip(self) -> None:
        self._state, self._probing = "open", False
        self._opened_at = self._clock()
        self.times_opened += 1

    def reset(self) -> None:
        self._outcomes.clear()
        self._state, self._probing = "closed", False

    def snapshot(self) -> dict:
        state = self.state
        retry_in = max(0.0, self.cooldown_s - (self._clock() - self._opened_at)) if state == "open" else 0.0
        return {
            "name": self.name, "state": state, "recent_calls": len(self._outcomes),
            "recent_failures": self._outcomes.count(False), "times_opened": self.times_opened,
            "retry_in_s": round(retry_in, 1),
        }
