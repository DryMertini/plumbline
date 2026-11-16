"""Per-request traces and process-level counters. Small, in-memory, dependency-free."""

from __future__ import annotations

import time
import uuid
from collections import Counter, deque
from collections.abc import Callable
from contextlib import contextmanager


class Trace:
    def __init__(self, on_event: Callable[[dict], None] | None = None):
        self.id = uuid.uuid4().hex[:12]
        self._t0 = time.perf_counter()
        self.spans: list[dict] = []
        self.events: list[dict] = []
        self._on_event = on_event

    def now_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 1)

    def emit(self, event: dict) -> None:
        event = {"t_ms": self.now_ms(), **event}
        self.events.append(event)
        if self._on_event:
            self._on_event(event)

    @contextmanager
    def span(self, name: str, **attrs):
        start = self.now_ms()
        span = {"name": name, "start_ms": start, "ms": 0.0, "status": "ok", **attrs}
        self.emit({"type": "stage_start", "stage": name})
        try:
            yield span
        except Exception:
            span["status"] = "error"
            raise
        finally:
            span["ms"] = round(self.now_ms() - start, 1)
            self.spans.append(span)
            self.emit({"type": "stage_end", "stage": name, "ms": span["ms"],
                       "status": span["status"]})

    def to_dict(self) -> dict:
        return {"id": self.id, "total_ms": self.now_ms(), "spans": self.spans}


class Metrics:
    def __init__(self, keep: int = 200):
        self.counters: Counter = Counter()
        self.latencies_ms: deque[float] = deque(maxlen=keep)
        self.recent: deque[dict] = deque(maxlen=30)

    def record(self, result: dict) -> None:
        c = self.counters
        c["requests"] += 1
        c[f"status.{result['status']}"] += 1
        if result.get("served_by"):
            c[f"served_by.{result['served_by']}"] += 1
        c["retries"] += result.get("retries", 0)
        c["fallbacks"] += int(bool(result.get("fallback_used")))
        for issue in (result.get("validation") or {}).get("issues", []):
            c[f"issue.{issue['code']}"] += 1
        for attempt in result.get("attempts", []):
            if attempt["outcome"] == "invalid":
                c["blocked_outputs"] += 1
        self.latencies_ms.append(result["trace"]["total_ms"])
        self.recent.appendleft({
            "id": result["id"], "question": result["question"][:120], "status": result["status"],
            "served_by": result.get("served_by"), "ms": result["trace"]["total_ms"],
            "retries": result.get("retries", 0),
        })

    def snapshot(self) -> dict:
        lat = sorted(self.latencies_ms)

        def pct(p: float) -> float:
            return lat[min(len(lat) - 1, int(p * len(lat)))] if lat else 0.0

        return {"counters": dict(self.counters), "p50_ms": pct(0.5), "p95_ms": pct(0.95),
                "recent": list(self.recent)}

    def reset(self) -> None:
        self.counters.clear()
        self.latencies_ms.clear()
        self.recent.clear()
