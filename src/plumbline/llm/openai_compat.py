"""Adapter for any OpenAI-compatible chat endpoint (a local Ollama / vLLM server, etc.).

Configured with PLUMBLINE_OPENAI_BASE_URL, PLUMBLINE_OPENAI_MODEL and, if the endpoint
needs one, PLUMBLINE_OPENAI_API_KEY.
"""

from __future__ import annotations

import os

import httpx

from plumbline.llm.base import LLMRequest, LLMResponse
from plumbline.llm.prompts import repair_message
from plumbline.resilience.errors import (
    AuthError,
    BadRequest,
    ConnectionFailed,
    LLMTimeout,
    ProviderUnavailable,
    RateLimited,
    ServerError,
)


class OpenAICompatLLM:
    name = "openai-compat"

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, *, timeout_s: float = 60.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = (base_url or os.environ.get("PLUMBLINE_OPENAI_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("PLUMBLINE_OPENAI_MODEL", "")
        self._api_key = api_key or os.environ.get("PLUMBLINE_OPENAI_API_KEY", "")
        self._timeout_s = timeout_s
        self._transport = transport

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if not self.base_url or not self.model:
            raise ProviderUnavailable("set PLUMBLINE_OPENAI_BASE_URL and PLUMBLINE_OPENAI_MODEL")
        messages = [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.prompt},
        ]
        if request.repair_feedback:
            messages += [
                {"role": "assistant", "content": request.previous_output or "{}"},
                {"role": "user", "content": repair_message(request.repair_feedback)},
            ]
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        body = {"model": self.model, "messages": messages,
                "response_format": {"type": "json_object"}}
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s, transport=self._transport) as http:
                resp = await http.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
        except httpx.TimeoutException as exc:
            raise LLMTimeout(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ConnectionFailed(str(exc)) from exc

        if resp.status_code == 429:
            raise RateLimited("HTTP 429", retry_after=_float(resp.headers.get("retry-after")))
        if resp.status_code in (401, 403):
            raise AuthError(f"HTTP {resp.status_code}")
        if resp.status_code >= 500 or resp.status_code in (408, 409):
            raise ServerError(f"HTTP {resp.status_code}")
        if resp.status_code >= 400:
            raise BadRequest(f"HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ServerError(f"unexpected response shape: {exc}") from exc
        return LLMResponse(text, self.name, data.get("model", self.model), data.get("usage") or {})


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
