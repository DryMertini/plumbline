"""Claude adapter (official Anthropic SDK).

SDK-level retries are switched off on purpose: retrying, backoff, circuit breaking and
fallback all live in `plumbline.resilience`, where they are observable and shared by every
provider. This adapter only translates - a request into a Messages API call with a JSON
schema-constrained output, and SDK exceptions into the plumbline failure taxonomy.
"""

from __future__ import annotations

import os

from plumbline.llm.base import LLMRequest, LLMResponse
from plumbline.llm.prompts import ANSWER_SCHEMA, repair_message
from plumbline.resilience.errors import (
    AuthError,
    BadRequest,
    ConnectionFailed,
    LLMTimeout,
    ProviderUnavailable,
    RateLimited,
    Refused,
    ServerError,
)

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str | None = None, *, timeout_s: float = 60.0, client=None,
                 effort: str | None = None, server_fallback: bool | None = None):
        self.model = model or os.environ.get("PLUMBLINE_ANTHROPIC_MODEL", DEFAULT_MODEL)
        self.effort = effort or os.environ.get("PLUMBLINE_ANTHROPIC_EFFORT", "medium")
        if server_fallback is None:
            server_fallback = os.environ.get("PLUMBLINE_ANTHROPIC_SERVER_FALLBACK", "1") != "0"
        # If Claude declines a request, let the API re-run it on a fallback model inside the
        # same call rather than surfacing a refusal for what is ordinary paper Q&A.
        self.server_fallback = server_fallback
        self._timeout_s = timeout_s
        self._client = client

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on the install
                raise ProviderUnavailable(
                    "the `anthropic` package is not installed (pip install 'plumbline[anthropic]')"
                ) from exc
            try:
                self._client = anthropic.AsyncAnthropic(max_retries=0, timeout=self._timeout_s)
            except anthropic.AnthropicError as exc:
                raise ProviderUnavailable(f"no Anthropic credentials found: {exc}") from exc
        return self._client

    async def complete(self, request: LLMRequest) -> LLMResponse:
        import anthropic

        client = self._get_client()
        messages = [{"role": "user", "content": request.prompt}]
        if request.repair_feedback:
            messages += [
                {"role": "assistant", "content": request.previous_output or "{}"},
                {"role": "user", "content": repair_message(request.repair_feedback)},
            ]
        params = dict(
            model=self.model,
            max_tokens=16000,
            system=request.system,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
            },
        )
        try:
            if self.server_fallback:
                response = await client.beta.messages.create(
                    betas=[FALLBACK_BETA], fallbacks="default", **params
                )
            else:
                response = await client.messages.create(**params)
        except anthropic.APITimeoutError as exc:
            raise LLMTimeout(str(exc)) from exc
        except anthropic.RateLimitError as exc:
            raise RateLimited(str(exc), retry_after=_retry_after(exc)) from exc
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise AuthError(str(exc)) from exc
        except (anthropic.BadRequestError, anthropic.NotFoundError) as exc:
            raise BadRequest(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500 or exc.status_code in (408, 409):
                raise ServerError(f"HTTP {exc.status_code}") from exc
            raise BadRequest(f"HTTP {exc.status_code}: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise ConnectionFailed(str(exc)) from exc
        except anthropic.AnthropicError as exc:  # e.g. credentials could not be resolved
            raise ProviderUnavailable(str(exc)) from exc

        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            raise Refused(f"model declined the request (category: {category})")
        text = next((b.text for b in response.content if b.type == "text"), "")
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "stop_reason": response.stop_reason,
        }
        return LLMResponse(text, self.name, response.model, usage)


def _retry_after(exc) -> float | None:
    try:
        return float(exc.response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None
