"""Hosted-model adapters, exercised against fakes - no network, no API key."""

import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from plumbline.llm.anthropic_llm import FALLBACK_BETA, AnthropicLLM
from plumbline.llm.base import Passage
from plumbline.llm.openai_compat import OpenAICompatLLM
from plumbline.llm.prompts import ANSWER_SCHEMA, build_request
from plumbline.resilience import errors

REQUEST = build_request("What is X?", [Passage("p1", "T", "S", "1", "text", "X is a thing.")])


def message(text="{}", stop_reason="end_turn", category=None):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason, stop_details=SimpleNamespace(category=category),
        model="claude-opus-5-5", usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


class FakeMessages:
    def __init__(self, outcome):
        self.outcome, self.calls = outcome, []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def fake_client(outcome):
    messages = FakeMessages(outcome)
    return SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=messages)), messages


def http_error(cls, status, headers=None):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx.Response(status, request=request, headers=headers or {}), body=None)


async def test_anthropic_request_shape_and_reply():
    client, messages = fake_client(message('{"abstained": true}'))
    response = await AnthropicLLM(client=client).complete(REQUEST)
    call = messages.calls[0]
    assert response.text == '{"abstained": true}' and response.usage["output_tokens"] == 5
    assert call["model"] == "claude-opus-5-5" and call["thinking"] == {"type": "adaptive"}
    assert call["output_config"]["format"] == {"type": "json_schema", "schema": ANSWER_SCHEMA}
    assert call["betas"] == [FALLBACK_BETA] and call["fallbacks"] == "default"
    assert call["messages"] == [{"role": "user", "content": REQUEST.prompt}]
    assert "temperature" not in call and "budget_tokens" not in json.dumps(call)


async def test_anthropic_repair_turn_replays_the_rejected_output():
    client, messages = fake_client(message())
    from dataclasses import replace

    repair = replace(REQUEST, previous_output='{"bad": 1}', repair_feedback="- quote not found")
    await AnthropicLLM(client=client, server_fallback=False).complete(repair)
    sent = messages.calls[0]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[1]["content"] == '{"bad": 1}' and "quote not found" in sent[2]["content"]
    assert "betas" not in messages.calls[0]


@pytest.mark.parametrize("exc,expected", [
    (anthropic.APITimeoutError(httpx.Request("POST", "https://x")), errors.LLMTimeout),
    (http_error(anthropic.RateLimitError, 429, {"retry-after": "7"}), errors.RateLimited),
    (http_error(anthropic.AuthenticationError, 401), errors.AuthError),
    (http_error(anthropic.BadRequestError, 400), errors.BadRequest),
    (http_error(anthropic.InternalServerError, 529), errors.ServerError),
    (anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")), errors.ConnectionFailed),
])
async def test_anthropic_errors_map_to_the_taxonomy(exc, expected):
    client, _ = fake_client(exc)
    with pytest.raises(expected) as err:
        await AnthropicLLM(client=client).complete(REQUEST)
    if expected is errors.RateLimited:
        assert err.value.retry_after == 7.0


async def test_anthropic_refusal_is_not_retried():
    client, _ = fake_client(message("", stop_reason="refusal", category="cyber"))
    with pytest.raises(errors.Refused) as err:
        await AnthropicLLM(client=client).complete(REQUEST)
    assert not err.value.retryable


def openai_llm(handler):
    return OpenAICompatLLM("http://llm.test/v1", "local-model", "k", transport=httpx.MockTransport(handler))


async def test_openai_compat_round_trip():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content), auth=request.headers["authorization"], url=str(request.url))
        return httpx.Response(200, json={"model": "local-model", "usage": {"total_tokens": 9},
                                         "choices": [{"message": {"content": '{"ok": 1}'}}]})

    response = await openai_llm(handler).complete(REQUEST)
    assert response.text == '{"ok": 1}' and seen["url"] == "http://llm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer k" and seen["messages"][0]["role"] == "system"
    assert seen["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("status,expected", [(429, errors.RateLimited), (401, errors.AuthError),
                                             (503, errors.ServerError), (400, errors.BadRequest)])
async def test_openai_compat_status_mapping(status, expected):
    with pytest.raises(expected):
        await openai_llm(lambda r: httpx.Response(status, headers={"retry-after": "2"})).complete(REQUEST)


async def test_openai_compat_transport_and_config_errors():
    def boom(request):
        raise httpx.ConnectTimeout("slow")

    with pytest.raises(errors.LLMTimeout):
        await openai_llm(boom).complete(REQUEST)
    with pytest.raises(errors.ProviderUnavailable):
        await OpenAICompatLLM("", "").complete(REQUEST)
    with pytest.raises(errors.ServerError):
        await openai_llm(lambda r: httpx.Response(200, json={"choices": []})).complete(REQUEST)
