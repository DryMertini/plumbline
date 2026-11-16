"""Failure taxonomy. The single most useful question about an error is: is it worth retrying?"""

from __future__ import annotations


class LLMError(Exception):
    retryable = False
    kind = "error"

    def __init__(self, message: str = "", *, retry_after: float | None = None):
        super().__init__(message or self.kind)
        self.retry_after = retry_after


class LLMTimeout(LLMError):
    retryable, kind = True, "timeout"


class RateLimited(LLMError):
    retryable, kind = True, "rate_limited"


class ServerError(LLMError):
    retryable, kind = True, "server_error"


class ConnectionFailed(LLMError):
    retryable, kind = True, "connection_error"


class AuthError(LLMError):
    kind = "auth_error"


class BadRequest(LLMError):
    kind = "bad_request"


class Refused(LLMError):
    kind = "refused"


class ProviderUnavailable(LLMError):
    """The provider is not configured (missing key / package). Skip it, do not retry."""

    kind = "unavailable"
