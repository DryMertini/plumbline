"""Tolerant JSON extraction: accept what models actually send, reject what is truly broken."""

from __future__ import annotations

import json
import re

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I | re.M)


class ParseFailure(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def extract_json(raw: str) -> dict:
    """Pull the first JSON object out of a model reply.

    Handles code fences and prose around the object. A reply that opens an object but never
    closes it is reported as truncated - the most common real failure (max_tokens, a cut
    stream) and one a retry usually fixes.
    """
    text = _FENCE.sub("", raw.strip()).strip()
    start = text.find("{")
    if start < 0:
        raise ParseFailure("no_json", "the reply contains no JSON object")
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start : i + 1])
                except ValueError as exc:
                    raise ParseFailure("malformed_json", f"invalid JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise ParseFailure("malformed_json", "top-level JSON value is not an object")
                return value
    raise ParseFailure("truncated_json", "the JSON object is cut off before it closes")
