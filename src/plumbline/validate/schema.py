"""The answer contract, as a typed model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Claim(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)
    quote: str = ""

    @field_validator("citations", mode="before")
    @classmethod
    def _strip_brackets(cls, value):
        if isinstance(value, list):
            return [str(v).strip().strip("[]") for v in value]
        return value


class Answer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    abstained: bool = False
    answer: str = ""
    claims: list[Claim]
    confidence: float = 0.0

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, value):
        try:
            return min(max(float(value), 0.0), 1.0)
        except (TypeError, ValueError):
            return 0.0
