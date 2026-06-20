from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FinishReason = Literal["length", "stop"]


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts for one generation request."""

    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Structured output for a completed generation."""

    text: str
    token_ids: list[int]
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    generated_token_logprobs: list[float]
    finish_reason: FinishReason
    usage: TokenUsage


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """A decoded token fragment produced during streaming generation."""

    text: str
    token_id: int
    index: int
    logprob: float
    finish_reason: FinishReason | None = None
