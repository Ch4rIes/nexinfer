from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True, slots=True)
class DecodeState:
    """Backend decode state after prefill or one decode step."""

    position: int
    backend_state: Any = None
    cache: Any = None


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Next-token logits and the backend-specific decode state."""

    logits: Sequence[float]
    state: DecodeState


@dataclass(frozen=True, slots=True)
class PrefillInput:
    """One scheduled prefill item for a backend batch."""

    token_ids: Sequence[int]
    num_cached_tokens: int = 0
    num_scheduled_tokens: int | None = None
    block_table: Sequence[int] = ()

    @property
    def scheduled_token_count(self) -> int:
        if self.num_scheduled_tokens is not None:
            return self.num_scheduled_tokens
        return max(len(self.token_ids) - self.num_cached_tokens, 0)


@dataclass(frozen=True, slots=True)
class DecodeInput:
    """One token and decode state to advance in a backend batch."""

    token_id: int
    state: DecodeState
    block_table: Sequence[int] = ()
    context_length: int | None = None
    num_scheduled_tokens: int = 1


@runtime_checkable
class DecoderOnlyBackend(Protocol):
    """Backend contract for autoregressive decoder-only models."""

    @property
    def vocab_size(self) -> int:
        """Number of token ids the backend can score."""

    def begin(self, input_ids: Sequence[int]) -> ModelOutput:
        """Return logits for the token after the prompt."""

    def step(self, token_id: int, state: DecodeState) -> ModelOutput:
        """Consume one generated token and return logits for the next one."""


@runtime_checkable
class BatchedDecoderOnlyBackend(DecoderOnlyBackend, Protocol):
    """Optional backend contract for grouped prefill and decode calls."""

    def begin_batch(self, inputs: Sequence[PrefillInput]) -> list[ModelOutput]:
        """Return next-token logits for a group of prompts."""

    def step_batch(self, inputs: Sequence[DecodeInput]) -> list[ModelOutput]:
        """Consume one generated token for each active sequence."""


@runtime_checkable
class Tokenizer(Protocol):
    """Tokenizer contract used by the generation engine."""

    def encode(self, text: str) -> list[int]:
        """Convert text into token ids."""

    def decode(self, token_ids: Sequence[int]) -> str:
        """Convert token ids back into text."""
