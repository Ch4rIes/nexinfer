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
class Tokenizer(Protocol):
    """Tokenizer contract used by the generation engine."""

    def encode(self, text: str) -> list[int]:
        """Convert text into token ids."""

    def decode(self, token_ids: Sequence[int]) -> str:
        """Convert token ids back into text."""
