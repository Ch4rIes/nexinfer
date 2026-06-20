from __future__ import annotations

from collections.abc import Mapping, Sequence

from nexinfer.protocols import DecodeState, ModelOutput


class BigramBackend:
    """A tiny backend that scores the next token from the previous token id."""

    def __init__(
        self,
        *,
        vocab_size: int,
        transitions: Mapping[int | None, Mapping[int, float]] | None = None,
        default_logit: float = -100.0,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self._vocab_size = vocab_size
        self._transitions = dict(transitions or {})
        self._default_logit = default_logit

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def begin(self, input_ids: Sequence[int]) -> ModelOutput:
        previous_token_id = input_ids[-1] if input_ids else None
        return ModelOutput(
            logits=self._logits_for(previous_token_id),
            state=DecodeState(
                position=len(input_ids),
                backend_state=previous_token_id,
            ),
        )

    def step(self, token_id: int, state: DecodeState) -> ModelOutput:
        return ModelOutput(
            logits=self._logits_for(token_id),
            state=DecodeState(
                position=state.position + 1,
                backend_state=token_id,
                cache=state.cache,
            ),
        )

    def _logits_for(self, previous_token_id: int | None) -> list[float]:
        logits = [self._default_logit] * self._vocab_size
        for token_id, logit in self._transitions.get(previous_token_id, {}).items():
            if not 0 <= token_id < self._vocab_size:
                raise ValueError(f"transition token id out of range: {token_id}")
            logits[token_id] = logit
        return logits
