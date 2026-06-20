from __future__ import annotations

from dataclasses import dataclass, field

from nexinfer.result import FinishReason


@dataclass(slots=True)
class SequenceState:
    """Mutable token state for one generation sequence."""

    prompt_token_ids: list[int]
    generated_token_ids: list[int] = field(default_factory=list)
    generated_token_logprobs: list[float] = field(default_factory=list)
    finish_reason: FinishReason | None = None

    @property
    def token_ids(self) -> list[int]:
        return [*self.prompt_token_ids, *self.generated_token_ids]

    @property
    def next_position(self) -> int:
        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    @property
    def completion_tokens(self) -> int:
        return len(self.generated_token_ids)

    def append(self, token_id: int, logprob: float) -> None:
        self.generated_token_ids.append(token_id)
        self.generated_token_logprobs.append(logprob)

    def finish(self, reason: FinishReason) -> None:
        self.finish_reason = reason

    def output_token_ids(self, *, include_prompt: bool) -> list[int]:
        if include_prompt:
            return self.token_ids
        return list(self.generated_token_ids)
