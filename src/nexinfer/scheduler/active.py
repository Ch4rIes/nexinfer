from __future__ import annotations

import random
from dataclasses import dataclass

from nexinfer.protocols import ModelOutput
from nexinfer.scheduler.request import GenerationRequest
from nexinfer.state import SequenceState


@dataclass(slots=True)
class ActiveSequence:
    """A request that has been admitted and is partway through decoding."""

    request: GenerationRequest
    sequence: SequenceState
    output: ModelOutput | None
    rng: random.Random
    max_new_tokens: int
    stop_token_ids: set[int]
    block_table: list[int] | None = None
    num_cached_tokens: int = 0
    num_scheduled_tokens: int = 0
    is_prefill: bool = True

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def is_finished(self) -> bool:
        return self.sequence.finish_reason is not None

    @property
    def can_decode(self) -> bool:
        return (
            not self.is_finished
            and self.output is not None
            and self.sequence.completion_tokens < self.max_new_tokens
        )
