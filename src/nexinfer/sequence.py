from __future__ import annotations

from enum import Enum, auto
from itertools import count

from nexinfer.errors import ConfigurationError
from nexinfer.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """Nano-VLLM-style lifecycle status for a scheduled sequence."""

    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """Nano-VLLM-compatible sequence payload for scheduling and runners."""

    block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams | None = None,
    ) -> None:
        if not token_ids:
            raise ConfigurationError("token_ids must not be empty")

        sampling_params = sampling_params or SamplingParams()
        self.seq_id = next(self.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = list(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table: list[int] = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, key: int | slice) -> int | list[int]:
        return self.token_ids[key]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self) -> list[int]:
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, index: int) -> list[int]:
        if not 0 <= index < self.num_blocks:
            raise IndexError("block index out of range")
        start = index * self.block_size
        return self.token_ids[start : start + self.block_size]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self) -> tuple[int, int, int, int, list[int], int | list[int]]:
        last_state = self.token_ids if self.is_prefill else self.last_token
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.block_table,
            last_state,
        )

    def __setstate__(
        self,
        state: tuple[int, int, int, int, list[int], int | list[int]],
    ) -> None:
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.block_table,
            last_state,
        ) = state
        self.seq_id = next(self.counter)
        self.status = SequenceStatus.WAITING
        self.is_prefill = isinstance(last_state, list)
        self.temperature = 1.0
        self.max_tokens = 0
        self.ignore_eos = False

        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state

    def __repr__(self) -> str:
        return (
            "Sequence("
            f"seq_id={self.seq_id}, "
            f"num_tokens={self.num_tokens}, "
            f"status={self.status.name}"
            ")"
        )


def reset_sequence_counter(value: int = 0) -> None:
    """Reset the Sequence id counter for deterministic tests and tools."""

    if value < 0:
        raise ConfigurationError("counter value must be non-negative")
    Sequence.counter = count(value)
