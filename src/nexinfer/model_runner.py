from __future__ import annotations

from collections.abc import Sequence as SequenceCollection
from dataclasses import dataclass
from typing import Any

from nexinfer.errors import ConfigurationError
from nexinfer.protocols import DecodeInput, DecodeState, PrefillInput
from nexinfer.sampling import Sampler
from nexinfer.sequence import Sequence as RunnerSequence


@dataclass(frozen=True, slots=True)
class PreparedPrefillBatch:
    """Flattened prefill metadata for a model runner."""

    input_ids: list[int]
    positions: list[int]
    cu_seqlens_q: list[int]
    cu_seqlens_k: list[int]
    max_seqlen_q: int
    max_seqlen_k: int
    slot_mapping: list[int]
    block_tables: list[list[int]]


@dataclass(frozen=True, slots=True)
class PreparedDecodeBatch:
    """Flattened decode metadata for a model runner."""

    input_ids: list[int]
    positions: list[int]
    slot_mapping: list[int]
    context_lengths: list[int]
    block_tables: list[list[int]]


@dataclass(frozen=True, slots=True)
class PreparedSampleBatch:
    """Per-sequence sampling metadata for a model runner."""

    temperatures: list[float]


@dataclass(frozen=True, slots=True)
class ModelRunnerContext:
    """Prepared runner context for the latest prefill or decode batch."""

    is_prefill: bool
    input_ids: list[int]
    positions: list[int]
    slot_mapping: list[int]
    block_tables: list[list[int]]
    cu_seqlens_q: list[int] | None = None
    cu_seqlens_k: list[int] | None = None
    max_seqlen_q: int | None = None
    max_seqlen_k: int | None = None
    context_lengths: list[int] | None = None

    @classmethod
    def from_prefill(cls, batch: PreparedPrefillBatch) -> "ModelRunnerContext":
        return cls(
            is_prefill=True,
            input_ids=batch.input_ids,
            positions=batch.positions,
            slot_mapping=batch.slot_mapping,
            block_tables=batch.block_tables,
            cu_seqlens_q=batch.cu_seqlens_q,
            cu_seqlens_k=batch.cu_seqlens_k,
            max_seqlen_q=batch.max_seqlen_q,
            max_seqlen_k=batch.max_seqlen_k,
        )

    @classmethod
    def from_decode(cls, batch: PreparedDecodeBatch) -> "ModelRunnerContext":
        return cls(
            is_prefill=False,
            input_ids=batch.input_ids,
            positions=batch.positions,
            slot_mapping=batch.slot_mapping,
            block_tables=batch.block_tables,
            context_lengths=batch.context_lengths,
        )


class ModelRunner:
    """Nano-VLLM-style runner orchestration around a logits-producing model."""

    def __init__(
        self,
        model: Any,
        *,
        block_size: int,
        sampler: Sampler | None = None,
    ) -> None:
        _validate_block_size(block_size)
        self.model = model
        self.block_size = block_size
        self.sampler = sampler or Sampler()
        self.last_context: ModelRunnerContext | None = None
        self.last_sample_batch: PreparedSampleBatch | None = None

    def run(
        self,
        sequences: SequenceCollection[RunnerSequence],
        is_prefill: bool,
    ) -> list[int]:
        """Prepare a sequence batch, run the model, and sample next tokens."""

        if not sequences:
            return []

        if is_prefill:
            prepared = prepare_prefill_sequences(sequences, block_size=self.block_size)
            self.last_context = ModelRunnerContext.from_prefill(prepared)
        else:
            prepared = prepare_decode_sequences(sequences, block_size=self.block_size)
            self.last_context = ModelRunnerContext.from_decode(prepared)

        sample_batch = prepare_sample_sequences(sequences)
        self.last_sample_batch = sample_batch
        logits = self.run_model(prepared.input_ids, prepared.positions, is_prefill)
        if len(logits) != len(sequences):
            raise ConfigurationError("model must return one logits row per sequence")
        return self.sampler(logits, sample_batch.temperatures)

    def run_model(
        self,
        input_ids: SequenceCollection[int],
        positions: SequenceCollection[int],
        is_prefill: bool,
    ) -> SequenceCollection[SequenceCollection[float]]:
        """Call the wrapped model object to produce logits."""

        run_model = getattr(self.model, "run_model", None)
        if callable(run_model):
            return run_model(input_ids, positions, is_prefill)
        if callable(self.model):
            return self.model(input_ids, positions, is_prefill)
        raise ConfigurationError("model must be callable or expose run_model")


def prepare_prefill_batch(
    inputs: SequenceCollection[PrefillInput],
    *,
    block_size: int,
) -> PreparedPrefillBatch:
    """Prepare Nano-VLLM-style flattened prefill inputs."""

    _validate_block_size(block_size)

    input_ids: list[int] = []
    positions: list[int] = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping: list[int] = []

    for item in inputs:
        start = item.num_cached_tokens
        seqlen_q = _scheduled_token_count(item)
        end = min(start + seqlen_q, len(item.token_ids))
        seqlen_q = max(end - start, 0)
        seqlen_k = end

        input_ids.extend(item.token_ids[start:end])
        positions.extend(range(start, end))
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(max_seqlen_q, seqlen_q)
        max_seqlen_k = max(max_seqlen_k, seqlen_k)
        slot_mapping.extend(
            _slot_mapping(
                block_table=item.block_table,
                block_size=block_size,
                start=start,
                end=end,
            )
        )

    return PreparedPrefillBatch(
        input_ids=input_ids,
        positions=positions,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        block_tables=_padded_block_tables(inputs),
    )


def prepare_decode_batch(
    inputs: SequenceCollection[DecodeInput],
    *,
    block_size: int,
) -> PreparedDecodeBatch:
    """Prepare Nano-VLLM-style flattened decode inputs."""

    _validate_block_size(block_size)

    input_ids: list[int] = []
    positions: list[int] = []
    slot_mapping: list[int] = []
    context_lengths: list[int] = []

    for item in inputs:
        context_length = item.context_length
        if context_length is None:
            context_length = item.state.position + item.num_scheduled_tokens
        if context_length <= 0:
            raise ConfigurationError("context_length must be positive")

        position = context_length - item.num_scheduled_tokens
        input_ids.append(item.token_id)
        positions.append(position)
        context_lengths.append(context_length)
        slot_mapping.extend(
            _slot_mapping(
                block_table=item.block_table,
                block_size=block_size,
                start=position,
                end=position + item.num_scheduled_tokens,
            )
        )

    return PreparedDecodeBatch(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lengths=context_lengths,
        block_tables=_padded_block_tables(inputs),
    )


def prepare_prefill_sequences(
    sequences: SequenceCollection[RunnerSequence],
    *,
    block_size: int,
) -> PreparedPrefillBatch:
    """Prepare flattened prefill metadata from Nano-VLLM-style sequences."""

    return prepare_prefill_batch(
        [
            PrefillInput(
                token_ids=sequence.token_ids,
                num_cached_tokens=sequence.num_cached_tokens,
                num_scheduled_tokens=sequence.num_scheduled_tokens,
                block_table=tuple(sequence.block_table),
            )
            for sequence in sequences
        ],
        block_size=block_size,
    )


def prepare_decode_sequences(
    sequences: SequenceCollection[RunnerSequence],
    *,
    block_size: int,
) -> PreparedDecodeBatch:
    """Prepare flattened decode metadata from Nano-VLLM-style sequences."""

    return prepare_decode_batch(
        [
            DecodeInput(
                token_id=sequence.last_token,
                state=DecodeState(position=len(sequence) - 1),
                block_table=tuple(sequence.block_table),
                context_length=len(sequence),
                num_scheduled_tokens=_decode_token_count(sequence),
            )
            for sequence in sequences
        ],
        block_size=block_size,
    )


def prepare_sample_sequences(
    sequences: SequenceCollection[RunnerSequence],
) -> PreparedSampleBatch:
    """Prepare per-sequence sampling temperatures."""

    return PreparedSampleBatch(
        temperatures=[float(sequence.temperature) for sequence in sequences]
    )


def _scheduled_token_count(item: PrefillInput) -> int:
    count = item.scheduled_token_count
    if count < 0:
        raise ConfigurationError("num_scheduled_tokens must be non-negative")
    if item.num_cached_tokens < 0:
        raise ConfigurationError("num_cached_tokens must be non-negative")
    return count


def _slot_mapping(
    *,
    block_table: SequenceCollection[int],
    block_size: int,
    start: int,
    end: int,
) -> list[int]:
    slots: list[int] = []
    for position in range(start, end):
        if not block_table:
            slots.append(-1)
            continue
        block_index = position // block_size
        if block_index >= len(block_table):
            raise ConfigurationError("block_table is too short for scheduled tokens")
        slots.append(block_table[block_index] * block_size + position % block_size)
    return slots


def _padded_block_tables(
    inputs: SequenceCollection[PrefillInput] | SequenceCollection[DecodeInput],
) -> list[list[int]]:
    max_length = max((len(item.block_table) for item in inputs), default=0)
    if max_length == 0:
        return []
    return [
        [*item.block_table, *([-1] * (max_length - len(item.block_table)))]
        for item in inputs
    ]


def _validate_block_size(block_size: int) -> None:
    if block_size <= 0:
        raise ConfigurationError("block_size must be positive")


def _decode_token_count(sequence: RunnerSequence) -> int:
    if sequence.num_scheduled_tokens < 0:
        raise ConfigurationError("num_scheduled_tokens must be non-negative")
    if sequence.num_scheduled_tokens > 1:
        raise ConfigurationError("decode num_scheduled_tokens must be 1")
    return sequence.num_scheduled_tokens or 1
