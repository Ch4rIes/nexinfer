from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from nexinfer.errors import ConfigurationError
from nexinfer.protocols import DecodeInput, PrefillInput


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


def prepare_prefill_batch(
    inputs: Sequence[PrefillInput],
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
    inputs: Sequence[DecodeInput],
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


def _scheduled_token_count(item: PrefillInput) -> int:
    count = item.scheduled_token_count
    if count < 0:
        raise ConfigurationError("num_scheduled_tokens must be non-negative")
    if item.num_cached_tokens < 0:
        raise ConfigurationError("num_cached_tokens must be non-negative")
    return count


def _slot_mapping(
    *,
    block_table: Sequence[int],
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
    inputs: Sequence[PrefillInput] | Sequence[DecodeInput],
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

