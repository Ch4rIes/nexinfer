import pytest

from nexinfer import (
    ConfigurationError,
    DecodeInput,
    DecodeState,
    PrefillInput,
    SamplingParams,
    Sequence,
    prepare_decode_batch,
    prepare_decode_sequences,
    prepare_prefill_batch,
    prepare_prefill_sequences,
    prepare_sample_sequences,
)


def test_prepare_prefill_batch_flattens_scheduled_prompt_tokens() -> None:
    prepared = prepare_prefill_batch(
        [
            PrefillInput(
                token_ids=[10, 11, 12, 13],
                num_cached_tokens=2,
                num_scheduled_tokens=2,
                block_table=(3, 4),
            ),
            PrefillInput(
                token_ids=[20, 21],
                num_cached_tokens=0,
                num_scheduled_tokens=1,
                block_table=(5,),
            ),
        ],
        block_size=2,
    )

    assert prepared.input_ids == [12, 13, 20]
    assert prepared.positions == [2, 3, 0]
    assert prepared.cu_seqlens_q == [0, 2, 3]
    assert prepared.cu_seqlens_k == [0, 4, 5]
    assert prepared.max_seqlen_q == 2
    assert prepared.max_seqlen_k == 4
    assert prepared.slot_mapping == [8, 9, 10]
    assert prepared.block_tables == [[3, 4], [5, -1]]


def test_prepare_prefill_batch_uses_remaining_tokens_by_default() -> None:
    prepared = prepare_prefill_batch(
        [
            PrefillInput(
                token_ids=[1, 2, 3],
                num_cached_tokens=1,
                block_table=(7, 8),
            )
        ],
        block_size=2,
    )

    assert prepared.input_ids == [2, 3]
    assert prepared.positions == [1, 2]
    assert prepared.slot_mapping == [15, 16]


def test_prepare_decode_batch_flattens_token_positions_and_context() -> None:
    prepared = prepare_decode_batch(
        [
            DecodeInput(
                token_id=99,
                state=DecodeState(position=3),
                block_table=(2, 4),
                context_length=4,
            ),
            DecodeInput(
                token_id=77,
                state=DecodeState(position=2),
                block_table=(6,),
                context_length=1,
            ),
        ],
        block_size=2,
    )

    assert prepared.input_ids == [99, 77]
    assert prepared.positions == [3, 0]
    assert prepared.context_lengths == [4, 1]
    assert prepared.slot_mapping == [9, 12]
    assert prepared.block_tables == [[2, 4], [6, -1]]


def test_prepare_decode_batch_infers_context_length_from_state() -> None:
    prepared = prepare_decode_batch(
        [
            DecodeInput(
                token_id=5,
                state=DecodeState(position=4),
                block_table=(1, 2, 3),
            )
        ],
        block_size=2,
    )

    assert prepared.positions == [4]
    assert prepared.context_lengths == [5]
    assert prepared.slot_mapping == [6]


def test_model_runner_preparation_validates_block_tables() -> None:
    with pytest.raises(ConfigurationError, match="too short"):
        prepare_prefill_batch(
            [
                PrefillInput(
                    token_ids=[1, 2, 3],
                    num_cached_tokens=0,
                    num_scheduled_tokens=3,
                    block_table=(0,),
                )
            ],
            block_size=2,
        )


def test_model_runner_preparation_allows_missing_block_table() -> None:
    prefill = prepare_prefill_batch(
        [PrefillInput(token_ids=[1, 2])],
        block_size=2,
    )
    decode = prepare_decode_batch(
        [DecodeInput(token_id=3, state=DecodeState(position=2))],
        block_size=2,
    )

    assert prefill.slot_mapping == [-1, -1]
    assert prefill.block_tables == []
    assert decode.slot_mapping == [-1]
    assert decode.block_tables == []


def test_prepare_prefill_sequences_uses_sequence_scheduling_metadata() -> None:
    first = Sequence([10, 11, 12, 13])
    first.num_cached_tokens = 2
    first.num_scheduled_tokens = 2
    first.block_table.extend([3, 4])
    second = Sequence([20, 21])
    second.num_cached_tokens = 0
    second.num_scheduled_tokens = 1
    second.block_table.extend([5])

    prepared = prepare_prefill_sequences([first, second], block_size=2)

    assert prepared.input_ids == [12, 13, 20]
    assert prepared.positions == [2, 3, 0]
    assert prepared.cu_seqlens_q == [0, 2, 3]
    assert prepared.cu_seqlens_k == [0, 4, 5]
    assert prepared.max_seqlen_q == 2
    assert prepared.max_seqlen_k == 4
    assert prepared.slot_mapping == [8, 9, 10]
    assert prepared.block_tables == [[3, 4], [5, -1]]


def test_prepare_decode_sequences_uses_last_token_and_context_length() -> None:
    first = Sequence([10, 11, 12])
    first.append_token(13)
    first.num_scheduled_tokens = 1
    first.block_table.extend([3, 4])
    second = Sequence([20])
    second.num_scheduled_tokens = 1
    second.block_table.extend([5])

    prepared = prepare_decode_sequences([first, second], block_size=2)

    assert prepared.input_ids == [13, 20]
    assert prepared.positions == [3, 0]
    assert prepared.context_lengths == [4, 1]
    assert prepared.slot_mapping == [9, 10]
    assert prepared.block_tables == [[3, 4], [5, -1]]


def test_prepare_decode_sequences_defaults_to_one_scheduled_token() -> None:
    sequence = Sequence([1, 2, 3])
    sequence.block_table.extend([7, 8])

    prepared = prepare_decode_sequences([sequence], block_size=2)

    assert prepared.input_ids == [3]
    assert prepared.positions == [2]
    assert prepared.context_lengths == [3]
    assert prepared.slot_mapping == [16]


def test_prepare_decode_sequences_validates_scheduled_token_count() -> None:
    sequence = Sequence([1, 2, 3])
    sequence.num_scheduled_tokens = 2
    sequence.block_table.extend([7, 8])

    with pytest.raises(ConfigurationError, match="decode num_scheduled_tokens"):
        prepare_decode_sequences([sequence], block_size=2)


def test_prepare_sample_sequences_returns_temperatures() -> None:
    first = Sequence([1], SamplingParams(temperature=0.5))
    second = Sequence([2], SamplingParams(temperature=1.25))

    prepared = prepare_sample_sequences([first, second])

    assert prepared.temperatures == [0.5, 1.25]
