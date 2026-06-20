import pickle

import pytest

from nexinfer import (
    ConfigurationError,
    SamplingParams,
    Sequence,
    SequenceStatus,
    reset_sequence_counter,
)


def test_sequence_tracks_prompt_completion_and_sampling_state() -> None:
    reset_sequence_counter()
    sequence = Sequence(
        [10, 11, 12],
        SamplingParams(temperature=0.5, max_tokens=4, ignore_eos=True),
    )

    assert sequence.seq_id == 0
    assert sequence.status == SequenceStatus.WAITING
    assert sequence.is_finished is False
    assert len(sequence) == 3
    assert sequence[0] == 10
    assert sequence[:2] == [10, 11]
    assert sequence.last_token == 12
    assert sequence.num_tokens == 3
    assert sequence.num_prompt_tokens == 3
    assert sequence.num_completion_tokens == 0
    assert sequence.prompt_token_ids == [10, 11, 12]
    assert sequence.completion_token_ids == []
    assert sequence.temperature == 0.5
    assert sequence.max_tokens == 4
    assert sequence.ignore_eos is True

    sequence.append_token(13)

    assert sequence.last_token == 13
    assert sequence.num_tokens == 4
    assert sequence.num_completion_tokens == 1
    assert sequence.completion_token_ids == [13]


def test_sequence_computes_block_views() -> None:
    reset_sequence_counter()
    old_block_size = Sequence.block_size
    Sequence.block_size = 2
    try:
        sequence = Sequence([1, 2, 3, 4, 5])

        assert sequence.num_blocks == 3
        assert sequence.last_block_num_tokens == 1
        assert sequence.block(0) == [1, 2]
        assert sequence.block(1) == [3, 4]
        assert sequence.block(2) == [5]
        with pytest.raises(IndexError, match="block index"):
            sequence.block(3)
    finally:
        Sequence.block_size = old_block_size


def test_sequence_pickle_state_keeps_prefill_token_ids() -> None:
    reset_sequence_counter()
    sequence = Sequence([1, 2, 3])
    sequence.block_table.extend([7, 8])
    sequence.num_cached_tokens = 2
    sequence.num_scheduled_tokens = 1

    restored = pickle.loads(pickle.dumps(sequence))

    assert restored.num_tokens == 3
    assert restored.num_prompt_tokens == 3
    assert restored.num_cached_tokens == 2
    assert restored.num_scheduled_tokens == 1
    assert restored.block_table == [7, 8]
    assert restored.is_prefill is True
    assert restored.token_ids == [1, 2, 3]
    assert restored.last_token == 3


def test_sequence_pickle_state_keeps_only_last_token_for_decode() -> None:
    reset_sequence_counter()
    sequence = Sequence([1, 2])
    sequence.append_token(3)
    sequence.block_table.extend([4, 5])
    sequence.num_cached_tokens = 2
    sequence.num_scheduled_tokens = 1
    sequence.is_prefill = False

    restored = pickle.loads(pickle.dumps(sequence))

    assert restored.num_tokens == 3
    assert restored.num_prompt_tokens == 2
    assert restored.num_cached_tokens == 2
    assert restored.num_scheduled_tokens == 1
    assert restored.block_table == [4, 5]
    assert restored.is_prefill is False
    assert restored.token_ids == []
    assert restored.last_token == 3


def test_sequence_status_finished_property_and_counter_reset() -> None:
    reset_sequence_counter(10)
    sequence = Sequence([1])

    assert sequence.seq_id == 10
    assert sequence.is_finished is False

    sequence.status = SequenceStatus.FINISHED

    assert sequence.is_finished is True


def test_sequence_rejects_empty_prompt_and_invalid_counter_reset() -> None:
    with pytest.raises(ConfigurationError, match="token_ids"):
        Sequence([])

    with pytest.raises(ConfigurationError, match="counter"):
        reset_sequence_counter(-1)
