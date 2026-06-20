from nexinfer import SamplingParams, Scheduler, Sequence, SequenceStatus


def test_scheduler_prefills_decodes_and_deallocates_finished_sequence() -> None:
    scheduler = Scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=4,
        eos=99,
        num_kvcache_blocks=4,
        kvcache_block_size=2,
    )
    seq = Sequence([1, 2], SamplingParams(temperature=1.0, max_tokens=2))
    scheduler.add(seq)

    prefill, is_prefill = scheduler.schedule()

    assert prefill == [seq]
    assert is_prefill is True
    assert seq.status == SequenceStatus.RUNNING
    assert seq.block_table == [0]
    assert seq.num_scheduled_tokens == 2

    scheduler.postprocess(prefill, [3], is_prefill)

    assert seq.token_ids == [1, 2, 3]
    assert seq.status == SequenceStatus.RUNNING

    decode, is_prefill = scheduler.schedule()

    assert decode == [seq]
    assert is_prefill is False
    assert seq.is_prefill is False
    assert seq.block_table == [0, 1]
    assert seq.num_scheduled_tokens == 1

    scheduler.postprocess(decode, [99], is_prefill)

    assert seq.is_finished is True
    assert scheduler.is_finished() is True
    assert scheduler.block_manager.used_blocks == 0


def test_scheduler_chunks_prefill_until_prompt_is_complete() -> None:
    scheduler = Scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=2,
        eos=99,
        num_kvcache_blocks=4,
        kvcache_block_size=2,
    )
    seq = Sequence([1, 2, 3, 4, 5], SamplingParams(temperature=1.0))
    scheduler.add(seq)

    first, is_prefill = scheduler.schedule()
    first_metadata = (first[0].num_cached_tokens, first[0].num_scheduled_tokens)
    scheduler.postprocess(first, [8], is_prefill)
    second, is_prefill = scheduler.schedule()
    second_metadata = (second[0].num_cached_tokens, second[0].num_scheduled_tokens)
    scheduler.postprocess(second, [8], is_prefill)
    third, is_prefill = scheduler.schedule()
    third_metadata = (third[0].num_cached_tokens, third[0].num_scheduled_tokens)
    scheduler.postprocess(third, [7], is_prefill)

    assert [first_metadata, second_metadata, third_metadata] == [
        (0, 2),
        (2, 2),
        (4, 1),
    ]
    assert seq.token_ids == [1, 2, 3, 4, 5, 7]
    assert seq.num_cached_tokens == 5
    assert seq.status == SequenceStatus.RUNNING
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == [seq]


def test_scheduler_reuses_deallocated_cached_prefix_blocks() -> None:
    scheduler = Scheduler(
        max_num_seqs=1,
        max_num_batched_tokens=8,
        eos=99,
        num_kvcache_blocks=4,
        kvcache_block_size=2,
    )
    first = Sequence([1, 2, 3, 4, 5], SamplingParams(temperature=1.0, max_tokens=1))
    scheduler.add(first)
    prefill, is_prefill = scheduler.schedule()
    prefix_blocks = tuple(first.block_table[:2])
    scheduler.postprocess(prefill, [99], is_prefill)

    second = Sequence([1, 2, 3, 4, 9], SamplingParams(temperature=1.0))
    scheduler.add(second)
    cached_prefill, is_prefill = scheduler.schedule()

    assert is_prefill is True
    assert cached_prefill == [second]
    assert second.num_cached_tokens == 4
    assert second.num_scheduled_tokens == 1
    assert tuple(second.block_table[:2]) == prefix_blocks


def test_scheduler_preempts_running_sequence_to_free_decode_block() -> None:
    scheduler = Scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=4,
        eos=99,
        num_kvcache_blocks=2,
        kvcache_block_size=2,
    )
    first = Sequence([1, 2], SamplingParams(temperature=1.0, max_tokens=2))
    second = Sequence([3, 4], SamplingParams(temperature=1.0, max_tokens=2))
    scheduler.add(first)
    scheduler.add(second)

    prefill, is_prefill = scheduler.schedule()
    scheduler.postprocess(prefill, [5, 6], is_prefill)
    decode, is_prefill = scheduler.schedule()

    assert is_prefill is False
    assert decode == [first]
    assert list(scheduler.waiting) == [second]
    assert list(scheduler.running) == [first]
    assert second.status == SequenceStatus.WAITING
    assert second.block_table == []
    assert scheduler.block_manager.used_blocks == 2
