import pytest

from nexinfer import (
    ActiveScheduler,
    GenerationConfig,
    GenerationRequest,
    LLMEngine,
    PrefixKVCacheBlockManager,
    SamplingConfig,
    VocabularyTokenizer,
)
from nexinfer.backends import BigramBackend


def _request(request_id: str, prompt: str, prompt_tokens: int) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt=prompt,
        config=GenerationConfig(sampling=SamplingConfig(temperature=0)),
        metadata={"token_ids": ",".join(str(index) for index in range(prompt_tokens))},
        prompt_token_count=prompt_tokens,
    )


def _engine() -> LLMEngine:
    tokenizer = VocabularyTokenizer(["a", "b", "x", "y", "<eos>"])
    backend = BigramBackend(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("a"): {tokenizer.token_id("b"): 5.0},
            tokenizer.token_id("b"): {tokenizer.token_id("<eos>"): 5.0},
            tokenizer.token_id("x"): {tokenizer.token_id("y"): 5.0},
            tokenizer.token_id("y"): {tokenizer.token_id("<eos>"): 5.0},
        },
    )
    return LLMEngine(backend, tokenizer)


def test_active_scheduler_prioritizes_prefill_before_decode() -> None:
    scheduler = ActiveScheduler(max_num_seqs=2, max_num_batched_tokens=2)
    scheduler.add_request(_request("one", "a", 1))
    scheduler.add_request(_request("two", "x", 1))

    prefill = scheduler.schedule()

    assert prefill.phase == "prefill"
    assert [request.request_id for request in prefill.requests] == ["one", "two"]
    assert prefill.num_tokens == 2

    active = tuple(_engine().start_requests(list(prefill.requests)))
    scheduler.postprocess_prefill(active)

    decode = scheduler.schedule()
    assert decode.phase == "decode"
    assert [active.request_id for active in decode.active_sequences] == ["one", "two"]
    assert decode.num_tokens == -2


def test_active_scheduler_limits_prefill_token_budget() -> None:
    scheduler = ActiveScheduler(max_num_seqs=8, max_num_batched_tokens=2)
    scheduler.add_request(_request("one", "a", 2))
    scheduler.add_request(_request("two", "x", 1))

    batch = scheduler.schedule()

    assert [request.request_id for request in batch.requests] == ["one"]
    assert scheduler.waiting_count == 1


def test_active_scheduler_requeues_unfinished_decode_sequences() -> None:
    scheduler = ActiveScheduler(max_num_seqs=2)
    scheduler.add_request(_request("one", "a", 1))
    engine = _engine()

    prefill = scheduler.schedule()
    active = tuple(engine.start_requests(list(prefill.requests)))
    scheduler.postprocess_prefill(active)
    decode = scheduler.schedule()
    engine.decode_active_batch(list(decode.active_sequences))
    finished = scheduler.postprocess_decode(decode.active_sequences)

    assert finished == ()
    assert scheduler.running_count == 1


def test_active_scheduler_removes_finished_sequences() -> None:
    scheduler = ActiveScheduler(max_num_seqs=2)
    request = GenerationRequest(
        request_id="one",
        prompt="a",
        config=GenerationConfig(
            max_new_tokens=1,
            sampling=SamplingConfig(temperature=0),
        ),
        prompt_token_count=1,
    )
    scheduler.add_request(request)
    engine = _engine()

    prefill = scheduler.schedule()
    active = tuple(engine.start_requests(list(prefill.requests)))
    scheduler.postprocess_prefill(active)
    decode = scheduler.schedule()
    engine.decode_active_batch(list(decode.active_sequences))
    finished = scheduler.postprocess_decode(decode.active_sequences)

    assert [item.request_id for item in finished] == ["one"]
    assert scheduler.is_idle() is True


def test_active_scheduler_rejects_duplicate_request_ids() -> None:
    scheduler = ActiveScheduler(max_num_seqs=2)
    scheduler.add_request(_request("one", "a", 1))

    with pytest.raises(ValueError, match="duplicate"):
        scheduler.add_request(_request("one", "x", 1))


def test_active_scheduler_allocates_blocks_for_prefill() -> None:
    block_manager = PrefixKVCacheBlockManager(num_blocks=4, block_size=2)
    scheduler = ActiveScheduler(max_num_seqs=2, block_manager=block_manager)
    scheduler.add_request(_request("one", "a", 3))

    batch = scheduler.schedule()

    assert [request.request_id for request in batch.requests] == ["one"]
    assert block_manager.allocation("one").block_table == [0, 1]
    assert block_manager.used_blocks == 2


def test_active_scheduler_uses_cached_prefix_tokens_in_prefill_budget() -> None:
    block_manager = PrefixKVCacheBlockManager(num_blocks=4, block_size=2)
    block_manager.allocate("cached", [0, 1, 2])
    block_manager.hash_blocks("cached", [0, 1, 2])
    block_manager.deallocate("cached")
    scheduler = ActiveScheduler(
        max_num_seqs=2,
        max_num_batched_tokens=1,
        block_manager=block_manager,
    )
    scheduler.add_request(_request("one", "a", 3))

    batch = scheduler.schedule()

    assert batch.cached_tokens == 2
    assert batch.num_tokens == 1


def test_active_scheduler_reserves_append_blocks_for_decode() -> None:
    block_manager = PrefixKVCacheBlockManager(num_blocks=2, block_size=1)
    scheduler = ActiveScheduler(max_num_seqs=1, block_manager=block_manager)
    request = GenerationRequest(
        request_id="one",
        prompt="a",
        config=GenerationConfig(
            max_new_tokens=2,
            sampling=SamplingConfig(temperature=0),
        ),
        metadata={"token_ids": "0"},
        prompt_token_count=1,
    )
    scheduler.add_request(request)
    engine = _engine()

    prefill = scheduler.schedule()
    active = tuple(engine.start_requests(list(prefill.requests)))
    scheduler.postprocess_prefill(active)
    decode = scheduler.schedule()

    assert [active.request_id for active in decode.active_sequences] == ["one"]
    assert block_manager.allocation("one").token_count == 2


def test_active_scheduler_deallocates_finished_blocks() -> None:
    block_manager = PrefixKVCacheBlockManager(num_blocks=2, block_size=2)
    scheduler = ActiveScheduler(max_num_seqs=1, block_manager=block_manager)
    request = GenerationRequest(
        request_id="one",
        prompt="a",
        config=GenerationConfig(
            max_new_tokens=1,
            sampling=SamplingConfig(temperature=0),
        ),
        metadata={"token_ids": "0"},
        prompt_token_count=1,
    )
    scheduler.add_request(request)
    engine = _engine()

    prefill = scheduler.schedule()
    active = tuple(engine.start_requests(list(prefill.requests)))
    scheduler.postprocess_prefill(active)
    decode = scheduler.schedule()
    engine.decode_active_batch(list(decode.active_sequences))
    scheduler.postprocess_decode(decode.active_sequences)

    assert block_manager.used_blocks == 0


def test_active_scheduler_cancel_deallocates_running_blocks() -> None:
    block_manager = PrefixKVCacheBlockManager(num_blocks=2, block_size=2)
    scheduler = ActiveScheduler(max_num_seqs=1, block_manager=block_manager)
    scheduler.add_request(_request("one", "a", 1))
    engine = _engine()

    prefill = scheduler.schedule()
    active = tuple(engine.start_requests(list(prefill.requests)))
    scheduler.postprocess_prefill(active)

    assert block_manager.used_blocks == 1
    assert scheduler.cancel("one") is True
    assert block_manager.used_blocks == 0
