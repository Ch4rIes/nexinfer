from nexinfer import (
    GenerationConfig,
    InferenceRuntime,
    LLMEngine,
    SamplingConfig,
    VocabularyTokenizer,
)
from nexinfer.backends import BigramBackend


def _runtime() -> InferenceRuntime:
    tokenizer = VocabularyTokenizer(["a", "b", "c", "<eos>"], eos_token="<eos>")
    backend = BigramBackend(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("a"): {tokenizer.token_id("b"): 5.0},
            tokenizer.token_id("b"): {tokenizer.eos_token_id: 5.0},
            tokenizer.token_id("c"): {tokenizer.eos_token_id: 5.0},
        },
    )
    engine = LLMEngine(backend, tokenizer)
    return InferenceRuntime(engine, max_batch_size=1)


def _budgeted_runtime() -> InferenceRuntime:
    tokenizer = VocabularyTokenizer(["a", "b", "c", "<eos>"], eos_token="<eos>")
    backend = BigramBackend(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("a"): {tokenizer.token_id("b"): 5.0},
            tokenizer.token_id("b"): {tokenizer.eos_token_id: 5.0},
            tokenizer.token_id("c"): {tokenizer.eos_token_id: 5.0},
        },
    )
    engine = LLMEngine(backend, tokenizer)
    return InferenceRuntime(engine, max_batch_size=8, max_batch_prompt_tokens=1)


def test_runtime_executes_one_scheduled_batch() -> None:
    runtime = _runtime()
    config = GenerationConfig(
        max_new_tokens=3,
        sampling=SamplingConfig(temperature=0),
        stop_token_ids=(3,),
    )
    runtime.submit("a", config, request_id="one")
    runtime.submit("c", config, request_id="two")

    completed = runtime.run_once()

    assert [item.request_id for item in completed] == ["one"]
    assert [item.result.text for item in completed] == ["b"]
    assert runtime.pending_requests == 1
    assert runtime.stats.batches == 1
    assert runtime.stats.requests == 1
    assert runtime.stats.prompt_tokens == 1
    assert runtime.stats.completion_tokens == 1


def test_runtime_preserves_submitted_metadata() -> None:
    runtime = _runtime()

    request = runtime.submit("a", request_id="one", metadata={"trace": "abc"})

    assert request.metadata == {"trace": "abc"}


def test_runtime_estimates_prompt_tokens_on_submit() -> None:
    runtime = _runtime()

    request = runtime.submit("a b", request_id="one")

    assert request.prompt_token_count == 2


def test_runtime_cancel_removes_pending_request() -> None:
    runtime = _runtime()
    runtime.submit("a", request_id="one")

    assert runtime.cancel("one") is True
    assert runtime.pending_requests == 0
    assert runtime.run_once() == ()


def test_runtime_can_drain_until_idle() -> None:
    runtime = _runtime()
    config = GenerationConfig(
        max_new_tokens=3,
        sampling=SamplingConfig(temperature=0),
        stop_token_ids=(3,),
    )
    runtime.submit("a", config, request_id="one")
    runtime.submit("c", config, request_id="two")

    completed = runtime.run_until_idle()

    assert [item.request_id for item in completed] == ["one", "two"]
    assert [item.result.text for item in completed] == ["b", ""]
    assert runtime.pending_requests == 0


def test_runtime_drain_can_stop_after_max_batches() -> None:
    runtime = _runtime()
    runtime.submit("a", request_id="one")
    runtime.submit("a", request_id="two")

    completed = runtime.run_until_idle(max_batches=1)

    assert [item.request_id for item in completed] == ["one"]
    assert runtime.pending_requests == 1


def test_runtime_honors_prompt_token_batch_budget() -> None:
    runtime = _budgeted_runtime()
    config = GenerationConfig(
        max_new_tokens=3,
        sampling=SamplingConfig(temperature=0),
        stop_token_ids=(3,),
    )
    runtime.submit("a", config, request_id="one")
    runtime.submit("c", config, request_id="two")

    completed = runtime.run_once()

    assert [item.request_id for item in completed] == ["one"]
    assert runtime.pending_requests == 1


def test_runtime_can_use_interleaved_decode_strategy() -> None:
    tokenizer = VocabularyTokenizer(["a", "b", "c", "x", "y", "z", "<eos>"])
    eos_id = tokenizer.token_id("<eos>")
    backend = BigramBackend(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("a"): {tokenizer.token_id("b"): 5.0},
            tokenizer.token_id("b"): {tokenizer.token_id("c"): 5.0},
            tokenizer.token_id("c"): {eos_id: 5.0},
            tokenizer.token_id("x"): {tokenizer.token_id("y"): 5.0},
            tokenizer.token_id("y"): {tokenizer.token_id("z"): 5.0},
            tokenizer.token_id("z"): {eos_id: 5.0},
        },
    )
    engine = LLMEngine(backend, tokenizer)
    runtime = InferenceRuntime(
        engine,
        max_batch_size=2,
        decode_strategy="interleaved",
    )
    config = GenerationConfig(
        max_new_tokens=8,
        sampling=SamplingConfig(temperature=0),
        stop_token_ids=(eos_id,),
    )
    runtime.submit("a", config, request_id="one")
    runtime.submit("x", config, request_id="two")

    completed = runtime.run_once()

    assert [item.request_id for item in completed] == ["one", "two"]
    assert [item.result.text for item in completed] == ["b c", "y z"]
