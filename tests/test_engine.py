import pytest

from nexinfer import (
    DecodeState,
    GenerationConfig,
    LLMEngine,
    RequestQueue,
    SamplingConfig,
    StreamChunk,
    VocabularyTokenizer,
)
from nexinfer.backends import BigramBackend
from nexinfer.protocols import ModelOutput


def test_generates_until_stop_token() -> None:
    tokenizer = VocabularyTokenizer(["hello", "world", "<eos>"], eos_token="<eos>")
    backend = BigramBackend(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("hello"): {tokenizer.token_id("world"): 5.0},
            tokenizer.token_id("world"): {tokenizer.eos_token_id: 5.0},
        },
    )
    engine = LLMEngine(backend, tokenizer)

    text = engine.generate(
        "hello",
        GenerationConfig(
            max_new_tokens=8,
            sampling=SamplingConfig(temperature=0),
            stop_token_ids=(tokenizer.eos_token_id,),
        ),
    )

    assert text == "world"


def test_can_include_prompt_and_stop_token() -> None:
    tokenizer = VocabularyTokenizer(["a", "b", "<eos>"], eos_token="<eos>")
    backend = BigramBackend(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("a"): {tokenizer.token_id("b"): 5.0},
            tokenizer.token_id("b"): {tokenizer.eos_token_id: 5.0},
        },
    )
    engine = LLMEngine(backend, tokenizer)

    ids = engine.generate_token_ids(
        "a",
        GenerationConfig(
            max_new_tokens=3,
            sampling=SamplingConfig(temperature=0),
            stop_token_ids=(tokenizer.eos_token_id,),
            include_prompt=True,
            include_stop_token=True,
        ),
    )

    assert ids == [
        tokenizer.token_id("a"),
        tokenizer.token_id("b"),
        tokenizer.eos_token_id,
    ]


def test_validates_backend_vocab_size() -> None:
    class BrokenBackend:
        vocab_size = 3

        def begin(self, input_ids: list[int]) -> ModelOutput:
            return ModelOutput([1.0, 2.0], DecodeState(position=0))

        def step(self, token_id: int, state: DecodeState) -> ModelOutput:
            return ModelOutput([1.0, 2.0], DecodeState(position=1))

    tokenizer = VocabularyTokenizer(["a"])
    engine = LLMEngine(BrokenBackend(), tokenizer)

    with pytest.raises(ValueError, match="expected vocab size"):
        engine.generate("a")


def test_validates_backend_decode_state() -> None:
    class BrokenBackend:
        vocab_size = 2

        def begin(self, input_ids: list[int]) -> ModelOutput:
            return ModelOutput([1.0, 2.0], None)  # type: ignore[arg-type]

        def step(self, token_id: int, state: DecodeState) -> ModelOutput:
            return ModelOutput([1.0, 2.0], DecodeState(position=1))

    tokenizer = VocabularyTokenizer(["a"])
    engine = LLMEngine(BrokenBackend(), tokenizer)

    with pytest.raises(ValueError, match="DecodeState"):
        engine.generate("a")


def test_complete_returns_structured_result() -> None:
    tokenizer = VocabularyTokenizer(["hello", "world", "<eos>"], eos_token="<eos>")
    backend = BigramBackend(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("hello"): {tokenizer.token_id("world"): 5.0},
            tokenizer.token_id("world"): {tokenizer.eos_token_id: 5.0},
        },
    )
    engine = LLMEngine(backend, tokenizer)

    result = engine.complete(
        "hello",
        GenerationConfig(
            max_new_tokens=8,
            sampling=SamplingConfig(temperature=0),
            stop_token_ids=(tokenizer.eos_token_id,),
        ),
    )

    assert result.text == "world"
    assert result.finish_reason == "stop"
    assert result.prompt_token_ids == [tokenizer.token_id("hello")]
    assert result.generated_token_ids == [tokenizer.token_id("world")]
    assert result.generated_token_logprobs == [0.0]
    assert result.usage.prompt_tokens == 1
    assert result.usage.completion_tokens == 1
    assert result.usage.total_tokens == 2


def test_stream_chunks_include_finish_reason_on_last_token() -> None:
    tokenizer = VocabularyTokenizer(["a", "b", "<eos>"], eos_token="<eos>")
    backend = BigramBackend(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("a"): {tokenizer.token_id("b"): 5.0},
            tokenizer.token_id("b"): {tokenizer.eos_token_id: 5.0},
        },
    )
    engine = LLMEngine(backend, tokenizer)

    chunks = list(
        engine.stream_chunks(
            "a",
            GenerationConfig(
                max_new_tokens=3,
                sampling=SamplingConfig(temperature=0),
                stop_token_ids=(tokenizer.eos_token_id,),
            ),
        )
    )

    assert chunks == [
        StreamChunk(
            text="b",
            token_id=tokenizer.token_id("b"),
            index=0,
            logprob=0.0,
            finish_reason="stop",
        )
    ]


def test_complete_batch_returns_one_result_per_prompt() -> None:
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

    results = engine.complete_batch(
        ["a", "c"],
        GenerationConfig(
            max_new_tokens=3,
            sampling=SamplingConfig(temperature=0),
            stop_token_ids=(tokenizer.eos_token_id,),
        ),
    )

    assert [result.text for result in results] == ["b", ""]
    assert [result.finish_reason for result in results] == ["stop", "stop"]


def test_complete_requests_uses_per_request_config() -> None:
    tokenizer = VocabularyTokenizer(["a", "b", "<eos>"], eos_token="<eos>")
    backend = BigramBackend(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("a"): {tokenizer.token_id("b"): 5.0},
            tokenizer.token_id("b"): {tokenizer.eos_token_id: 5.0},
        },
    )
    engine = LLMEngine(backend, tokenizer)
    queue = RequestQueue()
    queue.submit(
        "a",
        GenerationConfig(
            max_new_tokens=3,
            sampling=SamplingConfig(temperature=0),
            stop_token_ids=(tokenizer.eos_token_id,),
        ),
    )

    results = engine.complete_requests(list(queue.schedule(max_requests=1).requests))

    assert [result.text for result in results] == ["b"]
