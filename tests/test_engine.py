import pytest

from nexinfer import (
    GenerationConfig,
    LLMEngine,
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
            return ModelOutput([1.0, 2.0])

        def step(self, token_id: int, state: object) -> ModelOutput:
            return ModelOutput([1.0, 2.0])

    tokenizer = VocabularyTokenizer(["a"])
    engine = LLMEngine(BrokenBackend(), tokenizer)

    with pytest.raises(ValueError, match="expected vocab size"):
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
            finish_reason="stop",
        )
    ]
