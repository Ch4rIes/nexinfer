import pytest

from nexinfer import ConfigurationError, LLM, SamplingParams, VocabularyTokenizer
from nexinfer.backends import BigramBackend


def test_llm_generate_returns_nano_vllm_style_outputs() -> None:
    tokenizer = VocabularyTokenizer(["hello", "world", "<eos>"], eos_token="<eos>")
    backend = BigramBackend(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("hello"): {tokenizer.token_id("world"): 100.0},
            tokenizer.token_id("world"): {tokenizer.eos_token_id: 100.0},
        },
    )
    llm = LLM(backend=backend, tokenizer=tokenizer)

    outputs = llm.generate(
        ["hello"],
        SamplingParams(temperature=0.01, max_tokens=4),
        use_tqdm=False,
    )

    assert outputs == [
        {"text": "world", "token_ids": [tokenizer.token_id("world")]}
    ]


def test_llm_generate_accepts_token_id_prompts() -> None:
    tokenizer = VocabularyTokenizer(["hello", "world", "<eos>"], eos_token="<eos>")
    backend = BigramBackend(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("hello"): {tokenizer.token_id("world"): 100.0},
            tokenizer.token_id("world"): {tokenizer.eos_token_id: 100.0},
        },
    )
    llm = LLM(backend=backend, tokenizer=tokenizer)

    outputs = llm.generate(
        [[tokenizer.token_id("hello")]],
        SamplingParams(temperature=0.01, max_tokens=4),
        use_tqdm=False,
    )

    assert outputs[0]["text"] == "world"
    assert outputs[0]["token_ids"] == [tokenizer.token_id("world")]


def test_llm_generate_accepts_per_prompt_sampling_params() -> None:
    tokenizer = VocabularyTokenizer(["a", "b", "c", "<eos>"], eos_token="<eos>")
    backend = BigramBackend(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("a"): {tokenizer.token_id("b"): 100.0},
            tokenizer.token_id("b"): {tokenizer.eos_token_id: 100.0},
            tokenizer.token_id("c"): {tokenizer.eos_token_id: 100.0},
        },
    )
    llm = LLM(backend=backend, tokenizer=tokenizer)

    outputs = llm.generate(
        ["a", "c"],
        [
            SamplingParams(temperature=0.01, max_tokens=4),
            SamplingParams(temperature=0.01, max_tokens=1),
        ],
        use_tqdm=False,
    )

    assert outputs == [
        {"text": "b", "token_ids": [tokenizer.token_id("b")]},
        {"text": "", "token_ids": []},
    ]


def test_sampling_params_can_ignore_eos() -> None:
    tokenizer = VocabularyTokenizer(["hello", "world", "<eos>"], eos_token="<eos>")
    backend = BigramBackend(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("hello"): {tokenizer.eos_token_id: 100.0},
            tokenizer.eos_token_id: {tokenizer.token_id("world"): 100.0},
        },
    )
    llm = LLM(backend=backend, tokenizer=tokenizer)

    outputs = llm.generate(
        ["hello"],
        SamplingParams(temperature=0.01, max_tokens=2, ignore_eos=True),
        use_tqdm=False,
    )

    assert outputs == [
        {
            "text": "<eos> world",
            "token_ids": [tokenizer.eos_token_id, tokenizer.token_id("world")],
        }
    ]


def test_sampling_params_match_nano_vllm_greedy_validation() -> None:
    with pytest.raises(ConfigurationError, match="greedy sampling"):
        SamplingParams(temperature=0.0)


def test_llm_rejects_unsupported_tensor_parallel_size() -> None:
    tokenizer = VocabularyTokenizer(["a"])
    backend = BigramBackend(vocab_size=len(tokenizer))

    with pytest.raises(ConfigurationError, match="tensor_parallel_size"):
        LLM(backend=backend, tokenizer=tokenizer, tensor_parallel_size=2)
