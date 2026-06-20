import pytest

from nexinfer import (
    ConfigurationError,
    LLM,
    LLMConfig,
    SamplingParams,
    VocabularyTokenizer,
)
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


def test_llm_accepts_nano_vllm_config_kwargs() -> None:
    tokenizer = VocabularyTokenizer(["a", "b", "c", "<eos>"], eos_token="<eos>")
    backend = BigramBackend(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("a"): {tokenizer.token_id("b"): 100.0},
            tokenizer.token_id("b"): {tokenizer.token_id("c"): 100.0},
            tokenizer.token_id("c"): {tokenizer.eos_token_id: 100.0},
        },
    )

    llm = LLM(
        backend=backend,
        tokenizer=tokenizer,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        max_model_len=2,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        kvcache_block_size=256,
        num_kvcache_blocks=8,
    )

    assert llm.config.max_num_batched_tokens == 512
    assert llm.config.max_num_seqs == 4
    assert llm.config.max_model_len == 2
    assert llm.config.gpu_memory_utilization == 0.5
    assert llm.config.enforce_eager is True
    assert llm.config.eos == tokenizer.eos_token_id
    assert llm.config.num_kvcache_blocks == 8

    outputs = llm.generate(
        ["a"],
        SamplingParams(temperature=0.01, max_tokens=4),
        use_tqdm=False,
    )

    assert outputs == [{"text": "b", "token_ids": [tokenizer.token_id("b")]}]


def test_llm_accepts_explicit_config_object() -> None:
    tokenizer = VocabularyTokenizer(["a", "b", "<eos>"], eos_token="<eos>")
    backend = BigramBackend(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("a"): {tokenizer.token_id("b"): 100.0},
        },
    )
    config = LLMConfig("toy", max_num_seqs=2, enforce_eager=True)

    llm = LLM(backend=backend, tokenizer=tokenizer, config=config)

    assert llm.config is config
    assert llm.enforce_eager is True
    assert llm.config.eos == tokenizer.eos_token_id


def test_llm_rejects_config_object_with_config_kwargs() -> None:
    tokenizer = VocabularyTokenizer(["a"])
    backend = BigramBackend(vocab_size=len(tokenizer))

    with pytest.raises(ConfigurationError, match="config kwargs"):
        LLM(
            backend=backend,
            tokenizer=tokenizer,
            config=LLMConfig("toy"),
            max_num_seqs=2,
        )


def test_llm_rejects_config_object_with_explicit_tensor_parallel_size() -> None:
    tokenizer = VocabularyTokenizer(["a"])
    backend = BigramBackend(vocab_size=len(tokenizer))

    with pytest.raises(ConfigurationError, match="tensor_parallel_size"):
        LLM(
            backend=backend,
            tokenizer=tokenizer,
            config=LLMConfig("toy"),
            tensor_parallel_size=2,
        )


def test_llm_queue_step_matches_nano_vllm_loop_shape() -> None:
    tokenizer = VocabularyTokenizer(["a", "b", "x", "y", "<eos>"], eos_token="<eos>")
    backend = BigramBackend(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("a"): {tokenizer.token_id("b"): 100.0},
            tokenizer.token_id("b"): {tokenizer.eos_token_id: 100.0},
            tokenizer.token_id("x"): {tokenizer.token_id("y"): 100.0},
            tokenizer.token_id("y"): {tokenizer.eos_token_id: 100.0},
        },
    )
    llm = LLM(backend=backend, tokenizer=tokenizer, max_num_seqs=2)

    first_id = llm.add_request(
        "a",
        SamplingParams(temperature=0.01, max_tokens=4),
    )
    second_id = llm.add_request(
        "x",
        SamplingParams(temperature=0.01, max_tokens=4),
    )

    assert (first_id, second_id) == (0, 1)
    assert llm.is_finished() is False

    outputs: dict[int, list[int]] = {}
    token_counts: list[int] = []
    while not llm.is_finished():
        step_outputs, num_tokens = llm.step()
        token_counts.append(num_tokens)
        outputs.update(step_outputs)

    assert token_counts[0] == 2
    assert -2 in token_counts
    assert outputs == {
        first_id: [tokenizer.token_id("b")],
        second_id: [tokenizer.token_id("y")],
    }
    assert llm.step() == ([], 0)


def test_llm_queue_accepts_token_id_prompt_requests() -> None:
    tokenizer = VocabularyTokenizer(["hello", "world", "<eos>"], eos_token="<eos>")
    backend = BigramBackend(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("hello"): {tokenizer.token_id("world"): 100.0},
            tokenizer.token_id("world"): {tokenizer.eos_token_id: 100.0},
        },
    )
    llm = LLM(backend=backend, tokenizer=tokenizer)

    request_id = llm.add_request(
        [tokenizer.token_id("hello")],
        SamplingParams(temperature=0.01, max_tokens=4),
    )

    outputs: dict[int, list[int]] = {}
    while not llm.is_finished():
        step_outputs, _ = llm.step()
        outputs.update(step_outputs)

    assert outputs == {request_id: [tokenizer.token_id("world")]}
