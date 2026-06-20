import random
from collections.abc import Mapping, Sequence

import pytest

from nexinfer import (
    ConfigurationError,
    LLM,
    LLMConfig,
    ModelRunner,
    ModelOutput,
    NanoLLMEngine,
    PrefillInput,
    SamplingParams,
    Sampler,
    VocabularyTokenizer,
    get_context,
    reset_sequence_counter,
)
from nexinfer.backends import BigramBackend


class CountingBigramBackend(BigramBackend):
    def __init__(
        self,
        *,
        vocab_size: int,
        transitions: Mapping[int | None, Mapping[int, float]],
    ) -> None:
        super().__init__(vocab_size=vocab_size, transitions=transitions)
        self.begin_batch_sizes: list[int] = []

    def begin_batch(
        self,
        inputs: Sequence[PrefillInput | Sequence[int]],
    ) -> list[ModelOutput]:
        batch = list(inputs)
        self.begin_batch_sizes.append(len(batch))
        return super().begin_batch(batch)


class ClosingBigramBackend(BigramBackend):
    def __init__(
        self,
        *,
        vocab_size: int,
        transitions: Mapping[int | None, Mapping[int, float]] | None = None,
    ) -> None:
        super().__init__(vocab_size=vocab_size, transitions=transitions)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class FixedRaceRng(random.Random):
    def __init__(self) -> None:
        super().__init__()

    def expovariate(self, lambd: float) -> float:
        assert lambd == 1.0
        return 1.0


class ContextualRunnerModel:
    def __init__(
        self,
        *,
        vocab_size: int,
        transitions: Mapping[int, int],
    ) -> None:
        self.vocab_size = vocab_size
        self.transitions = dict(transitions)
        self.close_calls = 0

    def run_model(
        self,
        input_ids: list[int],
        positions: list[int],
        is_prefill: bool,
    ) -> list[list[float]]:
        token_ids = input_ids if not is_prefill else self._prefill_token_ids(input_ids)
        return [self._logits_for(token_id) for token_id in token_ids]

    def close(self) -> None:
        self.close_calls += 1

    def _prefill_token_ids(self, input_ids: list[int]) -> list[int]:
        context = get_context()
        boundaries = context.cu_seqlens_q or [0, len(input_ids)]
        return [
            input_ids[end - 1]
            for _, end in zip(boundaries[:-1], boundaries[1:], strict=True)
        ]

    def _logits_for(self, token_id: int) -> list[float]:
        logits = [-100.0] * self.vocab_size
        logits[self.transitions[token_id]] = 100.0
        return logits


def _runner_llm(
    model: ContextualRunnerModel,
    tokenizer: VocabularyTokenizer,
    *,
    max_num_seqs: int = 2,
) -> LLM:
    return LLM(
        model_runner=ModelRunner(
            model,
            block_size=256,
            sampler=Sampler(FixedRaceRng()),
        ),
        tokenizer=tokenizer,
        max_num_seqs=max_num_seqs,
        num_kvcache_blocks=8,
    )


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


def test_llm_generate_uses_queue_scheduler_batching() -> None:
    tokenizer = VocabularyTokenizer(["a", "b", "x", "y", "<eos>"], eos_token="<eos>")
    backend = CountingBigramBackend(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("a"): {tokenizer.token_id("b"): 100.0},
            tokenizer.token_id("b"): {tokenizer.eos_token_id: 100.0},
            tokenizer.token_id("x"): {tokenizer.token_id("y"): 100.0},
            tokenizer.token_id("y"): {tokenizer.eos_token_id: 100.0},
        },
    )
    llm = LLM(backend=backend, tokenizer=tokenizer, max_num_seqs=2)

    outputs = llm.generate(
        ["a", "x"],
        SamplingParams(temperature=0.01, max_tokens=4),
        use_tqdm=False,
    )

    assert outputs == [
        {"text": "b", "token_ids": [tokenizer.token_id("b")]},
        {"text": "y", "token_ids": [tokenizer.token_id("y")]},
    ]
    assert backend.begin_batch_sizes == [2]
    assert llm.add_request("a", SamplingParams(temperature=0.01, max_tokens=1)) == 2


def test_llm_generate_can_use_model_runner_path() -> None:
    reset_sequence_counter()
    tokenizer = VocabularyTokenizer(["a", "b", "x", "y"], eos_token=None)
    model = ContextualRunnerModel(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("a"): tokenizer.token_id("b"),
            tokenizer.token_id("x"): tokenizer.token_id("y"),
        },
    )
    llm = _runner_llm(model, tokenizer, max_num_seqs=2)

    outputs = llm.generate(
        ["a", "x"],
        SamplingParams(temperature=0.01, max_tokens=1),
        use_tqdm=False,
    )

    assert isinstance(llm.engine, NanoLLMEngine)
    assert outputs == [
        {"text": "b", "token_ids": [tokenizer.token_id("b")]},
        {"text": "y", "token_ids": [tokenizer.token_id("y")]},
    ]


def test_llm_model_runner_path_step_matches_nano_loop() -> None:
    reset_sequence_counter()
    tokenizer = VocabularyTokenizer(["a", "b", "<eos>"], eos_token="<eos>")
    model = ContextualRunnerModel(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("a"): tokenizer.token_id("b"),
            tokenizer.token_id("b"): tokenizer.eos_token_id,
        },
    )
    llm = _runner_llm(model, tokenizer, max_num_seqs=1)

    request_id = llm.add_request(
        "a",
        SamplingParams(temperature=0.01, max_tokens=2),
    )

    assert request_id == 0
    assert llm.step() == ([], 1)
    assert llm.step() == (
        [(request_id, [tokenizer.token_id("b"), tokenizer.eos_token_id])],
        -1,
    )
    assert llm.is_finished() is True


def test_llm_model_runner_path_validation_and_close() -> None:
    tokenizer = VocabularyTokenizer(["a", "b"], eos_token=None)
    model = ContextualRunnerModel(
        vocab_size=len(tokenizer),
        transitions={tokenizer.token_id("a"): tokenizer.token_id("b")},
    )

    with pytest.raises(ConfigurationError, match="tokenizer"):
        LLM(model_runner=ModelRunner(model, block_size=256))
    with pytest.raises(ConfigurationError, match="backend"):
        LLM(
            backend=BigramBackend(vocab_size=len(tokenizer)),
            model_runner=ModelRunner(model, block_size=256),
            tokenizer=tokenizer,
        )

    llm = _runner_llm(model, tokenizer)
    llm.close()

    assert model.close_calls == 1
    with pytest.raises(ConfigurationError, match="closed"):
        llm.step()


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


def test_llm_exit_closes_backend_and_rejects_later_use() -> None:
    tokenizer = VocabularyTokenizer(["a", "b", "<eos>"], eos_token="<eos>")
    backend = ClosingBigramBackend(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("a"): {tokenizer.token_id("b"): 100.0},
        },
    )
    llm = LLM(backend=backend, tokenizer=tokenizer)

    llm.add_request("a", SamplingParams(temperature=0.01, max_tokens=1))
    assert llm.is_finished() is False

    llm.exit()
    llm.exit()

    assert backend.close_calls == 1
    assert llm.is_finished() is True
    with pytest.raises(ConfigurationError, match="closed"):
        llm.add_request("a", SamplingParams(temperature=0.01, max_tokens=1))
    with pytest.raises(ConfigurationError, match="closed"):
        llm.step()
    with pytest.raises(ConfigurationError, match="closed"):
        llm.generate(["a"], SamplingParams(temperature=0.01, max_tokens=1))


def test_llm_context_manager_closes_backend() -> None:
    tokenizer = VocabularyTokenizer(["a", "b", "<eos>"], eos_token="<eos>")
    backend = ClosingBigramBackend(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("a"): {tokenizer.token_id("b"): 100.0},
        },
    )

    with LLM(backend=backend, tokenizer=tokenizer) as llm:
        outputs = llm.generate(
            ["a"],
            SamplingParams(temperature=0.01, max_tokens=1),
            use_tqdm=False,
        )

    assert outputs == [{"text": "b", "token_ids": [tokenizer.token_id("b")]}]
    assert backend.close_calls == 1
