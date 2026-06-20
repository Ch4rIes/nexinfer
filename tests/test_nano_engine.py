import random

import pytest

from nexinfer import (
    ConfigurationError,
    ModelRunner,
    NanoLLMEngine,
    SamplingParams,
    Sampler,
    Scheduler,
    VocabularyTokenizer,
    get_context,
    reset_sequence_counter,
)


class FixedRaceRng(random.Random):
    def __init__(self) -> None:
        super().__init__()

    def expovariate(self, lambd: float) -> float:
        assert lambd == 1.0
        return 1.0


class ContextualBigramModel:
    def __init__(
        self,
        *,
        vocab_size: int,
        transitions: dict[int, int],
    ) -> None:
        self.vocab_size = vocab_size
        self.transitions = transitions
        self.calls: list[tuple[list[int], list[int], bool]] = []

    def run_model(
        self,
        input_ids: list[int],
        positions: list[int],
        is_prefill: bool,
    ) -> list[list[float]]:
        self.calls.append((list(input_ids), list(positions), is_prefill))
        previous_token_ids = self._previous_token_ids(input_ids, is_prefill)
        return [self._logits_for(token_id) for token_id in previous_token_ids]

    def _previous_token_ids(self, input_ids: list[int], is_prefill: bool) -> list[int]:
        if not is_prefill:
            return list(input_ids)

        context = get_context()
        boundaries = context.cu_seqlens_q or [0, len(input_ids)]
        previous_token_ids: list[int] = []
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            if end <= start:
                raise AssertionError("prefill sequence must schedule at least one token")
            previous_token_ids.append(input_ids[end - 1])
        return previous_token_ids

    def _logits_for(self, token_id: int) -> list[float]:
        logits = [-100.0] * self.vocab_size
        logits[self.transitions[token_id]] = 100.0
        return logits


class CallingRunner:
    def __init__(self, runner: ModelRunner) -> None:
        self.runner = runner
        self.calls: list[str] = []

    def call(self, method_name: str, *args: object) -> object:
        self.calls.append(method_name)
        if method_name == "run":
            return self.runner.run(*args)  # type: ignore[arg-type]
        if method_name == "exit":
            return None
        raise AssertionError(f"unexpected runner method: {method_name}")


def test_nano_engine_step_runs_scheduler_and_model_runner_loop() -> None:
    reset_sequence_counter()
    tokenizer = VocabularyTokenizer(["a", "b", "x", "y", "<eos>"], eos_token="<eos>")
    model = ContextualBigramModel(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("a"): tokenizer.token_id("b"),
            tokenizer.token_id("b"): tokenizer.eos_token_id,
            tokenizer.token_id("x"): tokenizer.token_id("y"),
            tokenizer.token_id("y"): tokenizer.eos_token_id,
        },
    )
    engine = _engine(model, tokenizer, max_num_seqs=2)

    first_id = engine.add_request("a", SamplingParams(temperature=0.01, max_tokens=2))
    second_id = engine.add_request("x", SamplingParams(temperature=0.01, max_tokens=2))

    assert (first_id, second_id) == (0, 1)
    assert engine.step() == ([], 2)
    assert model.calls[0] == (
        [tokenizer.token_id("a"), tokenizer.token_id("x")],
        [0, 0],
        True,
    )

    outputs, num_tokens = engine.step()

    assert num_tokens == -2
    assert outputs == [
        (first_id, [tokenizer.token_id("b"), tokenizer.eos_token_id]),
        (second_id, [tokenizer.token_id("y"), tokenizer.eos_token_id]),
    ]
    assert engine.is_finished() is True
    assert engine.step() == ([], 0)


def test_nano_engine_generate_returns_prompt_order_outputs() -> None:
    reset_sequence_counter()
    tokenizer = VocabularyTokenizer(["a", "b", "x", "y", "<eos>"], eos_token="<eos>")
    model = ContextualBigramModel(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("a"): tokenizer.token_id("b"),
            tokenizer.token_id("x"): tokenizer.token_id("y"),
        },
    )
    engine = _engine(model, tokenizer, max_num_seqs=2)

    outputs = engine.generate(
        ["a", "x"],
        SamplingParams(temperature=0.01, max_tokens=1),
        use_tqdm=False,
    )

    assert outputs == [
        {"text": "b", "token_ids": [tokenizer.token_id("b")]},
        {"text": "y", "token_ids": [tokenizer.token_id("y")]},
    ]


def test_nano_engine_uses_runner_call_proxy_and_closes_runner() -> None:
    reset_sequence_counter()
    tokenizer = VocabularyTokenizer(["a", "b"], eos_token=None)
    model = ContextualBigramModel(
        vocab_size=len(tokenizer),
        transitions={tokenizer.token_id("a"): tokenizer.token_id("b")},
    )
    runner = CallingRunner(
        ModelRunner(model, block_size=2, sampler=Sampler(FixedRaceRng()))
    )
    scheduler = Scheduler(
        max_num_seqs=1,
        max_num_batched_tokens=2,
        eos=-1,
        num_kvcache_blocks=2,
        kvcache_block_size=2,
    )
    engine = NanoLLMEngine(runner, tokenizer, scheduler)

    outputs = engine.generate(
        [[tokenizer.token_id("a")]],
        SamplingParams(temperature=0.01, max_tokens=1),
        use_tqdm=False,
    )
    engine.close()

    assert outputs == [{"text": "b", "token_ids": [tokenizer.token_id("b")]}]
    assert runner.calls == ["run", "exit"]
    assert engine.is_finished() is True
    with pytest.raises(ConfigurationError, match="closed"):
        engine.step()


def _engine(
    model: ContextualBigramModel,
    tokenizer: VocabularyTokenizer,
    *,
    max_num_seqs: int,
) -> NanoLLMEngine:
    runner = ModelRunner(model, block_size=2, sampler=Sampler(FixedRaceRng()))
    scheduler = Scheduler(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=2,
        eos=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1,
        num_kvcache_blocks=8,
        kvcache_block_size=2,
    )
    return NanoLLMEngine(runner, tokenizer, scheduler)
