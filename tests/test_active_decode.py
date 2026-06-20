from nexinfer import (
    GenerationConfig,
    GenerationRequest,
    LLMEngine,
    PrefixKVCacheBlockManager,
    SamplingConfig,
)
from nexinfer.backends import BigramBackend
from nexinfer.protocols import DecodeInput, DecodeState, ModelOutput, PrefillInput
from nexinfer.scheduler import ActiveScheduler
from nexinfer.tokenizer import VocabularyTokenizer


def test_engine_decodes_active_sequence_one_step_at_a_time() -> None:
    tokenizer = VocabularyTokenizer(["a", "b", "c", "<eos>"], eos_token="<eos>")
    backend = BigramBackend(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("a"): {tokenizer.token_id("b"): 5.0},
            tokenizer.token_id("b"): {tokenizer.token_id("c"): 5.0},
            tokenizer.token_id("c"): {tokenizer.eos_token_id: 5.0},
        },
    )
    engine = LLMEngine(backend, tokenizer)
    request = GenerationRequest(
        request_id="one",
        prompt="a",
        config=GenerationConfig(
            max_new_tokens=8,
            sampling=SamplingConfig(temperature=0),
            stop_token_ids=(tokenizer.eos_token_id,),
        ),
    )

    active = engine.start_request(request)
    assert active.request_id == "one"
    assert active.sequence.generated_token_ids == []

    engine.decode_one(active)
    assert active.sequence.generated_token_ids == [tokenizer.token_id("b")]
    assert active.is_finished is False

    engine.decode_one(active)
    assert active.sequence.generated_token_ids == [
        tokenizer.token_id("b"),
        tokenizer.token_id("c"),
    ]
    assert active.is_finished is False

    engine.decode_one(active)
    assert active.sequence.finish_reason == "stop"
    assert active.output is None


def test_active_sequence_finishes_when_effective_limit_is_zero() -> None:
    tokenizer = VocabularyTokenizer(["a"])
    backend = BigramBackend(vocab_size=len(tokenizer))
    engine = LLMEngine(backend, tokenizer)
    request = GenerationRequest(
        request_id="done",
        prompt="a",
        config=GenerationConfig(max_new_tokens=0),
    )

    active = engine.start_request(request)

    assert active.is_finished is True
    assert active.sequence.finish_reason == "length"


def test_interleaved_requests_decode_round_robin() -> None:
    tokenizer = VocabularyTokenizer(["a", "b", "c", "x", "y", "z", "<eos>"])
    eos_id = tokenizer.token_id("<eos>")

    class TracingBackend(BigramBackend):
        def __init__(self) -> None:
            super().__init__(
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
            self.steps: list[int] = []

        def step(self, token_id: int, state: DecodeState) -> ModelOutput:
            self.steps.append(token_id)
            return super().step(token_id, state)

    backend = TracingBackend()
    engine = LLMEngine(backend, tokenizer)
    config = GenerationConfig(
        max_new_tokens=8,
        sampling=SamplingConfig(temperature=0),
        stop_token_ids=(eos_id,),
    )
    requests = [
        GenerationRequest("one", "a", config),
        GenerationRequest("two", "x", config),
    ]

    results = engine.complete_requests_interleaved(requests)

    assert [result.text for result in results] == ["b c", "y z"]
    assert backend.steps == [
        tokenizer.token_id("b"),
        tokenizer.token_id("y"),
        tokenizer.token_id("c"),
        tokenizer.token_id("z"),
    ]


def test_interleaved_requests_use_batched_backend_methods() -> None:
    tokenizer = VocabularyTokenizer(["a", "b", "x", "y", "<eos>"])
    eos_id = tokenizer.token_id("<eos>")

    class CountingBackend(BigramBackend):
        def __init__(self) -> None:
            super().__init__(
                vocab_size=len(tokenizer),
                transitions={
                    tokenizer.token_id("a"): {tokenizer.token_id("b"): 5.0},
                    tokenizer.token_id("b"): {eos_id: 5.0},
                    tokenizer.token_id("x"): {tokenizer.token_id("y"): 5.0},
                    tokenizer.token_id("y"): {eos_id: 5.0},
                },
            )
            self.begin_batch_calls = 0
            self.step_batch_sizes: list[int] = []

        def begin_batch(self, input_ids_batch):
            self.begin_batch_calls += 1
            return super().begin_batch(input_ids_batch)

        def step_batch(self, inputs):
            self.step_batch_sizes.append(len(inputs))
            return super().step_batch(inputs)

    backend = CountingBackend()
    engine = LLMEngine(backend, tokenizer)
    config = GenerationConfig(
        max_new_tokens=8,
        sampling=SamplingConfig(temperature=0),
        stop_token_ids=(eos_id,),
    )

    results = engine.complete_requests_interleaved(
        [
            GenerationRequest("one", "a", config),
            GenerationRequest("two", "x", config),
        ]
    )

    assert [result.text for result in results] == ["b", "y"]
    assert backend.begin_batch_calls == 1
    assert backend.step_batch_sizes == [2]


def test_scheduled_metadata_reaches_backend_inputs() -> None:
    tokenizer = VocabularyTokenizer(["a", "b", "c", "d"])

    class InspectingBackend(BigramBackend):
        def __init__(self) -> None:
            super().__init__(
                vocab_size=len(tokenizer),
                transitions={
                    tokenizer.token_id("c"): {tokenizer.token_id("d"): 5.0},
                },
            )
            self.prefill_inputs: list[PrefillInput] = []
            self.decode_inputs: list[DecodeInput] = []

        def begin_batch(self, inputs):
            self.prefill_inputs.extend(inputs)
            return super().begin_batch(inputs)

        def step_batch(self, inputs):
            self.decode_inputs.extend(inputs)
            return super().step_batch(inputs)

    backend = InspectingBackend()
    engine = LLMEngine(backend, tokenizer)
    block_manager = PrefixKVCacheBlockManager(num_blocks=4, block_size=2)
    scheduler = ActiveScheduler(
        max_num_seqs=1,
        max_num_batched_tokens=2,
        block_manager=block_manager,
    )
    request = GenerationRequest(
        request_id="one",
        prompt="a b c",
        config=GenerationConfig(
            max_new_tokens=2,
            sampling=SamplingConfig(temperature=0),
        ),
        metadata={
            "token_ids": ",".join(str(token_id) for token_id in tokenizer.encode("a b c"))
        },
        prompt_token_count=3,
    )
    scheduler.add_request(request)

    first = scheduler.schedule()
    assert first.requests == ()
    second = scheduler.schedule()
    scheduled = {item.request_id: item for item in second.scheduled_sequences}
    active = tuple(engine.start_requests(list(second.requests), scheduled=scheduled))
    scheduler.postprocess_prefill(active)
    decode = scheduler.schedule()
    engine.decode_active_batch(list(decode.active_sequences))

    assert backend.prefill_inputs[-1].num_cached_tokens == 2
    assert backend.prefill_inputs[-1].num_scheduled_tokens == 1
    assert tuple(backend.prefill_inputs[-1].block_table) == (0, 1)
    assert backend.decode_inputs[-1].num_scheduled_tokens == 1
    assert backend.decode_inputs[-1].context_length == 4
    assert tuple(backend.decode_inputs[-1].block_table) == (0, 1)
