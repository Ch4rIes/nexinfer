import random
from types import SimpleNamespace

import pytest

from nexinfer import (
    ConfigurationError,
    Context,
    CUDAGraphCapturePlan,
    DecodeInput,
    DecodeState,
    KVCacheLayout,
    KVCacheSlice,
    LLMConfig,
    ModelRunner,
    ModelRunnerCommand,
    ModelRunnerGroup,
    PrefillInput,
    SamplingParams,
    Sampler,
    Sequence,
    get_context,
    prepare_block_tables,
    prepare_decode_batch,
    prepare_decode_sequences,
    prepare_prefill_batch,
    prepare_prefill_sequences,
    prepare_sample_sequences,
    reset_context,
    reset_sequence_counter,
)


class FixedRaceRng(random.Random):
    def __init__(self) -> None:
        super().__init__()

    def expovariate(self, lambd: float) -> float:
        assert lambd == 1.0
        return 1.0


class FakeModel:
    def __init__(self, logits: list[list[float]]) -> None:
        self.logits = logits
        self.calls: list[tuple[list[int], list[int], bool]] = []

    def run_model(
        self,
        input_ids: list[int],
        positions: list[int],
        is_prefill: bool,
    ) -> list[list[float]]:
        self.calls.append((list(input_ids), list(positions), is_prefill))
        return self.logits


class ContextCapturingModel(FakeModel):
    def __init__(self, logits: list[list[float]]) -> None:
        super().__init__(logits)
        self.contexts: list[Context] = []

    def run_model(
        self,
        input_ids: list[int],
        positions: list[int],
        is_prefill: bool,
    ) -> list[list[float]]:
        self.contexts.append(get_context())
        return super().run_model(input_ids, positions, is_prefill)


class ClosingModel(FakeModel):
    def __init__(self) -> None:
        super().__init__([])
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class NanoStyleModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[int], list[int]]] = []
        self.compute_logits_calls: list[list[int]] = []

    def __call__(self, input_ids: list[int], positions: list[int]) -> list[int]:
        self.calls.append((list(input_ids), list(positions)))
        return [token_id + position for token_id, position in zip(input_ids, positions)]

    def compute_logits(self, hidden_states: list[int]) -> list[list[float]]:
        self.compute_logits_calls.append(list(hidden_states))
        return [[0.0, 10.0] for _ in hidden_states]


class KVCacheModule:
    def __init__(self) -> None:
        self.k_cache: object | None = None
        self.v_cache: object | None = None


class KVCacheModel(FakeModel):
    def __init__(self, num_modules: int) -> None:
        super().__init__([])
        self.kv_modules = [KVCacheModule() for _ in range(num_modules)]

    def modules(self) -> list[object]:
        return [object(), *self.kv_modules]


class MutatingWorkerRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[int], bool]] = []
        self.close_calls = 0

    def call(self, method_name: str, *args: object) -> None:
        if method_name == "run":
            sequences, is_prefill = args
            sequence = sequences[0]  # type: ignore[index]
            self.calls.append((method_name, list(sequence.token_ids), is_prefill))
            sequence.append_token(999)
            return None
        if method_name == "exit":
            self.close_calls += 1
            return None
        raise AssertionError(f"unexpected method: {method_name}")


class RecordingPrimaryRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[list[int], bool]] = []
        self.close_calls = 0

    def run(self, sequences: list[Sequence], is_prefill: bool) -> list[int]:
        self.calls.append((list(sequences[0].token_ids), is_prefill))
        return [7]

    def exit(self) -> None:
        self.close_calls += 1


def test_prepare_prefill_batch_flattens_scheduled_prompt_tokens() -> None:
    prepared = prepare_prefill_batch(
        [
            PrefillInput(
                token_ids=[10, 11, 12, 13],
                num_cached_tokens=2,
                num_scheduled_tokens=2,
                block_table=(3, 4),
            ),
            PrefillInput(
                token_ids=[20, 21],
                num_cached_tokens=0,
                num_scheduled_tokens=1,
                block_table=(5,),
            ),
        ],
        block_size=2,
    )

    assert prepared.input_ids == [12, 13, 20]
    assert prepared.positions == [2, 3, 0]
    assert prepared.cu_seqlens_q == [0, 2, 3]
    assert prepared.cu_seqlens_k == [0, 4, 5]
    assert prepared.max_seqlen_q == 2
    assert prepared.max_seqlen_k == 4
    assert prepared.slot_mapping == [8, 9, 10]
    assert prepared.block_tables == [[3, 4], [5, -1]]


def test_prepare_prefill_batch_uses_remaining_tokens_by_default() -> None:
    prepared = prepare_prefill_batch(
        [
            PrefillInput(
                token_ids=[1, 2, 3],
                num_cached_tokens=1,
                block_table=(7, 8),
            )
        ],
        block_size=2,
    )

    assert prepared.input_ids == [2, 3]
    assert prepared.positions == [1, 2]
    assert prepared.slot_mapping == [15, 16]


def test_prepare_decode_batch_flattens_token_positions_and_context() -> None:
    prepared = prepare_decode_batch(
        [
            DecodeInput(
                token_id=99,
                state=DecodeState(position=3),
                block_table=(2, 4),
                context_length=4,
            ),
            DecodeInput(
                token_id=77,
                state=DecodeState(position=2),
                block_table=(6,),
                context_length=1,
            ),
        ],
        block_size=2,
    )

    assert prepared.input_ids == [99, 77]
    assert prepared.positions == [3, 0]
    assert prepared.context_lengths == [4, 1]
    assert prepared.slot_mapping == [9, 12]
    assert prepared.block_tables == [[2, 4], [6, -1]]


def test_prepare_decode_batch_infers_context_length_from_state() -> None:
    prepared = prepare_decode_batch(
        [
            DecodeInput(
                token_id=5,
                state=DecodeState(position=4),
                block_table=(1, 2, 3),
            )
        ],
        block_size=2,
    )

    assert prepared.positions == [4]
    assert prepared.context_lengths == [5]
    assert prepared.slot_mapping == [6]


def test_model_runner_preparation_validates_block_tables() -> None:
    with pytest.raises(ConfigurationError, match="too short"):
        prepare_prefill_batch(
            [
                PrefillInput(
                    token_ids=[1, 2, 3],
                    num_cached_tokens=0,
                    num_scheduled_tokens=3,
                    block_table=(0,),
                )
            ],
            block_size=2,
        )


def test_model_runner_preparation_allows_missing_block_table() -> None:
    prefill = prepare_prefill_batch(
        [PrefillInput(token_ids=[1, 2])],
        block_size=2,
    )
    decode = prepare_decode_batch(
        [DecodeInput(token_id=3, state=DecodeState(position=2))],
        block_size=2,
    )

    assert prefill.slot_mapping == [-1, -1]
    assert prefill.block_tables == []
    assert decode.slot_mapping == [-1]
    assert decode.block_tables == []


def test_prepare_prefill_sequences_uses_sequence_scheduling_metadata() -> None:
    first = Sequence([10, 11, 12, 13])
    first.num_cached_tokens = 2
    first.num_scheduled_tokens = 2
    first.block_table.extend([3, 4])
    second = Sequence([20, 21])
    second.num_cached_tokens = 0
    second.num_scheduled_tokens = 1
    second.block_table.extend([5])

    prepared = prepare_prefill_sequences([first, second], block_size=2)

    assert prepared.input_ids == [12, 13, 20]
    assert prepared.positions == [2, 3, 0]
    assert prepared.cu_seqlens_q == [0, 2, 3]
    assert prepared.cu_seqlens_k == [0, 4, 5]
    assert prepared.max_seqlen_q == 2
    assert prepared.max_seqlen_k == 4
    assert prepared.slot_mapping == [8, 9, 10]
    assert prepared.block_tables == [[3, 4], [5, -1]]


def test_prepare_decode_sequences_uses_last_token_and_context_length() -> None:
    first = Sequence([10, 11, 12])
    first.append_token(13)
    first.num_scheduled_tokens = 1
    first.block_table.extend([3, 4])
    second = Sequence([20])
    second.num_scheduled_tokens = 1
    second.block_table.extend([5])

    prepared = prepare_decode_sequences([first, second], block_size=2)

    assert prepared.input_ids == [13, 20]
    assert prepared.positions == [3, 0]
    assert prepared.context_lengths == [4, 1]
    assert prepared.slot_mapping == [9, 10]
    assert prepared.block_tables == [[3, 4], [5, -1]]


def test_prepare_decode_sequences_defaults_to_one_scheduled_token() -> None:
    sequence = Sequence([1, 2, 3])
    sequence.block_table.extend([7, 8])

    prepared = prepare_decode_sequences([sequence], block_size=2)

    assert prepared.input_ids == [3]
    assert prepared.positions == [2]
    assert prepared.context_lengths == [3]
    assert prepared.slot_mapping == [16]


def test_prepare_decode_sequences_validates_scheduled_token_count() -> None:
    sequence = Sequence([1, 2, 3])
    sequence.num_scheduled_tokens = 2
    sequence.block_table.extend([7, 8])

    with pytest.raises(ConfigurationError, match="decode num_scheduled_tokens"):
        prepare_decode_sequences([sequence], block_size=2)


def test_prepare_sample_sequences_returns_temperatures() -> None:
    first = Sequence([1], SamplingParams(temperature=0.5))
    second = Sequence([2], SamplingParams(temperature=1.25))

    prepared = prepare_sample_sequences([first, second])

    assert prepared.temperatures == [0.5, 1.25]


def test_prepare_block_tables_pads_sequence_tables() -> None:
    first = Sequence([1, 2])
    first.block_table.extend([3, 4])
    second = Sequence([5])
    second.block_table.extend([9])

    assert prepare_block_tables([first, second]) == [[3, 4], [9, -1]]


def test_model_runner_prepare_block_tables_uses_sequence_metadata() -> None:
    first = Sequence([1, 2])
    first.block_table.extend([3, 4, 5])
    second = Sequence([6])
    second.block_table.extend([7])
    runner = ModelRunner(FakeModel([]), block_size=2)

    assert runner.prepare_block_tables([first, second]) == [[3, 4, 5], [7, -1, -1]]


def test_model_runner_prepare_methods_set_context_and_sampling_metadata() -> None:
    sequence = Sequence([10, 11, 12], SamplingParams(temperature=0.5))
    sequence.num_cached_tokens = 1
    sequence.num_scheduled_tokens = 2
    sequence.block_table.extend([3, 4])
    runner = ModelRunner(FakeModel([]), block_size=2)

    input_ids, positions = runner.prepare_prefill([sequence])
    sample_batch = runner.prepare_sample([sequence])

    assert (input_ids, positions) == ([11, 12], [1, 2])
    assert runner.last_context is not None
    assert runner.last_context.is_prefill is True
    assert get_context().is_prefill is True
    assert sample_batch.temperatures == [0.5]
    assert runner.last_sample_batch == sample_batch
    reset_context()


def test_model_runner_run_prefill_prepares_context_and_samples_tokens() -> None:
    sequence = Sequence([10, 11, 12], SamplingParams(temperature=0.5))
    sequence.num_cached_tokens = 1
    sequence.num_scheduled_tokens = 2
    sequence.block_table.extend([3, 4])
    model = FakeModel([[0.0, 10.0]])
    runner = ModelRunner(
        model,
        block_size=2,
        sampler=Sampler(FixedRaceRng()),
    )

    token_ids = runner.run([sequence], is_prefill=True)

    assert token_ids == [1]
    assert model.calls == [([11, 12], [1, 2], True)]
    assert runner.last_sample_batch is not None
    assert runner.last_sample_batch.temperatures == [0.5]
    assert runner.last_context is not None
    assert runner.last_context.is_prefill is True
    assert runner.last_context.input_ids == [11, 12]
    assert runner.last_context.positions == [1, 2]
    assert runner.last_context.slot_mapping == [7, 8]


def test_model_runner_sets_and_resets_prefill_attention_context() -> None:
    sequence = Sequence([10, 11, 12], SamplingParams(temperature=1.0))
    sequence.num_cached_tokens = 1
    sequence.num_scheduled_tokens = 2
    sequence.block_table.extend([3, 4])
    model = ContextCapturingModel([[0.0, 10.0]])
    runner = ModelRunner(
        model,
        block_size=2,
        sampler=Sampler(FixedRaceRng()),
    )

    runner.run([sequence], is_prefill=True)

    context = model.contexts[0]
    assert context.is_prefill is True
    assert context.cu_seqlens_q == [0, 2]
    assert context.cu_seqlens_k == [0, 3]
    assert context.max_seqlen_q == 2
    assert context.max_seqlen_k == 3
    assert context.slot_mapping == [7, 8]
    assert context.block_tables == [[3, 4]]
    assert get_context() == Context()


def test_model_runner_run_decode_prepares_context_and_samples_tokens() -> None:
    first = Sequence([10, 11], SamplingParams(temperature=1.0))
    first.append_token(12)
    first.num_scheduled_tokens = 1
    first.block_table.extend([3, 4])
    second = Sequence([20], SamplingParams(temperature=2.0))
    second.num_scheduled_tokens = 1
    second.block_table.extend([5])
    model = FakeModel([[10.0, 0.0], [0.0, 10.0]])
    runner = ModelRunner(
        model,
        block_size=2,
        sampler=Sampler(FixedRaceRng()),
    )

    token_ids = runner.run([first, second], is_prefill=False)

    assert token_ids == [0, 1]
    assert model.calls == [([12, 20], [2, 0], False)]
    assert runner.last_sample_batch is not None
    assert runner.last_sample_batch.temperatures == [1.0, 2.0]
    assert runner.last_context is not None
    assert runner.last_context.is_prefill is False
    assert runner.last_context.input_ids == [12, 20]
    assert runner.last_context.positions == [2, 0]
    assert runner.last_context.context_lengths == [3, 1]


def test_model_runner_sets_and_resets_decode_attention_context() -> None:
    sequence = Sequence([10, 11], SamplingParams(temperature=1.0))
    sequence.append_token(12)
    sequence.num_scheduled_tokens = 1
    sequence.block_table.extend([3, 4])
    model = ContextCapturingModel([[10.0, 0.0]])
    runner = ModelRunner(
        model,
        block_size=2,
        sampler=Sampler(FixedRaceRng()),
    )

    runner.run([sequence], is_prefill=False)

    context = model.contexts[0]
    assert context.is_prefill is False
    assert context.cu_seqlens_q is None
    assert context.cu_seqlens_k is None
    assert context.slot_mapping == [8]
    assert context.context_lens == [3]
    assert context.block_tables == [[3, 4]]
    assert get_context() == Context()


def test_model_runner_accepts_callable_model() -> None:
    calls: list[tuple[list[int], list[int], bool]] = []

    def model(
        input_ids: list[int],
        positions: list[int],
        is_prefill: bool,
    ) -> list[list[float]]:
        calls.append((list(input_ids), list(positions), is_prefill))
        return [[0.0, 10.0]]

    sequence = Sequence([1], SamplingParams(temperature=1.0))
    sequence.num_scheduled_tokens = 1
    runner = ModelRunner(
        model,
        block_size=2,
        sampler=Sampler(FixedRaceRng()),
    )

    assert runner.run([sequence], is_prefill=True) == [1]
    assert calls == [([1], [0], True)]


def test_model_runner_accepts_nano_style_model_compute_logits_contract() -> None:
    sequence = Sequence([10, 11], SamplingParams(temperature=1.0))
    sequence.num_scheduled_tokens = 1
    sequence.block_table.extend([3])
    model = NanoStyleModel()
    runner = ModelRunner(
        model,
        block_size=2,
        sampler=Sampler(FixedRaceRng()),
    )

    assert runner.run([sequence], is_prefill=True) == [1]
    assert model.calls == [([10], [0])]
    assert model.compute_logits_calls == [[10]]


def test_model_runner_call_dispatches_methods_by_name() -> None:
    sequence = Sequence([1], SamplingParams(temperature=1.0))
    sequence.num_scheduled_tokens = 1
    runner = ModelRunner(
        FakeModel([[0.0, 10.0]]),
        block_size=2,
        sampler=Sampler(FixedRaceRng()),
    )

    assert runner.call("run", [sequence], True) == [1]


def test_model_runner_warmup_uses_configured_synthetic_prefill_batch() -> None:
    model = FakeModel([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    runner = ModelRunner(
        model,
        block_size=2,
        sampler=Sampler(FixedRaceRng()),
        config=LLMConfig(
            "tiny",
            max_num_batched_tokens=6,
            max_model_len=2,
            max_num_seqs=4,
        ),
    )

    runner.call("warmup_model")

    assert model.calls == [([0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 1], True)]
    assert runner.last_context is not None
    assert runner.last_context.max_seqlen_q == 2
    assert runner.last_context.max_seqlen_k == 2
    assert runner.last_sample_batch is not None
    assert runner.last_sample_batch.temperatures == [1.0, 1.0, 1.0]


def test_model_runner_warmup_requires_config() -> None:
    runner = ModelRunner(FakeModel([]), block_size=2)

    with pytest.raises(ConfigurationError, match="config"):
        runner.warmup_model()


def test_model_runner_allocate_kv_cache_attaches_layer_slices() -> None:
    model = KVCacheModel(num_modules=2)
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=4,
            hidden_size=16,
            num_attention_heads=8,
        ),
        num_kvcache_blocks=3,
        tensor_parallel_size=2,
    )
    runner = ModelRunner(model, block_size=5, config=config)

    kv_cache = runner.allocate_kv_cache()

    assert isinstance(kv_cache, KVCacheLayout)
    assert runner.kv_cache is kv_cache
    assert runner.kv_cache_shape == (2, 2, 3, 5, 2, 2)
    assert model.kv_modules[0].k_cache == KVCacheSlice("k", 0, (3, 5, 2, 2))
    assert model.kv_modules[0].v_cache == KVCacheSlice("v", 0, (3, 5, 2, 2))
    assert model.kv_modules[1].k_cache == KVCacheSlice("k", 1, (3, 5, 2, 2))
    assert model.kv_modules[1].v_cache == KVCacheSlice("v", 1, (3, 5, 2, 2))


def test_model_runner_allocate_kv_cache_requires_config_and_hf_metadata() -> None:
    runner = ModelRunner(FakeModel([]), block_size=2)

    with pytest.raises(ConfigurationError, match="config"):
        runner.allocate_kv_cache()
    with pytest.raises(ConfigurationError, match="hf_config"):
        runner.allocate_kv_cache(SimpleNamespace(num_kvcache_blocks=1))


def test_model_runner_allocate_kv_cache_uses_explicit_head_dim() -> None:
    runner = ModelRunner(KVCacheModel(num_modules=1), block_size=4)
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            num_hidden_layers=1,
            num_key_value_heads=2,
            head_dim=7,
        ),
        num_kvcache_blocks=3,
        tensor_parallel_size=1,
    )

    runner.allocate_kv_cache(config)

    assert runner.kv_cache_shape == (2, 1, 3, 4, 2, 7)


def test_model_runner_capture_cudagraph_records_buffer_plan() -> None:
    config = SimpleNamespace(
        max_num_seqs=40,
        max_model_len=33,
        hf_config=SimpleNamespace(hidden_size=16),
    )
    runner = ModelRunner(FakeModel([]), block_size=8, config=config)

    plan = runner.capture_cudagraph()

    assert isinstance(plan, CUDAGraphCapturePlan)
    assert plan.batch_sizes == [1, 2, 4, 8, 16, 32]
    assert plan.max_batch_size == 40
    assert plan.max_num_blocks == 5
    assert plan.hidden_size == 16
    assert plan.buffer_shapes == {
        "input_ids": (40,),
        "positions": (40,),
        "slot_mapping": (40,),
        "context_lens": (40,),
        "block_tables": (40, 5),
        "outputs": (40, 16),
    }
    assert runner.graph_bs == plan.batch_sizes
    assert runner.graph_capture_plan == plan


def test_model_runner_capture_cudagraph_caps_max_batch_size() -> None:
    runner = ModelRunner(FakeModel([]), block_size=16)

    plan = runner.capture_cudagraph(
        SimpleNamespace(
            max_num_seqs=600,
            max_model_len=17,
        )
    )

    assert plan.max_batch_size == 512
    assert plan.batch_sizes[-1] == 512
    assert plan.max_num_blocks == 2
    assert plan.hidden_size is None
    assert "outputs" not in plan.buffer_shapes


def test_model_runner_capture_cudagraph_requires_config() -> None:
    runner = ModelRunner(FakeModel([]), block_size=2)

    with pytest.raises(ConfigurationError, match="config"):
        runner.capture_cudagraph()


def test_model_runner_call_rejects_unknown_method() -> None:
    runner = ModelRunner(FakeModel([]), block_size=2)

    with pytest.raises(ConfigurationError, match="unknown model runner method"):
        runner.call("missing")


def test_model_runner_exit_closes_wrapped_model() -> None:
    model = ClosingModel()
    runner = ModelRunner(model, block_size=2)

    runner.call("exit")
    runner.close()

    assert model.close_calls == 2


def test_model_runner_validates_model_logits_row_count() -> None:
    runner = ModelRunner(
        FakeModel([[0.0, 1.0]]),
        block_size=2,
        sampler=Sampler(FixedRaceRng()),
    )
    first = Sequence([1], SamplingParams(temperature=1.0))
    first.num_scheduled_tokens = 1
    second = Sequence([2], SamplingParams(temperature=1.0))
    second.num_scheduled_tokens = 1

    with pytest.raises(ConfigurationError, match="one logits row"):
        runner.run([first, second], is_prefill=True)


def test_model_runner_rank_worker_runs_without_sampling() -> None:
    sequence = Sequence([1], SamplingParams(temperature=0.5))
    sequence.num_scheduled_tokens = 1
    model = FakeModel([[0.0, 10.0]])
    runner = ModelRunner(
        model,
        block_size=2,
        sampler=Sampler(FixedRaceRng()),
        rank=1,
        world_size=2,
    )

    assert runner.run([sequence], is_prefill=True) is None
    assert model.calls == [([1], [0], True)]
    assert runner.last_sample_batch is None


def test_model_runner_validates_rank_metadata() -> None:
    with pytest.raises(ConfigurationError, match="world_size"):
        ModelRunner(FakeModel([]), block_size=2, world_size=0)

    with pytest.raises(ConfigurationError, match="rank"):
        ModelRunner(FakeModel([]), block_size=2, rank=2, world_size=2)


def test_model_runner_group_broadcasts_pickled_commands_to_workers() -> None:
    reset_sequence_counter()
    sequence = Sequence([1, 2])
    worker = MutatingWorkerRunner()
    primary = RecordingPrimaryRunner()
    group = ModelRunnerGroup(primary, [worker])

    assert group.world_size == 2
    assert group.call("run", [sequence], True) == [7]

    assert isinstance(group.commands[0], ModelRunnerCommand)
    assert group.commands[0].method_name == "run"
    assert group.commands[0].payload_size > 0
    assert worker.calls == [("run", [1, 2], True)]
    assert primary.calls == [([1, 2], True)]
    assert sequence.token_ids == [1, 2]


def test_model_runner_group_broadcasts_exit_to_workers() -> None:
    worker = MutatingWorkerRunner()
    primary = RecordingPrimaryRunner()
    group = ModelRunnerGroup(primary, [worker])

    group.exit()

    assert [command.method_name for command in group.commands] == ["exit"]
    assert worker.close_calls == 1
    assert primary.close_calls == 1


def test_model_runner_group_rejects_unknown_primary_method() -> None:
    group = ModelRunnerGroup(object())

    with pytest.raises(ConfigurationError, match="unknown model runner method"):
        group.call("missing")


def test_model_runner_empty_batch_returns_no_tokens() -> None:
    runner = ModelRunner(FakeModel([]), block_size=2)

    assert runner.run([], is_prefill=True) == []
    assert runner.last_context is None
    assert runner.last_sample_batch is None
