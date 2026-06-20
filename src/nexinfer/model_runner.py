from __future__ import annotations

from collections.abc import Sequence as SequenceCollection
from dataclasses import dataclass
import pickle
from typing import Any

from nexinfer.context import reset_context, set_context
from nexinfer.errors import ConfigurationError
from nexinfer.protocols import DecodeInput, DecodeState, PrefillInput
from nexinfer.sampling import Sampler
from nexinfer.sequence import Sequence as RunnerSequence


@dataclass(frozen=True, slots=True)
class PreparedPrefillBatch:
    """Flattened prefill metadata for a model runner."""

    input_ids: list[int]
    positions: list[int]
    cu_seqlens_q: list[int]
    cu_seqlens_k: list[int]
    max_seqlen_q: int
    max_seqlen_k: int
    slot_mapping: list[int]
    block_tables: list[list[int]]


@dataclass(frozen=True, slots=True)
class PreparedDecodeBatch:
    """Flattened decode metadata for a model runner."""

    input_ids: list[int]
    positions: list[int]
    slot_mapping: list[int]
    context_lengths: list[int]
    block_tables: list[list[int]]


@dataclass(frozen=True, slots=True)
class PreparedSampleBatch:
    """Per-sequence sampling metadata for a model runner."""

    temperatures: list[float]


@dataclass(frozen=True, slots=True)
class KVCacheSlice:
    """A lightweight reference to one layer's KV-cache tensor slice."""

    kind: str
    layer_id: int
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class KVCacheLayout:
    """CPU-safe KV-cache layout used when no tensor factory is supplied."""

    shape: tuple[int, ...]

    def __getitem__(self, index: tuple[int, int]) -> KVCacheSlice:
        cache_kind, layer_id = index
        if cache_kind not in (0, 1):
            raise IndexError("cache kind must be 0 for keys or 1 for values")
        if not 0 <= layer_id < self.shape[1]:
            raise IndexError("layer id out of range")
        kind = "k" if cache_kind == 0 else "v"
        return KVCacheSlice(kind=kind, layer_id=layer_id, shape=self.shape[2:])


@dataclass(frozen=True, slots=True)
class CUDAGraphCapturePlan:
    """CPU-safe metadata for Nano-VLLM-style CUDA graph capture buffers."""

    batch_sizes: list[int]
    max_batch_size: int
    max_num_blocks: int
    hidden_size: int | None
    buffer_shapes: dict[str, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class CUDAGraphReplayPlan:
    """CPU-safe metadata for one Nano-VLLM-style CUDA graph replay."""

    batch_size: int
    graph_batch_size: int
    input_ids: list[int]
    positions: list[int]
    slot_mapping: list[int]
    context_lengths: list[int]
    block_tables: list[list[int]]


@dataclass(frozen=True, slots=True)
class ModelRunnerCommand:
    """Serialized command metadata broadcast to model-runner workers."""

    method_name: str
    payload_size: int


@dataclass(frozen=True, slots=True)
class ModelRunnerContext:
    """Prepared runner context for the latest prefill or decode batch."""

    is_prefill: bool
    input_ids: list[int]
    positions: list[int]
    slot_mapping: list[int]
    block_tables: list[list[int]]
    cu_seqlens_q: list[int] | None = None
    cu_seqlens_k: list[int] | None = None
    max_seqlen_q: int | None = None
    max_seqlen_k: int | None = None
    context_lengths: list[int] | None = None

    @classmethod
    def from_prefill(cls, batch: PreparedPrefillBatch) -> "ModelRunnerContext":
        return cls(
            is_prefill=True,
            input_ids=batch.input_ids,
            positions=batch.positions,
            slot_mapping=batch.slot_mapping,
            block_tables=batch.block_tables,
            cu_seqlens_q=batch.cu_seqlens_q,
            cu_seqlens_k=batch.cu_seqlens_k,
            max_seqlen_q=batch.max_seqlen_q,
            max_seqlen_k=batch.max_seqlen_k,
        )

    @classmethod
    def from_decode(cls, batch: PreparedDecodeBatch) -> "ModelRunnerContext":
        return cls(
            is_prefill=False,
            input_ids=batch.input_ids,
            positions=batch.positions,
            slot_mapping=batch.slot_mapping,
            block_tables=batch.block_tables,
            context_lengths=batch.context_lengths,
        )


class ModelRunner:
    """Nano-VLLM-style runner orchestration around a logits-producing model."""

    def __init__(
        self,
        model: Any,
        *,
        block_size: int,
        sampler: Sampler | None = None,
        config: Any | None = None,
        rank: int = 0,
        world_size: int = 1,
        enforce_eager: bool | None = None,
    ) -> None:
        _validate_block_size(block_size)
        _validate_runner_rank(rank=rank, world_size=world_size)
        self.model = model
        self.block_size = block_size
        self.sampler = sampler or Sampler()
        self.config = config
        self.enforce_eager = _resolve_enforce_eager(enforce_eager, config)
        self.rank = rank
        self.world_size = world_size
        self.last_context: ModelRunnerContext | None = None
        self.last_sample_batch: PreparedSampleBatch | None = None
        self.last_graph_replay: CUDAGraphReplayPlan | None = None
        self.kv_cache: Any | None = None
        self.kv_cache_shape: tuple[int, ...] | None = None
        self.graph_bs: list[int] = []
        self.graph_capture_plan: CUDAGraphCapturePlan | None = None

    def call(self, method_name: str, *args: Any) -> Any:
        """Call a runner method by name, matching Nano-VLLM's control path."""

        method = getattr(self, method_name, None)
        if not callable(method):
            raise ConfigurationError(f"unknown model runner method: {method_name}")
        return method(*args)

    def exit(self) -> None:
        """Release resources owned by the wrapped model when it supports cleanup."""

        for method_name in ("exit", "close"):
            method = getattr(self.model, method_name, None)
            if callable(method):
                method()
                return

    close = exit

    def warmup_model(self, config: Any | None = None) -> None:
        """Run a Nano-VLLM-style synthetic prefill warmup batch."""

        config = config or self.config
        if config is None:
            raise ConfigurationError("config is required to warm up model")

        max_num_batched_tokens = _positive_config_int(
            config,
            "max_num_batched_tokens",
        )
        max_model_len = _positive_config_int(config, "max_model_len")
        max_num_seqs = _positive_config_int(config, "max_num_seqs")
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, max_num_seqs)
        sequences = [RunnerSequence([0] * seq_len) for _ in range(num_seqs)]
        for sequence in sequences:
            sequence.num_scheduled_tokens = seq_len
        self.run(sequences, True)

    def allocate_kv_cache(
        self,
        config: Any | None = None,
        *,
        cache_factory: Any | None = None,
    ) -> Any:
        """Allocate a Nano-VLLM-style KV-cache layout and attach layer slices."""

        config = config or self.config
        if config is None:
            raise ConfigurationError("config is required to allocate KV cache")
        hf_config = getattr(config, "hf_config", None)
        if hf_config is None:
            raise ConfigurationError("hf_config is required to allocate KV cache")
        num_blocks = _positive_config_int(config, "num_kvcache_blocks")
        world_size = _positive_config_int(config, "tensor_parallel_size")
        num_layers = _positive_attr_int(hf_config, "num_hidden_layers")
        num_key_value_heads = _positive_attr_int(
            hf_config,
            "num_key_value_heads",
        )
        if num_key_value_heads % world_size != 0:
            raise ConfigurationError("num_key_value_heads must divide world size")
        num_kv_heads = num_key_value_heads // world_size
        head_dim = getattr(hf_config, "head_dim", None)
        if head_dim is None:
            head_dim = _positive_attr_int(hf_config, "hidden_size") // _positive_attr_int(
                hf_config,
                "num_attention_heads",
            )
        head_dim = int(head_dim)
        if head_dim <= 0:
            raise ConfigurationError("head_dim must be positive")

        shape = (
            2,
            num_layers,
            num_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )
        self.kv_cache_shape = shape
        self.kv_cache = (
            cache_factory(shape) if cache_factory is not None else KVCacheLayout(shape)
        )
        layer_id = 0
        modules = getattr(self.model, "modules", None)
        for module in modules() if callable(modules) else ():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                if layer_id >= num_layers:
                    raise ConfigurationError("more KV-cache modules than layers")
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        return self.kv_cache

    def capture_cudagraph(self, config: Any | None = None) -> CUDAGraphCapturePlan:
        """Record Nano-VLLM-style CUDA graph capture buffer metadata."""

        config = config or self.config
        if config is None:
            raise ConfigurationError("config is required to capture CUDA graph")
        max_num_seqs = _positive_config_int(config, "max_num_seqs")
        max_model_len = _positive_config_int(config, "max_model_len")
        max_batch_size = min(max_num_seqs, 512)
        max_num_blocks = (max_model_len + self.block_size - 1) // self.block_size
        batch_sizes = [1, 2, 4, 8, *range(16, max_batch_size + 1, 16)]
        hidden_size = _optional_positive_hidden_size(config)
        buffer_shapes = {
            "input_ids": (max_batch_size,),
            "positions": (max_batch_size,),
            "slot_mapping": (max_batch_size,),
            "context_lens": (max_batch_size,),
            "block_tables": (max_batch_size, max_num_blocks),
        }
        if hidden_size is not None:
            buffer_shapes["outputs"] = (max_batch_size, hidden_size)

        plan = CUDAGraphCapturePlan(
            batch_sizes=batch_sizes,
            max_batch_size=max_batch_size,
            max_num_blocks=max_num_blocks,
            hidden_size=hidden_size,
            buffer_shapes=buffer_shapes,
        )
        self.graph_bs = batch_sizes
        self.graph_capture_plan = plan
        return plan

    def run(
        self,
        sequences: SequenceCollection[RunnerSequence],
        is_prefill: bool,
    ) -> list[int] | None:
        """Prepare a sequence batch, run the model, and sample next tokens."""

        if not sequences:
            return []

        if is_prefill:
            input_ids, positions = self.prepare_prefill(sequences)
        else:
            input_ids, positions = self.prepare_decode(sequences)

        sample_batch = self.prepare_sample(sequences) if self.rank == 0 else None
        try:
            logits = self.run_model(input_ids, positions, is_prefill)
        finally:
            reset_context()
        if self.rank != 0:
            return None
        if len(logits) != len(sequences):
            raise ConfigurationError("model must return one logits row per sequence")
        assert sample_batch is not None
        return self.sampler(logits, sample_batch.temperatures)

    def prepare_prefill(
        self,
        sequences: SequenceCollection[RunnerSequence],
    ) -> tuple[list[int], list[int]]:
        """Prepare prefill inputs and set the active attention context."""

        prepared = prepare_prefill_sequences(sequences, block_size=self.block_size)
        self.last_context = ModelRunnerContext.from_prefill(prepared)
        self._set_prefill_context(prepared)
        return prepared.input_ids, prepared.positions

    def prepare_decode(
        self,
        sequences: SequenceCollection[RunnerSequence],
    ) -> tuple[list[int], list[int]]:
        """Prepare decode inputs and set the active attention context."""

        prepared = prepare_decode_sequences(sequences, block_size=self.block_size)
        self.last_context = ModelRunnerContext.from_decode(prepared)
        self._set_decode_context(prepared)
        return prepared.input_ids, prepared.positions

    def prepare_sample(
        self,
        sequences: SequenceCollection[RunnerSequence],
    ) -> PreparedSampleBatch:
        """Prepare per-sequence sampling metadata."""

        self.last_sample_batch = prepare_sample_sequences(sequences)
        return self.last_sample_batch

    def prepare_block_tables(
        self,
        sequences: SequenceCollection[RunnerSequence],
    ) -> list[list[int]]:
        """Prepare padded block tables for a sequence batch."""

        return prepare_block_tables(sequences)

    def run_model(
        self,
        input_ids: SequenceCollection[int],
        positions: SequenceCollection[int],
        is_prefill: bool,
    ) -> SequenceCollection[SequenceCollection[float]]:
        """Call the wrapped model object to produce logits."""

        self.last_graph_replay = None
        if self._should_plan_cudagraph_replay(len(input_ids), is_prefill):
            self.last_graph_replay = self.prepare_cudagraph_replay(
                input_ids,
                positions,
            )

        run_model = getattr(self.model, "run_model", None)
        if callable(run_model):
            return run_model(input_ids, positions, is_prefill)
        compute_logits = getattr(self.model, "compute_logits", None)
        if callable(compute_logits) and callable(self.model):
            hidden_states = self.model(input_ids, positions)
            return compute_logits(hidden_states)
        if callable(self.model):
            return self.model(input_ids, positions, is_prefill)
        raise ConfigurationError("model must be callable or expose run_model")

    def prepare_cudagraph_replay(
        self,
        input_ids: SequenceCollection[int],
        positions: SequenceCollection[int],
    ) -> CUDAGraphReplayPlan:
        """Prepare CPU-safe metadata for a captured decode graph replay."""

        if self.graph_capture_plan is None:
            raise ConfigurationError("CUDA graph capture plan is required")
        batch_size = len(input_ids)
        graph_batch_size = self._cudagraph_batch_size(batch_size)
        context = self.last_context
        if context is None or context.is_prefill:
            raise ConfigurationError("decode context is required for CUDA graph replay")
        if context.context_lengths is None:
            raise ConfigurationError("decode context lengths are required")

        return CUDAGraphReplayPlan(
            batch_size=batch_size,
            graph_batch_size=graph_batch_size,
            input_ids=list(input_ids),
            positions=list(positions),
            slot_mapping=list(context.slot_mapping),
            context_lengths=list(context.context_lengths),
            block_tables=[list(row) for row in context.block_tables],
        )

    def _set_prefill_context(self, batch: PreparedPrefillBatch) -> None:
        set_context(
            True,
            cu_seqlens_q=batch.cu_seqlens_q,
            cu_seqlens_k=batch.cu_seqlens_k,
            max_seqlen_q=batch.max_seqlen_q,
            max_seqlen_k=batch.max_seqlen_k,
            slot_mapping=batch.slot_mapping,
            block_tables=batch.block_tables or None,
        )

    def _set_decode_context(self, batch: PreparedDecodeBatch) -> None:
        set_context(
            False,
            slot_mapping=batch.slot_mapping,
            context_lens=batch.context_lengths,
            block_tables=batch.block_tables or None,
        )

    def _should_plan_cudagraph_replay(self, batch_size: int, is_prefill: bool) -> bool:
        return (
            not is_prefill
            and not self.enforce_eager
            and self.graph_capture_plan is not None
            and 0 < batch_size <= self.graph_capture_plan.max_batch_size
        )

    def _cudagraph_batch_size(self, batch_size: int) -> int:
        for graph_batch_size in self.graph_bs:
            if graph_batch_size >= batch_size:
                return graph_batch_size
        raise ConfigurationError("batch size exceeds captured CUDA graph plan")


class ModelRunnerGroup:
    """Rank-0 style command fanout for a primary runner and worker runners."""

    def __init__(
        self,
        primary: Any,
        workers: SequenceCollection[Any] | None = None,
    ) -> None:
        self.primary = primary
        self.workers = list(workers or ())
        self.commands: list[ModelRunnerCommand] = []

    @property
    def world_size(self) -> int:
        return 1 + len(self.workers)

    def call(self, method_name: str, *args: Any) -> Any:
        """Broadcast a runner command to workers and return the primary result."""

        payload = _serialize_runner_command(method_name, args) if self.workers else b""
        self.commands.append(
            ModelRunnerCommand(
                method_name=method_name,
                payload_size=len(payload),
            )
        )
        for worker in self.workers:
            worker_method_name, worker_args = _deserialize_runner_command(payload)
            _call_runner_method(worker, worker_method_name, *worker_args)
        return _call_runner_method(self.primary, method_name, *args)

    def exit(self) -> None:
        self.call("exit")

    close = exit


def prepare_prefill_batch(
    inputs: SequenceCollection[PrefillInput],
    *,
    block_size: int,
) -> PreparedPrefillBatch:
    """Prepare Nano-VLLM-style flattened prefill inputs."""

    _validate_block_size(block_size)

    input_ids: list[int] = []
    positions: list[int] = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping: list[int] = []

    for item in inputs:
        start = item.num_cached_tokens
        seqlen_q = _scheduled_token_count(item)
        end = min(start + seqlen_q, len(item.token_ids))
        seqlen_q = max(end - start, 0)
        seqlen_k = end

        input_ids.extend(item.token_ids[start:end])
        positions.extend(range(start, end))
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(max_seqlen_q, seqlen_q)
        max_seqlen_k = max(max_seqlen_k, seqlen_k)
        slot_mapping.extend(
            _slot_mapping(
                block_table=item.block_table,
                block_size=block_size,
                start=start,
                end=end,
            )
        )

    return PreparedPrefillBatch(
        input_ids=input_ids,
        positions=positions,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        block_tables=prepare_block_tables(inputs),
    )


def prepare_decode_batch(
    inputs: SequenceCollection[DecodeInput],
    *,
    block_size: int,
) -> PreparedDecodeBatch:
    """Prepare Nano-VLLM-style flattened decode inputs."""

    _validate_block_size(block_size)

    input_ids: list[int] = []
    positions: list[int] = []
    slot_mapping: list[int] = []
    context_lengths: list[int] = []

    for item in inputs:
        context_length = item.context_length
        if context_length is None:
            context_length = item.state.position + item.num_scheduled_tokens
        if context_length <= 0:
            raise ConfigurationError("context_length must be positive")

        position = context_length - item.num_scheduled_tokens
        input_ids.append(item.token_id)
        positions.append(position)
        context_lengths.append(context_length)
        slot_mapping.extend(
            _slot_mapping(
                block_table=item.block_table,
                block_size=block_size,
                start=position,
                end=position + item.num_scheduled_tokens,
            )
        )

    return PreparedDecodeBatch(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lengths=context_lengths,
        block_tables=prepare_block_tables(inputs),
    )


def prepare_prefill_sequences(
    sequences: SequenceCollection[RunnerSequence],
    *,
    block_size: int,
) -> PreparedPrefillBatch:
    """Prepare flattened prefill metadata from Nano-VLLM-style sequences."""

    return prepare_prefill_batch(
        [
            PrefillInput(
                token_ids=sequence.token_ids,
                num_cached_tokens=sequence.num_cached_tokens,
                num_scheduled_tokens=sequence.num_scheduled_tokens,
                block_table=tuple(sequence.block_table),
            )
            for sequence in sequences
        ],
        block_size=block_size,
    )


def prepare_decode_sequences(
    sequences: SequenceCollection[RunnerSequence],
    *,
    block_size: int,
) -> PreparedDecodeBatch:
    """Prepare flattened decode metadata from Nano-VLLM-style sequences."""

    return prepare_decode_batch(
        [
            DecodeInput(
                token_id=sequence.last_token,
                state=DecodeState(position=len(sequence) - 1),
                block_table=tuple(sequence.block_table),
                context_length=len(sequence),
                num_scheduled_tokens=_decode_token_count(sequence),
            )
            for sequence in sequences
        ],
        block_size=block_size,
    )


def prepare_sample_sequences(
    sequences: SequenceCollection[RunnerSequence],
) -> PreparedSampleBatch:
    """Prepare per-sequence sampling temperatures."""

    return PreparedSampleBatch(
        temperatures=[float(sequence.temperature) for sequence in sequences]
    )


def prepare_block_tables(
    inputs: SequenceCollection[PrefillInput]
    | SequenceCollection[DecodeInput]
    | SequenceCollection[RunnerSequence],
) -> list[list[int]]:
    """Prepare Nano-VLLM-style padded block tables."""

    max_length = max((len(item.block_table) for item in inputs), default=0)
    if max_length == 0:
        return []
    return [
        [*item.block_table, *([-1] * (max_length - len(item.block_table)))]
        for item in inputs
    ]


def _scheduled_token_count(item: PrefillInput) -> int:
    count = item.scheduled_token_count
    if count < 0:
        raise ConfigurationError("num_scheduled_tokens must be non-negative")
    if item.num_cached_tokens < 0:
        raise ConfigurationError("num_cached_tokens must be non-negative")
    return count


def _slot_mapping(
    *,
    block_table: SequenceCollection[int],
    block_size: int,
    start: int,
    end: int,
) -> list[int]:
    slots: list[int] = []
    for position in range(start, end):
        if not block_table:
            slots.append(-1)
            continue
        block_index = position // block_size
        if block_index >= len(block_table):
            raise ConfigurationError("block_table is too short for scheduled tokens")
        slots.append(block_table[block_index] * block_size + position % block_size)
    return slots


def _validate_block_size(block_size: int) -> None:
    if block_size <= 0:
        raise ConfigurationError("block_size must be positive")


def _validate_runner_rank(*, rank: int, world_size: int) -> None:
    if world_size <= 0:
        raise ConfigurationError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ConfigurationError("rank must be in [0, world_size)")


def _resolve_enforce_eager(enforce_eager: bool | None, config: Any | None) -> bool:
    if enforce_eager is not None:
        return bool(enforce_eager)
    if config is None:
        return False
    return bool(getattr(config, "enforce_eager", False))


def _positive_config_int(config: Any, name: str) -> int:
    value = int(getattr(config, name))
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return value


def _positive_attr_int(target: Any, name: str) -> int:
    value = int(getattr(target, name))
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return value


def _optional_positive_hidden_size(config: Any) -> int | None:
    hf_config = getattr(config, "hf_config", None)
    if hf_config is None or getattr(hf_config, "hidden_size", None) is None:
        return None
    return _positive_attr_int(hf_config, "hidden_size")


def _serialize_runner_command(
    method_name: str,
    args: tuple[Any, ...],
) -> bytes:
    try:
        return pickle.dumps([method_name, *args])
    except Exception as exc:  # pragma: no cover - exact pickle errors vary
        raise ConfigurationError(
            "model runner worker command arguments must be pickleable"
        ) from exc


def _deserialize_runner_command(payload: bytes) -> tuple[str, tuple[Any, ...]]:
    try:
        method_name, *args = pickle.loads(payload)
    except Exception as exc:  # pragma: no cover - exact pickle errors vary
        raise ConfigurationError("invalid model runner worker command payload") from exc
    if not isinstance(method_name, str):
        raise ConfigurationError("model runner worker command name must be a string")
    return method_name, tuple(args)


def _call_runner_method(target: Any, method_name: str, *args: Any) -> Any:
    call = getattr(target, "call", None)
    if callable(call):
        return call(method_name, *args)
    method = getattr(target, method_name, None)
    if not callable(method):
        raise ConfigurationError(f"unknown model runner method: {method_name}")
    return method(*args)


def _decode_token_count(sequence: RunnerSequence) -> int:
    if sequence.num_scheduled_tokens < 0:
        raise ConfigurationError("num_scheduled_tokens must be non-negative")
    if sequence.num_scheduled_tokens > 1:
        raise ConfigurationError("decode num_scheduled_tokens must be 1")
    return sequence.num_scheduled_tokens or 1
