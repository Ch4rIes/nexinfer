from __future__ import annotations

import atexit
from collections.abc import Sequence
from dataclasses import fields, replace
from itertools import count
from time import perf_counter
from typing import Any, TypedDict

from nexinfer.cache import PrefixKVCacheBlockManager
from nexinfer.engine import LLMEngine
from nexinfer.errors import ConfigurationError
from nexinfer.model import LLMConfig, ModelConfig
from nexinfer.nano_engine import NanoLLMEngine
from nexinfer.protocols import DecoderOnlyBackend, Tokenizer
from nexinfer.runtime import InferenceRuntime
from nexinfer.sampling_params import SamplingParams
from nexinfer.scheduler import Scheduler
from nexinfer.tokenizer import HuggingFaceTokenizer


class LLMOutput(TypedDict):
    """Nano-VLLM-style generation output."""

    text: str
    token_ids: list[int]


LLMStepOutput = tuple[list[tuple[int, list[int]]], int]


class LLM:
    """Nano-VLLM-compatible facade over NexInfer's generation engine."""

    def __init__(
        self,
        model: str | None = None,
        *,
        backend: DecoderOnlyBackend | None = None,
        model_runner: Any | None = None,
        tokenizer: Tokenizer | None = None,
        scheduler: Scheduler | None = None,
        config: LLMConfig | None = None,
        model_config: ModelConfig | None = None,
        enforce_eager: bool = False,
        tensor_parallel_size: int = 1,
        **backend_kwargs: Any,
    ) -> None:
        config_kwargs = _pop_config_kwargs(backend_kwargs)
        self.config = _resolve_llm_config(
            model=model,
            config=config,
            model_config=model_config,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tensor_parallel_size,
            config_kwargs=config_kwargs,
        )
        self.enforce_eager = self.config.enforce_eager

        if model_runner is not None and backend is not None:
            raise ConfigurationError("backend cannot be combined with model_runner")
        if scheduler is not None and model_runner is None:
            raise ConfigurationError("scheduler requires model_runner")
        if model_runner is not None:
            _validate_model_runner_world_size(model_runner, self.config)
        elif self.config.tensor_parallel_size != 1:
            raise ConfigurationError(
                "tensor_parallel_size > 1 requires a model_runner group"
            )

        self._backend = backend
        self._nano_engine: NanoLLMEngine | None = None
        self._runtime: InferenceRuntime | None = None

        if model_runner is not None:
            if tokenizer is None:
                raise ConfigurationError("tokenizer is required with model_runner")
            eos_token_id = _eos_token_id(tokenizer)
            if eos_token_id is not None:
                self.config.eos = eos_token_id
            scheduler = scheduler or Scheduler(self.config)
            self.engine = NanoLLMEngine(model_runner, tokenizer, scheduler)
            self._nano_engine = self.engine
        elif backend is None or tokenizer is None:
            if model_config is None:
                model_config = self.config.to_model_config()
            elif model_config.max_context_tokens is None:
                model_config = replace(
                    model_config,
                    max_context_tokens=self.config.max_model_len,
                )
            backend, tokenizer = self._load_model(model_config, **backend_kwargs)

        if model_runner is None:
            assert backend is not None
            assert tokenizer is not None
            eos_token_id = _eos_token_id(tokenizer)
            if eos_token_id is not None:
                self.config.eos = eos_token_id
            self._backend = backend
            self.engine = LLMEngine(backend, tokenizer)
            self._runtime = self._build_runtime()

        self._request_ids = count()
        self._closed = False
        self._exit_callback = self.exit
        atexit.register(self._exit_callback)

    @property
    def tokenizer(self) -> Tokenizer:
        if self._nano_engine is not None:
            return self._nano_engine.tokenizer
        return self.engine.tokenizer

    def __enter__(self) -> "LLM":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.exit()

    def generate(
        self,
        prompts: Sequence[str] | Sequence[Sequence[int]],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        use_tqdm: bool = True,
    ) -> list[LLMOutput]:
        """Generate Nano-VLLM-style outputs for text or token-id prompts."""

        self._ensure_open()
        prompt_list = list(prompts)
        params = self._normalize_sampling_params(sampling_params, len(prompt_list))
        if self._nano_engine is not None:
            return self._nano_engine.generate(prompt_list, params, use_tqdm=use_tqdm)

        request_ids: list[int] = []
        for prompt, prompt_params in zip(prompt_list, params, strict=True):
            request_ids.append(self.add_request(prompt, prompt_params))

        outputs: dict[int, list[int]] = {}
        prefill_throughput = 0.0
        decode_throughput = 0.0
        progress = _progress_bar(total=len(prompt_list), enabled=use_tqdm)
        try:
            while not self.is_finished():
                start = perf_counter()
                step_outputs, num_tokens = self.step()
                elapsed = max(perf_counter() - start, 1e-12)
                if num_tokens > 0:
                    prefill_throughput = num_tokens / elapsed
                elif num_tokens < 0:
                    decode_throughput = -num_tokens / elapsed
                progress.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
                for seq_id, token_ids in step_outputs:
                    outputs[seq_id] = token_ids
                    progress.update(1)
        finally:
            progress.close()

        return [
            {
                "text": self.tokenizer.decode(outputs[request_id]),
                "token_ids": outputs[request_id],
            }
            for request_id in request_ids
        ]

    def add_request(
        self,
        prompt: str | Sequence[int],
        sampling_params: SamplingParams | None = None,
    ) -> int:
        """Add one request to the Nano-VLLM-style internal queue."""

        self._ensure_open()
        if self._nano_engine is not None:
            return self._nano_engine.add_request(
                prompt,
                sampling_params or SamplingParams(),
            )

        assert self._runtime is not None
        request_id = next(self._request_ids)
        config = (sampling_params or SamplingParams()).to_generation_config(
            eos_token_id=_eos_token_id(self.tokenizer),
            max_total_tokens=self.config.max_model_len,
        )
        self._runtime.submit(prompt, config, request_id=str(request_id))
        return request_id

    def step(self) -> LLMStepOutput:
        """Run one scheduler step and return completed outputs plus token count."""

        self._ensure_open()
        if self._nano_engine is not None:
            return self._nano_engine.step()

        assert self._runtime is not None
        completed = self._runtime.run_once()
        outputs = [
            (int(item.request_id), item.result.generated_token_ids)
            for item in completed
        ]
        return outputs, self._runtime.last_scheduled_tokens

    def is_finished(self) -> bool:
        """Return whether the internal request queue is empty."""

        if self._closed:
            return True
        if self._nano_engine is not None:
            return self._nano_engine.is_finished()
        assert self._runtime is not None
        return self._runtime.pending_requests == 0

    def exit(self) -> None:
        """Release backend resources owned by this LLM."""

        if self._closed:
            return
        self._closed = True
        try:
            atexit.unregister(self._exit_callback)
        except ValueError:
            pass
        if self._nano_engine is not None:
            self._nano_engine.close()
        elif self._backend is not None:
            _call_optional_cleanup(self._backend)

    close = exit

    def _normalize_sampling_params(
        self,
        sampling_params: SamplingParams | Sequence[SamplingParams] | None,
        prompt_count: int,
    ) -> list[SamplingParams]:
        if sampling_params is None:
            return [SamplingParams()] * prompt_count
        if isinstance(sampling_params, SamplingParams):
            return [sampling_params] * prompt_count

        params = list(sampling_params)
        if len(params) != prompt_count:
            raise ConfigurationError(
                "sampling_params must contain one item per prompt when provided as a list"
            )
        return params

    def _load_model(
        self,
        config: ModelConfig,
        **backend_kwargs: Any,
    ) -> tuple[DecoderOnlyBackend, Tokenizer]:
        from nexinfer.backends import TorchCausalLMBackend

        tokenizer = HuggingFaceTokenizer.from_pretrained(config.model_name_or_path)
        backend = TorchCausalLMBackend.from_pretrained(config, **backend_kwargs)
        return backend, tokenizer

    def _build_runtime(self) -> InferenceRuntime:
        block_manager = None
        if self.config.num_kvcache_blocks != -1:
            block_manager = PrefixKVCacheBlockManager(
                num_blocks=self.config.num_kvcache_blocks,
                block_size=self.config.kvcache_block_size,
            )
        return InferenceRuntime(
            self.engine,
            max_batch_size=self.config.max_num_seqs,
            max_batch_prompt_tokens=self.config.max_num_batched_tokens,
            decode_strategy="continuous",
            block_manager=block_manager,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise ConfigurationError("LLM is closed")


def _is_token_id_prompt(prompt: object) -> bool:
    if isinstance(prompt, str):
        return False
    if not isinstance(prompt, Sequence):
        return False
    return all(isinstance(token_id, int) for token_id in prompt)


def _eos_token_id(tokenizer: Tokenizer) -> int | None:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return None
    return int(eos_token_id)


def _call_optional_cleanup(target: object) -> None:
    for method_name in ("exit", "close"):
        method = getattr(target, method_name, None)
        if callable(method):
            method()
            return


def _validate_model_runner_world_size(
    model_runner: Any,
    config: LLMConfig,
) -> None:
    world_size = getattr(model_runner, "world_size", None)
    if world_size is None:
        if config.tensor_parallel_size == 1:
            return
        raise ConfigurationError(
            "model_runner must expose world_size for tensor_parallel_size > 1"
        )

    world_size = int(world_size)
    if world_size <= 0:
        raise ConfigurationError("model_runner world_size must be positive")
    if config.tensor_parallel_size == 1 and world_size > 1:
        config.tensor_parallel_size = world_size
        return
    if world_size != config.tensor_parallel_size:
        raise ConfigurationError(
            "model_runner world_size must match tensor_parallel_size"
        )


class _NoopProgress:
    def set_postfix(self, values: dict[str, str]) -> None:
        pass

    def update(self, amount: int) -> None:
        pass

    def close(self) -> None:
        pass


def _progress_bar(*, total: int, enabled: bool) -> Any:
    if not enabled:
        return _NoopProgress()
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return _NoopProgress()
    return tqdm(
        total=total,
        desc="Generating",
        dynamic_ncols=True,
        disable=not enabled,
    )


_CONFIG_FIELD_NAMES = {field.name for field in fields(LLMConfig)}


def _pop_config_kwargs(backend_kwargs: dict[str, Any]) -> dict[str, Any]:
    config_kwargs: dict[str, Any] = {}
    for key in list(backend_kwargs):
        if key in _CONFIG_FIELD_NAMES:
            config_kwargs[key] = backend_kwargs.pop(key)
    return config_kwargs


def _resolve_llm_config(
    *,
    model: str | None,
    config: LLMConfig | None,
    model_config: ModelConfig | None,
    enforce_eager: bool,
    tensor_parallel_size: int,
    config_kwargs: dict[str, Any],
) -> LLMConfig:
    if config is not None:
        if config_kwargs:
            keys = ", ".join(sorted(config_kwargs))
            raise ConfigurationError(
                f"config kwargs cannot be combined with config: {keys}"
            )
        if tensor_parallel_size != 1:
            raise ConfigurationError(
                "tensor_parallel_size cannot be combined with config"
            )
        if enforce_eager and not config.enforce_eager:
            raise ConfigurationError("enforce_eager cannot be combined with config")
        return config

    config_kwargs.setdefault("enforce_eager", enforce_eager)
    config_kwargs.setdefault("tensor_parallel_size", tensor_parallel_size)
    config_model = model or (
        model_config.model_name_or_path if model_config is not None else "injected"
    )
    return LLMConfig(config_model, **config_kwargs)
