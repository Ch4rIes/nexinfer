from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields, replace
from typing import Any, TypedDict

from nexinfer.engine import LLMEngine
from nexinfer.errors import ConfigurationError
from nexinfer.model import LLMConfig, ModelConfig
from nexinfer.protocols import DecoderOnlyBackend, Tokenizer
from nexinfer.sampling_params import SamplingParams
from nexinfer.tokenizer import HuggingFaceTokenizer


class LLMOutput(TypedDict):
    """Nano-VLLM-style generation output."""

    text: str
    token_ids: list[int]


class LLM:
    """Nano-VLLM-compatible facade over NexInfer's generation engine."""

    def __init__(
        self,
        model: str | None = None,
        *,
        backend: DecoderOnlyBackend | None = None,
        tokenizer: Tokenizer | None = None,
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
        if self.config.tensor_parallel_size != 1:
            raise ConfigurationError("tensor_parallel_size > 1 is not supported yet")
        self.enforce_eager = self.config.enforce_eager

        if backend is None or tokenizer is None:
            if model_config is None:
                model_config = self.config.to_model_config()
            elif model_config.max_context_tokens is None:
                model_config = replace(
                    model_config,
                    max_context_tokens=self.config.max_model_len,
                )
            backend, tokenizer = self._load_model(model_config, **backend_kwargs)

        eos_token_id = _eos_token_id(tokenizer)
        if eos_token_id is not None:
            self.config.eos = eos_token_id
        self.engine = LLMEngine(backend, tokenizer)

    @property
    def tokenizer(self) -> Tokenizer:
        return self.engine.tokenizer

    def generate(
        self,
        prompts: Sequence[str] | Sequence[Sequence[int]],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        use_tqdm: bool = True,
    ) -> list[LLMOutput]:
        """Generate Nano-VLLM-style outputs for text or token-id prompts."""

        del use_tqdm
        prompt_list = list(prompts)
        params = self._normalize_sampling_params(sampling_params, len(prompt_list))
        eos_token_id = _eos_token_id(self.tokenizer)

        outputs: list[LLMOutput] = []
        for prompt, prompt_params in zip(prompt_list, params, strict=True):
            config = prompt_params.to_generation_config(
                eos_token_id=eos_token_id,
                max_total_tokens=self.config.max_model_len,
            )
            if _is_token_id_prompt(prompt):
                result = self.engine.complete_token_ids(prompt, config)
            else:
                result = self.engine.complete(str(prompt), config)
            outputs.append({"text": result.text, "token_ids": result.token_ids})
        return outputs

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
