from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

from nexinfer.engine import LLMEngine
from nexinfer.errors import ConfigurationError
from nexinfer.model import ModelConfig
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
        model_config: ModelConfig | None = None,
        enforce_eager: bool = False,
        tensor_parallel_size: int = 1,
        **backend_kwargs: Any,
    ) -> None:
        if tensor_parallel_size != 1:
            raise ConfigurationError("tensor_parallel_size > 1 is not supported yet")
        self.enforce_eager = enforce_eager

        if backend is None or tokenizer is None:
            if model_config is None:
                if model is None:
                    raise ConfigurationError(
                        "model is required when backend and tokenizer are not provided"
                    )
                model_config = ModelConfig(model)
            backend, tokenizer = self._load_model(model_config, **backend_kwargs)

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
            config = prompt_params.to_generation_config(eos_token_id=eos_token_id)
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
