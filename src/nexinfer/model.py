from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from nexinfer.errors import ConfigurationError

Device = Literal["auto", "cpu", "cuda", "mps"]
DType = Literal["auto", "float32", "float16", "bfloat16"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for loading a model backend."""

    model_name_or_path: str
    device: Device = "auto"
    dtype: DType = "auto"
    trust_remote_code: bool = False
    revision: str | None = None
    max_context_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.model_name_or_path:
            raise ConfigurationError("model_name_or_path must not be empty")
        if self.max_context_tokens is not None and self.max_context_tokens <= 0:
            raise ConfigurationError(
                "max_context_tokens must be positive when configured"
            )


@dataclass(slots=True)
class LLMConfig:
    """Nano-VLLM-style runtime and model configuration."""

    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: Any | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self) -> None:
        if not self.model:
            raise ConfigurationError("model must not be empty")
        if self.max_num_batched_tokens <= 0:
            raise ConfigurationError("max_num_batched_tokens must be positive")
        if self.max_num_seqs <= 0:
            raise ConfigurationError("max_num_seqs must be positive")
        if self.max_model_len <= 0:
            raise ConfigurationError("max_model_len must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ConfigurationError(
                "gpu_memory_utilization must be in the interval (0, 1]"
            )
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ConfigurationError("tensor_parallel_size must be between 1 and 8")
        if self.kvcache_block_size <= 0:
            raise ConfigurationError("kvcache_block_size must be positive")
        if self.kvcache_block_size % 256 != 0:
            raise ConfigurationError("kvcache_block_size must be a multiple of 256")
        if self.num_kvcache_blocks != -1 and self.num_kvcache_blocks <= 0:
            raise ConfigurationError(
                "num_kvcache_blocks must be -1 or a positive integer"
            )
        if self.hf_config is not None:
            max_position_embeddings = getattr(
                self.hf_config,
                "max_position_embeddings",
                None,
            )
            if max_position_embeddings is not None:
                self.clamp_model_len(int(max_position_embeddings))

    def clamp_model_len(self, max_position_embeddings: int) -> None:
        if max_position_embeddings <= 0:
            raise ConfigurationError("max_position_embeddings must be positive")
        self.max_model_len = min(self.max_model_len, max_position_embeddings)

    def to_model_config(
        self,
        *,
        device: Device = "auto",
        dtype: DType = "auto",
        trust_remote_code: bool = False,
        revision: str | None = None,
    ) -> ModelConfig:
        return ModelConfig(
            self.model,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            revision=revision,
            max_context_tokens=self.max_model_len,
        )


Config = LLMConfig
