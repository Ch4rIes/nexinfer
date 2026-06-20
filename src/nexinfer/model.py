from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
