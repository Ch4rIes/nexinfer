from __future__ import annotations

from dataclasses import dataclass

from nexinfer.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Controls how next-token logits are converted into one token id."""

    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ConfigurationError("temperature must be non-negative")
        if self.top_k is not None and self.top_k <= 0:
            raise ConfigurationError("top_k must be positive when set")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ConfigurationError("top_p must be in the interval (0, 1]")


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Controls text generation."""

    max_new_tokens: int = 32
    max_prompt_tokens: int | None = None
    max_total_tokens: int | None = None
    sampling: SamplingConfig = SamplingConfig()
    stop_token_ids: tuple[int, ...] = ()
    include_prompt: bool = False
    include_stop_token: bool = False

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ConfigurationError("max_new_tokens must be non-negative")
        if self.max_prompt_tokens is not None and self.max_prompt_tokens < 0:
            raise ConfigurationError("max_prompt_tokens must be non-negative when set")
        if self.max_total_tokens is not None and self.max_total_tokens < 0:
            raise ConfigurationError("max_total_tokens must be non-negative when set")
