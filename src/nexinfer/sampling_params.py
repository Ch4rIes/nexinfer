from __future__ import annotations

from dataclasses import dataclass

from nexinfer.config import GenerationConfig, SamplingConfig
from nexinfer.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Nano-VLLM-compatible sampling parameters."""

    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.temperature <= 1e-10:
            raise ConfigurationError("greedy sampling is not permitted")
        if self.max_tokens < 0:
            raise ConfigurationError("max_tokens must be non-negative")

    def to_generation_config(
        self,
        *,
        eos_token_id: int | None = None,
        max_total_tokens: int | None = None,
    ) -> GenerationConfig:
        stop_token_ids: tuple[int, ...]
        if self.ignore_eos or eos_token_id is None:
            stop_token_ids = ()
        else:
            stop_token_ids = (eos_token_id,)

        return GenerationConfig(
            max_new_tokens=self.max_tokens,
            max_total_tokens=max_total_tokens,
            sampling=SamplingConfig(temperature=self.temperature),
            stop_token_ids=stop_token_ids,
        )
