from __future__ import annotations

from dataclasses import dataclass

from nexinfer.result import GenerationResult


@dataclass(frozen=True, slots=True)
class RuntimeStats:
    """Cumulative runtime execution counters."""

    batches: int = 0
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record_batch(self, results: tuple[GenerationResult, ...]) -> "RuntimeStats":
        return RuntimeStats(
            batches=self.batches + 1,
            requests=self.requests + len(results),
            prompt_tokens=self.prompt_tokens
            + sum(result.usage.prompt_tokens for result in results),
            completion_tokens=self.completion_tokens
            + sum(result.usage.completion_tokens for result in results),
        )
