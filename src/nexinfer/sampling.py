from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from nexinfer.config import SamplingConfig


@dataclass(frozen=True, slots=True)
class SampledToken:
    """A token selected from a sampling distribution."""

    token_id: int
    probability: float
    logprob: float


class Sampler:
    """Nano-VLLM-style batched temperature sampler."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def __call__(
        self,
        logits: Sequence[Sequence[float]],
        temperatures: Sequence[float],
    ) -> list[int]:
        if len(logits) != len(temperatures):
            raise ValueError("logits and temperatures must have the same length")
        return [
            self.sample_row(row, temperature)
            for row, temperature in zip(logits, temperatures, strict=True)
        ]

    def sample_row(self, logits: Sequence[float], temperature: float) -> int:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not logits:
            raise ValueError("logits must not be empty")
        _validate_logits(logits)
        probabilities = _softmax([value / temperature for value in logits])
        return max(
            range(len(probabilities)),
            key=lambda index: probabilities[index] / self._exponential_sample(),
        )

    def _exponential_sample(self) -> float:
        return max(self._rng.expovariate(1.0), 1e-10)


def sample_token(
    logits: Sequence[float],
    config: SamplingConfig | None = None,
    rng: random.Random | None = None,
) -> int:
    """Select a token id from logits using greedy, top-k, and nucleus sampling."""

    return sample_next(logits, config, rng).token_id


def sample_next(
    logits: Sequence[float],
    config: SamplingConfig | None = None,
    rng: random.Random | None = None,
) -> SampledToken:
    """Select the next token and return its sampling metadata."""

    if not logits:
        raise ValueError("logits must not be empty")

    config = config or SamplingConfig()
    rng = rng or random.Random(config.seed)
    _validate_logits(logits)

    if config.temperature == 0:
        token_id = max(range(len(logits)), key=logits.__getitem__)
        return SampledToken(token_id=token_id, probability=1.0, logprob=0.0)

    scaled = [value / config.temperature for value in logits]
    probabilities = _softmax(scaled)
    candidates = list(enumerate(probabilities))

    if config.top_k is not None:
        candidates = sorted(candidates, key=lambda item: item[1], reverse=True)[
            : config.top_k
        ]

    if config.top_p is not None and config.top_p < 1:
        sorted_candidates = sorted(candidates, key=lambda item: item[1], reverse=True)
        kept: list[tuple[int, float]] = []
        cumulative = 0.0
        for token_id, probability in sorted_candidates:
            kept.append((token_id, probability))
            cumulative += probability
            if cumulative >= config.top_p:
                break
        candidates = kept

    total = sum(probability for _, probability in candidates)
    if total <= 0:
        token_id = max(range(len(logits)), key=logits.__getitem__)
        return SampledToken(token_id=token_id, probability=1.0, logprob=0.0)

    threshold = rng.random() * total
    cumulative = 0.0
    for token_id, probability in candidates:
        cumulative += probability
        if cumulative >= threshold:
            normalized_probability = probability / total
            return SampledToken(
                token_id=token_id,
                probability=normalized_probability,
                logprob=math.log(normalized_probability),
            )

    token_id, probability = candidates[-1]
    normalized_probability = probability / total
    return SampledToken(
        token_id=token_id,
        probability=normalized_probability,
        logprob=math.log(normalized_probability),
    )


def _softmax(values: Sequence[float]) -> list[float]:
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def _validate_logits(logits: Sequence[float]) -> None:
    for value in logits:
        if not math.isfinite(value):
            raise ValueError("logits must contain only finite values")
