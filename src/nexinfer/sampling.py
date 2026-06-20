from __future__ import annotations

import math
import random
from collections.abc import Sequence

from nexinfer.config import SamplingConfig


def sample_token(
    logits: Sequence[float],
    config: SamplingConfig | None = None,
    rng: random.Random | None = None,
) -> int:
    """Select a token id from logits using greedy, top-k, and nucleus sampling."""

    if not logits:
        raise ValueError("logits must not be empty")

    config = config or SamplingConfig()
    rng = rng or random.Random(config.seed)
    _validate_logits(logits)

    if config.temperature == 0:
        return max(range(len(logits)), key=logits.__getitem__)

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
        return max(range(len(logits)), key=logits.__getitem__)

    threshold = rng.random() * total
    cumulative = 0.0
    for token_id, probability in candidates:
        cumulative += probability
        if cumulative >= threshold:
            return token_id

    return candidates[-1][0]


def _softmax(values: Sequence[float]) -> list[float]:
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def _validate_logits(logits: Sequence[float]) -> None:
    for value in logits:
        if not math.isfinite(value):
            raise ValueError("logits must contain only finite values")
