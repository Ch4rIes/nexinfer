import random

import pytest

from nexinfer import Sampler, SamplingConfig
from nexinfer.sampling import sample_next, sample_token


class FakeExponentialRng(random.Random):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self._values = values

    def expovariate(self, lambd: float) -> float:
        assert lambd == 1.0
        return self._values.pop(0)


def test_temperature_zero_is_greedy() -> None:
    token_id = sample_token([1.0, 3.0, 2.0], SamplingConfig(temperature=0))

    assert token_id == 1


def test_top_k_excludes_lower_ranked_tokens() -> None:
    rng = random.Random(7)

    for _ in range(100):
        token_id = sample_token(
            [10.0, 9.0, -100.0],
            SamplingConfig(top_k=2),
            rng,
        )
        assert token_id in {0, 1}


def test_top_p_keeps_highest_probability_prefix() -> None:
    rng = random.Random(13)

    for _ in range(100):
        token_id = sample_token(
            [100.0, 90.0, 80.0],
            SamplingConfig(top_p=0.5),
            rng,
        )
        assert token_id == 0


def test_rejects_invalid_logits() -> None:
    with pytest.raises(ValueError, match="finite"):
        sample_token([1.0, float("nan")])


def test_sample_next_returns_logprob_metadata() -> None:
    sampled = sample_next(
        [0.0, 0.0],
        SamplingConfig(seed=1),
    )

    assert sampled.token_id in {0, 1}
    assert sampled.probability == pytest.approx(0.5)
    assert sampled.logprob == pytest.approx(-0.6931471805599453)


def test_sampler_selects_batched_tokens_with_exponential_race() -> None:
    sampler = Sampler(FakeExponentialRng([10.0, 1.0, 1.0, 10.0]))

    token_ids = sampler(
        [
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        [1.0, 1.0],
    )

    assert token_ids == [1, 0]


def test_sampler_applies_per_row_temperature() -> None:
    sampler = Sampler(FakeExponentialRng([1.0, 1.0, 1.0, 1.0]))

    token_ids = sampler(
        [
            [0.0, 2.0],
            [3.0, 0.0],
        ],
        [0.5, 2.0],
    )

    assert token_ids == [1, 0]


def test_sampler_validates_batch_shapes_and_temperature() -> None:
    sampler = Sampler()

    with pytest.raises(ValueError, match="same length"):
        sampler([[0.0, 1.0]], [1.0, 1.0])

    with pytest.raises(ValueError, match="temperature"):
        sampler([[0.0, 1.0]], [0.0])

    with pytest.raises(ValueError, match="finite"):
        sampler([[float("inf"), 1.0]], [1.0])

    with pytest.raises(ValueError, match="empty"):
        sampler([[]], [1.0])
