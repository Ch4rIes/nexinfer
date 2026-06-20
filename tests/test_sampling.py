import random

import pytest

from nexinfer import SamplingConfig
from nexinfer.sampling import sample_token


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
