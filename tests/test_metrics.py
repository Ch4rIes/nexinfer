from nexinfer import RuntimeStats, TokenUsage
from nexinfer.result import GenerationResult


def _result(prompt_tokens: int, completion_tokens: int) -> GenerationResult:
    return GenerationResult(
        text="",
        token_ids=[],
        prompt_token_ids=[],
        generated_token_ids=[],
        generated_token_logprobs=[],
        finish_reason="length",
        usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def test_runtime_stats_record_batch() -> None:
    stats = RuntimeStats().record_batch((_result(2, 3), _result(1, 1)))

    assert stats.batches == 1
    assert stats.requests == 2
    assert stats.prompt_tokens == 3
    assert stats.completion_tokens == 4
    assert stats.total_tokens == 7
