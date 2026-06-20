import pytest

from nexinfer import GenerationConfig, RequestQueue, SamplingConfig


def test_request_queue_schedules_fifo_batches() -> None:
    queue = RequestQueue()
    first = queue.submit("first", request_id="a")
    second = queue.submit("second", request_id="b")
    queue.submit("third", request_id="c")

    batch = queue.schedule(max_requests=2)

    assert batch.requests == (first, second)
    assert len(batch) == 2
    assert len(queue) == 1


def test_request_queue_generates_request_ids() -> None:
    queue = RequestQueue()

    first = queue.submit("hello")
    second = queue.submit("world")

    assert first.request_id == "req-1"
    assert second.request_id == "req-2"


def test_request_queue_preserves_request_config() -> None:
    queue = RequestQueue()
    config = GenerationConfig(sampling=SamplingConfig(temperature=0))

    request = queue.submit("hello", config=config)

    assert request.config is config


def test_request_queue_preserves_request_metadata() -> None:
    queue = RequestQueue()

    request = queue.submit(
        "hello",
        request_id="traceable",
        metadata={"tenant": "demo", "trace": "abc"},
    )

    assert request.metadata == {"tenant": "demo", "trace": "abc"}


def test_request_queue_tracks_prompt_token_counts() -> None:
    queue = RequestQueue()
    queue.submit("a", request_id="a", prompt_token_count=2)
    queue.submit("b", request_id="b", prompt_token_count=3)

    batch = queue.schedule(max_requests=10, max_prompt_tokens=2)

    assert [request.request_id for request in batch.requests] == ["a"]
    assert batch.prompt_tokens == 2
    assert len(queue) == 1


def test_request_queue_allows_first_request_over_prompt_budget() -> None:
    queue = RequestQueue()
    queue.submit("large", request_id="large", prompt_token_count=8)

    batch = queue.schedule(max_requests=10, max_prompt_tokens=4)

    assert [request.request_id for request in batch.requests] == ["large"]


def test_request_queue_cancels_pending_request() -> None:
    queue = RequestQueue()
    queue.submit("first", request_id="a")
    queue.submit("second", request_id="b")

    assert queue.cancel("a") is True
    assert queue.cancel("missing") is False
    assert [request.request_id for request in queue.schedule(max_requests=10).requests] == [
        "b"
    ]


def test_request_queue_rejects_duplicate_pending_ids() -> None:
    queue = RequestQueue()
    queue.submit("first", request_id="a")

    with pytest.raises(ValueError, match="duplicate"):
        queue.submit("again", request_id="a")
