import pytest

from nexinfer import (
    BackendError,
    ConfigurationError,
    GenerationConfig,
    KVCacheBlockAllocator,
    NexInferError,
    RequestQueue,
    SchedulerError,
    SamplingConfig,
)


def test_configuration_errors_share_base_type() -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        SamplingConfig(temperature=-1)

    assert isinstance(exc_info.value, NexInferError)
    assert isinstance(exc_info.value, ValueError)


def test_scheduler_errors_share_base_type() -> None:
    queue = RequestQueue()

    with pytest.raises(SchedulerError):
        queue.schedule(max_requests=0)


def test_cache_errors_preserve_memory_error_compatibility() -> None:
    allocator = KVCacheBlockAllocator(block_size=1, max_blocks=1)
    allocator.allocate("one", 1)

    with pytest.raises(MemoryError) as exc_info:
        allocator.allocate("two", 1)

    assert isinstance(exc_info.value, NexInferError)


def test_backend_error_is_value_error_compatible() -> None:
    assert issubclass(BackendError, ValueError)


def test_generation_config_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        GenerationConfig(max_new_tokens=-1)

    with pytest.raises(ConfigurationError):
        GenerationConfig(max_prompt_tokens=-1)

    with pytest.raises(ConfigurationError):
        GenerationConfig(max_total_tokens=-1)
