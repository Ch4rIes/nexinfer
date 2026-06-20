import pytest

from nexinfer import ConfigurationError, LLMConfig, ModelConfig


def test_model_config_accepts_loading_options() -> None:
    config = ModelConfig(
        model_name_or_path="tiny",
        device="cpu",
        dtype="float32",
        trust_remote_code=True,
        revision="main",
        max_context_tokens=128,
    )

    assert config.model_name_or_path == "tiny"
    assert config.device == "cpu"
    assert config.dtype == "float32"
    assert config.trust_remote_code is True
    assert config.revision == "main"
    assert config.max_context_tokens == 128


def test_model_config_validates_required_name() -> None:
    with pytest.raises(ConfigurationError):
        ModelConfig(model_name_or_path="")


def test_model_config_validates_context_limit() -> None:
    with pytest.raises(ConfigurationError):
        ModelConfig(model_name_or_path="tiny", max_context_tokens=0)


def test_llm_config_matches_nano_vllm_defaults() -> None:
    config = LLMConfig("tiny")

    assert config.model == "tiny"
    assert config.max_num_batched_tokens == 16384
    assert config.max_num_seqs == 512
    assert config.max_model_len == 4096
    assert config.gpu_memory_utilization == 0.9
    assert config.tensor_parallel_size == 1
    assert config.enforce_eager is False
    assert config.eos == -1
    assert config.kvcache_block_size == 256
    assert config.num_kvcache_blocks == -1


def test_llm_config_validates_scheduler_and_cache_limits() -> None:
    with pytest.raises(ConfigurationError, match="max_num_batched_tokens"):
        LLMConfig("tiny", max_num_batched_tokens=0)

    with pytest.raises(ConfigurationError, match="max_num_seqs"):
        LLMConfig("tiny", max_num_seqs=0)

    with pytest.raises(ConfigurationError, match="max_model_len"):
        LLMConfig("tiny", max_model_len=0)

    with pytest.raises(ConfigurationError, match="gpu_memory_utilization"):
        LLMConfig("tiny", gpu_memory_utilization=1.1)

    with pytest.raises(ConfigurationError, match="tensor_parallel_size"):
        LLMConfig("tiny", tensor_parallel_size=9)

    with pytest.raises(ConfigurationError, match="multiple of 256"):
        LLMConfig("tiny", kvcache_block_size=128)

    with pytest.raises(ConfigurationError, match="num_kvcache_blocks"):
        LLMConfig("tiny", num_kvcache_blocks=0)


def test_llm_config_can_clamp_model_length_from_hf_config() -> None:
    config = LLMConfig("tiny", max_model_len=4096)

    config.clamp_model_len(1024)

    assert config.max_model_len == 1024


def test_llm_config_can_create_model_config() -> None:
    config = LLMConfig("tiny", max_model_len=1024)

    model_config = config.to_model_config(device="cpu", dtype="float32")

    assert model_config.model_name_or_path == "tiny"
    assert model_config.device == "cpu"
    assert model_config.dtype == "float32"
    assert model_config.max_context_tokens == 1024
