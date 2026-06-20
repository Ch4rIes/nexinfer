import pytest

from nexinfer import ConfigurationError, ModelConfig


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
