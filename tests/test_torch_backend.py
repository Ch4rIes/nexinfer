from nexinfer.backends import TorchCausalLMBackend


def test_torch_backend_exposes_vocab_size_and_device() -> None:
    backend = TorchCausalLMBackend(object(), vocab_size=32000, device="cpu")

    assert backend.vocab_size == 32000
    assert backend.device == "cpu"
