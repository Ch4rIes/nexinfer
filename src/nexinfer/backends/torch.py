from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from nexinfer.model import DType, Device, ModelConfig
from nexinfer.protocols import DecodeState, ModelOutput


class TorchCausalLMBackend:
    """Optional PyTorch backend for Hugging Face-style causal language models."""

    def __init__(self, model: Any, *, vocab_size: int, device: str = "cpu") -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self._model = model
        self._vocab_size = vocab_size
        self._device = device

    @classmethod
    def from_pretrained(cls, config: ModelConfig, **kwargs: Any) -> "TorchCausalLMBackend":
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                "Install NexInfer with the torch extra to load PyTorch models."
            ) from exc

        dtype = _resolve_torch_dtype(torch, config.dtype)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=dtype,
            **kwargs,
        )
        device = _resolve_torch_device(torch, config.device)
        model.to(device)
        model.eval()
        return cls(model, vocab_size=int(model.config.vocab_size), device=str(device))

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def device(self) -> str:
        return self._device

    def begin(self, input_ids: Sequence[int]) -> ModelOutput:
        return self._forward(input_ids, position=len(input_ids), past_key_values=None)

    def step(self, token_id: int, state: DecodeState) -> ModelOutput:
        return self._forward(
            [token_id],
            position=state.position + 1,
            past_key_values=state.backend_state,
        )

    def _forward(
        self,
        input_ids: Sequence[int],
        *,
        position: int,
        past_key_values: Any,
    ) -> ModelOutput:
        import torch

        if not input_ids:
            raise ValueError("input_ids must not be empty")

        tensor = torch.tensor([list(input_ids)], device=self._device)
        with torch.inference_mode():
            output = self._model(
                input_ids=tensor,
                past_key_values=past_key_values,
                use_cache=True,
            )

        logits = output.logits[0, -1].detach().float().cpu().tolist()
        cache = output.past_key_values
        return ModelOutput(
            logits=logits,
            state=DecodeState(
                position=position,
                backend_state=cache,
                cache=cache,
            ),
        )


def _resolve_torch_dtype(torch: Any, dtype: DType) -> Any:
    if dtype == "auto":
        return "auto"
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {dtype}")


def _resolve_torch_device(torch: Any, device: Device) -> Any:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
