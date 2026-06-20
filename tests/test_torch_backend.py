from collections.abc import Sequence
from typing import Any

from nexinfer import DecodeInput, DecodeState, ModelOutput, PrefillInput
from nexinfer.backends import TorchCausalLMBackend


class FakeTorchBackend(TorchCausalLMBackend):
    def __init__(self) -> None:
        super().__init__(object(), vocab_size=10, device="cpu", block_size=2)

    def _forward(
        self,
        input_ids: Sequence[int],
        *,
        position: int,
        past_key_values: Any,
    ) -> ModelOutput:
        return ModelOutput(
            logits=[0.0] * self.vocab_size,
            state=DecodeState(position=position, backend_state=past_key_values),
        )


def test_torch_backend_exposes_vocab_size_and_device() -> None:
    backend = TorchCausalLMBackend(object(), vocab_size=32000, device="cpu")

    assert backend.vocab_size == 32000
    assert backend.device == "cpu"


def test_torch_backend_begin_batch_records_prepared_prefill_context() -> None:
    backend = FakeTorchBackend()

    outputs = backend.begin_batch(
        [
            PrefillInput(
                token_ids=[10, 11, 12, 13],
                num_cached_tokens=2,
                num_scheduled_tokens=2,
                block_table=(3, 4),
            )
        ]
    )

    context = backend.last_context
    assert outputs[0].state.position == 4
    assert context is not None
    assert context.is_prefill is True
    assert context.input_ids == [12, 13]
    assert context.positions == [2, 3]
    assert context.slot_mapping == [8, 9]
    assert context.block_tables == [[3, 4]]
    assert context.cu_seqlens_q == [0, 2]
    assert context.cu_seqlens_k == [0, 4]
    assert context.max_seqlen_q == 2
    assert context.max_seqlen_k == 4
    assert context.context_lengths is None


def test_torch_backend_step_batch_records_prepared_decode_context() -> None:
    backend = FakeTorchBackend()

    outputs = backend.step_batch(
        [
            DecodeInput(
                token_id=7,
                state=DecodeState(position=4, backend_state="cache"),
                block_table=(1, 2, 3),
                context_length=5,
            )
        ]
    )

    context = backend.last_context
    assert outputs[0].state.position == 5
    assert context is not None
    assert context.is_prefill is False
    assert context.input_ids == [7]
    assert context.positions == [4]
    assert context.slot_mapping == [6]
    assert context.block_tables == [[1, 2, 3]]
    assert context.context_lengths == [5]
    assert context.cu_seqlens_q is None
    assert context.cu_seqlens_k is None
