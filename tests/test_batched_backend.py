from nexinfer import DecodeInput, PrefillInput
from nexinfer.backends import BigramBackend


def test_bigram_backend_supports_batched_prefill_and_decode() -> None:
    backend = BigramBackend(
        vocab_size=4,
        transitions={
            1: {2: 5.0},
            3: {0: 5.0},
        },
    )

    prefill_outputs = backend.begin_batch([[0, 1], [3]])
    decode_outputs = backend.step_batch(
        [
            DecodeInput(token_id=2, state=prefill_outputs[0].state),
            DecodeInput(token_id=0, state=prefill_outputs[1].state),
        ]
    )

    assert prefill_outputs[0].state.position == 2
    assert prefill_outputs[1].state.position == 1
    assert decode_outputs[0].state.position == 3
    assert decode_outputs[1].state.position == 2


def test_bigram_backend_accepts_structured_prefill_inputs() -> None:
    backend = BigramBackend(
        vocab_size=4,
        transitions={
            1: {2: 5.0},
        },
    )

    outputs = backend.begin_batch(
        [
            PrefillInput(
                token_ids=[0, 1],
                num_cached_tokens=1,
                num_scheduled_tokens=1,
                block_table=(7,),
            )
        ]
    )

    assert outputs[0].state.position == 2
