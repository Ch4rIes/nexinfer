from nexinfer.backends import BigramBackend


def test_bigram_backend_tracks_decode_position() -> None:
    backend = BigramBackend(
        vocab_size=3,
        transitions={
            1: {2: 5.0},
            2: {0: 5.0},
        },
    )

    prefill = backend.begin([0, 1])
    decode = backend.step(2, prefill.state)

    assert prefill.state.position == 2
    assert prefill.state.backend_state == 1
    assert decode.state.position == 3
    assert decode.state.backend_state == 2
