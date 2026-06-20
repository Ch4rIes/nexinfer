from nexinfer import SequenceState


def test_sequence_state_tracks_positions_and_output_tokens() -> None:
    state = SequenceState(prompt_token_ids=[10, 11])

    assert state.next_position == 2

    state.append(12, -0.5)
    state.append(13, -0.25)
    state.finish("length")

    assert state.token_ids == [10, 11, 12, 13]
    assert state.generated_token_ids == [12, 13]
    assert state.generated_token_logprobs == [-0.5, -0.25]
    assert state.next_position == 4
    assert state.completion_tokens == 2
    assert state.finish_reason == "length"
    assert state.output_token_ids(include_prompt=False) == [12, 13]
    assert state.output_token_ids(include_prompt=True) == [10, 11, 12, 13]
