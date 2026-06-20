from nexinfer import GenerationConfig, GenerationRequest, LLMEngine, SamplingConfig
from nexinfer.backends import BigramBackend
from nexinfer.tokenizer import VocabularyTokenizer


def test_engine_decodes_active_sequence_one_step_at_a_time() -> None:
    tokenizer = VocabularyTokenizer(["a", "b", "c", "<eos>"], eos_token="<eos>")
    backend = BigramBackend(
        vocab_size=len(tokenizer),
        transitions={
            tokenizer.token_id("a"): {tokenizer.token_id("b"): 5.0},
            tokenizer.token_id("b"): {tokenizer.token_id("c"): 5.0},
            tokenizer.token_id("c"): {tokenizer.eos_token_id: 5.0},
        },
    )
    engine = LLMEngine(backend, tokenizer)
    request = GenerationRequest(
        request_id="one",
        prompt="a",
        config=GenerationConfig(
            max_new_tokens=8,
            sampling=SamplingConfig(temperature=0),
            stop_token_ids=(tokenizer.eos_token_id,),
        ),
    )

    active = engine.start_request(request)
    assert active.request_id == "one"
    assert active.sequence.generated_token_ids == []

    engine.decode_one(active)
    assert active.sequence.generated_token_ids == [tokenizer.token_id("b")]
    assert active.is_finished is False

    engine.decode_one(active)
    assert active.sequence.generated_token_ids == [
        tokenizer.token_id("b"),
        tokenizer.token_id("c"),
    ]
    assert active.is_finished is False

    engine.decode_one(active)
    assert active.sequence.finish_reason == "stop"
    assert active.output is None


def test_active_sequence_finishes_when_effective_limit_is_zero() -> None:
    tokenizer = VocabularyTokenizer(["a"])
    backend = BigramBackend(vocab_size=len(tokenizer))
    engine = LLMEngine(backend, tokenizer)
    request = GenerationRequest(
        request_id="done",
        prompt="a",
        config=GenerationConfig(max_new_tokens=0),
    )

    active = engine.start_request(request)

    assert active.is_finished is True
    assert active.sequence.finish_reason == "length"
