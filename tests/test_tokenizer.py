from nexinfer import HuggingFaceTokenizer


class FakeHFTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        return "".join(chr(token_id) for token_id in token_ids)


def test_hugging_face_tokenizer_adapter() -> None:
    tokenizer = HuggingFaceTokenizer(FakeHFTokenizer())

    token_ids = tokenizer.encode("hi")

    assert token_ids == [104, 105]
    assert tokenizer.decode(token_ids) == "hi"
