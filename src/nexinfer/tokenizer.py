from __future__ import annotations

from collections.abc import Sequence


class VocabularyTokenizer:
    """A tiny whitespace tokenizer useful for examples, tests, and toy models."""

    def __init__(
        self,
        vocabulary: Sequence[str],
        *,
        unk_token: str = "<unk>",
        eos_token: str | None = None,
    ) -> None:
        tokens = list(dict.fromkeys(vocabulary))
        if unk_token not in tokens:
            tokens.append(unk_token)
        if eos_token is not None and eos_token not in tokens:
            tokens.append(eos_token)

        self._token_to_id = {token: index for index, token in enumerate(tokens)}
        self._id_to_token = {index: token for token, index in self._token_to_id.items()}
        self.unk_token = unk_token
        self.eos_token = eos_token
        self.unk_token_id = self._token_to_id[unk_token]
        self.eos_token_id = (
            self._token_to_id[eos_token] if eos_token is not None else None
        )

    def __len__(self) -> int:
        return len(self._token_to_id)

    def token_id(self, token: str) -> int:
        return self._token_to_id[token]

    def token(self, token_id: int) -> str:
        return self._id_to_token[token_id]

    def encode(self, text: str) -> list[int]:
        if not text.strip():
            return []
        return [
            self._token_to_id.get(piece, self.unk_token_id)
            for piece in text.strip().split()
        ]

    def decode(self, token_ids: Sequence[int]) -> str:
        tokens = [self._id_to_token[token_id] for token_id in token_ids]
        return " ".join(tokens)

