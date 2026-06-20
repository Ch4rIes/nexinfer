from __future__ import annotations

import random
from collections.abc import Iterator

from nexinfer.config import GenerationConfig
from nexinfer.protocols import DecoderOnlyBackend, Tokenizer
from nexinfer.sampling import sample_token


class LLMEngine:
    """Runs autoregressive generation against a decoder-only backend."""

    def __init__(self, backend: DecoderOnlyBackend, tokenizer: Tokenizer) -> None:
        self._backend = backend
        self._tokenizer = tokenizer

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        """Generate text from a prompt."""

        config = config or GenerationConfig()
        token_ids = self.generate_token_ids(prompt, config)
        return self._tokenizer.decode(token_ids)

    def stream(self, prompt: str, config: GenerationConfig | None = None) -> Iterator[str]:
        """Yield decoded text fragments as tokens are generated."""

        for token_id in self.generate_token_ids(prompt, config):
            yield self._tokenizer.decode([token_id])

    def generate_token_ids(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> list[int]:
        """Generate token ids from a prompt."""

        config = config or GenerationConfig()
        input_ids = self._tokenizer.encode(prompt)
        if config.max_new_tokens == 0:
            return input_ids if config.include_prompt else []

        output = self._backend.begin(input_ids)
        _validate_vocab_size(output.logits, self._backend.vocab_size)

        rng = random.Random(config.sampling.seed)
        generated: list[int] = []
        stop_token_ids = set(config.stop_token_ids)

        for _ in range(config.max_new_tokens):
            token_id = sample_token(output.logits, config.sampling, rng)
            if token_id in stop_token_ids:
                if config.include_stop_token:
                    generated.append(token_id)
                break

            generated.append(token_id)
            output = self._backend.step(token_id, output.state)
            _validate_vocab_size(output.logits, self._backend.vocab_size)

        if config.include_prompt:
            return [*input_ids, *generated]
        return generated


def _validate_vocab_size(logits: object, vocab_size: int) -> None:
    try:
        actual = len(logits)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("backend logits must be a sized sequence") from exc

    if actual != vocab_size:
        raise ValueError(
            f"backend returned {actual} logits, expected vocab size {vocab_size}"
        )
