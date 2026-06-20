from __future__ import annotations

import random
from collections.abc import Iterator

from nexinfer.config import GenerationConfig
from nexinfer.protocols import DecoderOnlyBackend, Tokenizer
from nexinfer.result import GenerationResult, StreamChunk, TokenUsage
from nexinfer.sampling import sample_token


class LLMEngine:
    """Runs autoregressive generation against a decoder-only backend."""

    def __init__(self, backend: DecoderOnlyBackend, tokenizer: Tokenizer) -> None:
        self._backend = backend
        self._tokenizer = tokenizer

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        """Generate text from a prompt."""

        return self.complete(prompt, config).text

    def complete(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate text and return structured metadata."""

        config = config or GenerationConfig()
        prompt_token_ids = self._tokenizer.encode(prompt)
        generated_token_ids, finish_reason = self._generate_completion_token_ids(
            prompt_token_ids,
            config,
        )
        token_ids = (
            [*prompt_token_ids, *generated_token_ids]
            if config.include_prompt
            else generated_token_ids
        )

        return GenerationResult(
            text=self._tokenizer.decode(token_ids),
            token_ids=token_ids,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            finish_reason=finish_reason,
            usage=TokenUsage(
                prompt_tokens=len(prompt_token_ids),
                completion_tokens=len(generated_token_ids),
            ),
        )

    def stream(self, prompt: str, config: GenerationConfig | None = None) -> Iterator[str]:
        """Yield decoded text fragments as tokens are generated."""

        for token_id in self.generate_token_ids(prompt, config):
            yield self._tokenizer.decode([token_id])

    def stream_chunks(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[StreamChunk]:
        """Yield decoded token fragments with per-token metadata."""

        config = config or GenerationConfig()
        prompt_token_ids = self._tokenizer.encode(prompt)
        generated_token_ids, finish_reason = self._generate_completion_token_ids(
            prompt_token_ids,
            config,
        )

        for index, token_id in enumerate(generated_token_ids):
            is_last = index == len(generated_token_ids) - 1
            yield StreamChunk(
                text=self._tokenizer.decode([token_id]),
                token_id=token_id,
                index=index,
                finish_reason=finish_reason if is_last else None,
            )

    def generate_token_ids(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> list[int]:
        """Generate token ids from a prompt."""

        config = config or GenerationConfig()
        prompt_token_ids = self._tokenizer.encode(prompt)
        generated_token_ids, _ = self._generate_completion_token_ids(
            prompt_token_ids,
            config,
        )
        if config.include_prompt:
            return [*prompt_token_ids, *generated_token_ids]
        return generated_token_ids

    def _generate_completion_token_ids(
        self,
        input_ids: list[int],
        config: GenerationConfig,
    ) -> tuple[list[int], str]:
        if config.max_new_tokens == 0:
            return [], "length"

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
                return generated, "stop"

            generated.append(token_id)
            output = self._backend.step(token_id, output.state)
            _validate_vocab_size(output.logits, self._backend.vocab_size)

        return generated, "length"


def _validate_vocab_size(logits: object, vocab_size: int) -> None:
    try:
        actual = len(logits)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("backend logits must be a sized sequence") from exc

    if actual != vocab_size:
        raise ValueError(
            f"backend returned {actual} logits, expected vocab size {vocab_size}"
        )
