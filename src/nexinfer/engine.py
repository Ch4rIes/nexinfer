from __future__ import annotations

import random
from collections.abc import Iterator

from nexinfer.config import GenerationConfig
from nexinfer.errors import BackendError, ConfigurationError
from nexinfer.protocols import (
    BatchedDecoderOnlyBackend,
    DecodeInput,
    DecodeState,
    DecoderOnlyBackend,
    ModelOutput,
    Tokenizer,
)
from nexinfer.result import GenerationResult, StreamChunk, TokenUsage
from nexinfer.sampling import sample_next
from nexinfer.scheduler import ActiveSequence, GenerationRequest
from nexinfer.state import SequenceState


class LLMEngine:
    """Runs autoregressive generation against a decoder-only backend."""

    def __init__(self, backend: DecoderOnlyBackend, tokenizer: Tokenizer) -> None:
        self._backend = backend
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        """Generate text from a prompt."""

        return self.complete(prompt, config).text

    def generate_batch(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
    ) -> list[str]:
        """Generate text for multiple prompts."""

        return [result.text for result in self.complete_batch(prompts, config)]

    def complete(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate text and return structured metadata."""

        config = config or GenerationConfig()
        prompt_token_ids = self._tokenizer.encode(prompt)
        _validate_prompt_limits(prompt_token_ids, config)
        sequence = self._generate_sequence(
            prompt_token_ids,
            config,
        )
        return self._result_from_sequence(sequence, config)

    def complete_batch(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
    ) -> list[GenerationResult]:
        """Generate structured results for multiple prompts."""

        return [self.complete(prompt, config) for prompt in prompts]

    def complete_requests(
        self,
        requests: list[GenerationRequest],
    ) -> list[GenerationResult]:
        """Generate structured results for scheduled requests."""

        return [self.complete(request.prompt, request.config) for request in requests]

    def complete_requests_interleaved(
        self,
        requests: list[GenerationRequest],
    ) -> list[GenerationResult]:
        """Generate scheduled requests by round-robin decoding active sequences."""

        active_sequences = self.start_requests(requests)
        while any(not active.is_finished for active in active_sequences):
            self.decode_active_batch(
                [active for active in active_sequences if not active.is_finished]
            )

        return [
            self._result_from_sequence(active.sequence, active.request.config)
            for active in active_sequences
        ]

    def start_request(self, request: GenerationRequest) -> ActiveSequence:
        """Admit a scheduled request and run its prefill step."""

        prompt_token_ids = self._tokenizer.encode(request.prompt)
        _validate_prompt_limits(prompt_token_ids, request.config)
        sequence = SequenceState(prompt_token_ids=prompt_token_ids)
        max_new_tokens = _effective_max_new_tokens(prompt_token_ids, request.config)
        if max_new_tokens == 0:
            sequence.finish("length")
            return ActiveSequence(
                request=request,
                sequence=sequence,
                output=None,
                rng=random.Random(request.config.sampling.seed),
                max_new_tokens=0,
                stop_token_ids=set(request.config.stop_token_ids),
            )

        output = self._backend.begin(prompt_token_ids)
        _validate_model_output(output, self._backend.vocab_size)
        return ActiveSequence(
            request=request,
            sequence=sequence,
            output=output,
            rng=random.Random(request.config.sampling.seed),
            max_new_tokens=max_new_tokens,
            stop_token_ids=set(request.config.stop_token_ids),
        )

    def start_requests(self, requests: list[GenerationRequest]) -> list[ActiveSequence]:
        """Admit scheduled requests and batch their prefill step when possible."""

        active_sequences: list[ActiveSequence] = []
        prefill_prompts: list[list[int]] = []
        prefill_indices: list[int] = []

        for request in requests:
            prompt_token_ids = self._tokenizer.encode(request.prompt)
            _validate_prompt_limits(prompt_token_ids, request.config)
            sequence = SequenceState(prompt_token_ids=prompt_token_ids)
            max_new_tokens = _effective_max_new_tokens(prompt_token_ids, request.config)
            active = ActiveSequence(
                request=request,
                sequence=sequence,
                output=None,
                rng=random.Random(request.config.sampling.seed),
                max_new_tokens=max_new_tokens,
                stop_token_ids=set(request.config.stop_token_ids),
            )
            active_sequences.append(active)
            if max_new_tokens == 0:
                sequence.finish("length")
            else:
                prefill_prompts.append(prompt_token_ids)
                prefill_indices.append(len(active_sequences) - 1)

        if prefill_prompts:
            outputs = self._begin_batch(prefill_prompts)
            if len(outputs) != len(prefill_prompts):
                raise BackendError("backend returned wrong number of prefill outputs")
            for active_index, output in zip(prefill_indices, outputs, strict=True):
                _validate_model_output(output, self._backend.vocab_size)
                active_sequences[active_index].output = output

        return active_sequences

    def prefill_active_sequences(
        self,
        active_sequences: list[ActiveSequence],
    ) -> list[ActiveSequence]:
        """Re-prefill preempted active sequences from their current token history."""

        prefill_inputs: list[list[int]] = []
        prefill_indices: list[int] = []
        for index, active in enumerate(active_sequences):
            if active.is_finished:
                continue
            if active.sequence.completion_tokens >= active.max_new_tokens:
                active.sequence.finish("length")
                active.output = None
                continue
            prefill_inputs.append(active.sequence.token_ids)
            prefill_indices.append(index)

        if prefill_inputs:
            outputs = self._begin_batch(prefill_inputs)
            if len(outputs) != len(prefill_inputs):
                raise BackendError("backend returned wrong number of prefill outputs")
            for active_index, output in zip(prefill_indices, outputs, strict=True):
                _validate_model_output(output, self._backend.vocab_size)
                active_sequences[active_index].output = output

        return active_sequences

    def decode_one(self, active: ActiveSequence) -> ActiveSequence:
        """Decode at most one token for an active sequence."""

        if not active.can_decode:
            if not active.is_finished:
                active.sequence.finish("length")
            return active

        assert active.output is not None
        sampled = sample_next(
            active.output.logits,
            active.request.config.sampling,
            active.rng,
        )
        token_id = sampled.token_id
        if token_id in active.stop_token_ids:
            if active.request.config.include_stop_token:
                active.sequence.append(token_id, sampled.logprob)
            active.sequence.finish("stop")
            active.output = None
            return active

        active.sequence.append(token_id, sampled.logprob)
        if active.sequence.completion_tokens >= active.max_new_tokens:
            active.sequence.finish("length")
            active.output = None
            return active

        active.output = self._backend.step(token_id, active.output.state)
        _validate_model_output(active.output, self._backend.vocab_size)
        return active

    def decode_active_batch(
        self,
        active_sequences: list[ActiveSequence],
    ) -> list[ActiveSequence]:
        """Decode at most one token for each active sequence in a batch."""

        decode_inputs: list[DecodeInput] = []
        decode_indices: list[int] = []

        for index, active in enumerate(active_sequences):
            if not active.can_decode:
                if not active.is_finished:
                    active.sequence.finish("length")
                continue

            assert active.output is not None
            sampled = sample_next(
                active.output.logits,
                active.request.config.sampling,
                active.rng,
            )
            token_id = sampled.token_id
            if token_id in active.stop_token_ids:
                if active.request.config.include_stop_token:
                    active.sequence.append(token_id, sampled.logprob)
                active.sequence.finish("stop")
                active.output = None
                continue

            active.sequence.append(token_id, sampled.logprob)
            if active.sequence.completion_tokens >= active.max_new_tokens:
                active.sequence.finish("length")
                active.output = None
                continue

            decode_inputs.append(DecodeInput(token_id=token_id, state=active.output.state))
            decode_indices.append(index)

        if decode_inputs:
            outputs = self._step_batch(decode_inputs)
            if len(outputs) != len(decode_inputs):
                raise BackendError("backend returned wrong number of decode outputs")
            for active_index, output in zip(decode_indices, outputs, strict=True):
                _validate_model_output(output, self._backend.vocab_size)
                active_sequences[active_index].output = output

        return active_sequences

    def result_from_active(self, active: ActiveSequence) -> GenerationResult:
        """Build a generation result from an active sequence."""

        return self._result_from_sequence(active.sequence, active.request.config)

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
        _validate_prompt_limits(prompt_token_ids, config)
        sequence = self._generate_sequence(
            prompt_token_ids,
            config,
        )

        for index, token_id in enumerate(sequence.generated_token_ids):
            is_last = index == len(sequence.generated_token_ids) - 1
            yield StreamChunk(
                text=self._tokenizer.decode([token_id]),
                token_id=token_id,
                index=index,
                logprob=sequence.generated_token_logprobs[index],
                finish_reason=sequence.finish_reason if is_last else None,
            )

    def generate_token_ids(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> list[int]:
        """Generate token ids from a prompt."""

        config = config or GenerationConfig()
        prompt_token_ids = self._tokenizer.encode(prompt)
        _validate_prompt_limits(prompt_token_ids, config)
        sequence = self._generate_sequence(
            prompt_token_ids,
            config,
        )
        return sequence.output_token_ids(include_prompt=config.include_prompt)

    def _generate_sequence(
        self,
        input_ids: list[int],
        config: GenerationConfig,
    ) -> SequenceState:
        sequence = SequenceState(prompt_token_ids=input_ids)
        max_new_tokens = _effective_max_new_tokens(input_ids, config)
        if max_new_tokens == 0:
            sequence.finish("length")
            return sequence

        output = self._backend.begin(input_ids)
        _validate_model_output(output, self._backend.vocab_size)

        rng = random.Random(config.sampling.seed)
        stop_token_ids = set(config.stop_token_ids)

        for _ in range(max_new_tokens):
            sampled = sample_next(output.logits, config.sampling, rng)
            token_id = sampled.token_id
            if token_id in stop_token_ids:
                if config.include_stop_token:
                    sequence.append(token_id, sampled.logprob)
                sequence.finish("stop")
                return sequence

            sequence.append(token_id, sampled.logprob)
            output = self._backend.step(token_id, output.state)
            _validate_model_output(output, self._backend.vocab_size)

        sequence.finish("length")
        return sequence

    def _result_from_sequence(
        self,
        sequence: SequenceState,
        config: GenerationConfig,
    ) -> GenerationResult:
        token_ids = sequence.output_token_ids(include_prompt=config.include_prompt)

        return GenerationResult(
            text=self._tokenizer.decode(token_ids),
            token_ids=token_ids,
            prompt_token_ids=sequence.prompt_token_ids,
            generated_token_ids=sequence.generated_token_ids,
            generated_token_logprobs=sequence.generated_token_logprobs,
            finish_reason=sequence.finish_reason or "length",
            usage=TokenUsage(
                prompt_tokens=len(sequence.prompt_token_ids),
                completion_tokens=sequence.completion_tokens,
            ),
        )

    def _begin_batch(self, input_ids_batch: list[list[int]]) -> list[ModelOutput]:
        if isinstance(self._backend, BatchedDecoderOnlyBackend):
            return self._backend.begin_batch(input_ids_batch)
        return [self._backend.begin(input_ids) for input_ids in input_ids_batch]

    def _step_batch(self, inputs: list[DecodeInput]) -> list[ModelOutput]:
        if isinstance(self._backend, BatchedDecoderOnlyBackend):
            return self._backend.step_batch(inputs)
        return [self._backend.step(item.token_id, item.state) for item in inputs]


def _validate_prompt_limits(input_ids: list[int], config: GenerationConfig) -> None:
    if (
        config.max_prompt_tokens is not None
        and len(input_ids) > config.max_prompt_tokens
    ):
        raise ConfigurationError(
            f"prompt has {len(input_ids)} tokens, exceeds max_prompt_tokens "
            f"{config.max_prompt_tokens}"
        )


def _effective_max_new_tokens(
    input_ids: list[int],
    config: GenerationConfig,
) -> int:
    if config.max_total_tokens is None:
        return config.max_new_tokens

    remaining_context = max(config.max_total_tokens - len(input_ids), 0)
    return min(config.max_new_tokens, remaining_context)


def _validate_model_output(output: ModelOutput, vocab_size: int) -> None:
    _validate_vocab_size(output.logits, vocab_size)
    if not isinstance(output.state, DecodeState):
        raise BackendError("backend state must be a DecodeState")


def _validate_vocab_size(logits: object, vocab_size: int) -> None:
    try:
        actual = len(logits)  # type: ignore[arg-type]
    except TypeError as exc:
        raise BackendError("backend logits must be a sized sequence") from exc

    if actual != vocab_size:
        raise BackendError(
            f"backend returned {actual} logits, expected vocab size {vocab_size}"
        )
