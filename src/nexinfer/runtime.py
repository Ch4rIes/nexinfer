from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import count
from typing import Literal

from nexinfer.cache import PrefixKVCacheBlockManager
from nexinfer.config import GenerationConfig
from nexinfer.engine import LLMEngine
from nexinfer.errors import ConfigurationError
from nexinfer.metrics import RuntimeStats
from nexinfer.result import GenerationResult
from nexinfer.scheduler import ActiveScheduler, GenerationRequest, RequestQueue

DecodeStrategy = Literal["sequential", "interleaved", "continuous"]


@dataclass(frozen=True, slots=True)
class CompletedRequest:
    """A finished scheduler request and its generation result."""

    request_id: str
    result: GenerationResult


class InferenceRuntime:
    """A small queued runtime that bridges scheduling and generation."""

    def __init__(
        self,
        engine: LLMEngine,
        *,
        max_batch_size: int = 8,
        max_batch_prompt_tokens: int | None = None,
        decode_strategy: DecodeStrategy = "sequential",
        block_manager: PrefixKVCacheBlockManager | None = None,
        queue: RequestQueue | None = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ConfigurationError("max_batch_size must be positive")
        if max_batch_prompt_tokens is not None and max_batch_prompt_tokens <= 0:
            raise ConfigurationError(
                "max_batch_prompt_tokens must be positive when set"
            )
        if decode_strategy not in {"sequential", "interleaved", "continuous"}:
            raise ConfigurationError(
                "decode_strategy must be sequential, interleaved, or continuous"
            )

        self._engine = engine
        self._max_batch_size = max_batch_size
        self._max_batch_prompt_tokens = max_batch_prompt_tokens
        self._decode_strategy = decode_strategy
        self._queue = queue or RequestQueue()
        self._active_scheduler = ActiveScheduler(
            max_num_seqs=max_batch_size,
            max_num_batched_tokens=max_batch_prompt_tokens,
            block_manager=block_manager,
        )
        self._stats = RuntimeStats()
        self._request_ids = count(1)

    @property
    def pending_requests(self) -> int:
        if self._decode_strategy == "continuous":
            return self._active_scheduler.pending_count
        return len(self._queue)

    @property
    def stats(self) -> RuntimeStats:
        return self._stats

    def submit(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        *,
        request_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> GenerationRequest:
        request_id = request_id or f"req-{next(self._request_ids)}"
        prompt_token_ids = self._engine.tokenizer.encode(prompt)
        prompt_token_count = len(prompt_token_ids)
        metadata = dict(metadata or {})
        metadata.setdefault(
            "token_ids",
            ",".join(str(token_id) for token_id in prompt_token_ids),
        )
        if self._decode_strategy == "continuous":
            request = GenerationRequest(
                request_id=request_id,
                prompt=prompt,
                config=config or GenerationConfig(),
                metadata=metadata,
                prompt_token_count=prompt_token_count,
            )
            self._active_scheduler.add_request(request)
            return request

        return self._queue.submit(
            prompt,
            config,
            request_id=request_id,
            metadata=metadata,
            prompt_token_count=prompt_token_count,
        )

    def cancel(self, request_id: str) -> bool:
        if self._decode_strategy == "continuous":
            return self._active_scheduler.cancel(request_id)
        return self._queue.cancel(request_id)

    def run_once(self) -> tuple[CompletedRequest, ...]:
        if self._decode_strategy == "continuous":
            return self._run_continuous_once()

        batch = self._queue.schedule(
            max_requests=self._max_batch_size,
            max_prompt_tokens=self._max_batch_prompt_tokens,
        )
        requests = list(batch.requests)
        if self._decode_strategy == "interleaved":
            results = self._engine.complete_requests_interleaved(requests)
        else:
            results = self._engine.complete_requests(requests)
        result_tuple = tuple(results)
        if result_tuple:
            self._stats = self._stats.record_batch(result_tuple)
        return tuple(
            CompletedRequest(request_id=request.request_id, result=result)
            for request, result in zip(batch.requests, result_tuple, strict=True)
        )

    def run_until_idle(self, *, max_batches: int | None = None) -> tuple[CompletedRequest, ...]:
        if max_batches is not None and max_batches <= 0:
            raise ConfigurationError("max_batches must be positive when set")

        completed: list[CompletedRequest] = []
        batches_run = 0
        while self.pending_requests:
            if max_batches is not None and batches_run >= max_batches:
                break
            completed.extend(self.run_once())
            batches_run += 1

        return tuple(completed)

    def _run_continuous_once(self) -> tuple[CompletedRequest, ...]:
        scheduled = self._active_scheduler.schedule()
        if scheduled.phase == "idle":
            return ()

        if scheduled.phase == "prefill":
            active_sequences = (
                *self._engine.start_requests(list(scheduled.requests)),
                *self._engine.prefill_active_sequences(
                    list(scheduled.active_sequences)
                ),
            )
            finished = self._active_scheduler.postprocess_prefill(active_sequences)
        else:
            active_list = list(scheduled.active_sequences)
            self._engine.decode_active_batch(active_list)
            finished = self._active_scheduler.postprocess_decode(tuple(active_list))

        completed = tuple(
            CompletedRequest(
                request_id=active.request_id,
                result=self._engine.result_from_active(active),
            )
            for active in finished
        )
        result_tuple = tuple(item.result for item in completed)
        if result_tuple:
            self._stats = self._stats.record_batch(result_tuple)
        return completed
