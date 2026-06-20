from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from nexinfer.config import GenerationConfig
from nexinfer.engine import LLMEngine
from nexinfer.errors import ConfigurationError
from nexinfer.result import GenerationResult
from nexinfer.scheduler import GenerationRequest, RequestQueue


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
        queue: RequestQueue | None = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ConfigurationError("max_batch_size must be positive")

        self._engine = engine
        self._max_batch_size = max_batch_size
        self._queue = queue or RequestQueue()

    @property
    def pending_requests(self) -> int:
        return len(self._queue)

    def submit(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        *,
        request_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> GenerationRequest:
        return self._queue.submit(
            prompt,
            config,
            request_id=request_id,
            metadata=metadata,
        )

    def cancel(self, request_id: str) -> bool:
        return self._queue.cancel(request_id)

    def run_once(self) -> tuple[CompletedRequest, ...]:
        batch = self._queue.schedule(max_requests=self._max_batch_size)
        results = self._engine.complete_requests(list(batch.requests))
        return tuple(
            CompletedRequest(request_id=request.request_id, result=result)
            for request, result in zip(batch.requests, results, strict=True)
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
