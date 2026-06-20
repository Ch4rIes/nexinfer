from __future__ import annotations

from collections.abc import Mapping
from collections import deque
from dataclasses import dataclass, field
from itertools import count

from nexinfer.config import GenerationConfig
from nexinfer.errors import SchedulerError


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """A pending generation request submitted to the scheduler."""

    request_id: str
    prompt: str
    config: GenerationConfig
    metadata: Mapping[str, str] = field(default_factory=dict)
    prompt_token_count: int | None = None


@dataclass(frozen=True, slots=True)
class ScheduledBatch:
    """A small batch of requests selected for execution."""

    requests: tuple[GenerationRequest, ...]

    def __len__(self) -> int:
        return len(self.requests)

    @property
    def prompt_tokens(self) -> int:
        return sum(request.prompt_token_count or 0 for request in self.requests)


class RequestQueue:
    """FIFO request queue for the first scheduler layer."""

    def __init__(self) -> None:
        self._ids = count(1)
        self._pending: deque[GenerationRequest] = deque()
        self._pending_ids: set[str] = set()

    def __len__(self) -> int:
        return len(self._pending)

    def submit(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        *,
        request_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
        prompt_token_count: int | None = None,
    ) -> GenerationRequest:
        request_id = request_id or f"req-{next(self._ids)}"
        if request_id in self._pending_ids:
            raise SchedulerError(f"duplicate request id: {request_id}")
        if prompt_token_count is not None and prompt_token_count < 0:
            raise SchedulerError("prompt_token_count must be non-negative when set")

        request = GenerationRequest(
            request_id=request_id,
            prompt=prompt,
            config=config or GenerationConfig(),
            metadata=dict(metadata or {}),
            prompt_token_count=prompt_token_count,
        )
        self._pending.append(request)
        self._pending_ids.add(request_id)
        return request

    def schedule(
        self,
        *,
        max_requests: int,
        max_prompt_tokens: int | None = None,
    ) -> ScheduledBatch:
        if max_requests <= 0:
            raise SchedulerError("max_requests must be positive")
        if max_prompt_tokens is not None and max_prompt_tokens <= 0:
            raise SchedulerError("max_prompt_tokens must be positive when set")

        requests: list[GenerationRequest] = []
        prompt_tokens = 0
        while self._pending and len(requests) < max_requests:
            next_request = self._pending[0]
            next_prompt_tokens = next_request.prompt_token_count or 0
            if (
                max_prompt_tokens is not None
                and requests
                and prompt_tokens + next_prompt_tokens > max_prompt_tokens
            ):
                break

            request = self._pending.popleft()
            self._pending_ids.remove(request.request_id)
            requests.append(request)
            prompt_tokens += next_prompt_tokens

        return ScheduledBatch(requests=tuple(requests))

    def cancel(self, request_id: str) -> bool:
        if request_id not in self._pending_ids:
            return False

        self._pending = deque(
            request for request in self._pending if request.request_id != request_id
        )
        self._pending_ids.remove(request_id)
        return True
