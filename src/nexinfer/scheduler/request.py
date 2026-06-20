from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import count

from nexinfer.config import GenerationConfig
from nexinfer.errors import SchedulerError


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """A pending generation request submitted to the scheduler."""

    request_id: str
    prompt: str
    config: GenerationConfig


@dataclass(frozen=True, slots=True)
class ScheduledBatch:
    """A small batch of requests selected for execution."""

    requests: tuple[GenerationRequest, ...]

    def __len__(self) -> int:
        return len(self.requests)


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
    ) -> GenerationRequest:
        request_id = request_id or f"req-{next(self._ids)}"
        if request_id in self._pending_ids:
            raise SchedulerError(f"duplicate request id: {request_id}")

        request = GenerationRequest(
            request_id=request_id,
            prompt=prompt,
            config=config or GenerationConfig(),
        )
        self._pending.append(request)
        self._pending_ids.add(request_id)
        return request

    def schedule(self, *, max_requests: int) -> ScheduledBatch:
        if max_requests <= 0:
            raise SchedulerError("max_requests must be positive")

        requests: list[GenerationRequest] = []
        while self._pending and len(requests) < max_requests:
            request = self._pending.popleft()
            self._pending_ids.remove(request.request_id)
            requests.append(request)

        return ScheduledBatch(requests=tuple(requests))

    def cancel(self, request_id: str) -> bool:
        if request_id not in self._pending_ids:
            return False

        self._pending = deque(
            request for request in self._pending if request.request_id != request_id
        )
        self._pending_ids.remove(request_id)
        return True
