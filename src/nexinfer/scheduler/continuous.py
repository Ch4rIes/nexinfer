from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

from nexinfer.errors import SchedulerError
from nexinfer.scheduler.active import ActiveSequence
from nexinfer.scheduler.request import GenerationRequest

SchedulePhase = Literal["idle", "prefill", "decode"]


@dataclass(frozen=True, slots=True)
class ScheduledActiveBatch:
    """A scheduler decision for one prefill or decode phase."""

    phase: SchedulePhase
    requests: tuple[GenerationRequest, ...] = ()
    active_sequences: tuple[ActiveSequence, ...] = ()
    num_tokens: int = 0

    def __len__(self) -> int:
        return len(self.requests) + len(self.active_sequences)


class ActiveScheduler:
    """Nano-VLLM-style waiting/running scheduler for active sequences."""

    def __init__(
        self,
        *,
        max_num_seqs: int,
        max_num_batched_tokens: int | None = None,
    ) -> None:
        if max_num_seqs <= 0:
            raise SchedulerError("max_num_seqs must be positive")
        if max_num_batched_tokens is not None and max_num_batched_tokens <= 0:
            raise SchedulerError("max_num_batched_tokens must be positive when set")

        self._max_num_seqs = max_num_seqs
        self._max_num_batched_tokens = max_num_batched_tokens
        self._waiting: deque[GenerationRequest] = deque()
        self._running: deque[ActiveSequence] = deque()
        self._request_ids: set[str] = set()

    @property
    def waiting_count(self) -> int:
        return len(self._waiting)

    @property
    def running_count(self) -> int:
        return len(self._running)

    @property
    def pending_count(self) -> int:
        return self.waiting_count + self.running_count

    def is_idle(self) -> bool:
        return not self._waiting and not self._running

    def has_request(self, request_id: str) -> bool:
        return request_id in self._request_ids

    def add_request(self, request: GenerationRequest) -> None:
        if request.request_id in self._request_ids:
            raise SchedulerError(f"duplicate request id: {request.request_id}")
        self._waiting.append(request)
        self._request_ids.add(request.request_id)

    def cancel(self, request_id: str) -> bool:
        if request_id not in self._request_ids:
            return False

        self._waiting = deque(
            request for request in self._waiting if request.request_id != request_id
        )
        self._running = deque(
            active for active in self._running if active.request_id != request_id
        )
        self._request_ids.remove(request_id)
        return True

    def schedule(self) -> ScheduledActiveBatch:
        prefill = self._schedule_prefill()
        if prefill:
            return prefill
        decode = self._schedule_decode()
        if decode:
            return decode
        return ScheduledActiveBatch(phase="idle")

    def postprocess_prefill(
        self,
        active_sequences: tuple[ActiveSequence, ...],
    ) -> tuple[ActiveSequence, ...]:
        return self._postprocess(active_sequences)

    def postprocess_decode(
        self,
        active_sequences: tuple[ActiveSequence, ...],
    ) -> tuple[ActiveSequence, ...]:
        return self._postprocess(active_sequences)

    def _schedule_prefill(self) -> ScheduledActiveBatch | None:
        if not self._waiting:
            return None

        requests: list[GenerationRequest] = []
        num_tokens = 0
        while self._waiting and len(requests) < self._max_num_seqs:
            next_request = self._waiting[0]
            next_tokens = next_request.prompt_token_count or 0
            would_exceed = (
                self._max_num_batched_tokens is not None
                and num_tokens + next_tokens > self._max_num_batched_tokens
            )
            if would_exceed and requests:
                break

            request = self._waiting.popleft()
            requests.append(request)
            num_tokens += next_tokens

        if not requests:
            return None
        return ScheduledActiveBatch(
            phase="prefill",
            requests=tuple(requests),
            num_tokens=num_tokens,
        )

    def _schedule_decode(self) -> ScheduledActiveBatch | None:
        active_sequences: list[ActiveSequence] = []
        while self._running and len(active_sequences) < self._max_num_seqs:
            active = self._running.popleft()
            if not active.is_finished:
                active_sequences.append(active)

        if not active_sequences:
            return None
        return ScheduledActiveBatch(
            phase="decode",
            active_sequences=tuple(active_sequences),
            num_tokens=-len(active_sequences),
        )

    def _postprocess(
        self,
        active_sequences: tuple[ActiveSequence, ...],
    ) -> tuple[ActiveSequence, ...]:
        finished: list[ActiveSequence] = []
        for active in active_sequences:
            if active.is_finished:
                finished.append(active)
                self._request_ids.remove(active.request_id)
            else:
                self._running.append(active)
        return tuple(finished)
