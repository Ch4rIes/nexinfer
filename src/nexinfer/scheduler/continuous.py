from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

from nexinfer.cache import PrefixKVCacheBlockManager
from nexinfer.errors import CacheError
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
    cached_tokens: int = 0

    def __len__(self) -> int:
        return len(self.requests) + len(self.active_sequences)


class ActiveScheduler:
    """Nano-VLLM-style waiting/running scheduler for active sequences."""

    def __init__(
        self,
        *,
        max_num_seqs: int,
        max_num_batched_tokens: int | None = None,
        block_manager: PrefixKVCacheBlockManager | None = None,
    ) -> None:
        if max_num_seqs <= 0:
            raise SchedulerError("max_num_seqs must be positive")
        if max_num_batched_tokens is not None and max_num_batched_tokens <= 0:
            raise SchedulerError("max_num_batched_tokens must be positive when set")

        self._max_num_seqs = max_num_seqs
        self._max_num_batched_tokens = max_num_batched_tokens
        self._block_manager = block_manager
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
        was_running = any(
            active.request_id == request_id for active in self._running
        )
        self._running = deque(
            active for active in self._running if active.request_id != request_id
        )
        if was_running:
            self._deallocate(request_id)
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
        for active in active_sequences:
            self._attach_block_table(active)
            self._hash_blocks(active)
        return self._postprocess(active_sequences)

    def postprocess_decode(
        self,
        active_sequences: tuple[ActiveSequence, ...],
    ) -> tuple[ActiveSequence, ...]:
        for active in active_sequences:
            self._attach_block_table(active)
            self._hash_blocks(active)
        return self._postprocess(active_sequences)

    def _schedule_prefill(self) -> ScheduledActiveBatch | None:
        if not self._waiting:
            return None

        requests: list[GenerationRequest] = []
        num_tokens = 0
        cached_tokens = 0
        while self._waiting and len(requests) < self._max_num_seqs:
            next_request = self._waiting[0]
            next_tokens = next_request.prompt_token_count or 0
            next_plan = self._allocation_plan(next_request)
            would_exceed = (
                self._max_num_batched_tokens is not None
                and num_tokens + max(next_tokens - next_plan.num_cached_tokens, 0)
                > self._max_num_batched_tokens
            )
            if would_exceed and requests:
                break
            if not next_plan.can_allocate:
                if requests:
                    break
                raise CacheError("not enough free KV-cache blocks")

            request = self._waiting.popleft()
            self._allocate_request(request, next_plan.num_cached_blocks)
            requests.append(request)
            num_tokens += max(next_tokens - next_plan.num_cached_tokens, 0)
            cached_tokens += next_plan.num_cached_tokens

        if not requests:
            return None
        return ScheduledActiveBatch(
            phase="prefill",
            requests=tuple(requests),
            num_tokens=num_tokens,
            cached_tokens=cached_tokens,
        )

    def _schedule_decode(self) -> ScheduledActiveBatch | None:
        active_sequences: list[ActiveSequence] = []
        while self._running and len(active_sequences) < self._max_num_seqs:
            active = self._running.popleft()
            if active.is_finished:
                continue
            if self._can_append(active):
                self._reserve_append(active)
                active_sequences.append(active)
            else:
                self._running.appendleft(active)
                break

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
                self._deallocate(active.request_id)
                self._request_ids.remove(active.request_id)
            else:
                self._running.append(active)
        return tuple(finished)

    def _allocation_plan(self, request: GenerationRequest) -> _SchedulerAllocationPlan:
        if self._block_manager is None:
            return _SchedulerAllocationPlan(
                can_allocate=True,
                num_cached_blocks=0,
                num_cached_tokens=0,
            )

        token_ids = self._request_token_ids(request)
        plan = self._block_manager.can_allocate(token_ids)
        return _SchedulerAllocationPlan(
            can_allocate=plan.can_allocate,
            num_cached_blocks=plan.num_cached_blocks,
            num_cached_tokens=plan.num_cached_blocks * self._block_manager.block_size,
        )

    def _allocate_request(
        self,
        request: GenerationRequest,
        num_cached_blocks: int,
    ) -> None:
        if self._block_manager is None:
            return
        self._block_manager.allocate(
            request.request_id,
            self._request_token_ids(request),
            num_cached_blocks=num_cached_blocks,
        )

    def _attach_block_table(self, active: ActiveSequence) -> None:
        if self._block_manager is None:
            return
        allocation = self._block_manager.allocation(active.request_id)
        active.block_table = list(allocation.block_table)

    def _hash_blocks(self, active: ActiveSequence) -> None:
        if self._block_manager is None:
            return
        self._block_manager.hash_blocks(active.request_id, active.sequence.token_ids)

    def _can_append(self, active: ActiveSequence) -> bool:
        if self._block_manager is None:
            return True
        return self._block_manager.can_append(active.request_id, additional_tokens=1)

    def _reserve_append(self, active: ActiveSequence) -> None:
        if self._block_manager is None:
            return
        self._block_manager.reserve(active.request_id, active.sequence.next_position + 1)
        self._attach_block_table(active)

    def _deallocate(self, request_id: str) -> None:
        if self._block_manager is None:
            return
        self._block_manager.deallocate(request_id)

    def _request_token_ids(self, request: GenerationRequest) -> list[int]:
        raw_token_ids = request.metadata.get("token_ids")
        if raw_token_ids:
            return [int(token_id) for token_id in raw_token_ids.split(",") if token_id]
        return [0] * (request.prompt_token_count or 0)


@dataclass(frozen=True, slots=True)
class _SchedulerAllocationPlan:
    can_allocate: bool
    num_cached_blocks: int
    num_cached_tokens: int
