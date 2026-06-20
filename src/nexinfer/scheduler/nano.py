from __future__ import annotations

from collections import deque
from typing import Any

from nexinfer.cache import BlockManager
from nexinfer.errors import SchedulerError
from nexinfer.sequence import Sequence, SequenceStatus


class Scheduler:
    """Nano-VLLM-compatible scheduler over Sequence objects."""

    def __init__(
        self,
        config: Any | None = None,
        *,
        max_num_seqs: int | None = None,
        max_num_batched_tokens: int | None = None,
        eos: int | None = None,
        num_kvcache_blocks: int | None = None,
        kvcache_block_size: int | None = None,
        block_manager: BlockManager | None = None,
    ) -> None:
        self.max_num_seqs = self._value(config, "max_num_seqs", max_num_seqs, 512)
        self.max_num_batched_tokens = self._value(
            config,
            "max_num_batched_tokens",
            max_num_batched_tokens,
            16384,
        )
        self.eos = self._value(config, "eos", eos, -1)
        self.block_size = self._value(
            config,
            "kvcache_block_size",
            kvcache_block_size,
            256,
        )
        num_blocks = self._value(
            config,
            "num_kvcache_blocks",
            num_kvcache_blocks,
            1024,
        )

        if self.max_num_seqs <= 0:
            raise SchedulerError("max_num_seqs must be positive")
        if self.max_num_batched_tokens <= 0:
            raise SchedulerError("max_num_batched_tokens must be positive")
        if self.block_size <= 0:
            raise SchedulerError("kvcache_block_size must be positive")
        if block_manager is None and num_blocks <= 0:
            raise SchedulerError("num_kvcache_blocks must be positive")

        self.block_manager = block_manager or BlockManager(num_blocks, self.block_size)
        self.block_size = self.block_manager.block_size
        Sequence.block_size = self.block_size
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs: list[Sequence] = []
        num_batched_tokens = 0

        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break

            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = len(seq) - num_cached_blocks * self.block_size
            else:
                num_cached_blocks = 0
                num_tokens = len(seq) - seq.num_cached_tokens

            if remaining < num_tokens and scheduled_seqs:
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == len(seq):
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.num_scheduled_tokens = 0
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        is_prefill: bool,
    ) -> None:
        for seq, token_id in zip(seqs, token_ids, strict=True):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < len(seq):
                continue

            seq.append_token(token_id)
            if (
                (not seq.ignore_eos and token_id == self.eos)
                or seq.num_completion_tokens == seq.max_tokens
            ):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    @staticmethod
    def _value(
        config: Any | None,
        name: str,
        explicit_value: int | None,
        default: int,
    ) -> int:
        if explicit_value is not None:
            return explicit_value
        if config is not None:
            return int(getattr(config, name))
        return default
