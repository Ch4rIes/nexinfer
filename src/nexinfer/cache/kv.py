from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from nexinfer.errors import CacheError, ConfigurationError


@dataclass(frozen=True, slots=True)
class KVCacheBlock:
    """One fixed-size block of KV-cache token capacity."""

    block_id: int
    block_size: int


@dataclass(frozen=True, slots=True)
class KVCacheAllocation:
    """Block-table metadata for one sequence's KV cache."""

    sequence_id: str
    blocks: tuple[KVCacheBlock, ...]
    token_capacity: int
    token_count: int = 0

    @property
    def block_table(self) -> tuple[int, ...]:
        return tuple(block.block_id for block in self.blocks)

    @property
    def remaining_tokens(self) -> int:
        return self.token_capacity - self.token_count


class KVCacheBlockAllocator:
    """A simple fixed-size block allocator for future paged KV-cache work."""

    def __init__(self, *, block_size: int, max_blocks: int) -> None:
        if block_size <= 0:
            raise ConfigurationError("block_size must be positive")
        if max_blocks <= 0:
            raise ConfigurationError("max_blocks must be positive")

        self._block_size = block_size
        self._free_block_ids = list(range(max_blocks))
        self._allocations: dict[str, KVCacheAllocation] = {}

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def free_blocks(self) -> int:
        return len(self._free_block_ids)

    @property
    def used_blocks(self) -> int:
        return len(self._allocations_block_ids())

    def allocate(self, sequence_id: str, token_count: int) -> KVCacheAllocation:
        if token_count < 0:
            raise ConfigurationError("token_count must be non-negative")
        if sequence_id in self._allocations:
            raise CacheError(f"sequence already has an allocation: {sequence_id}")

        block_count = ceil(token_count / self._block_size) if token_count else 0
        if block_count > self.free_blocks:
            raise CacheError("not enough free KV-cache blocks")

        block_ids = [self._free_block_ids.pop(0) for _ in range(block_count)]
        allocation = KVCacheAllocation(
            sequence_id=sequence_id,
            blocks=tuple(
                KVCacheBlock(block_id=block_id, block_size=self._block_size)
                for block_id in block_ids
            ),
            token_capacity=block_count * self._block_size,
            token_count=token_count,
        )
        self._allocations[sequence_id] = allocation
        return allocation

    def reserve(self, sequence_id: str, additional_tokens: int) -> KVCacheAllocation:
        if additional_tokens < 0:
            raise ConfigurationError("additional_tokens must be non-negative")

        allocation = self._allocation_for(sequence_id)
        needed_tokens = allocation.token_count + additional_tokens
        if needed_tokens <= allocation.token_capacity:
            updated = KVCacheAllocation(
                sequence_id=allocation.sequence_id,
                blocks=allocation.blocks,
                token_capacity=allocation.token_capacity,
                token_count=needed_tokens,
            )
            self._allocations[sequence_id] = updated
            return updated

        additional_blocks = ceil(
            (needed_tokens - allocation.token_capacity) / self._block_size
        )
        if additional_blocks > self.free_blocks:
            raise CacheError("not enough free KV-cache blocks")

        new_block_ids = [
            self._free_block_ids.pop(0) for _ in range(additional_blocks)
        ]
        new_blocks = tuple(
            KVCacheBlock(block_id=block_id, block_size=self._block_size)
            for block_id in new_block_ids
        )
        updated = KVCacheAllocation(
            sequence_id=allocation.sequence_id,
            blocks=(*allocation.blocks, *new_blocks),
            token_capacity=allocation.token_capacity
            + additional_blocks * self._block_size,
            token_count=needed_tokens,
        )
        self._allocations[sequence_id] = updated
        return updated

    def free(self, sequence_id: str) -> None:
        allocation = self._allocation_for(sequence_id)
        self._free_block_ids.extend(allocation.block_table)
        self._free_block_ids.sort()
        del self._allocations[sequence_id]

    def allocation(self, sequence_id: str) -> KVCacheAllocation:
        return self._allocation_for(sequence_id)

    def _allocation_for(self, sequence_id: str) -> KVCacheAllocation:
        try:
            return self._allocations[sequence_id]
        except KeyError as exc:
            raise KeyError(f"unknown sequence allocation: {sequence_id}") from exc

    def _allocations_block_ids(self) -> set[int]:
        return {
            block_id
            for allocation in self._allocations.values()
            for block_id in allocation.block_table
        }
