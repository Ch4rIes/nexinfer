from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
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


@dataclass(slots=True)
class ManagedKVCacheBlock:
    """A ref-counted block that can hold a cached prefix hash."""

    block_id: int
    ref_count: int = 0
    hash: int = -1
    token_ids: tuple[int, ...] = ()

    def reset(self) -> None:
        self.ref_count = 1
        self.hash = -1
        self.token_ids = ()

    def update(self, block_hash: int, token_ids: Sequence[int]) -> None:
        self.hash = block_hash
        self.token_ids = tuple(token_ids)


@dataclass(frozen=True, slots=True)
class KVCacheAllocationPlan:
    """Allocation feasibility and prefix-cache hit information."""

    can_allocate: bool
    num_cached_blocks: int
    num_required_blocks: int
    num_new_blocks: int


@dataclass(slots=True)
class ManagedKVCacheAllocation:
    """Mutable block-table metadata owned by one sequence."""

    sequence_id: str
    block_table: list[int]
    block_size: int
    token_count: int
    num_cached_tokens: int = 0
    num_hashed_tokens: int = 0

    @property
    def token_capacity(self) -> int:
        return len(self.block_table) * self.block_size


class PrefixKVCacheBlockManager:
    """Ref-counted KV block manager with prefix-cache hashing."""

    def __init__(self, *, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0:
            raise ConfigurationError("num_blocks must be positive")
        if block_size <= 0:
            raise ConfigurationError("block_size must be positive")

        self.block_size = block_size
        self.blocks = [ManagedKVCacheBlock(block_id) for block_id in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self._allocations: dict[str, ManagedKVCacheAllocation] = {}

    @property
    def free_blocks(self) -> int:
        return len(self.free_block_ids)

    @property
    def used_blocks(self) -> int:
        return len(self.used_block_ids)

    @classmethod
    def compute_hash(
        cls,
        token_ids: Sequence[int],
        prefix: int = -1,
    ) -> int:
        digest = hashlib.blake2b(digest_size=8)
        if prefix != -1:
            digest.update(prefix.to_bytes(8, "little", signed=False))
        for token_id in token_ids:
            digest.update(int(token_id).to_bytes(8, "little", signed=True))
        return int.from_bytes(digest.digest(), "little", signed=False)

    def can_allocate(self, token_ids: Sequence[int]) -> KVCacheAllocationPlan:
        num_required_blocks = self._num_blocks_for_tokens(len(token_ids))
        num_cached_blocks = self._num_cached_prefix_blocks(token_ids)
        num_new_blocks = num_required_blocks

        h = -1
        for block_index in range(num_cached_blocks):
            block_token_ids = self._block(token_ids, block_index)
            h = self.compute_hash(block_token_ids, h)
            block_id = self.hash_to_block_id[h]
            if block_id in self.used_block_ids:
                num_new_blocks -= 1

        return KVCacheAllocationPlan(
            can_allocate=self.free_blocks >= num_new_blocks,
            num_cached_blocks=num_cached_blocks,
            num_required_blocks=num_required_blocks,
            num_new_blocks=num_new_blocks,
        )

    def allocate(
        self,
        sequence_id: str,
        token_ids: Sequence[int],
        *,
        num_cached_blocks: int | None = None,
    ) -> ManagedKVCacheAllocation:
        if sequence_id in self._allocations:
            raise CacheError(f"sequence already has an allocation: {sequence_id}")

        plan = self.can_allocate(token_ids)
        if not plan.can_allocate:
            raise CacheError("not enough free KV-cache blocks")

        if num_cached_blocks is None:
            num_cached_blocks = plan.num_cached_blocks
        if not 0 <= num_cached_blocks <= plan.num_cached_blocks:
            raise CacheError("num_cached_blocks exceeds available prefix cache")

        block_table: list[int] = []
        h = -1
        for block_index in range(num_cached_blocks):
            block_token_ids = self._block(token_ids, block_index)
            h = self.compute_hash(block_token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            block_table.append(block_id)

        for _ in range(num_cached_blocks, plan.num_required_blocks):
            block_table.append(self._allocate_block())

        allocation = ManagedKVCacheAllocation(
            sequence_id=sequence_id,
            block_table=block_table,
            block_size=self.block_size,
            token_count=len(token_ids),
            num_cached_tokens=num_cached_blocks * self.block_size,
            num_hashed_tokens=num_cached_blocks * self.block_size,
        )
        self._allocations[sequence_id] = allocation
        return allocation

    def can_append(self, sequence_id: str, additional_tokens: int = 1) -> bool:
        if additional_tokens < 0:
            raise ConfigurationError("additional_tokens must be non-negative")

        allocation = self.allocation(sequence_id)
        needed_blocks = self._num_blocks_for_tokens(
            allocation.token_count + additional_tokens
        )
        return self.free_blocks >= max(needed_blocks - len(allocation.block_table), 0)

    def reserve(self, sequence_id: str, token_count: int) -> ManagedKVCacheAllocation:
        allocation = self.allocation(sequence_id)
        if token_count < allocation.token_count:
            raise ConfigurationError("token_count cannot shrink an allocation")

        needed_blocks = self._num_blocks_for_tokens(token_count)
        additional_blocks = needed_blocks - len(allocation.block_table)
        if additional_blocks > self.free_blocks:
            raise CacheError("not enough free KV-cache blocks")

        for _ in range(max(additional_blocks, 0)):
            allocation.block_table.append(self._allocate_block())
        allocation.token_count = token_count
        return allocation

    def hash_blocks(
        self,
        sequence_id: str,
        token_ids: Sequence[int],
    ) -> tuple[int, ...]:
        allocation = self.allocation(sequence_id)
        end = min(len(allocation.block_table), len(token_ids) // self.block_size)
        start = allocation.num_hashed_tokens // self.block_size
        if start >= end:
            return ()

        h = self.blocks[allocation.block_table[start - 1]].hash if start > 0 else -1
        hashes: list[int] = []
        for block_index in range(start, end):
            block_id = allocation.block_table[block_index]
            block_token_ids = self._block(token_ids, block_index)
            h = self.compute_hash(block_token_ids, h)
            self.blocks[block_id].update(h, block_token_ids)
            self.hash_to_block_id[h] = block_id
            hashes.append(h)

        allocation.num_hashed_tokens = end * self.block_size
        return tuple(hashes)

    def deallocate(self, sequence_id: str) -> None:
        allocation = self.allocation(sequence_id)
        for block_id in reversed(allocation.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        del self._allocations[sequence_id]

    def allocation(self, sequence_id: str) -> ManagedKVCacheAllocation:
        try:
            return self._allocations[sequence_id]
        except KeyError as exc:
            raise KeyError(f"unknown sequence allocation: {sequence_id}") from exc

    def _allocate_block(self) -> int:
        try:
            block_id = self.free_block_ids.popleft()
        except IndexError as exc:
            raise CacheError("not enough free KV-cache blocks") from exc

        block = self.blocks[block_id]
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int) -> None:
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def _num_cached_prefix_blocks(self, token_ids: Sequence[int]) -> int:
        h = -1
        num_cached_blocks = 0
        for block_index in range(max(self._num_blocks_for_tokens(len(token_ids)) - 1, 0)):
            block_token_ids = self._block(token_ids, block_index)
            h = self.compute_hash(block_token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != tuple(
                block_token_ids
            ):
                break
            num_cached_blocks += 1
        return num_cached_blocks

    def _num_blocks_for_tokens(self, token_count: int) -> int:
        return ceil(token_count / self.block_size) if token_count else 0

    def _block(self, token_ids: Sequence[int], block_index: int) -> list[int]:
        start = block_index * self.block_size
        end = start + self.block_size
        return list(token_ids[start:end])
