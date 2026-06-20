import pytest

from nexinfer import KVCacheBlockAllocator


def test_allocates_block_table_for_sequence() -> None:
    allocator = KVCacheBlockAllocator(block_size=4, max_blocks=3)

    allocation = allocator.allocate("seq-1", token_count=5)

    assert allocation.sequence_id == "seq-1"
    assert allocation.block_table == (0, 1)
    assert allocation.token_capacity == 8
    assert allocation.token_count == 5
    assert allocation.remaining_tokens == 3
    assert allocator.free_blocks == 1
    assert allocator.used_blocks == 2


def test_reserve_extends_allocation_only_when_needed() -> None:
    allocator = KVCacheBlockAllocator(block_size=4, max_blocks=3)
    allocator.allocate("seq-1", token_count=3)

    same_block = allocator.reserve("seq-1", additional_tokens=1)
    extended = allocator.reserve("seq-1", additional_tokens=1)

    assert same_block.block_table == (0,)
    assert same_block.token_count == 4
    assert extended.block_table == (0, 1)
    assert extended.token_count == 5


def test_free_reuses_blocks() -> None:
    allocator = KVCacheBlockAllocator(block_size=4, max_blocks=2)
    allocator.allocate("seq-1", token_count=4)
    allocator.free("seq-1")

    allocation = allocator.allocate("seq-2", token_count=4)

    assert allocation.block_table == (0,)
    assert allocator.free_blocks == 1


def test_raises_when_cache_is_full() -> None:
    allocator = KVCacheBlockAllocator(block_size=4, max_blocks=1)
    allocator.allocate("seq-1", token_count=4)

    with pytest.raises(MemoryError, match="not enough free"):
        allocator.allocate("seq-2", token_count=1)
