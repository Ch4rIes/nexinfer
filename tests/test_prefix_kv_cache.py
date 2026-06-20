import pytest

from nexinfer import BlockManager, CacheError, PrefixKVCacheBlockManager, Sequence


def test_prefix_manager_reuses_deallocated_cached_prefix() -> None:
    manager = PrefixKVCacheBlockManager(num_blocks=4, block_size=2)
    first = manager.allocate("first", [1, 2, 3, 4, 5])
    manager.hash_blocks("first", [1, 2, 3, 4, 5])
    first_prefix_blocks = tuple(first.block_table[:2])
    manager.deallocate("first")

    plan = manager.can_allocate([1, 2, 3, 4, 9])
    second = manager.allocate("second", [1, 2, 3, 4, 9])

    assert plan.can_allocate is True
    assert plan.num_cached_blocks == 2
    assert second.block_table[:2] == list(first_prefix_blocks)
    assert second.num_cached_tokens == 4


def test_prefix_manager_shares_running_cached_blocks_with_refcounts() -> None:
    manager = PrefixKVCacheBlockManager(num_blocks=4, block_size=2)
    first = manager.allocate("first", [1, 2, 3])
    manager.hash_blocks("first", [1, 2, 3])
    prefix_block_id = first.block_table[0]

    plan = manager.can_allocate([1, 2, 9])
    second = manager.allocate("second", [1, 2, 9])

    assert plan.num_cached_blocks == 1
    assert plan.num_new_blocks == 1
    assert second.block_table[0] == prefix_block_id
    assert manager.blocks[prefix_block_id].ref_count == 2

    manager.deallocate("second")
    assert manager.blocks[prefix_block_id].ref_count == 1


def test_prefix_manager_reserves_new_blocks_for_append() -> None:
    manager = PrefixKVCacheBlockManager(num_blocks=2, block_size=2)
    allocation = manager.allocate("seq", [1, 2])

    assert allocation.token_capacity == 2
    assert manager.can_append("seq", 1) is True

    updated = manager.reserve("seq", token_count=3)

    assert updated.token_count == 3
    assert updated.token_capacity == 4
    assert len(updated.block_table) == 2


def test_prefix_manager_reports_allocation_failure() -> None:
    manager = PrefixKVCacheBlockManager(num_blocks=1, block_size=2)
    manager.allocate("seq", [1, 2])

    assert manager.can_allocate([3, 4]).can_allocate is False
    with pytest.raises(CacheError, match="not enough free"):
        manager.allocate("other", [3, 4])


def test_prefix_manager_hashes_only_complete_blocks_once() -> None:
    manager = PrefixKVCacheBlockManager(num_blocks=3, block_size=2)
    manager.allocate("seq", [1, 2, 3])

    first_hashes = manager.hash_blocks("seq", [1, 2, 3])
    second_hashes = manager.hash_blocks("seq", [1, 2, 3])

    assert len(first_hashes) == 1
    assert second_hashes == ()


def test_block_manager_allocates_sequence_block_table() -> None:
    manager = BlockManager(4, 2)
    seq = Sequence([1, 2, 3, 4, 5])

    cached_blocks = manager.can_allocate(seq)
    manager.allocate(seq, cached_blocks)

    assert cached_blocks == 0
    assert seq.block_table == [0, 1, 2]
    assert seq.num_cached_tokens == 0
    assert manager.used_blocks == 3


def test_block_manager_reuses_deallocated_sequence_prefix() -> None:
    manager = BlockManager(4, 2)
    first = Sequence([1, 2, 3, 4, 5])
    manager.allocate(first, manager.can_allocate(first))
    first.num_scheduled_tokens = 4
    manager.hash_blocks(first)
    first_prefix_blocks = tuple(first.block_table[:2])
    manager.deallocate(first)

    second = Sequence([1, 2, 3, 4, 9])
    cached_blocks = manager.can_allocate(second)
    manager.allocate(second, cached_blocks)

    assert cached_blocks == 2
    assert second.block_table[:2] == list(first_prefix_blocks)
    assert second.num_cached_tokens == 4


def test_block_manager_shares_running_sequence_prefix_with_refcounts() -> None:
    manager = BlockManager(4, 2)
    first = Sequence([1, 2, 3])
    manager.allocate(first, manager.can_allocate(first))
    first.num_scheduled_tokens = 2
    manager.hash_blocks(first)
    prefix_block_id = first.block_table[0]

    second = Sequence([1, 2, 9])
    cached_blocks = manager.can_allocate(second)
    manager.allocate(second, cached_blocks)

    assert cached_blocks == 1
    assert second.block_table[0] == prefix_block_id
    assert manager.blocks[prefix_block_id].ref_count == 2

    manager.deallocate(second)

    assert manager.blocks[prefix_block_id].ref_count == 1


def test_block_manager_append_allocates_when_sequence_enters_new_block() -> None:
    manager = BlockManager(2, 2)
    seq = Sequence([1, 2])
    manager.allocate(seq, manager.can_allocate(seq))

    seq.append_token(3)

    assert manager.can_append(seq) is True
    manager.may_append(seq)

    assert seq.block_table == [0, 1]
    assert manager.allocation(str(seq.seq_id)).token_count == 3


def test_block_manager_append_reports_capacity_pressure() -> None:
    manager = BlockManager(1, 2)
    seq = Sequence([1, 2])
    manager.allocate(seq, manager.can_allocate(seq))
    seq.append_token(3)

    assert manager.can_append(seq) is False
    with pytest.raises(CacheError, match="not enough free"):
        manager.may_append(seq)


def test_block_manager_deallocate_mutates_sequence() -> None:
    manager = BlockManager(2, 2)
    seq = Sequence([1, 2])
    manager.allocate(seq, manager.can_allocate(seq))
    seq.num_cached_tokens = 2

    manager.deallocate(seq)

    assert seq.block_table == []
    assert seq.num_cached_tokens == 0
    assert manager.used_blocks == 0
