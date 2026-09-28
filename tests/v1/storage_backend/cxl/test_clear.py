# SPDX-License-Identifier: Apache-2.0
"""Tests for the bulk node-clear path.

Covers ``CXLIndexWriter.clear_owned_slots`` — the writer-level primitive that
``CXLStore.clear`` builds on:

  - Only VALID slots owned by *this* node are tombstoned; their offsets are
    returned so the caller can free them from the node heap.
  - Slots owned by other nodes (donor entries in the shared index) are left
    untouched.
  - Busy slots (pinned / nonzero ref_count) are skipped and counted, never
    force-freed.
  - After the freed offsets go back to a real ``NodeHeap``, its regions become
    fully free and ``trim()`` releases them to the pool.
"""

# Standard
import os
import tempfile

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.cxl.bootstrap import (
    CXLBootstrapConfig,
    bootstrap_pool,
)
from lmcache.v1.storage_backend.cxl.heap import NodeHeap
from lmcache.v1.storage_backend.cxl.index import CXLIndex
from lmcache.v1.storage_backend.cxl.index_writer import (
    CommitPin,
    CXLIndexWriter,
    ReserveOutcome,
)
from lmcache.v1.storage_backend.cxl.layout import (
    SLOT_STATE_TOMB,
    SLOT_STATE_VALID,
)
from lmcache.v1.storage_backend.cxl.lock_manager import LockManager
from lmcache.v1.storage_backend.cxl.locks import TwoTierLock
from lmcache.v1.storage_backend.cxl.regions import RegionAllocator
from lmcache.v1.storage_backend.cxl.store import (
    object_key_to_chunk_hash,
    object_key_to_tenant_digest,
)

POOL_SIZE = 64 * (1 << 20)
REGION_SIZE = 2 * (1 << 20)
# A chunk size that evenly divides REGION_SIZE, giving a small slots-per-region
# so a handful of chunks span more than one region (exercising multi-region
# trim).
CHUNK_SIZE = REGION_SIZE // 4  # 4 slots per region


def _metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="cxl-clear-test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(4, 2, 16, 4, 64),
        chunk_size=16,
    )


def _make_digest(h: int) -> bytes:
    """The 16-byte tenant digest the store stamps for `_make_key(h)`."""
    return object_key_to_tenant_digest(_make_key(h))


def _make_hash(h: int) -> int:
    """The u64 index hash the store derives for `_make_key(h)`."""
    return object_key_to_chunk_hash(_make_key(h))


def _make_key(h: int) -> ObjectKey:
    md = _metadata()
    return ObjectKey(
        chunk_hash=h.to_bytes(8, "little"),
        model_name=md.model_name,
        kv_rank=0,
        cache_salt="",
    )


@pytest.fixture
def ctx():
    """Bootstrap a pool plus region allocator, index, writer, and a heap.

    The lock manager is started so writes can progress. Each test gets a
    fresh tmpfile. Yields everything a node-clear test needs.
    """
    with tempfile.NamedTemporaryFile(prefix="cxl-clear-", delete=False) as f:
        f.truncate(POOL_SIZE)
        path = f.name
    cfg = CXLBootstrapConfig(dev_path=path, region_size=REGION_SIZE, initialize=True)
    handle = bootstrap_pool(cfg)
    lock = TwoTierLock(handle, node_id=0)
    mgr = LockManager(handle)
    mgr.start()
    region_alloc = RegionAllocator(handle, lock)
    index = CXLIndex(handle)
    iw = CXLIndexWriter(handle, index, lock, node_id=0)
    heap = NodeHeap(region_alloc, node_id=0, chunk_size=CHUNK_SIZE)
    try:
        yield handle, region_alloc, index, iw, heap
    finally:
        mgr.stop()
        handle.close()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def _store_chunk(iw: CXLIndexWriter, heap: NodeHeap, key_hash: int, *, pin=False):
    """Alloc a heap chunk and publish a VALID slot pointing at it.

    Returns (slot_idx, chunk_offset).
    """
    offset = heap.alloc()
    res = iw.reserve_slot(_make_hash(key_hash), _make_digest(key_hash))
    assert res.outcome == ReserveOutcome.RESERVED
    iw.commit_slot(
        res.slot_idx,
        chunk_offset=offset,
        chunk_len=CHUNK_SIZE,
        fmt=MemoryFormat.KV_2LTD,
        pin=CommitPin.BORN_PINNED if pin else CommitPin.UNPINNED,
    )
    return res.slot_idx, offset


def test_clear_owned_slots_tombstones_and_returns_offsets(ctx):
    """Every VALID slot this node owns is tombstoned; its offset is returned."""
    handle, _region_alloc, _index, iw, heap = ctx
    stored = [_store_chunk(iw, heap, 0x1000 + i) for i in range(6)]
    expected_offsets = sorted(off for _slot, off in stored)

    freed, skipped = iw.clear_owned_slots()

    assert skipped == 0
    assert sorted(freed) == expected_offsets
    slots = handle.slots()
    for slot_idx, _off in stored:
        assert slots[slot_idx].line0.state == SLOT_STATE_TOMB


def test_clear_owned_slots_skips_other_nodes(ctx):
    """A slot owned by a different node is left VALID and its offset omitted."""
    handle, region_alloc, index, iw, heap = ctx
    mine_slot, mine_off = _store_chunk(iw, heap, 0x2001)

    # A foreign node commits a VALID slot into the shared index. It draws its
    # chunk from its OWN region, so the offset is not in this heap.
    foreign_id = 7
    iw_foreign = CXLIndexWriter(
        handle, index, TwoTierLock(handle, node_id=foreign_id), node_id=foreign_id
    )
    foreign_heap = NodeHeap(region_alloc, node_id=foreign_id, chunk_size=CHUNK_SIZE)
    foreign_off = foreign_heap.alloc()
    r = iw_foreign.reserve_slot(_make_hash(0x2002), _make_digest(0x2002))
    assert r.outcome == ReserveOutcome.RESERVED
    iw_foreign.commit_slot(
        r.slot_idx,
        chunk_offset=foreign_off,
        chunk_len=CHUNK_SIZE,
        fmt=MemoryFormat.KV_2LTD,
    )

    freed, skipped = iw.clear_owned_slots()

    assert freed == [mine_off]
    assert foreign_off not in freed
    assert skipped == 0
    slots = handle.slots()
    assert slots[mine_slot].line0.state == SLOT_STATE_TOMB
    assert slots[r.slot_idx].line0.state == SLOT_STATE_VALID  # foreign untouched


def test_clear_owned_slots_skips_busy(ctx):
    """A pinned slot is skipped (counted), not tombstoned or freed."""
    handle, _region_alloc, _index, iw, heap = ctx
    free_slot, free_off = _store_chunk(iw, heap, 0x3001)
    pinned_slot, pinned_off = _store_chunk(iw, heap, 0x3002, pin=True)

    freed, skipped = iw.clear_owned_slots()

    assert freed == [free_off]
    assert pinned_off not in freed
    assert skipped == 1
    slots = handle.slots()
    assert slots[free_slot].line0.state == SLOT_STATE_TOMB
    assert slots[pinned_slot].line0.state == SLOT_STATE_VALID


def test_clear_owned_slots_empty_index(ctx):
    """Clearing with nothing stored returns empty, skips nothing."""
    _handle, _region_alloc, _index, iw, _heap = ctx
    freed, skipped = iw.clear_owned_slots()
    assert freed == []
    assert skipped == 0


def test_clear_frees_heap_and_trims_regions(ctx):
    """End-to-end: freed offsets returned to the heap empty its regions,
    and trim() releases every one back to the pool."""
    _handle, _region_alloc, _index, iw, heap = ctx
    # Span more than one region: 4 slots/region, store 9 -> 3 regions.
    for i in range(9):
        _store_chunk(iw, heap, 0x4000 + i)
    regions_before = heap.stats().total_regions
    assert regions_before >= 2

    freed, skipped = iw.clear_owned_slots()
    assert skipped == 0
    heap.free_batch(freed)
    released = heap.trim()

    # Every region the heap held is released; nothing left owned or free.
    assert len(released) == regions_before
    assert heap.stats().total_regions == 0
    assert heap.stats().free_slots == 0


def test_clear_keeps_region_with_busy_chunk(ctx):
    """A region holding a pinned chunk is NOT trimmed after clear."""
    _handle, _region_alloc, _index, iw, heap = ctx
    # One pinned chunk keeps its region alive; fill the rest of that region and
    # a second full region that should trim.
    _store_chunk(iw, heap, 0x5001, pin=True)  # region 0, stays
    for i in range(3):
        _store_chunk(iw, heap, 0x5100 + i)  # rest of region 0
    for i in range(4):
        _store_chunk(iw, heap, 0x5200 + i)  # region 1, fully clearable

    regions_before = heap.stats().total_regions
    assert regions_before >= 2

    freed, skipped = iw.clear_owned_slots()
    assert skipped == 1
    heap.free_batch(freed)
    released = heap.trim()

    # The pinned chunk's region survives; at least one full region trimmed.
    assert len(released) >= 1
    assert heap.stats().total_regions == regions_before - len(released)
    assert heap.stats().total_regions >= 1
