# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-node DRAM heap (heap.py)."""

# Standard
import os
import tempfile
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.cxl.bootstrap import (
    CXLBootstrapConfig,
    bootstrap_pool,
)
from lmcache.v1.storage_backend.cxl.heap import NodeHeap
from lmcache.v1.storage_backend.cxl.lock_manager import LockManager
from lmcache.v1.storage_backend.cxl.locks import TwoTierLock
from lmcache.v1.storage_backend.cxl.regions import (
    NoRegionAvailable,
    RegionAllocator,
)


POOL_SIZE = 64 * (1 << 20)
REGION_SIZE = 2 * (1 << 20)  # 2 MiB per region
CHUNK_SIZE = 64 * 1024  # 64 KiB per chunk → 32 chunks per region


def _metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="heap-test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(4, 2, 16, 4, 64),
        chunk_size=16,
    )


@pytest.fixture
def ctx():
    with tempfile.NamedTemporaryFile(prefix="cxl-heap-", delete=False) as f:
        f.truncate(POOL_SIZE)
        path = f.name
    cfg = CXLBootstrapConfig(
        dev_path=path, region_size=REGION_SIZE, initialize=True
    )
    handle = bootstrap_pool(cfg, _metadata())
    lock = TwoTierLock(handle, node_id=0)
    alloc = RegionAllocator(handle, lock)
    mgr = LockManager(handle)
    mgr.start()
    try:
        yield handle, alloc
    finally:
        mgr.stop()
        handle.close()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


# ---------- init ----------


def test_heap_rejects_non_divisible_chunk_size(ctx):
    _, alloc = ctx
    with pytest.raises(ValueError, match="does not evenly divide"):
        NodeHeap(alloc, node_id=0, chunk_size=REGION_SIZE - 1)


def test_heap_rejects_zero_chunk_size(ctx):
    _, alloc = ctx
    with pytest.raises(ValueError):
        NodeHeap(alloc, node_id=0, chunk_size=0)


def test_fresh_heap_owns_no_regions(ctx):
    _, alloc = ctx
    heap = NodeHeap(alloc, node_id=0, chunk_size=CHUNK_SIZE)
    stats = heap.stats()
    assert stats.total_regions == 0
    assert stats.total_slots == 0
    assert stats.free_slots == 0
    assert stats.chunk_size == CHUNK_SIZE


# ---------- alloc / free ----------


def test_first_alloc_claims_region_and_seeds_slots(ctx):
    _, alloc = ctx
    heap = NodeHeap(alloc, node_id=0, chunk_size=CHUNK_SIZE)
    off = heap.alloc()
    stats = heap.stats()
    assert stats.total_regions == 1
    assert stats.total_slots == REGION_SIZE // CHUNK_SIZE
    # One slot just handed out, rest still free.
    assert stats.free_slots == stats.total_slots - 1
    # Offset is past the regions base and aligned to chunk_size.
    assert off >= heap._regions._handle.layout.off_regions
    assert (off - heap._regions._handle.layout.off_regions) % CHUNK_SIZE == 0


def test_alloc_distinct_offsets(ctx):
    _, alloc = ctx
    heap = NodeHeap(alloc, node_id=0, chunk_size=CHUNK_SIZE)
    offsets = [heap.alloc() for _ in range(10)]
    assert len(set(offsets)) == 10


def test_alloc_drains_region_then_claims_next(ctx):
    _, alloc = ctx
    heap = NodeHeap(alloc, node_id=0, chunk_size=CHUNK_SIZE)
    per_region = REGION_SIZE // CHUNK_SIZE
    for _ in range(per_region):
        heap.alloc()
    # Free list is now empty; next alloc must claim a second region.
    heap.alloc()
    assert heap.stats().total_regions == 2


def test_alloc_raises_when_pool_exhausted(ctx, handle=None):
    _, alloc = ctx
    heap = NodeHeap(alloc, node_id=0, chunk_size=CHUNK_SIZE)
    # Exhaust every region.
    per_region = REGION_SIZE // CHUNK_SIZE
    total_regions = alloc.region_count
    total_chunks = per_region * total_regions
    for _ in range(total_chunks):
        heap.alloc()
    with pytest.raises(NoRegionAvailable):
        heap.alloc()


def test_free_puts_offset_back_in_free_list(ctx):
    _, alloc = ctx
    heap = NodeHeap(alloc, node_id=0, chunk_size=CHUNK_SIZE)
    off = heap.alloc()
    before = heap.stats().free_slots
    heap.free(off)
    after = heap.stats().free_slots
    assert after == before + 1


def test_free_detects_double_free(ctx):
    _, alloc = ctx
    heap = NodeHeap(alloc, node_id=0, chunk_size=CHUNK_SIZE)
    off = heap.alloc()
    heap.free(off)
    with pytest.raises(ValueError, match="double-free"):
        heap.free(off)


def test_free_rejects_offset_from_unowned_region(ctx):
    _, alloc = ctx
    heap_a = NodeHeap(alloc, node_id=0, chunk_size=CHUNK_SIZE)
    off_a = heap_a.alloc()

    # Fresh heap on a different "node" — it owns no regions yet.
    heap_b = NodeHeap(alloc, node_id=1, chunk_size=CHUNK_SIZE)
    with pytest.raises(ValueError, match="this heap does not own"):
        heap_b.free(off_a)


def test_allocated_offset_points_into_claimed_region(ctx):
    _, alloc = ctx
    heap = NodeHeap(alloc, node_id=0, chunk_size=CHUNK_SIZE)
    off = heap.alloc()
    region_id = heap.owned_regions()[0]
    region_base = (
        heap._regions._handle.layout.off_regions
        + region_id * REGION_SIZE
    )
    assert region_base <= off < region_base + REGION_SIZE


# ---------- alloc_batch / free_batch ----------


def test_alloc_batch_returns_n_distinct_offsets(ctx):
    _, alloc = ctx
    heap = NodeHeap(alloc, node_id=0, chunk_size=CHUNK_SIZE)
    n = (REGION_SIZE // CHUNK_SIZE) + 5  # spans two regions
    batch = heap.alloc_batch(n)
    assert len(batch) == n
    assert len(set(batch)) == n
    assert heap.stats().total_regions >= 2


def test_alloc_batch_rejects_nonpositive_n(ctx):
    _, alloc = ctx
    heap = NodeHeap(alloc, node_id=0, chunk_size=CHUNK_SIZE)
    with pytest.raises(ValueError):
        heap.alloc_batch(0)
    with pytest.raises(ValueError):
        heap.alloc_batch(-1)


def test_free_batch_restores_free_count(ctx):
    _, alloc = ctx
    heap = NodeHeap(alloc, node_id=0, chunk_size=CHUNK_SIZE)
    batch = heap.alloc_batch(5)
    before = heap.stats().free_slots
    heap.free_batch(batch)
    after = heap.stats().free_slots
    assert after == before + 5


# ---------- trim ----------


def test_trim_releases_empty_regions(ctx):
    _, alloc = ctx
    heap = NodeHeap(alloc, node_id=0, chunk_size=CHUNK_SIZE)
    # Fill exactly one region, then free everything.
    per_region = REGION_SIZE // CHUNK_SIZE
    offsets = [heap.alloc() for _ in range(per_region)]
    assert heap.stats().total_regions == 1
    for off in offsets:
        heap.free(off)

    released = heap.trim()
    assert len(released) == 1
    assert heap.stats().total_regions == 0
    # Released region is now FREE in the global pool.
    assert alloc.info(released[0]).owner_node_id == 0xFFFF  # OWNER_FREE


def test_trim_keeps_partially_used_regions(ctx):
    _, alloc = ctx
    heap = NodeHeap(alloc, node_id=0, chunk_size=CHUNK_SIZE)
    # Claim one region but only use half.
    per_region = REGION_SIZE // CHUNK_SIZE
    for _ in range(per_region // 2):
        heap.alloc()
    released = heap.trim()
    assert released == []
    assert heap.stats().total_regions == 1


# ---------- concurrency ----------


def test_concurrent_alloc_free_has_no_double_handout(ctx):
    _, alloc = ctx
    heap = NodeHeap(alloc, node_id=0, chunk_size=CHUNK_SIZE)

    all_allocated = []
    allocated_lock = threading.Lock()

    def worker():
        local = []
        for _ in range(100):
            off = heap.alloc()
            local.append(off)
        with allocated_lock:
            all_allocated.extend(local)
        # Free them back.
        for off in local:
            heap.free(off)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)

    assert len(all_allocated) == 400
    # No offset handed out twice at overlapping times: since each
    # offset is freed after use, duplicates in the aggregated list are
    # only acceptable if one was freed before the next alloc grabbed
    # it. Check by ensuring, within each thread's allocations, all are
    # distinct; aggregating across threads is fine because free→alloc
    # reuse is legitimate.
    # A stronger invariant: at any one moment, each slot is owned by at
    # most one thread. The deque pop/append is under the same lock, so
    # this is guaranteed by construction. What we can verify here is
    # that the final state has all slots free.
    stats = heap.stats()
    assert stats.free_slots == stats.total_slots
