# SPDX-License-Identifier: Apache-2.0
"""Tests for the global region allocator (regions.py)."""

# Standard
import os
import tempfile

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.cxl.bootstrap import (
    CXLBootstrapConfig,
    bootstrap_pool,
)
from lmcache.v1.storage_backend.cxl.layout import (
    OWNER_FREE,
    OWNER_ORPHANED,
)
from lmcache.v1.storage_backend.cxl.lock_manager import LockManager
from lmcache.v1.storage_backend.cxl.locks import TwoTierLock
from lmcache.v1.storage_backend.cxl.regions import (
    NoRegionAvailable,
    REGION_LOCK_ID,
    RegionAllocator,
)


POOL_SIZE = 64 * (1 << 20)
REGION_SIZE = 2 * (1 << 20)


def _metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="region-test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(4, 2, 16, 4, 64),
        chunk_size=16,
    )


@pytest.fixture
def handle():
    with tempfile.NamedTemporaryFile(prefix="cxl-rgn-", delete=False) as f:
        f.truncate(POOL_SIZE)
        path = f.name
    cfg = CXLBootstrapConfig(
        dev_path=path, region_size=REGION_SIZE, initialize=True, generation=11
    )
    h = bootstrap_pool(cfg, _metadata())
    try:
        yield h
    finally:
        h.close()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


@pytest.fixture
def allocator_and_manager(handle):
    lock = TwoTierLock(handle, node_id=0)
    mgr = LockManager(handle)
    mgr.start()
    try:
        yield RegionAllocator(handle, lock), mgr
    finally:
        mgr.stop()


# ---------- basics ----------


def test_fresh_pool_has_all_regions_free(allocator_and_manager, handle):
    alloc, _ = allocator_and_manager
    infos = list(alloc.iter_regions())
    assert len(infos) == handle.layout.region_count
    for info in infos:
        assert info.owner_node_id == OWNER_FREE
        assert info.claim_epoch == 0


def test_claim_marks_region_owned(allocator_and_manager):
    alloc, _ = allocator_and_manager
    r = alloc.claim(node_id=5)
    info = alloc.info(r)
    assert info.owner_node_id == 5
    assert info.claim_epoch == 11  # header.gen


def test_claim_returns_distinct_regions(allocator_and_manager):
    alloc, _ = allocator_and_manager
    seen = set()
    for _ in range(5):
        r = alloc.claim(node_id=0)
        assert r not in seen
        seen.add(r)


def test_claim_raises_when_pool_exhausted(allocator_and_manager, handle):
    alloc, _ = allocator_and_manager
    total = handle.layout.region_count
    for _ in range(total):
        alloc.claim(node_id=0)
    with pytest.raises(NoRegionAvailable):
        alloc.claim(node_id=0)


# ---------- release ----------


def test_release_returns_region_to_free_pool(allocator_and_manager):
    alloc, _ = allocator_and_manager
    r = alloc.claim(node_id=3)
    alloc.release(r, node_id=3)
    assert alloc.info(r).owner_node_id == OWNER_FREE
    # The released region is now re-claimable. Because `search_hint`
    # has advanced past `r`, we may or may not get `r` back on the
    # very next claim — the contract is only that it eventually
    # becomes available again.
    seen = set()
    for _ in range(alloc.region_count):
        seen.add(alloc.claim(node_id=3))
    assert r in seen


def test_release_rejects_wrong_owner(allocator_and_manager):
    alloc, _ = allocator_and_manager
    r = alloc.claim(node_id=3)
    with pytest.raises(RuntimeError, match="owner is 3"):
        alloc.release(r, node_id=4)


def test_release_rejects_reserved_node_ids(allocator_and_manager):
    alloc, _ = allocator_and_manager
    r = alloc.claim(node_id=1)
    with pytest.raises(ValueError):
        alloc.release(r, node_id=OWNER_FREE)
    with pytest.raises(ValueError):
        alloc.release(r, node_id=OWNER_ORPHANED)


# ---------- GC ----------


def test_gc_dead_node_orphans_all_its_regions(allocator_and_manager):
    alloc, _ = allocator_and_manager
    r1 = alloc.claim(node_id=7)
    r2 = alloc.claim(node_id=7)
    r3 = alloc.claim(node_id=8)

    orphaned = alloc.gc_dead_node(dead_node_id=7)
    assert set(orphaned) == {r1, r2}
    assert alloc.info(r1).owner_node_id == OWNER_ORPHANED
    assert alloc.info(r2).owner_node_id == OWNER_ORPHANED
    # Node 8's region is untouched.
    assert alloc.info(r3).owner_node_id == 8


def test_gc_does_not_free_bitmap_bit(allocator_and_manager):
    """Orphaned regions stay in the bitmap so no one re-claims them.

    The point: a VALID slot might still point into this region; if we
    cleared the bit, a peer could re-claim and overwrite.
    """
    alloc, _ = allocator_and_manager
    r = alloc.claim(node_id=9)
    alloc.gc_dead_node(dead_node_id=9)
    # A subsequent claim MUST NOT return r until it's promoted.
    others = {alloc.claim(node_id=0) for _ in range(5)}
    assert r not in others


# ---------- promote_orphaned ----------


def test_promote_orphaned_requires_orphaned_state(allocator_and_manager):
    alloc, _ = allocator_and_manager
    r = alloc.claim(node_id=1)
    # Region is OWNED, not ORPHANED. Promote should refuse.
    assert alloc.promote_orphaned(r, is_drained=lambda _: True) is False
    assert alloc.info(r).owner_node_id == 1


def test_promote_orphaned_respects_drain_predicate(allocator_and_manager):
    alloc, _ = allocator_and_manager
    r = alloc.claim(node_id=1)
    alloc.gc_dead_node(dead_node_id=1)
    assert alloc.info(r).owner_node_id == OWNER_ORPHANED

    # Predicate says not drained → no promotion.
    assert alloc.promote_orphaned(r, is_drained=lambda _: False) is False
    assert alloc.info(r).owner_node_id == OWNER_ORPHANED

    # Predicate says drained → promoted to FREE.
    assert alloc.promote_orphaned(r, is_drained=lambda _: True) is True
    assert alloc.info(r).owner_node_id == OWNER_FREE
    # And the bitmap bit is cleared — re-claimable.
    r2 = alloc.claim(node_id=2)
    # search_hint may skip forward; verify that eventually we can
    # reach the promoted region after enough claims wrap.
    reclaim_seen = {r2}
    while r not in reclaim_seen and len(reclaim_seen) < allocator_and_manager[0]._region_count:
        try:
            rid = alloc.claim(node_id=2)
        except NoRegionAvailable:
            break
        reclaim_seen.add(rid)
    assert r in reclaim_seen


# ---------- owned-by / info ----------


def test_regions_owned_by_lists_only_that_nodes_regions(allocator_and_manager):
    alloc, _ = allocator_and_manager
    alloc.claim(node_id=4)
    alloc.claim(node_id=4)
    alloc.claim(node_id=5)
    assert len(alloc.regions_owned_by(4)) == 2
    assert len(alloc.regions_owned_by(5)) == 1
    assert alloc.regions_owned_by(99) == []


def test_region_address_bounds(allocator_and_manager, handle):
    alloc, _ = allocator_and_manager
    r = alloc.claim(node_id=0)
    addr = alloc.region_address(r)
    assert addr >= handle.base + handle.layout.off_regions
    assert addr + REGION_SIZE <= handle.base + handle.size


# ---------- concurrency ----------


def test_claim_under_contention_has_no_double_alloc(handle):
    """Two TwoTierLock instances from different "nodes" claim concurrently.

    Every claimed region_id must be distinct; no over-allocation.
    """
    # Standard
    import threading

    lock_a = TwoTierLock(handle, node_id=0)
    lock_b = TwoTierLock(handle, node_id=1)
    alloc_a = RegionAllocator(handle, lock_a)
    alloc_b = RegionAllocator(handle, lock_b)

    all_claimed = []
    errors = []
    claim_lock = threading.Lock()

    def worker(allocator, node_id, n):
        local = []
        for _ in range(n):
            try:
                local.append(allocator.claim(node_id=node_id))
            except NoRegionAvailable:
                break
        with claim_lock:
            all_claimed.extend(local)

    mgr = LockManager(handle)
    mgr.start()
    try:
        ta = threading.Thread(target=worker, args=(alloc_a, 0, 5))
        tb = threading.Thread(target=worker, args=(alloc_b, 1, 5))
        ta.start()
        tb.start()
        ta.join(5)
        tb.join(5)
    finally:
        mgr.stop()

    # No region claimed twice.
    assert len(all_claimed) == len(set(all_claimed)), (
        f"double-allocation detected: {all_claimed}"
    )
    assert errors == []
