# SPDX-License-Identifier: Apache-2.0
"""Tests for the periodic GC thread."""

# Standard
import os
import tempfile
import threading
import time
from typing import FrozenSet

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.cxl.bootstrap import (
    CXLBootstrapConfig,
    bootstrap_pool,
)
from lmcache.v1.storage_backend.cxl.gc import (
    CXLGarbageCollector,
    CXLGCConfig,
)
from lmcache.v1.storage_backend.cxl.index import CXLIndex
from lmcache.v1.storage_backend.cxl.index_writer import (
    CommitPin,
    CXLIndexWriter,
    ReserveOutcome,
)
from lmcache.v1.storage_backend.cxl.layout import (
    OWNER_FREE,
    OWNER_ORPHANED,
    SLOT_STATE_ALLOCATING,
    SLOT_STATE_TOMB,
    SLOT_STATE_VALID,
)
from lmcache.v1.storage_backend.cxl.lock_manager import LockManager
from lmcache.v1.storage_backend.cxl.locks import TwoTierLock
from lmcache.v1.storage_backend.cxl.regions import RegionAllocator


POOL_SIZE = 64 * (1 << 20)
REGION_SIZE = 2 * (1 << 20)


def _metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="cxl-gc-test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(4, 2, 16, 4, 64),
        chunk_size=16,
    )


def _make_key(h: int) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="cxl-gc-test",
        world_size=1,
        worker_id=0,
        chunk_hash=h,
        dtype=torch.float16,
    )


@pytest.fixture
def ctx():
    """Bootstrap a pool plus the lower layers needed for GC.

    The lock manager is started so writes (reserve_slot etc.) can
    progress. Each test gets a fresh tmpfile.
    """
    with tempfile.NamedTemporaryFile(prefix="cxl-gc-", delete=False) as f:
        f.truncate(POOL_SIZE)
        path = f.name
    cfg = CXLBootstrapConfig(dev_path=path, region_size=REGION_SIZE, initialize=True)
    handle = bootstrap_pool(cfg, _metadata())
    lock = TwoTierLock(handle, node_id=0)
    mgr = LockManager(handle)
    mgr.start()
    region_alloc = RegionAllocator(handle, lock)
    index = CXLIndex(handle)
    iw = CXLIndexWriter(handle, index, lock, node_id=0)
    try:
        yield handle, region_alloc, index, iw
    finally:
        mgr.stop()
        handle.close()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def _make_gc(
    ctx,
    alive: set,
    *,
    enforce_singleton: bool = False,
    sweep_interval_s: float = 0.05,
):
    handle, region_alloc, index, iw = ctx
    box = {"alive": frozenset(alive)}

    def liveness() -> FrozenSet[int]:
        return box["alive"]

    gc = CXLGarbageCollector(
        allocator=region_alloc,
        index_writer=iw,
        pool=handle,
        liveness=liveness,
        config=CXLGCConfig(
            sweep_interval_s=sweep_interval_s,
            enforce_singleton=enforce_singleton,
        ),
    )
    return gc, box


# ---------- discovery ----------


def test_no_op_when_all_owners_alive(ctx):
    _, region_alloc, _, _ = ctx
    region_alloc.claim(node_id=3)
    region_alloc.claim(node_id=5)
    gc, _ = _make_gc(ctx, alive={3, 5})
    stats = gc.sweep_once()
    assert stats["dead"] == []
    assert stats["regions_orphaned"] == 0
    assert stats["regions_freed"] == 0


def test_dead_owner_regions_become_orphaned(ctx):
    """Dead-owner regions are recognized and orphaned in one sweep.

    In this test, the dead node owned no VALID slots, so the same
    sweep also promotes the orphans to FREE. The intermediate
    ORPHANED state is observable in the stats dict.
    """
    _, region_alloc, _, _ = ctx
    r1 = region_alloc.claim(node_id=7)
    r2 = region_alloc.claim(node_id=7)
    r3 = region_alloc.claim(node_id=8)
    gc, _ = _make_gc(ctx, alive={8})  # node 7 is "dead"
    stats = gc.sweep_once()
    assert stats["dead"] == [7]
    assert stats["regions_orphaned"] == 2
    assert stats["regions_freed"] == 2  # immediately promoted (no VALID slots)
    # Final state: FREE (alive node 8's region untouched).
    assert region_alloc.info(r1).owner_node_id == OWNER_FREE
    assert region_alloc.info(r2).owner_node_id == OWNER_FREE
    assert region_alloc.info(r3).owner_node_id == 8


def test_orphaned_regions_are_not_re_orphaned_next_sweep(ctx):
    """Stamp a VALID slot in a dead-owned region so its orphan state
    persists across sweeps (otherwise it'd be promoted to FREE
    immediately). Confirm the ORPHANED state isn't re-orphaned.
    """
    handle, region_alloc, index, iw = ctx
    region_id = region_alloc.claim(node_id=4)
    layout = handle.layout
    region_lo = layout.off_regions + region_id * layout.region_size
    iw_dead = CXLIndexWriter(handle, index, TwoTierLock(handle, node_id=4), node_id=4)
    r = iw_dead.reserve_slot(_make_key(0xEE01))
    iw_dead.commit_slot(
        r.slot_idx,
        chunk_offset=region_lo + 256,
        chunk_len=128,
        fmt=MemoryFormat.KV_2LTD,
    )

    gc, _ = _make_gc(ctx, alive=set())
    first = gc.sweep_once()
    assert first["regions_orphaned"] == 1
    second = gc.sweep_once()
    # OWNER_ORPHANED != real node ids, so it's filtered out of dead set.
    assert second["regions_orphaned"] == 0
    assert second["dead"] == []
    # Region stays ORPHANED because the VALID slot keeps it pinned.
    assert region_alloc.info(region_id).owner_node_id == OWNER_ORPHANED


# ---------- ALLOCATING slot sweep ----------


def test_sweep_flips_dead_owner_allocating_to_tomb(ctx):
    handle, region_alloc, index, iw = ctx
    # Manually stamp two ALLOCATING slots: one owned by a dead node,
    # one owned by an alive node. Scan should affect only the dead one.
    iw_alive = CXLIndexWriter(handle, index, TwoTierLock(handle, node_id=2), node_id=2)
    iw_dead = CXLIndexWriter(handle, index, TwoTierLock(handle, node_id=9), node_id=9)
    r_alive = iw_alive.reserve_slot(_make_key(0xAA01))
    r_dead = iw_dead.reserve_slot(_make_key(0xAA02))
    assert r_alive.outcome == ReserveOutcome.RESERVED
    assert r_dead.outcome == ReserveOutcome.RESERVED

    gc, _ = _make_gc(ctx, alive={2})  # node 9 is dead

    freed = iw.sweep_dead_owner_allocating(9)
    assert len(freed) == 1
    assert freed[0][0] == r_dead.slot_idx

    # Verify the slot is now TOMB.
    slots = handle.slots()
    assert slots[r_dead.slot_idx].line0.state == SLOT_STATE_TOMB
    # Alive node's ALLOCATING slot is untouched.
    assert slots[r_alive.slot_idx].line0.state == SLOT_STATE_ALLOCATING


def test_sweep_does_not_touch_valid_slots(ctx):
    """Dead-owner GC must not flip VALID slots — peers may still read them.

    The `promote_orphaned` step is what eventually reclaims the
    region, after the VALID slots drain naturally.
    """
    handle, region_alloc, index, iw = ctx
    # Use an alive writer to stamp a VALID slot, then mark the writer's
    # node id as dead and confirm sweep leaves the slot alone.
    dead_node_id = 11
    iw_dead = CXLIndexWriter(
        handle, index, TwoTierLock(handle, node_id=dead_node_id), node_id=dead_node_id
    )
    r = iw_dead.reserve_slot(_make_key(0xBB10))
    assert r.outcome == ReserveOutcome.RESERVED
    iw_dead.commit_slot(
        r.slot_idx, chunk_offset=4096, chunk_len=512, fmt=MemoryFormat.KV_2LTD
    )

    iw.sweep_dead_owner_allocating(dead_node_id)

    slots = handle.slots()
    assert slots[r.slot_idx].line0.state == SLOT_STATE_VALID


# ---------- promote_orphaned ----------


def test_orphan_promotion_when_no_live_slots(ctx):
    """A dead-owned region with no VALID slots is orphaned and freed
    in the same sweep — the GC's three steps are pipelined per tick."""
    handle, region_alloc, index, iw = ctx
    region_id = region_alloc.claim(node_id=12)
    gc, _ = _make_gc(ctx, alive=set())
    stats = gc.sweep_once()
    assert stats["regions_orphaned"] == 1
    assert stats["regions_freed"] == 1
    assert region_alloc.info(region_id).owner_node_id == OWNER_FREE


def test_orphan_kept_until_valid_slot_drains(ctx):
    handle, region_alloc, index, iw = ctx
    region_id = region_alloc.claim(node_id=13)
    layout = handle.layout
    region_lo = layout.off_regions + region_id * layout.region_size

    # Stamp a VALID slot pointing into this region.
    iw_dead = CXLIndexWriter(handle, index, TwoTierLock(handle, node_id=13), node_id=13)
    r = iw_dead.reserve_slot(_make_key(0xCC10))
    iw_dead.commit_slot(
        r.slot_idx,
        chunk_offset=region_lo + 1024,
        chunk_len=256,
        fmt=MemoryFormat.KV_2LTD,
    )

    gc, _ = _make_gc(ctx, alive=set())
    s1 = gc.sweep_once()
    s2 = gc.sweep_once()
    # First sweep orphans, second sees the VALID slot still present.
    assert s1["regions_orphaned"] == 1
    assert s2["regions_freed"] == 0
    assert region_alloc.info(region_id).owner_node_id == OWNER_ORPHANED

    # Evict the slot — drain the orphan.
    ok, _view = iw.evict(r.slot_idx)
    assert ok
    s3 = gc.sweep_once()
    assert s3["regions_freed"] == 1
    assert region_alloc.info(region_id).owner_node_id == OWNER_FREE


# ---------- background thread ----------


def test_background_thread_runs_sweeps_periodically(ctx):
    _, region_alloc, _, _ = ctx
    region_alloc.claim(node_id=20)

    gc, alive_box = _make_gc(ctx, alive={20}, sweep_interval_s=0.02)
    gc.start()
    try:
        # Initially node 20 is alive — no orphans.
        time.sleep(0.05)
        assert gc.stats["regions_orphaned"] == 0

        # Mark it dead; the next sweep should orphan its region.
        alive_box["alive"] = frozenset()
        deadline = time.time() + 1.0
        while time.time() < deadline:
            if gc.stats["regions_orphaned"] >= 1:
                break
            time.sleep(0.01)
        assert gc.stats["regions_orphaned"] >= 1
    finally:
        gc.stop()


def test_singleton_enforcement_rejects_second_gc(ctx):
    gc1, _ = _make_gc(ctx, alive=set(), enforce_singleton=True)
    gc1.start()
    try:
        gc2, _ = _make_gc(ctx, alive=set(), enforce_singleton=True)
        with pytest.raises(RuntimeError, match="already running"):
            gc2.start()
    finally:
        gc1.stop()


def test_singleton_can_be_disabled_for_tests(ctx):
    gc1, _ = _make_gc(ctx, alive=set(), enforce_singleton=False)
    gc2, _ = _make_gc(ctx, alive=set(), enforce_singleton=False)
    gc1.start()
    try:
        gc2.start()
        gc2.stop()
    finally:
        gc1.stop()


def test_idempotent_concurrent_sweeps(ctx):
    """Two sweep_once() calls converge to the same state.

    Models the brief overlap during a singleton failover. The first
    sweep orphans + frees in one tick (no VALID slots); the second
    sweep finds nothing to do.
    """
    _, region_alloc, _, _ = ctx
    r = region_alloc.claim(node_id=30)
    gc1, _ = _make_gc(ctx, alive=set())
    gc2, _ = _make_gc(ctx, alive=set())

    s1 = gc1.sweep_once()
    assert s1["regions_orphaned"] == 1
    assert s1["regions_freed"] == 1
    assert region_alloc.info(r).owner_node_id == OWNER_FREE

    # Second sweep: nothing to do.
    s2 = gc2.sweep_once()
    assert s2["regions_orphaned"] == 0
    assert s2["regions_freed"] == 0


# ---------- region_has_no_live_slots ----------


def test_region_has_no_live_slots_returns_true_for_empty_region(ctx):
    handle, region_alloc, _, iw = ctx
    region_id = region_alloc.claim(node_id=0)
    layout = handle.layout
    lo = layout.off_regions + region_id * layout.region_size
    hi = lo + layout.region_size
    assert iw.region_has_no_live_slots(region_id, lo, hi)


def test_region_has_no_live_slots_returns_false_when_valid_slot_inside(ctx):
    handle, region_alloc, index, iw = ctx
    region_id = region_alloc.claim(node_id=0)
    layout = handle.layout
    lo = layout.off_regions + region_id * layout.region_size
    hi = lo + layout.region_size

    r = iw.reserve_slot(_make_key(0xDD01))
    iw.commit_slot(
        r.slot_idx, chunk_offset=lo + 256, chunk_len=128, fmt=MemoryFormat.KV_2LTD
    )
    assert not iw.region_has_no_live_slots(region_id, lo, hi)


# ---------- batched commit ----------


def test_commit_slot_batch_commits_all_born_pinned(ctx):
    """commit_slot_batch publishes every slot VALID with pin_count==1."""
    handle, region_alloc, index, iw = ctx
    region_id = region_alloc.claim(node_id=0)
    layout = handle.layout
    lo = layout.off_regions + region_id * layout.region_size

    n = 8
    reserved = [iw.reserve_slot(_make_key(0xE100 + i)) for i in range(n)]
    slot_idxs = [r.slot_idx for r in reserved]
    results = iw.commit_slot_batch(
        slot_idxs=slot_idxs,
        chunk_offsets=[lo + 256 * i for i in range(n)],
        chunk_lens=[128] * n,
        fmts=[MemoryFormat.KV_2LTD] * n,
        pin=CommitPin.BORN_PINNED,
    )
    assert results == [True] * n
    slots = handle.slots()
    for s in slot_idxs:
        assert slots[s].line0.state == SLOT_STATE_VALID
        assert slots[s].line1.pin_count == 1


def test_commit_slot_batch_unpinned_default(ctx):
    """Default pin mode leaves committed slots VALID with pin_count==0."""
    handle, region_alloc, index, iw = ctx
    region_id = region_alloc.claim(node_id=0)
    lo = handle.layout.off_regions + region_id * handle.layout.region_size

    reserved = [iw.reserve_slot(_make_key(0xE200 + i)) for i in range(3)]
    slot_idxs = [r.slot_idx for r in reserved]
    results = iw.commit_slot_batch(
        slot_idxs=slot_idxs,
        chunk_offsets=[lo + 256 * i for i in range(3)],
        chunk_lens=[128] * 3,
        fmts=[MemoryFormat.KV_2LTD] * 3,
    )
    assert results == [True, True, True]
    slots = handle.slots()
    for s in slot_idxs:
        assert slots[s].line0.state == SLOT_STATE_VALID
        assert slots[s].line1.pin_count == 0


def test_commit_slot_batch_skips_non_allocating(ctx):
    """A slot that is not ALLOCATING is reported False, others still commit."""
    handle, region_alloc, index, iw = ctx
    region_id = region_alloc.claim(node_id=0)
    lo = handle.layout.off_regions + region_id * handle.layout.region_size

    reserved = [iw.reserve_slot(_make_key(0xE300 + i)) for i in range(3)]
    slot_idxs = [r.slot_idx for r in reserved]
    # Pre-commit the middle slot so it is VALID (not ALLOCATING) when the
    # batch runs.
    iw.commit_slot(
        slot_idxs[1], chunk_offset=lo, chunk_len=128, fmt=MemoryFormat.KV_2LTD
    )

    results = iw.commit_slot_batch(
        slot_idxs=slot_idxs,
        chunk_offsets=[lo + 256 * i for i in range(3)],
        chunk_lens=[128] * 3,
        fmts=[MemoryFormat.KV_2LTD] * 3,
        pin=CommitPin.BORN_PINNED,
    )
    # Middle slot was already VALID -> skipped; the other two commit.
    assert results == [True, False, True]
    slots = handle.slots()
    assert slots[slot_idxs[0]].line1.pin_count == 1
    assert slots[slot_idxs[2]].line1.pin_count == 1


def test_commit_slot_batch_rejects_unequal_lengths(ctx):
    _, _, _, iw = ctx
    with pytest.raises(ValueError, match="equal length"):
        iw.commit_slot_batch(
            slot_idxs=[0, 1],
            chunk_offsets=[0],
            chunk_lens=[1, 2],
            fmts=[MemoryFormat.KV_2LTD, MemoryFormat.KV_2LTD],
        )


def test_commit_slot_batch_empty_is_noop(ctx):
    _, _, _, iw = ctx
    assert (
        iw.commit_slot_batch(slot_idxs=[], chunk_offsets=[], chunk_lens=[], fmts=[])
        == []
    )
