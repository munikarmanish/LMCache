# SPDX-License-Identifier: Apache-2.0
"""Per-geometry heap classes (`HeapSet`).

One pool serves several KV geometries at once, so it needs several chunk
sizes. `HeapSet` creates one `NodeHeap` per distinct byte size on first
use and routes `free(offset)` back to the owning class.

Covers §3.3 of
``docs/design/v1/distributed/l2_adapters/cxl_multi_tenant.md``.
"""

# Standard
import os
import tempfile

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.cxl.bootstrap import (
    CXLBootstrapConfig,
    bootstrap_pool,
)
from lmcache.v1.storage_backend.cxl.heap import OutOfChunks
from lmcache.v1.storage_backend.cxl.heap_set import HeapSet
from lmcache.v1.storage_backend.cxl.lock_manager import LockManager
from lmcache.v1.storage_backend.cxl.locks import TwoTierLock
from lmcache.v1.storage_backend.cxl.regions import RegionAllocator

POOL_SIZE = 32 * (1 << 20)
REGION_SIZE = 1 << 20
NODE = 0


@pytest.fixture
def heaps():
    with tempfile.NamedTemporaryFile(prefix="cxl-hs-", delete=False) as f:
        f.truncate(POOL_SIZE)
        path = f.name
    handle = bootstrap_pool(
        CXLBootstrapConfig(dev_path=path, region_size=REGION_SIZE, initialize=True)
    )
    mgr = LockManager(handle)
    mgr.start()
    lock = TwoTierLock(handle, node_id=NODE)
    alloc = RegionAllocator(handle, lock)
    try:
        yield HeapSet(alloc, node_id=NODE), alloc
    finally:
        mgr.stop()
        handle.close()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


# ---------- class creation ----------


def test_classes_are_created_on_first_use(heaps):
    hs, _ = heaps
    assert hs.classes() == []
    hs.alloc(4096)
    hs.alloc(8192)
    assert hs.classes() == [4096, 8192]


def test_same_size_reuses_one_class(heaps):
    """Two geometries that compute to the same size share a class.

    The tenant digest keeps their contents distinct, so sharing a slab
    is safe and bounds the class count by distinct sizes, not models.
    """
    hs, _ = heaps
    hs.alloc(4096)
    hs.alloc(4096)
    assert hs.classes() == [4096]
    assert hs.heap_for(4096) is hs.heap_for(4096)


def test_rejects_non_positive_size(heaps):
    hs, _ = heaps
    with pytest.raises(ValueError, match="must be positive"):
        hs.alloc(0)


def test_rejects_size_larger_than_region(heaps):
    """A class that cannot fit one slot in a region fails loudly."""
    hs, _ = heaps
    with pytest.raises(ValueError, match="exceeds region_size"):
        hs.alloc(REGION_SIZE + 1)


# ---------- routing ----------


def test_free_routes_to_the_owning_class(heaps):
    hs, _ = heaps
    small = hs.alloc(4096)
    large = hs.alloc(16384)

    hs.free(small)
    hs.free(large)

    # Each freed offset went back to its own class's free-list.
    stats = hs.stats()
    assert stats[4096].free_slots == stats[4096].total_slots
    assert stats[16384].free_slots == stats[16384].total_slots


def test_free_rejects_a_foreign_offset(heaps):
    hs, _ = heaps
    hs.alloc(4096)
    with pytest.raises(ValueError, match="not fall in any region|fall in no region"):
        hs.free(0)  # inside the metadata section, not any region


def test_free_batch_groups_by_class(heaps):
    hs, _ = heaps
    offs = [hs.alloc(4096) for _ in range(3)]
    offs += [hs.alloc(16384) for _ in range(2)]

    hs.free_batch(offs)

    stats = hs.stats()
    assert stats[4096].free_slots == stats[4096].total_slots
    assert stats[16384].free_slots == stats[16384].total_slots


def test_free_batch_frees_known_offsets_before_raising(heaps):
    """A bad offset must not strand the good ones in the same call."""
    hs, _ = heaps
    good = hs.alloc(4096)
    with pytest.raises(ValueError, match="not fall in any region|fall in no region"):
        hs.free_batch([good, 0])
    stats = hs.stats()
    assert stats[4096].free_slots == stats[4096].total_slots


# ---------- allocation semantics ----------


def test_alloc_batch_is_per_class(heaps):
    hs, _ = heaps
    offs = hs.alloc_batch(4096, 5)
    assert len(offs) == 5
    assert len(set(offs)) == 5
    assert hs.classes() == [4096]


def test_alloc_no_claim_is_scoped_to_its_class(heaps):
    """An empty class does not borrow another class's free slots."""
    hs, _ = heaps
    # Populate the 4096 class so the *pool* has free slots somewhere.
    hs.free(hs.alloc(4096))
    with pytest.raises(OutOfChunks):
        hs.alloc_no_claim(16384)


def test_occupancy_sums_across_classes(heaps):
    hs, _ = heaps
    hs.alloc(4096)
    hs.alloc(16384)
    occupied, total = hs.occupancy()
    assert occupied == 2
    # Two classes, each holding one region's worth of slots.
    per_region = (REGION_SIZE // 4096) + (REGION_SIZE // 16384)
    assert total == per_region


def test_trim_releases_empty_regions_across_classes(heaps):
    hs, alloc = heaps
    a = hs.alloc(4096)
    b = hs.alloc(16384)
    assert len(hs.owned_regions()) == 2

    hs.free(a)
    hs.free(b)
    released = hs.trim()

    assert len(released) == 2
    assert hs.owned_regions() == []
    assert len(released) == len(set(released))


# ---------- exact-fit sizing ----------


def test_non_divisible_size_leaves_only_a_tail(heaps):
    """Exact-fit sizes waste a bounded region tail, not per-chunk bytes."""
    hs, _ = heaps
    size = (REGION_SIZE // 3) + 1
    heap = hs.heap_for(size)
    assert heap.slots_per_region == REGION_SIZE // size
    tail = REGION_SIZE - heap.slots_per_region * size
    assert tail < size  # never more than one slot's worth
