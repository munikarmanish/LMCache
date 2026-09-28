# SPDX-License-Identifier: Apache-2.0
"""Tests for node-local LRU eviction on CXL region exhaustion.

When a store cannot claim a new region because the global pool is exhausted,
the node evicts its own coldest chunks (node-local LRU) back into its heap
free-list and retries the store from there, capping its footprint instead of
dropping the put. Covered here:

  - Claiming is preferred: while the pool has FREE regions, stores never evict.
  - On exhaustion, the coldest chunk is evicted and the new store lands;
    recently-read (hot) chunks survive.
  - Reads refresh recency (true LRU), so a chunk kept warm by reads is not the
    victim even if it was stored first.
  - A pinned / in-flight chunk is never evicted; if every candidate is pinned,
    the store batch is dropped whole (no partial store).
  - The node-local LRU tracker's own ordering / touch / forget semantics.
"""

# Standard
import os
import tempfile

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.cxl.lru_tracker import NodeLRUTracker
from lmcache.v1.storage_backend.cxl.store import CXLStore, CXLStoreConfig

# A deliberately tiny pool so the store path reaches region exhaustion within a
# handful of chunks. With region_size == chunk_size, each region holds exactly
# one chunk, so the region count is the chunk capacity — easy to reason about.
# The CXL pool size must be 2 MiB-aligned, so the region size (and the override)
# are multiples of 2 MiB. With max_nodes=2 the fixed header/locks/index overhead
# is small; a 10 MiB pool over 2 MiB regions bootstraps to exactly 4 regions.
REGION_SIZE = 2 << 20  # 2 MiB
CHUNK_SIZE = REGION_SIZE  # 1 chunk per region
POOL_SIZE = 64 * (1 << 20)  # backing file; region count is set by the override
POOL_OVERRIDE = 10 << 20  # 2 MiB-aligned -> 4 regions
POOL_REGIONS = 4
POOL_MAX_NODES = 2


def _metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="cxl-evict-test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(4, 2, 16, 4, 64),
        chunk_size=16,
    )


def _make_key(h: int) -> ObjectKey:
    md = _metadata()
    return ObjectKey(
        chunk_hash=h.to_bytes(8, "little"),
        model_name=md.model_name,
        kv_rank=0,
        cache_salt="",
    )


def _make_source_obj(size_bytes: int, fill_byte: int = 0xAB) -> TensorMemoryObj:
    data = torch.full((size_bytes,), fill_byte, dtype=torch.uint8)
    meta = MemoryObjMetadata(
        shape=torch.Size([size_bytes]),
        dtype=torch.uint8,
        address=data.data_ptr(),
        phy_size=size_bytes,
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.KV_2LTD,
    )
    return TensorMemoryObj(raw_data=data, metadata=meta, parent_allocator=None)


# Heap classes are exact-fit, so the payload size *is* the slab size.
# Storing a full region's worth per chunk keeps this file's "1 chunk per
# region, 4 regions" premise, which the eviction ladder is exercised
# against; a smaller payload would carve hundreds of slots per region and
# the pool would never exhaust.
PAYLOAD = REGION_SIZE


def _put(backend: CXLStore, key_hash: int) -> ObjectKey:
    key = _make_key(key_hash)
    backend.put_batch([key], [_make_source_obj(PAYLOAD)])
    return key


@pytest.fixture
def backend():
    """A CXLStore over a 4-region pool (1 chunk per region)."""
    with tempfile.NamedTemporaryFile(prefix="cxl-evict-", delete=False) as f:
        f.truncate(POOL_SIZE)
        path = f.name
    cfg = CXLStoreConfig(
        dev_path=path,
        node_id=0,
        max_chunk_size_bytes=CHUNK_SIZE,
        region_size=REGION_SIZE,
        initialize=True,
        pool_size_override=POOL_OVERRIDE,
        max_nodes=POOL_MAX_NODES,
        evict_low_watermark=0.5,
    )
    b = CXLStore(cfg)
    try:
        yield b
    finally:
        b.close()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


# ---------- tracker unit tests ----------


def test_tracker_orders_coldest_first():
    lru = NodeLRUTracker()
    for s in (5, 3, 9, 1):
        lru.touch(s)
    assert lru.coldest(2) == [5, 3]
    assert lru.coldest(10) == [5, 3, 9, 1]


def test_tracker_touch_promotes_to_hot():
    lru = NodeLRUTracker()
    for s in (5, 3, 9):
        lru.touch(s)
    lru.touch(5)  # 5 is now most-recent
    assert lru.coldest(1) == [3]
    assert lru.coldest(3) == [3, 9, 5]


def test_tracker_forget_and_clear():
    lru = NodeLRUTracker()
    for s in (1, 2, 3):
        lru.touch(s)
    lru.forget(2)
    assert lru.coldest(3) == [1, 3]
    assert lru.tracked_count() == 2
    lru.clear()
    assert lru.coldest(1) == []
    assert lru.tracked_count() == 0


def test_tracker_coldest_nonpositive():
    lru = NodeLRUTracker()
    lru.touch(1)
    assert lru.coldest(0) == []
    assert lru.coldest(-1) == []


# ---------- eviction integration ----------


def test_no_eviction_while_pool_has_free_regions(backend):
    """Stores up to capacity claim regions; nothing is evicted."""
    keys = [_put(backend, 0x100 + i) for i in range(POOL_REGIONS)]
    # Every chunk fit by claiming a fresh region — all still resident.
    for k in keys:
        assert backend.contains(k)


def test_exhaustion_evicts_coldest(backend):
    """Once the pool is full, a new store evicts coldest-first to the floor.

    With evict_low_watermark=0.5 over 4 owned slots, the eviction floor is 2:
    when the pool is exhausted, eviction drains down to the floor (freeing 2
    coldest chunks) rather than one at a time, so it doesn't re-trigger on the
    very next store. The two coldest go; the two warmest survive alongside the
    new chunk.
    """
    keys = [_put(backend, 0x200 + i) for i in range(POOL_REGIONS)]
    # Pool is now full (4 chunks, 4 regions). Store one more.
    new_key = _put(backend, 0x2FF)

    assert backend.contains(new_key), "new store must land after eviction"
    # The two coldest (first-stored, never-read) chunks were evicted to floor.
    assert not backend.contains(keys[0])
    assert not backend.contains(keys[1])
    # The two warmest survive.
    assert backend.contains(keys[2])
    assert backend.contains(keys[3])


def test_read_keeps_chunk_warm(backend):
    """A chunk kept warm by a read is not the eviction victim."""
    keys = [_put(backend, 0x300 + i) for i in range(POOL_REGIONS)]
    # Read the first-stored chunk so it becomes most-recently-used.
    assert backend.contains(keys[0], pin=False)
    # Trigger eviction with a new store.
    new_key = _put(backend, 0x3FF)

    assert backend.contains(new_key)
    # keys[0] was refreshed by the read, so keys[1] (now coldest) is the victim.
    assert backend.contains(keys[0]), "recently-read chunk must survive"
    assert not backend.contains(keys[1])


def test_pinned_chunk_never_evicted(backend):
    """A pinned chunk is skipped; the next-coldest is evicted instead."""
    keys = [_put(backend, 0x400 + i) for i in range(POOL_REGIONS)]
    # Pin the coldest chunk (first stored). Pinning also refreshes recency,
    # so re-touch order: pin keys[0] then verify it survives as pinned.
    assert backend.pin(keys[0])

    new_key = _put(backend, 0x4FF)
    assert backend.contains(new_key)
    # keys[0] is pinned -> never evicted even though it was oldest by store.
    assert backend.contains(keys[0])
    backend.unpin(keys[0])


def test_batch_dropped_when_all_pinned(backend):
    """If every resident chunk is pinned, an exhausting store batch is dropped."""
    keys = [_put(backend, 0x500 + i) for i in range(POOL_REGIONS)]
    for k in keys:
        assert backend.pin(k)

    # Pool full, every chunk pinned -> eviction frees nothing -> batch dropped.
    dropped_key = _make_key(0x5FF)
    backend.put_batch([dropped_key], [_make_source_obj(PAYLOAD)])

    assert not backend.contains(dropped_key), "batch must be dropped, not stored"
    # All original (pinned) chunks are intact.
    for k in keys:
        assert backend.contains(k)
        backend.unpin(k)


def test_evicted_key_can_be_restored(backend):
    """After a chunk is evicted, storing it again succeeds (slot reused)."""
    keys = [_put(backend, 0x600 + i) for i in range(POOL_REGIONS)]
    _put(backend, 0x6FF)  # evicts keys[0]
    assert not backend.contains(keys[0])

    # Re-store the evicted key; it should land by evicting the next-coldest.
    backend.put_batch([keys[0]], [_make_source_obj(PAYLOAD)])
    assert backend.contains(keys[0])
