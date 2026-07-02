# SPDX-License-Identifier: Apache-2.0
"""Integration tests for CXLBackend (single-process skeleton).

These exercise the full put → get → evict → pin/unpin lifecycle with
a tmpfile-backed pool. The backend runs its own lock manager; the
StubFence handles cross-host visibility (which is a no-op on a single
host anyway).
"""

# Standard
import asyncio
import ctypes
import os
import tempfile
import threading

# Third Party
import numpy as np
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
from lmcache.v1.storage_backend.cxl_backend import CXLBackend, CXLBackendConfig


POOL_SIZE = 64 * (1 << 20)
REGION_SIZE = 2 * (1 << 20)
CHUNK_SIZE = 64 * 1024  # 64 KiB — 32 chunks per region


def _metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="cxl-backend-test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(4, 2, 16, 4, 64),
        chunk_size=16,
    )


def _make_key(h: int) -> CacheEngineKey:
    md = _metadata()
    return CacheEngineKey(
        model_name=md.model_name,
        world_size=md.world_size,
        worker_id=md.worker_id,
        chunk_hash=h,
        dtype=md.kv_dtype,
    )


def _make_source_obj(size_bytes: int, fill_byte: int = 0xAB) -> TensorMemoryObj:
    """A CPU-resident MemoryObj of `size_bytes` of uniform fill.

    Mimics the kind of payload a prefill worker would hand to put().
    """
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


@pytest.fixture
def backend():
    with tempfile.NamedTemporaryFile(prefix="cxl-backend-", delete=False) as f:
        f.truncate(POOL_SIZE)
        path = f.name
    cfg = CXLBackendConfig(
        dev_path=path,
        node_id=0,
        chunk_size_bytes=CHUNK_SIZE,
        region_size=REGION_SIZE,
        initialize=True,
    )
    b = CXLBackend(cfg, _metadata())
    try:
        yield b
    finally:
        b.close()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


# ---------- basic lifecycle ----------


def test_fresh_backend_has_no_entries(backend):
    assert not backend.contains(_make_key(0xDEAD))


def test_put_then_contains_hits(backend):
    key = _make_key(0x1111)
    src = _make_source_obj(1024, fill_byte=0x55)
    backend.batched_submit_put_task([key], [src])
    assert backend.contains(key)


def test_put_then_get_returns_same_bytes(backend):
    key = _make_key(0x2222)
    payload = (np.arange(1024, dtype=np.uint8) & 0xFF).astype(np.uint8)
    src = TensorMemoryObj(
        raw_data=torch.from_numpy(payload.copy()),
        metadata=MemoryObjMetadata(
            shape=torch.Size([1024]),
            dtype=torch.uint8,
            address=0,
            phy_size=1024,
            ref_count=1,
            pin_count=0,
            fmt=MemoryFormat.KV_2LTD,
        ),
        parent_allocator=None,
    )
    backend.batched_submit_put_task([key], [src])

    got = backend.get_blocking(key)
    assert got is not None
    # First 1024 bytes of got.raw_data must equal payload.
    got_bytes = got.raw_data[:1024].numpy()
    np.testing.assert_array_equal(got_bytes, payload)
    got.ref_count_down()  # release the pin


def test_get_misses_after_remove(backend):
    key = _make_key(0x3333)
    backend.batched_submit_put_task([key], [_make_source_obj(512)])
    assert backend.contains(key)
    assert backend.remove(key)
    assert not backend.contains(key)
    assert backend.get_blocking(key) is None


def test_remove_returns_false_for_missing_key(backend):
    assert not backend.remove(_make_key(0x4444))


def test_put_dedupes_same_key(backend):
    """A second put for the same key must not take a new chunk."""
    key = _make_key(0x5555)
    backend.batched_submit_put_task([key], [_make_source_obj(1024, fill_byte=0x01)])
    stats_before = backend._heap.stats()
    backend.batched_submit_put_task([key], [_make_source_obj(1024, fill_byte=0x02)])
    stats_after = backend._heap.stats()
    assert stats_before.free_slots == stats_after.free_slots
    # Content is whatever was written first; the second put is ignored.
    got = backend.get_blocking(key)
    assert got is not None
    assert int(got.raw_data[0]) == 0x01
    got.ref_count_down()


def test_batched_put_multiple_keys(backend):
    keys = [_make_key(0x6000 + i) for i in range(5)]
    srcs = [_make_source_obj(1024, fill_byte=i) for i in range(5)]
    backend.batched_submit_put_task(keys, srcs)
    for i, k in enumerate(keys):
        got = backend.get_blocking(k)
        assert got is not None
        assert int(got.raw_data[0]) == i
        got.ref_count_down()


def test_batched_contains_returns_prefix_hit_count(backend):
    keys = [_make_key(0x7000 + i) for i in range(4)]
    # Put only the first two.
    backend.batched_submit_put_task(keys[:2], [_make_source_obj(512) for _ in range(2)])
    hit = backend.batched_contains(keys)
    assert hit == 2  # stops at first miss


def test_on_complete_callback_fires_per_key(backend):
    keys = [_make_key(0x8000 + i) for i in range(3)]
    srcs = [_make_source_obj(512) for _ in range(3)]
    fired = []
    backend.batched_submit_put_task(
        keys, srcs, on_complete_callback=lambda k: fired.append(k.chunk_hash)
    )
    assert fired == [k.chunk_hash for k in keys]


# ---------- pin / unpin ----------


def test_pin_blocks_evict(backend):
    key = _make_key(0x9001)
    backend.batched_submit_put_task([key], [_make_source_obj(512)])
    assert backend.pin(key)
    # remove should refuse because pin_count > 0.
    assert not backend.remove(key)
    assert backend.contains(key)
    # After unpin, remove succeeds.
    assert backend.unpin(key)
    assert backend.remove(key)


def test_unpin_of_unpinned_key_returns_false(backend):
    key = _make_key(0x9002)
    backend.batched_submit_put_task([key], [_make_source_obj(512)])
    assert not backend.unpin(key)


def test_pin_on_missing_key_returns_false(backend):
    assert not backend.pin(_make_key(0x9003))


def test_held_memoryobj_blocks_evict_until_released(backend):
    """Holding a MemoryObj returned by get_blocking keeps ref_count>0,
    so concurrent remove must refuse until the obj is dropped."""
    key = _make_key(0xA001)
    backend.batched_submit_put_task([key], [_make_source_obj(512)])
    obj = backend.get_blocking(key)
    assert obj is not None

    # While held, remove fails.
    assert not backend.remove(key)
    # Release the obj — ref_count hits zero and TensorMemoryObj.free
    # is invoked via the _SlotRefcountAdapter, dropping our slot ref.
    obj.ref_count_down()
    # Now remove succeeds.
    assert backend.remove(key)


# ---------- async surface ----------


def test_batched_async_contains_matches_sync(backend):
    keys = [_make_key(0xB000 + i) for i in range(3)]
    backend.batched_submit_put_task(keys[:2], [_make_source_obj(512) for _ in range(2)])
    hit = asyncio.run(backend.batched_async_contains("lookup-1", keys))
    assert hit == 2


def test_batched_get_non_blocking_returns_hit_prefix(backend):
    keys = [_make_key(0xC000 + i) for i in range(4)]
    backend.batched_submit_put_task(keys[:3], [_make_source_obj(512) for _ in range(3)])
    got = asyncio.run(backend.batched_get_non_blocking("lookup-2", keys))
    assert len(got) == 3
    for obj in got:
        obj.ref_count_down()


# ---------- allocator surface ----------


def test_allocate_returns_memoryobj_within_chunk_size(backend):
    shapes = torch.Size([1024])
    obj = backend.allocate(shapes, torch.uint8, fmt=MemoryFormat.KV_2LTD)
    assert obj is not None
    assert obj.get_size() == 1024
    # Address is in the CXL pool.
    assert obj.meta.address >= backend._pool.base + backend._pool.layout.off_regions
    assert obj.meta.address + obj.get_size() <= backend._pool.base + backend._pool.size
    obj.ref_count_down()


def test_allocate_rejects_oversize_request(backend):
    with pytest.raises(ValueError):
        backend.allocate(torch.Size([CHUNK_SIZE + 1]), torch.uint8)


def test_calculate_chunk_budget_is_positive(backend):
    n = backend.calculate_chunk_budget()
    assert n > 0
    assert n == backend._heap.slots_per_region * backend._pool.layout.region_count


# ---------- concurrency ----------


def test_concurrent_puts_and_gets_do_not_corrupt(backend):
    """Many threads putting and fetching distinct keys.

    Each key's payload byte pattern is derived from its chunk_hash, so
    a successful get must see bytes matching that pattern.
    """
    num_threads = 4
    per_thread = 50

    def worker(tid: int):
        for i in range(per_thread):
            h = 0x10000 * (tid + 1) + i
            key = _make_key(h)
            fill = h & 0xFF
            backend.batched_submit_put_task(
                [key], [_make_source_obj(512, fill_byte=fill)]
            )
            got = backend.get_blocking(key)
            assert got is not None, f"miss after put for h={h}"
            assert int(got.raw_data[0]) == fill
            got.ref_count_down()

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)


def test_double_get_holds_two_refs(backend):
    """Two concurrent get_blocking calls both get valid objs; remove
    blocks until both are released."""
    key = _make_key(0xD001)
    backend.batched_submit_put_task([key], [_make_source_obj(512)])
    o1 = backend.get_blocking(key)
    o2 = backend.get_blocking(key)
    assert o1 is not None and o2 is not None
    assert not backend.remove(key)
    o1.ref_count_down()
    assert not backend.remove(key)
    o2.ref_count_down()
    assert backend.remove(key)


# ---------- memcheck ----------


def test_backend_close_cleans_up(backend):
    # Just running close() via fixture teardown is enough; this
    # placeholder ensures at least one test exercises close() without
    # prior put activity.
    assert backend._pool is not None


# ---------- batched pin / unpin ----------


def _pin_count(backend, key):
    """Read the on-CXL pin_count for ``key`` (-1 if no VALID slot)."""
    view = backend._index.lookup(key)
    if view is None:
        return -1
    return int(backend._pool.slots()[view.slot_idx].line1.pin_count)


def test_pin_batch_pins_only_present_keys(backend):
    """pin_batch returns per-key success and pins exactly the VALID slots."""
    present = [_make_key(0x6100 + i) for i in range(3)]
    for k in present:
        backend.batched_submit_put_task([k], [_make_source_obj(512)])
    missing = _make_key(0x61FF)

    # Interleave a missing key in the middle.
    keys = [present[0], missing, present[1], present[2]]
    results = backend.pin_batch(keys)
    assert results == [True, False, True, True]
    assert _pin_count(backend, present[0]) == 1
    assert _pin_count(backend, present[1]) == 1
    assert _pin_count(backend, present[2]) == 1
    assert _pin_count(backend, missing) == -1


def test_pin_batch_then_unpin_batch_round_trip(backend):
    """unpin_batch reverses pin_batch; the slot is reclaimable afterward."""
    keys = [_make_key(0x6200 + i) for i in range(4)]
    for k in keys:
        backend.batched_submit_put_task([k], [_make_source_obj(256)])

    backend.pin_batch(keys)
    for k in keys:
        assert _pin_count(backend, k) == 1
        # Pinned slots refuse eviction.
        assert not backend.remove(k)

    results = backend.unpin_batch(keys)
    assert results == [True, True, True, True]
    for k in keys:
        assert _pin_count(backend, k) == 0
        # Now reclaimable.
        assert backend.remove(k)


def test_pin_batch_duplicate_keys_bump_twice(backend):
    """A key appearing twice in pin_batch is pinned twice (one per slot)."""
    key = _make_key(0x6300)
    backend.batched_submit_put_task([key], [_make_source_obj(256)])

    results = backend.pin_batch([key, key])
    assert results == [True, True]
    assert _pin_count(backend, key) == 2

    backend.unpin_batch([key, key])
    assert _pin_count(backend, key) == 0


def test_pin_batch_empty_is_noop(backend):
    assert backend.pin_batch([]) == []
    assert backend.unpin_batch([]) == []
