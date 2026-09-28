# SPDX-License-Identifier: Apache-2.0
"""Integration tests for CXLStore (single-process skeleton).

These exercise the full put → get → evict → pin/unpin lifecycle with
a tmpfile-backed pool. The backend runs its own lock manager; the
StubFence handles cross-host visibility (which is a no-op on a single
host anyway).
"""

# Standard
import hashlib
import os
import tempfile
import threading

# Third Party
import numpy as np
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
from lmcache.v1.storage_backend.cxl.store import CXLStore, CXLStoreConfig

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


def _make_key(h: int) -> ObjectKey:
    md = _metadata()
    return ObjectKey(
        chunk_hash=h.to_bytes(8, "little"),
        model_name=md.model_name,
        kv_rank=0,
        cache_salt="",
    )


def _content_hash(*parts: int) -> int:
    """A realistic 64-bit chunk_hash derived from `parts`.

    Production `chunk_hash` values come from hashing token content
    (sha256_cbor, or Python's builtin hash), so they are full-width 64-bit
    and uniformly spread across the index's `chunk_hash % slot_count` home
    slots. Small hand-picked integers are not: a base that happens to be a
    multiple of the slot count aliases every key onto the same home slot,
    which piles keys into one probe run and can exhaust `max_probe` while
    the index is still nearly empty.

    Tests that store many keys at once should use this instead of a literal
    so their probe behavior matches production.
    """
    payload = b":".join(str(p).encode() for p in parts)
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


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
    cfg = CXLStoreConfig(
        dev_path=path,
        node_id=0,
        max_chunk_size_bytes=CHUNK_SIZE,
        region_size=REGION_SIZE,
        initialize=True,
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


# ---------- basic lifecycle ----------


def _read(backend, key, size: int = 1024):
    """Read a chunk into a fresh buffer. Returns None on miss."""
    buf = torch.empty(size, dtype=torch.uint8)
    n = backend.read_into(key, buf.data_ptr(), buf.numel())
    return None if n == 0 else buf[:n]


def test_fresh_backend_has_no_entries(backend):
    assert not backend.contains(_make_key(0xDEAD))


def test_put_then_contains_hits(backend):
    key = _make_key(0x1111)
    src = _make_source_obj(1024, fill_byte=0x55)
    backend.put_batch([key], [src])
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
    backend.put_batch([key], [src])

    got = _read(backend, key, 1024)
    assert got is not None
    np.testing.assert_array_equal(got[:1024].numpy(), payload)


def test_get_misses_after_remove(backend):
    key = _make_key(0x3333)
    backend.put_batch([key], [_make_source_obj(512)])
    assert backend.contains(key)
    assert backend.remove(key)
    assert not backend.contains(key)
    assert _read(backend, key) is None


def test_remove_returns_false_for_missing_key(backend):
    assert not backend.remove(_make_key(0x4444))


def test_put_dedupes_same_key(backend):
    """A second put for the same key must not take a new chunk."""
    key = _make_key(0x5555)
    backend.put_batch([key], [_make_source_obj(1024, fill_byte=0x01)])
    stats_before = backend.heaps.total_stats()
    backend.put_batch([key], [_make_source_obj(1024, fill_byte=0x02)])
    stats_after = backend.heaps.total_stats()
    assert stats_before.free_slots == stats_after.free_slots
    # Content is whatever was written first; the second put is ignored.
    got = _read(backend, key)
    assert got is not None
    assert int(got[0]) == 0x01


def test_batched_put_multiple_keys(backend):
    keys = [_make_key(0x6000 + i) for i in range(5)]
    srcs = [_make_source_obj(1024, fill_byte=i) for i in range(5)]
    backend.put_batch(keys, srcs)
    for i, k in enumerate(keys):
        got = _read(backend, k)
        assert got is not None
        assert int(got[0]) == i


def test_partial_batch_stores_only_put_keys(backend):
    """Keys never stored stay misses; stored ones hit."""
    keys = [_make_key(0x7000 + i) for i in range(4)]
    # Put only the first two.
    backend.put_batch(keys[:2], [_make_source_obj(512) for _ in range(2)])
    assert [backend.contains(k) for k in keys] == [True, True, False, False]


# ---------- pin / unpin ----------


def test_pin_blocks_evict(backend):
    key = _make_key(0x9001)
    backend.put_batch([key], [_make_source_obj(512)])
    assert backend.pin(key)
    # remove should refuse because pin_count > 0.
    assert not backend.remove(key)
    assert backend.contains(key)
    # After unpin, remove succeeds.
    assert backend.unpin(key)
    assert backend.remove(key)


def test_unpin_of_unpinned_key_returns_false(backend):
    key = _make_key(0x9002)
    backend.put_batch([key], [_make_source_obj(512)])
    assert not backend.unpin(key)


def test_pin_on_missing_key_returns_false(backend):
    assert not backend.pin(_make_key(0x9003))


def test_pinned_slot_blocks_evict_until_unpinned(backend):
    """A pin holds a slot alive: remove must refuse until it is released.

    This is the lookup->retrieve window the L2 adapter relies on — the
    pin taken at lookup_and_lock keeps the chunk readable for the H2D
    copy even if another node's store wants to evict it.
    """
    key = _make_key(0xA001)
    backend.put_batch([key], [_make_source_obj(512)])
    assert backend.pin(key)

    # While pinned, remove fails.
    assert not backend.remove(key)
    assert backend.unpin(key)
    # Now remove succeeds.
    assert backend.remove(key)


# ---------- concurrency ----------


def test_concurrent_puts_and_gets_do_not_corrupt(backend):
    """Many threads putting and fetching distinct keys.

    Each key's payload byte pattern is derived from its chunk_hash, so
    a successful get must see bytes matching that pattern.
    """
    num_threads = 4
    per_thread = 50

    # Failures are collected rather than asserted in-thread: an assert inside
    # a worker only surfaces as a PytestUnhandledThreadExceptionWarning, which
    # does NOT fail the test. The main thread asserts on this list instead.
    failures: list[str] = []
    failures_lock = threading.Lock()

    def worker(tid: int):
        for i in range(per_thread):
            # Realistic content-derived hash: full-width and uniformly spread
            # over the index's home slots, matching production key behavior.
            h = _content_hash(tid, i)
            key = _make_key(h)
            fill = h & 0xFF
            try:
                backend.put_batch([key], [_make_source_obj(512, fill_byte=fill)])
                got = _read(backend, key, 512)
                if got is None:
                    with failures_lock:
                        failures.append(f"miss after put for h={h:#x}")
                    continue
                if int(got[0]) != fill:
                    with failures_lock:
                        failures.append(
                            f"corrupt payload for h={h:#x}: "
                            f"got {int(got[0])}, want {fill}"
                        )
            except Exception as e:  # noqa: BLE001 - reported below
                with failures_lock:
                    failures.append(f"exception for h={h:#x}: {e!r}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not any(t.is_alive() for t in threads), "worker thread did not finish"
    assert not failures, f"{len(failures)} failure(s): {failures[:10]}"


def test_double_pin_holds_two_refs(backend):
    """Two pins on one slot both count; remove blocks until both drop.

    Mirrors an MLA / multi-worker retrieve where several consumers each
    take their own pin on the same chunk.
    """
    key = _make_key(0xD001)
    backend.put_batch([key], [_make_source_obj(512)])
    assert backend.pin(key)
    assert backend.pin(key)
    assert not backend.remove(key)
    backend.unpin(key)
    assert not backend.remove(key)
    backend.unpin(key)
    assert backend.remove(key)


# ---------- memcheck ----------


def test_backend_close_cleans_up(backend):
    # Just running close() via fixture teardown is enough; this
    # placeholder ensures at least one test exercises close() without
    # prior put activity.
    assert backend.pool is not None


# ---------- batched pin / unpin ----------


def _pin_count(backend, key):
    """Read the on-CXL pin_count for ``key`` (-1 if no VALID slot)."""
    slot_idx = backend.slot_index_of(key)
    if slot_idx is None:
        return -1
    return int(backend.pool.slots()[slot_idx].line1.pin_count)


def test_pin_batch_pins_only_present_keys(backend):
    """pin_batch returns per-key success and pins exactly the VALID slots."""
    present = [_make_key(0x6100 + i) for i in range(3)]
    for k in present:
        backend.put_batch([k], [_make_source_obj(512)])
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
        backend.put_batch([k], [_make_source_obj(256)])

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
    backend.put_batch([key], [_make_source_obj(256)])

    results = backend.pin_batch([key, key])
    assert results == [True, True]
    assert _pin_count(backend, key) == 2

    backend.unpin_batch([key, key])
    assert _pin_count(backend, key) == 0


def test_pin_batch_empty_is_noop(backend):
    assert backend.pin_batch([]) == []
    assert backend.unpin_batch([]) == []
