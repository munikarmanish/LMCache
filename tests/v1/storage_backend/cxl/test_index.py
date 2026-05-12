# SPDX-License-Identifier: Apache-2.0
"""Lockless CXL index lookup tests.

The writer path (reserve_slot / commit / EVICT) is implemented in a
later slice, so here we stamp slot state directly via the struct
view. That's fine: the reader doesn't care how a slot got written,
only what bytes it sees.
"""

# Standard
import ctypes
import os
import tempfile
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.cxl.bootstrap import (
    CXLBootstrapConfig,
    bootstrap_pool,
    compute_geom_hash,
)
from lmcache.v1.storage_backend.cxl.index import (
    CXLIndex,
    DEFAULT_MAX_PROBE,
    SlotView,
    _slot_probe_order,
)
from lmcache.v1.storage_backend.cxl.layout import (
    GEOM_HASH_SIZE,
    SLOT_STATE_ALLOCATING,
    SLOT_STATE_EMPTY,
    SLOT_STATE_TOMB,
    SLOT_STATE_VALID,
)


POOL_SIZE = 64 * (1 << 20)
REGION_SIZE = 2 * (1 << 20)


def _metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test-model",
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
    with tempfile.NamedTemporaryFile(prefix="cxl-idx-", delete=False) as f:
        f.truncate(POOL_SIZE)
        path = f.name
    cfg = CXLBootstrapConfig(
        dev_path=path, region_size=REGION_SIZE, initialize=True, generation=7
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


def _make_key(chunk_hash: int) -> CacheEngineKey:
    md = _metadata()
    return CacheEngineKey(
        model_name=md.model_name,
        world_size=md.world_size,
        worker_id=md.worker_id,
        chunk_hash=chunk_hash,
        dtype=md.kv_dtype,
    )


def _write_slot(
    handle,
    slot_idx: int,
    chunk_hash: int,
    state: int,
    *,
    chunk_offset: int = 0,
    chunk_len: int = 0,
    fmt: int = MemoryFormat.KV_2LTD.value,
    owner: int = 0,
    generation: int | None = None,
    geom_hash: bytes | None = None,
):
    """Test-only: directly write a slot's line0. Simulates a writer."""
    slot = handle.slots()[slot_idx]
    slot.line0.chunk_hash = chunk_hash & 0xFFFFFFFFFFFFFFFF
    slot.line0.chunk_offset = chunk_offset
    slot.line0.chunk_len = chunk_len
    slot.line0.state = state
    slot.line0.fmt = fmt
    slot.line0.owner_node_id = owner
    slot.line0.generation = (
        generation if generation is not None else handle.header.gen
    )
    gh = geom_hash if geom_hash is not None else bytes(handle.header.geom_hash)
    assert len(gh) == GEOM_HASH_SIZE
    ctypes.memmove(slot.line0.geom_hash, gh, GEOM_HASH_SIZE)


# ---------- probe order ----------


def test_probe_order_wraps_around():
    order = list(_slot_probe_order(chunk_hash=10, slot_count=4, max_probe=8))
    # 10 % 4 = 2 → 2, 3, 0, 1, 2, 3, 0, 1
    assert order == [2, 3, 0, 1, 2, 3, 0, 1]


def test_probe_order_starts_at_hash_mod_n():
    order = list(_slot_probe_order(chunk_hash=42, slot_count=17, max_probe=3))
    assert order[0] == 42 % 17


# ---------- basic lookup ----------


def test_empty_index_lookup_misses(handle):
    idx = CXLIndex(handle)
    assert idx.lookup(_make_key(0x1234)) is None


def test_valid_slot_hit(handle):
    idx = CXLIndex(handle)
    key = _make_key(0xCAFEBABE)
    slot_idx = key.chunk_hash % idx.slot_count
    _write_slot(
        handle,
        slot_idx,
        chunk_hash=key.chunk_hash,
        state=SLOT_STATE_VALID,
        chunk_offset=1024,
        chunk_len=512,
    )
    view = idx.lookup(key)
    assert view is not None
    assert isinstance(view, SlotView)
    assert view.slot_idx == slot_idx
    assert view.chunk_hash == key.chunk_hash
    assert view.chunk_offset == 1024
    assert view.chunk_len == 512
    assert view.state == SLOT_STATE_VALID


def test_empty_slot_terminates_probe(handle):
    """A VALID match that lives past an EMPTY must NOT be found.

    Open-addressing invariant: inserts place at the first EMPTY after
    probing from the hash. So a VALID slot past an EMPTY cannot have
    been the target of this probe chain — it's for some other hash.
    """
    idx = CXLIndex(handle)
    key = _make_key(1)
    start = key.chunk_hash % idx.slot_count
    # Leave slot `start` empty; place the key two slots later.
    _write_slot(
        handle,
        (start + 2) % idx.slot_count,
        chunk_hash=key.chunk_hash,
        state=SLOT_STATE_VALID,
    )
    assert idx.lookup(key) is None


def test_tomb_slot_is_skipped(handle):
    """TOMB does not terminate probing — a real hit can follow."""
    idx = CXLIndex(handle)
    key = _make_key(2)
    start = key.chunk_hash % idx.slot_count
    _write_slot(handle, start, chunk_hash=0, state=SLOT_STATE_TOMB)
    _write_slot(
        handle,
        (start + 1) % idx.slot_count,
        chunk_hash=key.chunk_hash,
        state=SLOT_STATE_VALID,
    )
    view = idx.lookup(key)
    assert view is not None
    assert view.slot_idx == (start + 1) % idx.slot_count


def test_allocating_slot_is_skipped(handle):
    """ALLOCATING is a partial write; readers skip and keep probing.

    A concurrent INSERT is publishing a slot; committing will flip it
    to VALID. If we returned the ALLOCATING slot, callers could DMA
    from an uninitialized chunk.
    """
    idx = CXLIndex(handle)
    key = _make_key(3)
    start = key.chunk_hash % idx.slot_count
    _write_slot(handle, start, chunk_hash=key.chunk_hash, state=SLOT_STATE_ALLOCATING)
    _write_slot(
        handle,
        (start + 1) % idx.slot_count,
        chunk_hash=key.chunk_hash,
        state=SLOT_STATE_VALID,
    )
    view = idx.lookup(key)
    assert view is not None
    assert view.slot_idx == (start + 1) % idx.slot_count


def test_different_hash_on_probe_chain_is_skipped(handle):
    """Wrong-hash VALID slots don't cause false hits; probing continues."""
    idx = CXLIndex(handle)
    target = _make_key(4)
    intruder = _make_key(4 + idx.slot_count)  # same slot, different hash
    start = target.chunk_hash % idx.slot_count
    # Intruder sits at `start`, target one after.
    _write_slot(
        handle, start, chunk_hash=intruder.chunk_hash, state=SLOT_STATE_VALID
    )
    _write_slot(
        handle,
        (start + 1) % idx.slot_count,
        chunk_hash=target.chunk_hash,
        state=SLOT_STATE_VALID,
    )
    view = idx.lookup(target)
    assert view is not None
    assert view.slot_idx == (start + 1) % idx.slot_count


# ---------- defensive checks ----------


def test_stale_generation_slot_is_skipped(handle):
    """A slot with an older generation is a fossil from a previous pool epoch."""
    idx = CXLIndex(handle)
    key = _make_key(5)
    start = key.chunk_hash % idx.slot_count
    # Writer stamps an older gen — simulates survivor from gen bump.
    _write_slot(
        handle,
        start,
        chunk_hash=key.chunk_hash,
        state=SLOT_STATE_VALID,
        generation=handle.header.gen - 1,
    )
    assert idx.lookup(key) is None


def test_geom_mismatch_slot_is_skipped(handle):
    """A slot whose geom_hash doesn't match the header must not be returned.

    Catches misconfigured peers that wrote chunks with a different KV
    layout. Returning such a slot would cause a DMA of bytes that
    can't be reinterpreted as the reader's KV geometry.
    """
    idx = CXLIndex(handle)
    key = _make_key(6)
    start = key.chunk_hash % idx.slot_count
    bogus = bytes(GEOM_HASH_SIZE)  # all zeros, definitely not our header's
    _write_slot(
        handle,
        start,
        chunk_hash=key.chunk_hash,
        state=SLOT_STATE_VALID,
        geom_hash=bogus,
    )
    assert idx.lookup(key) is None


def test_chunk_hash_zero_does_not_match_empty_slot(handle):
    """A key with chunk_hash=0 must not be confused with an EMPTY slot.

    EMPTY slots have all-zero memory, so chunk_hash defaults to 0.
    The state check is what disambiguates.
    """
    idx = CXLIndex(handle)
    key_zero = _make_key(0)
    # Do NOT write anything — the whole index is EMPTY with chunk_hash=0.
    assert idx.lookup(key_zero) is None

    # Now stamp a real slot at probe start with chunk_hash=0 and VALID.
    slot_idx = 0  # 0 % N = 0
    _write_slot(handle, slot_idx, chunk_hash=0, state=SLOT_STATE_VALID)
    view = idx.lookup(key_zero)
    assert view is not None and view.slot_idx == 0


# ---------- probe bound ----------


def test_max_probe_bounds_lookup_cost(handle):
    idx = CXLIndex(handle, max_probe=4)
    key = _make_key(7)
    start = key.chunk_hash % idx.slot_count
    # Place the match 5 slots out — beyond max_probe=4.
    _write_slot(
        handle,
        (start + 5) % idx.slot_count,
        chunk_hash=key.chunk_hash,
        state=SLOT_STATE_VALID,
    )
    # Fill the preceding 4 slots with TOMBs to force full probing.
    for j in range(5):
        if j == 0:
            _write_slot(handle, start, chunk_hash=0, state=SLOT_STATE_TOMB)
        else:
            _write_slot(
                handle, (start + j) % idx.slot_count, chunk_hash=0,
                state=SLOT_STATE_TOMB,
            )
    # max_probe=4 means we scan slots [start, start+3], miss the match.
    assert idx.lookup(key) is None

    # With a generous max_probe we find it.
    idx2 = CXLIndex(handle, max_probe=16)
    assert idx2.lookup(key) is not None


def test_max_probe_clamped_to_slot_count(handle):
    """Asking for more probes than slots is silently capped."""
    idx = CXLIndex(handle, max_probe=DEFAULT_MAX_PROBE)
    assert idx.max_probe <= idx.slot_count


# ---------- contains convenience ----------


def test_contains_matches_lookup(handle):
    idx = CXLIndex(handle)
    key = _make_key(99)
    assert not idx.contains(key)
    _write_slot(
        handle,
        key.chunk_hash % idx.slot_count,
        chunk_hash=key.chunk_hash,
        state=SLOT_STATE_VALID,
    )
    assert idx.contains(key)


# ---------- concurrency sanity ----------


def test_concurrent_reads_do_not_block(handle):
    """Many reader threads hitting the same VALID slot get consistent views.

    This is the "lock-free read" property: readers should never
    observe a torn value, and they should never get stuck.
    """
    idx = CXLIndex(handle)
    key = _make_key(0xABCDEF)
    _write_slot(
        handle,
        key.chunk_hash % idx.slot_count,
        chunk_hash=key.chunk_hash,
        state=SLOT_STATE_VALID,
        chunk_offset=4096,
        chunk_len=8192,
    )

    results = []
    results_lock = threading.Lock()

    def worker():
        for _ in range(1000):
            v = idx.lookup(key)
            with results_lock:
                results.append(v)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - start

    assert len(results) == 8 * 1000
    for v in results:
        assert v is not None
        assert v.chunk_offset == 4096
        assert v.chunk_len == 8192
    # Loose bound — 8000 pure-python lookups over a 128B read per slot
    # should finish in well under a second.
    assert elapsed < 5.0, f"lookups took {elapsed:.2f}s; concurrency regression?"


def test_reader_and_writer_race_is_safe(handle):
    """A writer flipping a slot ALLOCATING↔VALID while readers probe.

    The reader must see one of:
      (a) None / skip-past (ALLOCATING or wrong-hash VALID intermediate states),
      (b) a consistent VALID snapshot with the right offset/len.
    Never a corrupted (torn) view.
    """
    idx = CXLIndex(handle)
    key = _make_key(0xDEAD)
    slot_idx = key.chunk_hash % idx.slot_count

    stop = threading.Event()
    corruption = []

    def writer():
        offsets = [(1024, 512), (2048, 1024), (4096, 256)]
        i = 0
        slot = handle.slots()[slot_idx]
        # Seed the slot VALID with the first tuple so readers have
        # something to find. Subsequent iterations simulate the real
        # INSERT protocol: flip state to ALLOCATING (reader skips),
        # write offset/len (still not visible as VALID), then flip
        # state back to VALID (publish).
        _write_slot(
            handle,
            slot_idx,
            chunk_hash=key.chunk_hash,
            state=SLOT_STATE_VALID,
            chunk_offset=offsets[0][0],
            chunk_len=offsets[0][1],
        )
        while not stop.is_set():
            off, ln = offsets[i % len(offsets)]
            i += 1
            # Simulate reserve_slot → DMA → commit:
            # 1) writer takes the slot to ALLOCATING
            slot.line0.state = SLOT_STATE_ALLOCATING
            # 2) writer stages the new offset/len (not observable as VALID)
            slot.line0.chunk_offset = off
            slot.line0.chunk_len = ln
            # 3) writer publishes VALID last, as a single u32 store
            slot.line0.state = SLOT_STATE_VALID

    def reader():
        valid_pairs = {(1024, 512), (2048, 1024), (4096, 256)}
        for _ in range(5000):
            v = idx.lookup(key)
            if v is None:
                continue  # ALLOCATING or pre-first-write
            if (v.chunk_offset, v.chunk_len) not in valid_pairs:
                corruption.append((v.chunk_offset, v.chunk_len))

    w = threading.Thread(target=writer)
    w.start()
    try:
        readers = [threading.Thread(target=reader) for _ in range(4)]
        for r in readers:
            r.start()
        for r in readers:
            r.join()
    finally:
        stop.set()
        w.join()

    assert corruption == [], f"observed torn reads: {corruption[:5]}"
