# SPDX-License-Identifier: Apache-2.0
"""Batched slot reservation (`reserve_slot_for_donor_batch`).

The cross-node fetch reserves one slot per chunk. Doing that one key at a
time cost one distributed-lock acquisition — roughly one arbiter sweep —
per chunk, which dominated the cold path at long prompts (117 chunks ≈
92 ms measured). The batched form takes all the start-slot locks in one
`acquire_batch`.

These tests pin the *semantics*, which must match the per-key method
exactly: same outcomes, same claimed slots, same probe-chain behavior.
The speedup itself is a property of `acquire_batch` and is covered by the
lock tests.
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
from lmcache.v1.storage_backend.cxl.index import CXLIndex
from lmcache.v1.storage_backend.cxl.index_writer import (
    CXLIndexWriter,
    ReserveOutcome,
)
from lmcache.v1.storage_backend.cxl.layout import (
    SLOT_STATE_ALLOCATING,
    SLOT_STATE_VALID,
)
from lmcache.v1.storage_backend.cxl.lock_manager import LockManager
from lmcache.v1.storage_backend.cxl.locks import TwoTierLock
from lmcache.v1.storage_backend.cxl.store import (
    object_key_to_chunk_hash,
    object_key_to_tenant_digest,
)

POOL_SIZE = 64 * (1 << 20)
REGION_SIZE = 2 * (1 << 20)
DONOR_NODE = 1


def _metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="cxl-reserve-batch-test",
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
    """The u64 index hash the store derives for this key."""
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
def writer():
    """A CXLIndexWriter over a fresh pool, with the arbiter running."""
    with tempfile.NamedTemporaryFile(prefix="cxl-resv-", delete=False) as f:
        f.truncate(POOL_SIZE)
        path = f.name
    cfg = CXLBootstrapConfig(dev_path=path, region_size=REGION_SIZE, initialize=True)
    handle = bootstrap_pool(cfg)
    lock = TwoTierLock(handle, node_id=0)
    mgr = LockManager(handle)
    mgr.start()
    index = CXLIndex(handle)
    iw = CXLIndexWriter(handle, index, lock, node_id=0)
    try:
        yield iw, handle
    finally:
        mgr.stop()
        handle.close()
        try:
            os.unlink(path)
        except OSError:
            pass


def test_batch_reserves_every_key(writer):
    iw, handle = writer
    keys = [_make_hash(1000 + i) for i in range(8)]
    digests = [_make_digest(1000 + i) for i in range(8)]

    results = iw.reserve_slot_for_donor_batch(keys, digests, DONOR_NODE)

    assert len(results) == len(keys)
    assert all(r.outcome is ReserveOutcome.RESERVED for r in results)
    # Distinct slots, each ALLOCATING and owned by the donor.
    slots = [r.slot_idx for r in results]
    assert len(set(slots)) == len(slots)
    for slot_idx in slots:
        line0 = handle.slots()[slot_idx].line0
        assert line0.state == SLOT_STATE_ALLOCATING
        assert line0.owner_node_id == DONOR_NODE


def test_batch_matches_per_key_method(writer):
    """The batch must produce the same outcomes and slots as the loop."""
    iw, _handle = writer
    keys = [_make_hash(2000 + i) for i in range(6)]
    digests = [_make_digest(2000 + i) for i in range(6)]

    # Reserve one at a time on a fresh pool...
    per_key = [
        iw.reserve_slot_for_donor(k, d, DONOR_NODE)
        for k, d in zip(keys, digests, strict=True)
    ]
    per_key_slots = [r.slot_idx for r in per_key]
    # ...release them so the pool returns to its starting state...
    for slot_idx in per_key_slots:
        iw.release_slot_for_donor(slot_idx, DONOR_NODE)

    # ...then batch the same keys and compare.
    batched = iw.reserve_slot_for_donor_batch(keys, digests, DONOR_NODE)

    assert [r.outcome for r in batched] == [r.outcome for r in per_key]
    assert [r.slot_idx for r in batched] == per_key_slots


def test_batch_empty_is_noop(writer):
    iw, _handle = writer
    assert iw.reserve_slot_for_donor_batch([], [], DONOR_NODE) == []


def test_batch_reports_already_present(writer):
    """A key whose slot is already VALID reports ALREADY_PRESENT, and the
    batch keeps going rather than terminating."""
    iw, handle = writer
    present = _make_hash(3000)
    present_digest = _make_digest(3000)

    # Reserve + commit one key so its slot is VALID.
    r = iw.reserve_slot_for_donor(present, present_digest, DONOR_NODE)
    assert r.outcome is ReserveOutcome.RESERVED
    handle.slots()[r.slot_idx].line0.owner_node_id = 0  # commit as self
    iw.commit_slot(r.slot_idx, chunk_offset=0, chunk_len=64, fmt=MemoryFormat.KV_2LTD)
    assert handle.slots()[r.slot_idx].line0.state == SLOT_STATE_VALID

    keys = [_make_hash(3001), present, _make_hash(3002)]
    digests = [_make_digest(3001), present_digest, _make_digest(3002)]
    results = iw.reserve_slot_for_donor_batch(keys, digests, DONOR_NODE)

    assert results[0].outcome is ReserveOutcome.RESERVED
    assert results[1].outcome is ReserveOutcome.ALREADY_PRESENT
    assert results[1].slot_idx == r.slot_idx
    # The batch did NOT stop at the ALREADY_PRESENT key.
    assert results[2].outcome is ReserveOutcome.RESERVED


def test_batch_handles_colliding_start_slots(writer):
    """Keys whose start slots share a lock_id must all still resolve.

    They are covered by one lock in the batch; the probe chains must not
    interfere or double-claim.
    """
    iw, handle = writer
    slot_count = iw._slot_count  # noqa: SLF001 - test asserts internal geometry
    # Same start slot => same lock_id, forcing the collision path. The
    # index hash is a digest, so search for keys that actually collide
    # rather than constructing them arithmetically.
    by_start: dict[int, list[tuple[int, bytes]]] = {}
    for i in range(100000):
        h = _make_hash(5000 + i)
        start = (h & 0xFFFFFFFFFFFFFFFF) % slot_count
        by_start.setdefault(start, []).append((h, _make_digest(5000 + i)))
        if len(by_start[start]) == 4:
            break
    found = next(v for v in by_start.values() if len(v) == 4)
    keys = [h for h, _ in found]
    digests = [d for _, d in found]
    starts = {(h & 0xFFFFFFFFFFFFFFFF) % slot_count for h in keys}
    assert len(starts) == 1, "test setup: keys must share a start slot"

    results = iw.reserve_slot_for_donor_batch(keys, digests, DONOR_NODE)

    assert all(r.outcome is ReserveOutcome.RESERVED for r in results)
    slots = [r.slot_idx for r in results]
    assert len(set(slots)) == len(slots), "each key must get its own slot"
    for slot_idx in slots:
        assert handle.slots()[slot_idx].line0.state == SLOT_STATE_ALLOCATING
