# SPDX-License-Identifier: Apache-2.0
"""Cross-node CXL tests (step 6): two CXLBackends on the same pool.

These exercise:

- A node-A CXL HIT visible to node B via its own attach + CXL_LOOKUP.
  This is the warm-path property: any peer reading CXL after A
  publishes VALID sees the chunk without the controller in the loop.

- The cold-path PushKVToCXL fallback: A has the chunk only in its
  local tier; B reserves CXL slots on A's behalf and asks A to push.
  Modeled in-process — the transport-layer ZMQ wrapping is decoupled
  from this logic and lives outside the CXL package.
"""

# Standard
import os
import tempfile
import threading
from typing import Optional

# Third Party
import numpy as np
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.cxl.cross_node import (
    CXLDonor,
    PushStatus,
    remote_fetch,
)
from lmcache.v1.storage_backend.cxl.p2p_messages import (
    PushKVToCXLMsg,
    PushKVToCXLRetMsg,
)
from lmcache.v1.storage_backend.cxl_backend import CXLBackend, CXLBackendConfig


POOL_SIZE = 64 * (1 << 20)
REGION_SIZE = 2 * (1 << 20)
CHUNK_SIZE = 64 * 1024


def _metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="cross-node-test",
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


def _make_source_obj(size_bytes: int, fill_byte: int) -> TensorMemoryObj:
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
def two_node_pool():
    """Yield (node_a_backend, node_b_backend) on the same pool.

    Node A initializes the pool and runs the lock manager (rack-wide
    arbiter). Node B attaches read/write but does NOT run a lock
    manager — the plan calls for one lock manager per rack.
    """
    with tempfile.NamedTemporaryFile(prefix="cxl-xnode-", delete=False) as f:
        f.truncate(POOL_SIZE)
        path = f.name

    cfg_a = CXLBackendConfig(
        dev_path=path,
        node_id=0,
        chunk_size_bytes=CHUNK_SIZE,
        region_size=REGION_SIZE,
        initialize=True,
        run_lock_manager=True,
    )
    cfg_b = CXLBackendConfig(
        dev_path=path,
        node_id=1,
        chunk_size_bytes=CHUNK_SIZE,
        region_size=REGION_SIZE,
        initialize=False,
        run_lock_manager=False,
    )
    a = CXLBackend(cfg_a, _metadata())
    b = CXLBackend(cfg_b, _metadata())
    try:
        yield a, b
    finally:
        b.close()
        a.close()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


# ---------- warm CXL: A writes, B reads ----------


def test_node_b_reads_what_node_a_put_into_cxl(two_node_pool):
    """The defining CXL property: any peer can read a chunk A published.

    No controller, no PushKVToCXL — just CXL_LOOKUP on the shared pool.
    """
    a, b = two_node_pool
    key = _make_key(0xAA01)
    payload = (np.arange(2048, dtype=np.uint8) & 0xFF).astype(np.uint8)
    src = _make_source_obj(2048, fill_byte=0xCC)
    src.raw_data = torch.from_numpy(payload.copy())  # overwrite fill
    a.batched_submit_put_task([key], [src])

    # Node B sees the slot through its own attach.
    assert b.contains(key)
    got = b.get_blocking(key)
    assert got is not None
    np.testing.assert_array_equal(got.raw_data[:2048].numpy(), payload)
    got.ref_count_down()


def test_node_a_can_evict_after_node_b_releases_get(two_node_pool):
    """Cross-node ref counting: B's get pins the slot, A's remove must wait.

    Caveat: this models the in-process case. With true cross-host
    coherence, B's ref_count++ is visible to A only after the CLFLUSH+
    MFENCE. Our StubFence is no-op on a single host, so this is
    automatic; the test documents the intent.
    """
    a, b = two_node_pool
    key = _make_key(0xAA02)
    a.batched_submit_put_task([key], [_make_source_obj(512, 0x11)])

    held = b.get_blocking(key)
    assert held is not None
    # B is holding a ref; A's remove must refuse.
    assert not a.remove(key)
    held.ref_count_down()
    # Now A can evict.
    assert a.remove(key)


# ---------- cold path: A has it locally, B fetches via PushKVToCXL ----------


class _FakeLocalTier:
    """Tiny stand-in for the donor's local L0/L1.

    Maps key string -> bytes payload + format. CXLDonor's
    LocalCopyProvider returns a TensorMemoryObj wrapping those bytes.
    """

    def __init__(self):
        self._store: dict[str, tuple[bytes, MemoryFormat]] = {}

    def put(self, key: CacheEngineKey, payload: bytes, fmt=MemoryFormat.KV_2LTD):
        self._store[key.to_string()] = (payload, fmt)

    def evict(self, key: CacheEngineKey):
        self._store.pop(key.to_string(), None)

    def __call__(self, key_str: str) -> Optional[MemoryObj]:
        record = self._store.get(key_str)
        if record is None:
            return None
        payload, fmt = record
        size = len(payload)
        data = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        meta = MemoryObjMetadata(
            shape=torch.Size([size]),
            dtype=torch.uint8,
            address=data.data_ptr(),
            phy_size=size,
            ref_count=1,  # caller will ref_count_down
            pin_count=0,
            fmt=fmt,
        )
        return TensorMemoryObj(
            raw_data=data, metadata=meta, parent_allocator=None
        )


def _build_donor(backend: CXLBackend, local: _FakeLocalTier) -> CXLDonor:
    return CXLDonor(
        handle=backend._pool,
        index_writer=backend._index_writer,
        heap=backend._heap,
        node_id=backend._node_id,
        local_copy_provider=local,
    )


def test_remote_fetch_publishes_donor_local_copy_into_cxl(two_node_pool):
    a, b = two_node_pool
    a_local = _FakeLocalTier()
    a_donor = _build_donor(a, a_local)

    keys = [_make_key(0xBB01 + i) for i in range(3)]
    payloads = [bytes([i & 0xFF] * 1024) for i in range(3)]
    for k, p in zip(keys, payloads):
        a_local.put(k, p)

    # B starts cold — no CXL_LOOKUP hits.
    assert all(not b.contains(k) for k in keys)

    # B runs the cold-path fallback, asking A to push.
    result = remote_fetch(
        requester_node_id=b._node_id,
        keys=keys,
        index_writer=b._index_writer,
        donor_node_id=a._node_id,
        donor=a_donor,
        sender_id="node-b",
        epoch=int(b._pool.header.gen),
    )
    assert result.num_satisfied == 3
    assert result.status == PushStatus.OK

    # Now B's CXL_LOOKUP hits all three.
    for i, k in enumerate(keys):
        got = b.get_blocking(k)
        assert got is not None
        assert int(got.raw_data[0]) == i
        got.ref_count_down()


def test_remote_fetch_truncates_when_donor_evicted_some(two_node_pool):
    a, b = two_node_pool
    a_local = _FakeLocalTier()
    a_donor = _build_donor(a, a_local)

    keys = [_make_key(0xBB10 + i) for i in range(4)]
    payloads = [bytes([0x40 + i] * 512) for i in range(4)]
    # Only the first two keys exist locally on A.
    a_local.put(keys[0], payloads[0])
    a_local.put(keys[1], payloads[1])

    result = remote_fetch(
        requester_node_id=b._node_id,
        keys=keys,
        index_writer=b._index_writer,
        donor_node_id=a._node_id,
        donor=a_donor,
        sender_id="node-b",
        epoch=int(b._pool.header.gen),
    )
    # Donor's local hit prefix is 2 → status PARTIAL.
    assert result.num_satisfied == 2
    assert result.status == PushStatus.PARTIAL

    assert b.contains(keys[0])
    assert b.contains(keys[1])
    assert not b.contains(keys[2])
    assert not b.contains(keys[3])


def test_remote_fetch_all_nack_when_donor_has_nothing(two_node_pool):
    a, b = two_node_pool
    a_local = _FakeLocalTier()  # empty
    a_donor = _build_donor(a, a_local)

    keys = [_make_key(0xBB20 + i) for i in range(2)]
    result = remote_fetch(
        requester_node_id=b._node_id,
        keys=keys,
        index_writer=b._index_writer,
        donor_node_id=a._node_id,
        donor=a_donor,
        sender_id="node-b",
        epoch=int(b._pool.header.gen),
    )
    assert result.num_satisfied == 0
    assert result.status == PushStatus.ALL_NACK
    assert not b.contains(keys[0])


def test_remote_fetch_rejects_stale_epoch(two_node_pool):
    a, b = two_node_pool
    a_local = _FakeLocalTier()
    a_donor = _build_donor(a, a_local)

    key = _make_key(0xBB30)
    a_local.put(key, b"x" * 256)
    real_epoch = int(b._pool.header.gen)

    # Pass a deliberately-stale epoch (real_epoch - 1).
    result = remote_fetch(
        requester_node_id=b._node_id,
        keys=[key],
        index_writer=b._index_writer,
        donor_node_id=a._node_id,
        donor=a_donor,
        sender_id="node-b",
        epoch=real_epoch - 1,
    )
    assert result.num_satisfied == 0
    assert result.status == PushStatus.EPOCH_STALE
    # Slot is released — no orphan.
    assert not b.contains(key)


def test_remote_fetch_skips_keys_already_in_cxl(two_node_pool):
    """If a key is ALREADY_PRESENT in CXL, the donor isn't asked.

    Prior put by node A places key in CXL directly (the warm path).
    A subsequent remote_fetch from B for the same key hits
    ALREADY_PRESENT and counts as satisfied without any DMA.
    """
    a, b = two_node_pool
    a_local = _FakeLocalTier()
    a_donor = _build_donor(a, a_local)

    key = _make_key(0xBB40)
    a.batched_submit_put_task([key], [_make_source_obj(512, 0x77)])

    # We don't even need to put it in a_local — the requester sees
    # ALREADY_PRESENT and short-circuits.
    result = remote_fetch(
        requester_node_id=b._node_id,
        keys=[key],
        index_writer=b._index_writer,
        donor_node_id=a._node_id,
        donor=a_donor,
        sender_id="node-b",
        epoch=int(b._pool.header.gen),
    )
    assert result.num_satisfied == 1
    # No DMA happened, but we still consider the prefix satisfied.
    assert result.status == PushStatus.OK


def test_remote_fetch_releases_slots_when_donor_partial(two_node_pool):
    """Slots that the donor didn't commit must be released by the requester.

    Sanity check that the slot pool is not "leaked" after a PARTIAL
    push — subsequent puts from B for those (now-EMPTY) slot positions
    succeed without INDEX_FULL.
    """
    a, b = two_node_pool
    a_local = _FakeLocalTier()
    a_donor = _build_donor(a, a_local)

    keys = [_make_key(0xBB50 + i) for i in range(3)]
    a_local.put(keys[0], b"\x10" * 256)  # only 1 of 3

    result = remote_fetch(
        requester_node_id=b._node_id,
        keys=keys,
        index_writer=b._index_writer,
        donor_node_id=a._node_id,
        donor=a_donor,
        sender_id="node-b",
        epoch=int(b._pool.header.gen),
    )
    assert result.num_satisfied == 1
    # B can now put fresh data for the un-fetched keys without
    # tripping ALREADY_PRESENT or running out of slots.
    src = _make_source_obj(256, 0x99)
    b.batched_submit_put_task(keys[1:], [_make_source_obj(256, 0x99) for _ in range(2)])
    for k in keys[1:]:
        assert b.contains(k)
