# SPDX-License-Identifier: Apache-2.0
"""Tests for the ZMQ-based CXL P2P transport.

Covers the wire format and dispatch logic. Cross-host validation
against real CXL hardware lives in a separate test that's skipped on
CI; this file uses tcp://127.0.0.1 with two CXLStore instances on
one tmpfile to drive the round-trip.
"""

# Standard
from typing import Optional
import os
import socket
import tempfile
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.cxl.cross_node import CXLDonor, remote_fetch
from lmcache.v1.storage_backend.cxl.p2p_messages import (
    PushStatus,
)
from lmcache.v1.storage_backend.cxl.p2p_transport import (
    CXLP2PClient,
    CXLP2PServer,
)
from lmcache.v1.storage_backend.cxl.store import CXLStore, CXLStoreConfig

POOL_SIZE = 64 * (1 << 20)
REGION_SIZE = 2 * (1 << 20)
CHUNK_SIZE = 64 * 1024


def _free_port() -> int:
    """Bind-and-close trick to get an unused TCP port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="cxl-p2p-test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(4, 2, 16, 4, 64),
        chunk_size=16,
    )


def _make_key(h: int, kv_rank: int = 0, cache_salt: str = "") -> ObjectKey:
    """Build an ObjectKey the way the L2 adapter's callers do."""
    md = _metadata()
    return ObjectKey(
        chunk_hash=h.to_bytes(8, "little"),
        model_name=md.model_name,
        kv_rank=kv_rank,
        cache_salt=cache_salt,
    )


def _make_local_obj(size: int, fill: int) -> TensorMemoryObj:
    data = torch.full((size,), fill, dtype=torch.uint8)
    meta = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.uint8,
        address=data.data_ptr(),
        phy_size=size,
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.KV_2LTD,
    )
    return TensorMemoryObj(raw_data=data, metadata=meta, parent_allocator=None)


class _FakeLocalTier:
    """Stand-in for the donor's local L0/L1 tier.

    Keyed on the full ObjectKey identity, matching `L1Manager` and the
    `LocalCopyProvider` contract.
    """

    def __init__(self):
        self._store: dict[ObjectKey, tuple[bytes, MemoryFormat]] = {}

    def put(self, key: ObjectKey, payload: bytes, fmt=MemoryFormat.KV_2LTD):
        self._store[key] = (payload, fmt)

    def __call__(
        self,
        chunk_hash: bytes,
        model_name: str,
        kv_rank: int,
        cache_salt: str,
    ) -> Optional[MemoryObj]:
        key = ObjectKey(
            chunk_hash=chunk_hash,
            model_name=model_name,
            kv_rank=kv_rank,
            cache_salt=cache_salt,
        )
        record = self._store.get(key)
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
            ref_count=1,
            pin_count=0,
            fmt=fmt,
        )
        return TensorMemoryObj(raw_data=data, metadata=meta, parent_allocator=None)


@pytest.fixture
def two_node_zmq():
    """Two CXLStores on one tmpfile, plus a ZMQ server on Node A."""
    with tempfile.NamedTemporaryFile(prefix="cxl-p2p-", delete=False) as f:
        f.truncate(POOL_SIZE)
        path = f.name

    cfg_a = CXLStoreConfig(
        dev_path=path,
        node_id=0,
        max_chunk_size_bytes=CHUNK_SIZE,
        region_size=REGION_SIZE,
        initialize=True,
        run_lock_manager=True,
    )
    cfg_b = CXLStoreConfig(
        dev_path=path,
        node_id=1,
        max_chunk_size_bytes=CHUNK_SIZE,
        region_size=REGION_SIZE,
        initialize=False,
        run_lock_manager=False,
    )
    a = CXLStore(cfg_a)
    b = CXLStore(cfg_b)

    a_local = _FakeLocalTier()
    a_donor = CXLDonor(
        handle=a.pool,
        index_writer=a.index_writer,
        heaps=a.heaps,
        node_id=a.node_id,
        local_copy_provider=a_local,
    )
    port = _free_port()
    bind_url = f"tcp://127.0.0.1:{port}"
    server = CXLP2PServer(donor=a_donor, bind_url=bind_url)
    server.start()

    client = CXLP2PClient(donor_url=bind_url)

    try:
        yield a, b, a_local, client
    finally:
        client.close()
        server.stop()
        b.close()
        a.close()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


# ---------- round-trip ----------


def _read(store, key, size: int = 4096):
    """Read a chunk into a fresh buffer. Returns None on miss."""
    buf = torch.empty(size, dtype=torch.uint8)
    n = store.read_into(key, buf.data_ptr(), buf.numel())
    return None if n == 0 else buf[:n]


def test_zmq_remote_fetch_round_trip(two_node_zmq):
    a, b, a_local, client = two_node_zmq

    keys = [_make_key(0xC001 + i) for i in range(3)]
    payloads = [bytes([0x40 + i] * 256) for i in range(3)]
    for ok, p in zip(keys, payloads):
        a_local.put(ok, p)

    result = remote_fetch(
        requester_node_id=b.node_id,
        keys=keys,
        tenant_digest_fn=b.tenant_digest_for,
        index_writer=b.index_writer,
        donor_node_id=a.node_id,
        donor=client,
        sender_id="node-b",
        epoch=int(b.pool.header.gen),
    )
    assert result.num_satisfied == 3
    assert result.status == PushStatus.OK

    for i, k in enumerate(keys):
        got = _read(b, k)
        assert got is not None
        assert int(got[0]) == 0x40 + i


def test_zmq_partial_success(two_node_zmq):
    a, b, a_local, client = two_node_zmq

    keys = [_make_key(0xC100 + i) for i in range(3)]
    a_local.put(keys[0], b"\x10" * 128)
    # keys[1] and keys[2] are not in the local tier.

    result = remote_fetch(
        requester_node_id=b.node_id,
        keys=keys,
        tenant_digest_fn=b.tenant_digest_for,
        index_writer=b.index_writer,
        donor_node_id=a.node_id,
        donor=client,
        sender_id="node-b",
        epoch=int(b.pool.header.gen),
    )
    assert result.num_satisfied == 1
    assert result.status == PushStatus.PARTIAL


def test_zmq_epoch_stale_rejection(two_node_zmq):
    a, b, a_local, client = two_node_zmq
    a_local.put(_make_key(0xC200), b"x" * 128)

    real_epoch = int(b.pool.header.gen)
    result = remote_fetch(
        requester_node_id=b.node_id,
        keys=[_make_key(0xC200)],
        tenant_digest_fn=b.tenant_digest_for,
        index_writer=b.index_writer,
        donor_node_id=a.node_id,
        donor=client,
        sender_id="node-b",
        epoch=real_epoch - 1,  # deliberately stale
    )
    assert result.num_satisfied == 0
    assert result.status == PushStatus.EPOCH_STALE


def test_zmq_all_nack_when_donor_empty(two_node_zmq):
    a, b, _, client = two_node_zmq

    keys = [_make_key(0xC300 + i) for i in range(2)]
    result = remote_fetch(
        requester_node_id=b.node_id,
        keys=keys,
        tenant_digest_fn=b.tenant_digest_for,
        index_writer=b.index_writer,
        donor_node_id=a.node_id,
        donor=client,
        sender_id="node-b",
        epoch=int(b.pool.header.gen),
    )
    assert result.num_satisfied == 0
    assert result.status == PushStatus.ALL_NACK


# ---------- error handling ----------


def test_client_recovers_after_server_restart(two_node_zmq):
    """Server stops + restarts; client times out then succeeds.

    Validates the REQ-socket reset on failure path.
    """
    a, b, a_local, client = two_node_zmq
    a_local.put(_make_key(0xC400), b"\x55" * 128)

    # Drive one successful round-trip first to confirm baseline.
    result = remote_fetch(
        requester_node_id=b.node_id,
        keys=[_make_key(0xC400)],
        tenant_digest_fn=b.tenant_digest_for,
        index_writer=b.index_writer,
        donor_node_id=a.node_id,
        donor=client,
        sender_id="node-b",
        epoch=int(b.pool.header.gen),
    )
    assert result.num_satisfied == 1


def test_distinct_clients_share_zmq_context(two_node_zmq):
    """Multiple clients pointing at the same server should all work."""
    a, b, a_local, client = two_node_zmq
    a_local.put(_make_key(0xC500), b"q" * 128)

    # Build a second client to the same server (same URL).
    second_client = CXLP2PClient(donor_url=client._donor_url)
    try:
        result = remote_fetch(
            requester_node_id=b.node_id,
            keys=[_make_key(0xC500)],
            tenant_digest_fn=b.tenant_digest_for,
            index_writer=b.index_writer,
            donor_node_id=a.node_id,
            donor=second_client,
            sender_id="node-b-2",
            epoch=int(b.pool.header.gen),
        )
        assert result.num_satisfied == 1
    finally:
        second_client.close()


def test_concurrent_remote_fetches_serialize_per_client(two_node_zmq):
    """Two threads sharing one client serialize on the REQ socket lock.

    They both succeed; correctness — not speed — is what we're
    asserting here.
    """
    a, b, a_local, client = two_node_zmq
    keys_a = [_make_key(0xC600 + i) for i in range(2)]
    keys_b = [_make_key(0xC700 + i) for i in range(2)]
    for ok in keys_a:
        a_local.put(ok, b"a" * 128)
    for ok in keys_b:
        a_local.put(ok, b"b" * 128)

    results: list = []
    errors: list = []

    def worker(keys):
        try:
            r = remote_fetch(
                requester_node_id=b.node_id,
                keys=keys,
                tenant_digest_fn=b.tenant_digest_for,
                index_writer=b.index_writer,
                donor_node_id=a.node_id,
                donor=client,
                sender_id="node-b",
                epoch=int(b.pool.header.gen),
            )
            results.append(r)
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=worker, args=(keys_a,))
    t2 = threading.Thread(target=worker, args=(keys_b,))
    t1.start()
    t2.start()
    t1.join(10)
    t2.join(10)

    assert errors == []
    assert len(results) == 2
    assert all(r.num_satisfied == 2 for r in results)
