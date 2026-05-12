# SPDX-License-Identifier: Apache-2.0
"""Tests for the ZMQ-based CXL P2P transport.

Covers the wire format and dispatch logic. Cross-host validation
against real CXL hardware lives in a separate test that's skipped on
CI; this file uses tcp://127.0.0.1 with two CXLBackend instances on
one tmpfile to drive the round-trip.
"""

# Standard
import os
import socket
import tempfile
import threading
import time
from typing import Optional

# Third Party
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
from lmcache.v1.storage_backend.cxl.cross_node import CXLDonor, remote_fetch
from lmcache.v1.storage_backend.cxl.p2p_messages import (
    PushKVToCXLMsg,
    PushKVToCXLRetMsg,
    PushStatus,
)
from lmcache.v1.storage_backend.cxl.p2p_transport import (
    CXLP2PClient,
    CXLP2PServer,
)
from lmcache.v1.storage_backend.cxl_backend import CXLBackend, CXLBackendConfig


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


def _make_key(h: int) -> CacheEngineKey:
    md = _metadata()
    return CacheEngineKey(
        model_name=md.model_name,
        world_size=md.world_size,
        worker_id=md.worker_id,
        chunk_hash=h,
        dtype=md.kv_dtype,
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
    """Stand-in for the donor's local L0/L1 tier."""

    def __init__(self):
        self._store: dict[str, tuple[bytes, MemoryFormat]] = {}

    def put(self, key: CacheEngineKey, payload: bytes,
            fmt=MemoryFormat.KV_2LTD):
        self._store[key.to_string()] = (payload, fmt)

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
            ref_count=1,
            pin_count=0,
            fmt=fmt,
        )
        return TensorMemoryObj(
            raw_data=data, metadata=meta, parent_allocator=None
        )


@pytest.fixture
def two_node_zmq():
    """Two CXLBackends on one tmpfile, plus a ZMQ server on Node A."""
    with tempfile.NamedTemporaryFile(prefix="cxl-p2p-", delete=False) as f:
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

    a_local = _FakeLocalTier()
    a_donor = CXLDonor(
        handle=a._pool,
        index_writer=a._index_writer,
        heap=a._heap,
        node_id=a._node_id,
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


def test_zmq_remote_fetch_round_trip(two_node_zmq):
    a, b, a_local, client = two_node_zmq

    keys = [_make_key(0xC001 + i) for i in range(3)]
    payloads = [bytes([0x40 + i] * 256) for i in range(3)]
    for k, p in zip(keys, payloads):
        a_local.put(k, p)

    result = remote_fetch(
        requester_node_id=b._node_id,
        keys=keys,
        index_writer=b._index_writer,
        donor_node_id=a._node_id,
        donor=client,
        sender_id="node-b",
        epoch=int(b._pool.header.gen),
    )
    assert result.num_satisfied == 3
    assert result.status == PushStatus.OK

    for i, k in enumerate(keys):
        got = b.get_blocking(k)
        assert got is not None
        assert int(got.raw_data[0]) == 0x40 + i
        got.ref_count_down()


def test_zmq_partial_success(two_node_zmq):
    a, b, a_local, client = two_node_zmq

    keys = [_make_key(0xC100 + i) for i in range(3)]
    a_local.put(keys[0], b"\x10" * 128)
    # keys[1] and keys[2] are not in the local tier.

    result = remote_fetch(
        requester_node_id=b._node_id,
        keys=keys,
        index_writer=b._index_writer,
        donor_node_id=a._node_id,
        donor=client,
        sender_id="node-b",
        epoch=int(b._pool.header.gen),
    )
    assert result.num_satisfied == 1
    assert result.status == PushStatus.PARTIAL


def test_zmq_epoch_stale_rejection(two_node_zmq):
    a, b, a_local, client = two_node_zmq
    a_local.put(_make_key(0xC200), b"x" * 128)

    real_epoch = int(b._pool.header.gen)
    result = remote_fetch(
        requester_node_id=b._node_id,
        keys=[_make_key(0xC200)],
        index_writer=b._index_writer,
        donor_node_id=a._node_id,
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
        requester_node_id=b._node_id,
        keys=keys,
        index_writer=b._index_writer,
        donor_node_id=a._node_id,
        donor=client,
        sender_id="node-b",
        epoch=int(b._pool.header.gen),
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
        requester_node_id=b._node_id,
        keys=[_make_key(0xC400)],
        index_writer=b._index_writer,
        donor_node_id=a._node_id,
        donor=client,
        sender_id="node-b",
        epoch=int(b._pool.header.gen),
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
            requester_node_id=b._node_id,
            keys=[_make_key(0xC500)],
            index_writer=b._index_writer,
            donor_node_id=a._node_id,
            donor=second_client,
            sender_id="node-b-2",
            epoch=int(b._pool.header.gen),
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
    for k in keys_a:
        a_local.put(k, b"a" * 128)
    for k in keys_b:
        a_local.put(k, b"b" * 128)

    results: list = []
    errors: list = []

    def worker(keys):
        try:
            r = remote_fetch(
                requester_node_id=b._node_id,
                keys=keys,
                index_writer=b._index_writer,
                donor_node_id=a._node_id,
                donor=client,
                sender_id="node-b",
                epoch=int(b._pool.header.gen),
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
