# SPDX-License-Identifier: Apache-2.0
"""Tests for the RDMA/NIXL peer L2 adapter (MP-mode integration).

These exercise the adapter, its control-plane protocol, the donor's
read-lock lifecycle, and the lookup -> load -> unlock orchestration
WITHOUT real RDMA hardware or CUDA:

- Config parsing + validation + factory registration.
- ``NixlPeerDonor`` read-lock lifecycle: lookup pins, unlock releases,
  lease expiry sweeps, close drains held pins. Driven against a tiny
  fake L1 that reproduces the reserve_read / unsafe_read / finish_read
  read-lock semantics.
- ``NixlPeerControlServer`` / ``NixlPeerControlClient`` round-trip over a
  real in-process ZMQ REP/REQ pair.
- ``NixlPeerL2Adapter`` end to end with a fake data channel: lookup
  remote-locks hits, load copies bytes via a fake one-sided READ, unlock
  releases the remote pins. Verifies the eventfd + bitmap surface and
  that the inert store path never touches a peer.

Real-NIXL wiring (``NixlChannel`` + handshake) is covered by the
two-node manual smoke path, not these unit tests.
"""

# Standard
from dataclasses import dataclass, field
import os
import select
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l2_adapters.nixl_peer_donor import (
    NixlPeerDonor,
    object_key_to_wire_key,
    wire_key_to_object_key,
)
from lmcache.v1.distributed.l2_adapters.nixl_peer_l2_adapter import (
    NixlPeerL2Adapter,
    NixlPeerL2AdapterConfig,
    _NixlReadChannel,
    _Peer,
    _strip_tcp_scheme,
)
from lmcache.v1.distributed.l2_adapters.nixl_peer_messages import (
    RemoteLookupReq,
    RemoteUnlockReq,
)
from lmcache.v1.distributed.l2_adapters.nixl_peer_transport import (
    NixlPeerControlClient,
    NixlPeerControlServer,
    _normalize_zmq_url,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)

MODEL = "nixl-peer-test"

# The L1 allocator stores ``meta.address`` as a byte offset; the donor and
# read channel convert it to a NIXL descriptor index by ``// page_size``.
# Use a page size > 1 so the conversion is actually exercised (a chunk at
# page_index N sits at byte offset N * PAGE_SIZE).
PAGE_SIZE = 4096


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _make_object_key(int_hash: int, kv_rank: int = 0) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(int_hash),
        model_name=MODEL,
        kv_rank=kv_rank,
    )


def _make_mem_obj(page_index: int, size_bytes: int, fill_byte: int) -> TensorMemoryObj:
    """A MemoryObj at ``page_index``, with ``meta.address`` the byte offset
    ``page_index * PAGE_SIZE`` (byte-offset-allocator semantics), so the
    donor/read-channel must divide by PAGE_SIZE to recover the index."""
    data = torch.full((size_bytes,), fill_byte, dtype=torch.uint8)
    meta = MemoryObjMetadata(
        shape=torch.Size([size_bytes]),
        dtype=torch.uint8,
        address=page_index * PAGE_SIZE,
        phy_size=size_bytes,
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.KV_2LTD,
    )
    return TensorMemoryObj(raw_data=data, metadata=meta, parent_allocator=None)


@dataclass
class _FakeL1Entry:
    obj: TensorMemoryObj
    read_locks: int = 0


class FakeL1Manager:
    """Minimal stand-in for L1Manager that reproduces the read-lock
    semantics the donor depends on, without CUDA.

    Maps ``ObjectKey`` -> a MemoryObj with a fixed page index. Tracks a
    per-key read-lock count so tests can assert that a remote lookup
    holds a lock and a remote unlock releases it.
    """

    def __init__(self) -> None:
        self._entries: dict[ObjectKey, _FakeL1Entry] = {}
        self._lock = threading.Lock()

    def put(self, key: ObjectKey, page_index: int, size_bytes: int, fill: int) -> None:
        with self._lock:
            self._entries[key] = _FakeL1Entry(
                obj=_make_mem_obj(page_index, size_bytes, fill)
            )

    def read_lock_count(self, key: ObjectKey) -> int:
        with self._lock:
            entry = self._entries.get(key)
            return entry.read_locks if entry else 0

    # ---- L1Manager surface the donor uses ----

    def reserve_read(self, keys, extra_count: int = 0):
        ret = {}
        with self._lock:
            for key in keys:
                entry = self._entries.get(key)
                if entry is None:
                    ret[key] = (L1Error.KEY_NOT_EXIST, None)
                    continue
                entry.read_locks += 1
                ret[key] = (L1Error.SUCCESS, entry.obj)
        return ret

    def unsafe_read(self, keys):
        ret = {}
        with self._lock:
            for key in keys:
                entry = self._entries.get(key)
                if entry is None or entry.read_locks <= 0:
                    ret[key] = (L1Error.KEY_NOT_EXIST, None)
                    continue
                ret[key] = (L1Error.SUCCESS, entry.obj)
        return ret

    def finish_read(self, keys, extra_count: int = 0):
        ret = {}
        with self._lock:
            for key in keys:
                entry = self._entries.get(key)
                if entry is None:
                    ret[key] = L1Error.KEY_NOT_EXIST
                    continue
                if entry.read_locks > 0:
                    entry.read_locks -= 1
                ret[key] = L1Error.SUCCESS
        return ret


@dataclass
class FakeDataChannel:
    """In-process stand-in for the NIXL data plane (``_NixlReadChannel``).

    ``read_chunks`` mimics a one-sided RDMA READ: it copies bytes from a
    per-peer "remote L1" (keyed by descriptor index) into the destination
    MemoryObjs, exactly as the real channel would land bytes into our L1.
    ``lazy_init_peer_connection`` records each handshake so tests can
    assert connections happen off the request path; set
    ``connect_fails=True`` to simulate an unreachable peer (the handshake
    raises). ``set_on_peer_registered`` captures the inbound-handshake
    callback so a test can simulate a peer connecting to us.
    """

    # peer_id -> {remote_index: bytes}
    remote_pages: dict[str, dict[int, bytes]] = field(default_factory=dict)
    closed: bool = False
    reads: list[tuple[str, list[int]]] = field(default_factory=list)
    connects: list[str] = field(default_factory=list)
    connect_fails: bool = False
    on_peer_registered: object = None

    def lazy_init_peer_connection(self, local_id, peer_id, peer_init_url):
        if self.connect_fails:
            raise RuntimeError(f"cannot reach peer {peer_id} at {peer_init_url}")
        self.connects.append(peer_id)

    def set_on_peer_registered(self, callback):
        self.on_peer_registered = callback

    def read_chunks(self, buffers, remote_page_indices, peer_id):
        self.reads.append((peer_id, list(remote_page_indices)))
        pages = self.remote_pages.get(peer_id, {})
        for buf, ri in zip(buffers, remote_page_indices, strict=True):
            src = pages[ri]
            dst = buf.raw_data
            if dst.dtype != torch.uint8:
                dst = dst.view(torch.uint8)
            n = min(len(src), dst.numel())
            dst[:n] = torch.frombuffer(bytearray(src[:n]), dtype=torch.uint8)
        return len(buffers)

    def close(self):
        self.closed = True


class _StubControlClient:
    """Control client stub for connection-only tests (no RPCs expected)."""

    def lookup(self, req):
        raise AssertionError("lookup not expected in this test")

    def unlock(self, req):
        return None

    def close(self):
        pass


def _wait_efd(efd: int, timeout_s: float = 5.0) -> None:
    poller = select.poll()
    poller.register(efd, select.POLLIN)
    events = poller.poll(int(timeout_s * 1000))
    assert events, f"eventfd not signalled within {timeout_s}s"
    os.eventfd_read(efd)


def _poll_until(fn, timeout_s: float = 5.0):
    """Poll ``fn`` until it returns a truthy value or time out."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = fn()
        if result is not None:
            return result
        time.sleep(0.005)
    raise AssertionError("condition not met within timeout")


# ---------------------------------------------------------------------------
# Config parsing / validation
# ---------------------------------------------------------------------------


def test_config_from_dict_minimal():
    cfg = NixlPeerL2AdapterConfig.from_dict({"type": "nixl_peer", "node_id": 0})
    assert cfg.node_id == 0
    assert cfg.peers == []
    assert cfg.nixl_backends == ["UCX"]


def test_config_from_dict_full():
    cfg = NixlPeerL2AdapterConfig.from_dict(
        {
            "type": "nixl_peer",
            "node_id": 0,
            "peers": [
                {
                    "node_id": 1,
                    "control_url": "tcp://hostB:8500",
                    "init_url": "tcp://hostB:8501",
                }
            ],
            "nixl_backends": ["UCX"],
            "lease_ms": 1000,
        }
    )
    assert len(cfg.peers) == 1
    assert cfg.peers[0]["node_id"] == 1
    assert cfg.lease_ms == 1000


def test_config_rejects_self_peer():
    with pytest.raises(ValueError, match="must differ"):
        NixlPeerL2AdapterConfig.from_dict(
            {
                "type": "nixl_peer",
                "node_id": 1,
                "peers": [
                    {
                        "node_id": 1,
                        "control_url": "tcp://x:1",
                        "init_url": "tcp://x:2",
                    }
                ],
            }
        )


def test_config_rejects_missing_peer_url():
    with pytest.raises(ValueError, match="control_url"):
        NixlPeerL2AdapterConfig.from_dict(
            {
                "type": "nixl_peer",
                "node_id": 0,
                "peers": [{"node_id": 1, "init_url": "tcp://x:2"}],
            }
        )


def test_config_rejects_bad_lease():
    with pytest.raises(ValueError, match="lease_ms"):
        NixlPeerL2AdapterConfig.from_dict(
            {"type": "nixl_peer", "node_id": 0, "lease_ms": 0}
        )


def test_config_registered_in_factory():
    # The registry should know "nixl_peer" once the module is imported.
    # First Party
    from lmcache.v1.distributed.l2_adapters.config import (
        get_registered_l2_adapter_types,
    )

    assert "nixl_peer" in get_registered_l2_adapter_types()


def test_help_string_mentions_key_fields():
    help_text = NixlPeerL2AdapterConfig.help()
    assert "peers" in help_text
    assert "lease_ms" in help_text


def test_strip_tcp_scheme():
    # NixlChannel prepends tcp:// itself, so init URLs must reach it bare.
    # Accept either form in config and normalize.
    assert _strip_tcp_scheme("tcp://host:8501") == "host:8501"
    assert _strip_tcp_scheme("host:8501") == "host:8501"
    assert _strip_tcp_scheme("0.0.0.0:8501") == "0.0.0.0:8501"
    # Only one leading scheme is stripped (defensive, not recursive).
    assert _strip_tcp_scheme("tcp://tcp://x:1") == "tcp://x:1"


def test_normalize_zmq_url():
    # Control sockets bind/connect directly, so they need a scheme; add
    # tcp:// when absent and leave an explicit scheme untouched.
    assert _normalize_zmq_url("host:8500") == "tcp://host:8500"
    assert _normalize_zmq_url("0.0.0.0:8500") == "tcp://0.0.0.0:8500"
    assert _normalize_zmq_url("tcp://host:8500") == "tcp://host:8500"
    # A non-TCP scheme is preserved.
    assert _normalize_zmq_url("ipc:///tmp/x") == "ipc:///tmp/x"


# ---------------------------------------------------------------------------
# Wire key round-trip
# ---------------------------------------------------------------------------


def test_wire_key_round_trips_byte_exact():
    # A 32-byte digest must survive the round-trip (no truncation).
    key = ObjectKey(
        chunk_hash=bytes(range(32)),
        model_name=MODEL,
        kv_rank=7,
        cache_salt="user-a",
    )
    restored = wire_key_to_object_key(object_key_to_wire_key(key))
    assert restored == key


# ---------------------------------------------------------------------------
# Donor read-lock lifecycle
# ---------------------------------------------------------------------------


def test_donor_lookup_pins_then_unlock_releases():
    l1 = FakeL1Manager()
    k0, k1, k2 = _make_object_key(0), _make_object_key(1), _make_object_key(2)
    l1.put(k0, page_index=5, size_bytes=64, fill=0xAA)
    l1.put(k2, page_index=9, size_bytes=64, fill=0xCC)
    # k1 deliberately absent (a true miss).

    donor = NixlPeerDonor(
        l1, peer_agent_id="node-0", lease_seconds=60, page_size=PAGE_SIZE
    )
    req = RemoteLookupReq(
        sender_id="node-1",
        lease_id="lease-1",
        keys=[object_key_to_wire_key(k) for k in (k0, k1, k2)],
    )
    resp = donor.handle_lookup(req)

    assert resp.found == [True, False, True]
    assert resp.page_indices == [5, -1, 9]
    assert resp.sizes == [64, 0, 64]
    assert resp.peer_agent_id == "node-0"
    # Hits are read-locked; the miss is not.
    assert l1.read_lock_count(k0) == 1
    assert l1.read_lock_count(k1) == 0
    assert l1.read_lock_count(k2) == 1

    unlock = RemoteUnlockReq(
        sender_id="node-1",
        lease_id="lease-1",
        keys=[object_key_to_wire_key(k) for k in (k0, k2)],
    )
    ack = donor.handle_unlock(unlock)
    assert ack.num_released == 2
    assert l1.read_lock_count(k0) == 0
    assert l1.read_lock_count(k2) == 0


def test_donor_unlock_is_idempotent():
    l1 = FakeL1Manager()
    k0 = _make_object_key(0)
    l1.put(k0, page_index=0, size_bytes=64, fill=1)
    donor = NixlPeerDonor(
        l1, peer_agent_id="node-0", lease_seconds=60, page_size=PAGE_SIZE
    )

    donor.handle_lookup(
        RemoteLookupReq("node-1", "lease-1", [object_key_to_wire_key(k0)])
    )
    first = donor.handle_unlock(
        RemoteUnlockReq("node-1", "lease-1", [object_key_to_wire_key(k0)])
    )
    second = donor.handle_unlock(
        RemoteUnlockReq("node-1", "lease-1", [object_key_to_wire_key(k0)])
    )
    assert first.num_released == 1
    assert second.num_released == 0
    assert l1.read_lock_count(k0) == 0


def test_donor_distinct_leases_on_same_key_are_independent():
    # The same chunk locked under two leases holds two read-locks; each
    # lease releases exactly its own. This is why unlocks must be grouped
    # by (peer, lease), not just by peer.
    l1 = FakeL1Manager()
    k0 = _make_object_key(0)
    l1.put(k0, page_index=0, size_bytes=64, fill=1)
    donor = NixlPeerDonor(
        l1, peer_agent_id="node-0", lease_seconds=60, page_size=PAGE_SIZE
    )

    donor.handle_lookup(
        RemoteLookupReq("node-1", "lease-A", [object_key_to_wire_key(k0)])
    )
    donor.handle_lookup(
        RemoteLookupReq("node-1", "lease-B", [object_key_to_wire_key(k0)])
    )
    assert l1.read_lock_count(k0) == 2

    # Unlocking under lease-A drops one; lease-B's pin survives.
    donor.handle_unlock(
        RemoteUnlockReq("node-1", "lease-A", [object_key_to_wire_key(k0)])
    )
    assert l1.read_lock_count(k0) == 1
    donor.handle_unlock(
        RemoteUnlockReq("node-1", "lease-B", [object_key_to_wire_key(k0)])
    )
    assert l1.read_lock_count(k0) == 0


def test_donor_lease_expiry_releases_stranded_pin():
    l1 = FakeL1Manager()
    k0 = _make_object_key(0)
    l1.put(k0, page_index=0, size_bytes=64, fill=1)
    donor = NixlPeerDonor(
        l1, peer_agent_id="node-0", lease_seconds=0.05, page_size=PAGE_SIZE
    )

    donor.handle_lookup(
        RemoteLookupReq("node-1", "lease-1", [object_key_to_wire_key(k0)])
    )
    assert l1.read_lock_count(k0) == 1

    # Before expiry: sweep is a no-op.
    assert donor.sweep_expired() == 0
    assert l1.read_lock_count(k0) == 1

    time.sleep(0.08)
    assert donor.sweep_expired() == 1
    assert l1.read_lock_count(k0) == 0


def test_donor_close_drains_held_pins():
    l1 = FakeL1Manager()
    k0, k1 = _make_object_key(0), _make_object_key(1)
    l1.put(k0, page_index=0, size_bytes=64, fill=1)
    l1.put(k1, page_index=1, size_bytes=64, fill=2)
    donor = NixlPeerDonor(
        l1, peer_agent_id="node-0", lease_seconds=60, page_size=PAGE_SIZE
    )

    donor.handle_lookup(
        RemoteLookupReq(
            "node-1",
            "lease-1",
            [object_key_to_wire_key(k0), object_key_to_wire_key(k1)],
        )
    )
    assert l1.read_lock_count(k0) == 1
    assert l1.read_lock_count(k1) == 1

    donor.close()
    assert l1.read_lock_count(k0) == 0
    assert l1.read_lock_count(k1) == 0


# ---------------------------------------------------------------------------
# Transport round-trip (real in-process ZMQ)
# ---------------------------------------------------------------------------


@pytest.fixture
def transport_pair():
    """A live donor server + a client connected to it, over loopback ZMQ."""
    l1 = FakeL1Manager()
    donor = NixlPeerDonor(
        l1, peer_agent_id="node-0", lease_seconds=60, page_size=PAGE_SIZE
    )
    # Port 0 lets the OS pick a free port; read it back from the socket.
    server = NixlPeerControlServer(donor, bind_url="tcp://127.0.0.1:0")
    server.start()
    url = server.bound_endpoint()
    client = NixlPeerControlClient(url, recv_timeout_ms=5000, send_timeout_ms=5000)
    try:
        yield l1, donor, client
    finally:
        client.close()
        server.stop()


def test_transport_lookup_and_unlock_round_trip(transport_pair):
    l1, _donor, client = transport_pair
    k0 = _make_object_key(0)
    l1.put(k0, page_index=3, size_bytes=128, fill=7)

    resp = client.lookup(
        RemoteLookupReq("node-1", "lease-1", [object_key_to_wire_key(k0)])
    )
    assert resp.found == [True]
    assert resp.page_indices == [3]
    assert l1.read_lock_count(k0) == 1

    ack = client.unlock(
        RemoteUnlockReq("node-1", "lease-1", [object_key_to_wire_key(k0)])
    )
    assert ack.num_released == 1
    assert l1.read_lock_count(k0) == 0


def test_transport_lookup_miss(transport_pair):
    _l1, _donor, client = transport_pair
    resp = client.lookup(
        RemoteLookupReq(
            "node-1", "lease-1", [object_key_to_wire_key(_make_object_key(99))]
        )
    )
    assert resp.found == [False]


# ---------------------------------------------------------------------------
# Full adapter: lookup -> load -> unlock with a fake data channel
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter_with_peer():
    """A NixlPeerL2Adapter wired to one peer's live control server + a
    fake data channel that serves that peer's 'remote L1'."""
    # Donor side (the peer): a server over a fake L1 holding two chunks.
    peer_l1 = FakeL1Manager()
    payload0 = bytes([0xAB]) * 64
    payload1 = bytes([0xCD]) * 64
    peer_l1.put(_make_object_key(10), page_index=2, size_bytes=64, fill=0xAB)
    peer_l1.put(_make_object_key(11), page_index=4, size_bytes=64, fill=0xCD)
    donor = NixlPeerDonor(
        peer_l1, peer_agent_id="node-1", lease_seconds=60, page_size=PAGE_SIZE
    )
    server = NixlPeerControlServer(donor, bind_url="tcp://127.0.0.1:0")
    server.start()
    url = server.bound_endpoint()
    control = NixlPeerControlClient(url, recv_timeout_ms=5000, send_timeout_ms=5000)

    # Fake data channel: the peer "node-1" exposes its pages by index.
    channel = FakeDataChannel(remote_pages={"node-1": {2: payload0, 4: payload1}})

    peers = [
        _Peer(
            node_id=1,
            peer_id="node-1",
            control=control,
            init_url="127.0.0.1:9999",
            local_id="node-0",
        )
    ]
    # Isolate the lazy-on-hit path: no eager connect, no inbound callback.
    # (Eager + inbound are covered by their own tests below.)
    adapter = NixlPeerL2Adapter(
        peers=peers,
        data_channel=channel,
        node_id=0,
        control_server=None,
        eager_connect=False,
        register_peer_callback=False,
    )
    try:
        yield adapter, peer_l1, channel, payload0, payload1
    finally:
        adapter.close()
        server.stop()


def _run_lookup(adapter, keys):
    task_id = adapter.submit_lookup_and_lock_task(keys)
    _wait_efd(adapter.get_lookup_and_lock_event_fd())
    return _poll_until(lambda: adapter.query_lookup_and_lock_result(task_id))


def _run_load(adapter, keys, objs):
    task_id = adapter.submit_load_task(keys, objs)
    _wait_efd(adapter.get_load_event_fd())
    return _poll_until(lambda: adapter.query_load_result(task_id))


def test_adapter_lookup_load_unlock_end_to_end(adapter_with_peer):
    adapter, peer_l1, channel, payload0, payload1 = adapter_with_peer
    k_hit0 = _make_object_key(10)
    k_miss = _make_object_key(999)
    k_hit1 = _make_object_key(11)
    keys = [k_hit0, k_miss, k_hit1]

    # No NIXL handshake until the first lookup that hits this peer.
    assert channel.connects == []

    # --- lookup: two hits, one miss; remote pins held for the hits ---
    bitmap = _run_lookup(adapter, keys)
    assert isinstance(bitmap, Bitmap)
    assert bitmap.get_indices_set() == {0, 2}
    assert adapter.debug_held_pin_count() == 2
    assert peer_l1.read_lock_count(k_hit0) == 1
    assert peer_l1.read_lock_count(k_hit1) == 1
    # The hit lazily established the peer connection (exactly once).
    assert channel.connects == ["node-1"]

    # --- load: RDMA-READ the two hits into destination L1 buffers ---
    dst0 = _make_mem_obj(page_index=0, size_bytes=64, fill_byte=0)
    dst1 = _make_mem_obj(page_index=1, size_bytes=64, fill_byte=0)
    # Load is called with only the hit keys (what the controller plans).
    load_bitmap = _run_load(adapter, [k_hit0, k_hit1], [dst0, dst1])
    assert load_bitmap.get_indices_set() == {0, 1}
    assert bytes(dst0.raw_data.tolist()) == payload0
    assert bytes(dst1.raw_data.tolist()) == payload1
    assert channel.reads == [("node-1", [2, 4])]

    # --- unlock: releases the remote read-locks ---
    adapter.submit_unlock([k_hit0, k_hit1])
    _poll_until(lambda: True if adapter.debug_held_pin_count() == 0 else None)
    _poll_until(lambda: True if peer_l1.read_lock_count(k_hit0) == 0 else None)
    assert peer_l1.read_lock_count(k_hit1) == 0


def test_adapter_two_lookups_same_peer_then_unlock(adapter_with_peer):
    # Two separate lookup tasks each hit the same peer (under distinct
    # leases). A single combined unlock must release BOTH leases' pins —
    # this exercises the (peer, lease) unlock grouping.
    adapter, peer_l1, _channel, _p0, _p1 = adapter_with_peer
    k_hit0 = _make_object_key(10)
    k_hit1 = _make_object_key(11)

    bm0 = _run_lookup(adapter, [k_hit0])
    bm1 = _run_lookup(adapter, [k_hit1])
    assert bm0.get_indices_set() == {0}
    assert bm1.get_indices_set() == {0}
    assert adapter.debug_held_pin_count() == 2
    assert peer_l1.read_lock_count(k_hit0) == 1
    assert peer_l1.read_lock_count(k_hit1) == 1

    adapter.submit_unlock([k_hit0, k_hit1])
    _poll_until(lambda: True if adapter.debug_held_pin_count() == 0 else None)
    _poll_until(lambda: True if peer_l1.read_lock_count(k_hit0) == 0 else None)
    _poll_until(lambda: True if peer_l1.read_lock_count(k_hit1) == 0 else None)


def test_adapter_connects_peer_at_most_once(adapter_with_peer):
    # Two lookups that both hit the peer should handshake only once
    # (the connection is memoized).
    adapter, _peer_l1, channel, _p0, _p1 = adapter_with_peer
    _run_lookup(adapter, [_make_object_key(10)])
    _run_lookup(adapter, [_make_object_key(11)])
    assert channel.connects == ["node-1"]


def test_adapter_unreachable_peer_misses_and_releases(adapter_with_peer):
    # The peer answers the control lookup (its control server is up) but
    # the NIXL data-plane handshake fails. The lookup must fall back to
    # MISS and release the remote read-lock the peer took, not hang.
    adapter, peer_l1, channel, _p0, _p1 = adapter_with_peer
    channel.connect_fails = True
    k_hit0 = _make_object_key(10)

    bitmap = _run_lookup(adapter, [k_hit0])
    assert bitmap.popcount() == 0
    assert adapter.debug_held_pin_count() == 0
    # The donor's read-lock was released (not stranded until lease expiry).
    _poll_until(lambda: True if peer_l1.read_lock_count(k_hit0) == 0 else None)


def test_adapter_eager_connect_at_startup():
    # With eager_connect, the adapter handshakes each peer in the
    # background at startup — no request needed. A later lookup hit then
    # finds the peer already connected (no handshake in the critical path).
    channel = FakeDataChannel(remote_pages={"node-1": {2: bytes([1]) * 64}})
    control = _StubControlClient()  # never asked anything yet
    peers = [
        _Peer(
            node_id=1,
            peer_id="node-1",
            control=control,
            init_url="127.0.0.1:9999",
            local_id="node-0",
        )
    ]
    adapter = NixlPeerL2Adapter(
        peers=peers, data_channel=channel, node_id=0, control_server=None
    )
    try:
        # Eager handshake happens off-path, shortly after construction.
        _poll_until(lambda: True if channel.connects == ["node-1"] else None)
        assert peers[0].connected
    finally:
        adapter.close()


def test_adapter_inbound_handshake_marks_connected():
    # When a peer connects to US (inbound NIXL handshake), the channel's
    # callback marks that peer connected here — so we never do our own
    # outbound handshake. Eager connect is disabled to isolate the inbound
    # path; the fake channel "fails" outbound to prove inbound alone works.
    channel = FakeDataChannel(
        remote_pages={"node-1": {2: bytes([1]) * 64}}, connect_fails=True
    )
    peers = [
        _Peer(
            node_id=1,
            peer_id="node-1",
            control=_StubControlClient(),
            init_url="127.0.0.1:9999",
            local_id="node-0",
        )
    ]
    adapter = NixlPeerL2Adapter(
        peers=peers,
        data_channel=channel,
        node_id=0,
        control_server=None,
        eager_connect=False,
    )
    try:
        assert channel.on_peer_registered is not None
        assert not peers[0].connected
        # Simulate node-1 handshaking us: the channel fires the callback
        # with the requester's local_id, which equals our peer_id "node-1".
        channel.on_peer_registered("node-1")
        assert peers[0].connected
        # An unlisted peer id is ignored, not an error.
        channel.on_peer_registered("node-9")
    finally:
        adapter.close()


def test_adapter_all_miss(adapter_with_peer):
    adapter, _peer_l1, _channel, _p0, _p1 = adapter_with_peer
    bitmap = _run_lookup(adapter, [_make_object_key(7), _make_object_key(8)])
    assert bitmap.popcount() == 0
    assert adapter.debug_held_pin_count() == 0


def test_adapter_store_is_inert(adapter_with_peer):
    adapter, _peer_l1, channel, _p0, _p1 = adapter_with_peer
    k = _make_object_key(10)
    obj = _make_mem_obj(page_index=0, size_bytes=64, fill_byte=1)
    task_id = adapter.submit_store_task([k], [obj])
    _wait_efd(adapter.get_store_event_fd())
    results = adapter.pop_completed_store_tasks()
    assert task_id in results
    assert results[task_id].is_successful()
    assert results[task_id].bytes_transferred() == 0
    # Store touched no peer.
    assert channel.reads == []


def test_adapter_no_peers_returns_all_miss():
    channel = FakeDataChannel()
    adapter = NixlPeerL2Adapter(
        peers=[], data_channel=channel, node_id=0, control_server=None
    )
    try:
        bitmap = _run_lookup(adapter, [_make_object_key(1), _make_object_key(2)])
        assert bitmap.popcount() == 0
    finally:
        adapter.close()


# ---------------------------------------------------------------------------
# Standalone operation: a peer that is down must not block lookups, and the
# background prober must pick it up when it comes online.
# ---------------------------------------------------------------------------


class _DeadControlClient:
    """Control client for a peer whose server is down.

    ``lookup`` raises (as a real timed-out RPC would), ``ping`` reports
    unreachable, and a test can flip ``up`` to simulate the peer coming
    online — after which both succeed.
    """

    def __init__(self):
        self.up = False
        self.lookups = 0
        self.pings = 0

    def lookup(self, req):
        self.lookups += 1
        if not self.up:
            raise RuntimeError("peer control server down")
        # Alive but holds nothing: empty positional response.
        # First Party
        from lmcache.v1.distributed.l2_adapters.nixl_peer_messages import (
            RemoteLookupResp,
        )

        n = len(req.keys)
        return RemoteLookupResp(
            found=[False] * n,
            page_indices=[-1] * n,
            sizes=[0] * n,
            peer_agent_id="node-1",
        )

    def ping(self, sender_id, timeout_ms):
        self.pings += 1
        return self.up

    def unlock(self, req):
        return None

    def close(self):
        pass


def _make_adapter_with_dead_peer(control, *, probe_interval_s=0.05):
    # First Party
    from lmcache.v1.distributed.l2_adapters.peer_health import PeerHealthMonitor

    peers = [
        _Peer(
            node_id=1,
            peer_id="node-1",
            control=control,
            init_url="127.0.0.1:9999",
            local_id="node-0",
        )
    ]
    monitor = PeerHealthMonitor(
        num_peers=1,
        probe_fn=lambda i: control.ping("node-0", 200),
        probe_interval_s=probe_interval_s,
        name="test-nixl-health",
    )
    channel = FakeDataChannel()
    adapter = NixlPeerL2Adapter(
        peers=peers,
        data_channel=channel,
        node_id=0,
        control_server=None,
        eager_connect=False,
        register_peer_callback=False,
        health_monitor=monitor,
    )
    return adapter, control


def test_adapter_skips_dead_peer_without_calling_lookup():
    # A peer that is down starts DEAD; the monitor's first probe fails, so
    # lookups skip it entirely (no control.lookup, no timeout stall).
    control = _DeadControlClient()  # up=False
    adapter, control = _make_adapter_with_dead_peer(control)
    try:
        # Let the prober run at least one (failing) sweep.
        time.sleep(0.2)
        bitmap = _run_lookup(adapter, [_make_object_key(10)])
        assert bitmap.popcount() == 0
        # The dead peer was skipped: lookup() was never issued to it.
        assert control.lookups == 0
        # But it WAS probed in the background.
        assert control.pings >= 1
    finally:
        adapter.close()


def test_adapter_picks_up_peer_when_it_comes_online():
    # The peer is down at first, then comes up. The background prober must
    # promote it so a subsequent lookup consults it (control.lookup runs).
    control = _DeadControlClient()  # up=False
    adapter, control = _make_adapter_with_dead_peer(control)
    try:
        time.sleep(0.2)
        # Peer comes online.
        control.up = True
        # Prober should promote it within a couple of intervals.
        _poll_until(
            lambda: True if adapter._health_monitor.is_alive(0) else None,
            timeout_s=3.0,
        )
        # Now a lookup consults the (now-alive) peer.
        _run_lookup(adapter, [_make_object_key(10)])
        assert control.lookups >= 1
    finally:
        adapter.close()


def test_adapter_demotes_peer_on_lookup_failure():
    # A peer that is "alive" per the monitor but fails a live lookup must
    # be demoted so the NEXT lookup skips it instead of paying the timeout
    # again.
    # First Party
    from lmcache.v1.distributed.l2_adapters.peer_health import PeerHealthMonitor

    control = _DeadControlClient()  # up=False -> lookup raises
    peers = [
        _Peer(
            node_id=1,
            peer_id="node-1",
            control=control,
            init_url="127.0.0.1:9999",
            local_id="node-0",
        )
    ]
    # Start the peer ALIVE and give the prober a long interval so it does
    # not interfere: we want to observe the request-path demotion.
    monitor = PeerHealthMonitor(
        num_peers=1,
        probe_fn=lambda i: control.ping("node-0", 200),
        probe_interval_s=100.0,
        start_alive=True,
    )
    channel = FakeDataChannel()
    adapter = NixlPeerL2Adapter(
        peers=peers,
        data_channel=channel,
        node_id=0,
        control_server=None,
        eager_connect=False,
        register_peer_callback=False,
        health_monitor=monitor,
    )
    try:
        # First lookup: peer is "alive", so lookup IS attempted and fails,
        # demoting the peer.
        _run_lookup(adapter, [_make_object_key(10)])
        assert control.lookups == 1
        assert not adapter._health_monitor.is_alive(0)
        # Second lookup: peer now dead -> skipped, no new lookup attempt.
        _run_lookup(adapter, [_make_object_key(11)])
        assert control.lookups == 1
    finally:
        adapter.close()


# ---------------------------------------------------------------------------
# _NixlReadChannel descriptor-index expansion (the chunk-vs-page granularity
# that previously transferred only the first page of each chunk)
# ---------------------------------------------------------------------------


class _FakeNixlAgent:
    """Captures the indices passed to make_prepped_xfer; reports DONE."""

    def __init__(self):
        self.local_indices = None
        self.remote_indices = None

    def make_prepped_xfer(
        self, op, local_handler, local_indices, remote_handler, remote_indices
    ):
        self.local_indices = list(local_indices)
        self.remote_indices = list(remote_indices)
        return "handle"

    def transfer(self, handle):
        return "DONE"

    def check_xfer_state(self, handle):
        return "DONE"


class _FakeNixlWrapper:
    xfer_handler = "local_handler"


class _FakeChannelForRead:
    """Minimal channel exposing what _NixlReadChannel.read_chunks touches."""

    def __init__(self):
        self.nixl_agent = _FakeNixlAgent()
        self.nixl_wrapper = _FakeNixlWrapper()
        self.remote_xfer_handlers_dict = {"node-1": "remote_handler"}


class _FakeBuf:
    """MemoryObj stand-in: only meta.address and get_size() are used."""

    def __init__(self, address: int, size: int):
        self.meta = type("M", (), {"address": address})()
        self._size = size

    def get_size(self) -> int:
        return self._size


def test_read_chunks_expands_chunk_into_all_descriptors():
    # A chunk that spans multiple NIXL descriptors must expand into ALL of
    # its consecutive descriptor indices (local + remote, paired) — not one
    # per chunk. This is the regression guard for the bug where only the
    # first page of each chunk transferred.
    page = 2 * 1024 * 1024  # 2 MiB descriptor
    chunk = 32 * 1024 * 1024  # 32 MiB chunk -> 16 descriptors
    pages_per_chunk = chunk // page

    fake = _FakeChannelForRead()
    rc = _NixlReadChannel(fake, page_size=page)

    # Two chunks at byte offsets 0 and 32 MiB; remote bases 0 and 16.
    buffers = [_FakeBuf(0, chunk), _FakeBuf(chunk, chunk)]
    remote_bases = [0, pages_per_chunk]  # donor returns base = addr // page

    rc.read_chunks(buffers, remote_bases, "node-1")

    agent = fake.nixl_agent
    # Each chunk -> pages_per_chunk consecutive indices.
    assert agent.local_indices == (
        list(range(0, pages_per_chunk))
        + list(range(pages_per_chunk, 2 * pages_per_chunk))
    )
    # Remote bases expand the same way and stay paired with local.
    assert agent.remote_indices == agent.local_indices
    assert len(agent.local_indices) == 2 * pages_per_chunk


def test_read_chunks_rejects_misaligned_address():
    page = 2 * 1024 * 1024
    rc = _NixlReadChannel(_FakeChannelForRead(), page_size=page)
    with pytest.raises(ValueError):
        rc.read_chunks([_FakeBuf(page + 1, page)], [0], "node-1")


def test_read_chunks_rejects_non_multiple_size():
    page = 2 * 1024 * 1024
    rc = _NixlReadChannel(_FakeChannelForRead(), page_size=page)
    with pytest.raises(ValueError):
        rc.read_chunks([_FakeBuf(0, page + 1)], [0], "node-1")
