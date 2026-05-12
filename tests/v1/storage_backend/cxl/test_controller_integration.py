# SPDX-License-Identifier: Apache-2.0
"""Tests for the cluster-controller integration adapters.

Two adapters under test:
  - ControllerLivenessProvider: query worker registry, distill alive
    cxl_node_ids per the InstanceMapping.
  - ControllerDonorRouter: BatchedP2PLookup -> DonorRoute -> cached
    CXLP2PClient.

We use a fake controller worker (`_FakeWorker`) that runs an asyncio
loop in a thread, mimicking the real LMCacheWorker shape closely
enough that the adapters' `asyncio.run_coroutine_threadsafe` path
exercises the same code as production.
"""

# Standard
import asyncio
import os
import socket
import tempfile
import threading
import time
from typing import Any, Optional

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_controller.message import (
    BatchedP2PLookupMsg,
    BatchedP2PLookupRetMsg,
    QueryWorkerInfoMsg,
    QueryWorkerInfoRetMsg,
    WorkerInfo,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.cxl.controller_integration import (
    ControllerBackedFetch,
    ControllerDonorRouter,
    ControllerLivenessProvider,
    InstanceMapping,
)
from lmcache.v1.storage_backend.cxl.cross_node import CXLDonor
from lmcache.v1.storage_backend.cxl.p2p_messages import PushStatus
from lmcache.v1.storage_backend.cxl.p2p_transport import CXLP2PServer
from lmcache.v1.storage_backend.cxl_backend import CXLBackend, CXLBackendConfig


# ---------- fakes ----------


class _FakeWorker:
    """Mimics the slice of LMCacheWorker our adapters use.

    Owns an asyncio loop in a daemon thread, exposes
    `async_put_and_wait_msg` that returns canned replies. Tests
    install handlers per message type.
    """

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self.loop.run_forever, daemon=True, name="fake-worker-loop"
        )
        self._thread.start()
        self._handlers: dict[type, Any] = {}
        self.calls: list[Any] = []  # for assertion
        self.calls_lock = threading.Lock()

    def install(self, msg_type, fn):
        """Register a sync function that returns the reply for `msg_type`."""
        self._handlers[msg_type] = fn

    async def async_put_and_wait_msg(self, msg):
        with self.calls_lock:
            self.calls.append(msg)
        for cls, fn in self._handlers.items():
            if isinstance(msg, cls):
                return fn(msg)
        raise RuntimeError(f"no handler for {type(msg).__name__}")

    def close(self):
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass
        self._thread.join(timeout=2)


@pytest.fixture
def fake_worker():
    fw = _FakeWorker()
    try:
        yield fw
    finally:
        fw.close()


# ---------- ControllerLivenessProvider ----------


def _make_worker_info(
    instance_id: str,
    worker_id: int,
    last_heartbeat_time: float,
    *,
    ip: str = "10.0.0.1",
    port: int = 1234,
) -> WorkerInfo:
    return WorkerInfo(
        instance_id=instance_id,
        worker_id=worker_id,
        ip=ip,
        port=port,
        peer_init_url=f"{ip}:{port}",
        registration_time=last_heartbeat_time - 1,
        last_heartbeat_time=last_heartbeat_time,
    )


def test_liveness_returns_alive_node_ids(fake_worker):
    """Two instances in the registry, both fresh -> both node ids alive."""
    now = time.time()
    fake_worker.install(
        QueryWorkerInfoMsg,
        lambda msg: QueryWorkerInfoRetMsg(
            event_id=msg.event_id,
            worker_infos=[
                _make_worker_info("inst-A", 0, now),
                _make_worker_info("inst-B", 0, now),
            ],
        ),
    )
    provider = ControllerLivenessProvider(
        worker=fake_worker,
        mapping=[
            InstanceMapping("inst-A", cxl_node_id=0),
            InstanceMapping("inst-B", cxl_node_id=1),
        ],
    )
    assert provider() == frozenset({0, 1})


def test_liveness_excludes_stale_instances(fake_worker):
    """Heartbeat older than stale_after_s -> instance treated as dead."""
    now = time.time()
    fake_worker.install(
        QueryWorkerInfoMsg,
        lambda msg: QueryWorkerInfoRetMsg(
            event_id=msg.event_id,
            worker_infos=[
                _make_worker_info("inst-fresh", 0, now),
                _make_worker_info("inst-stale", 0, now - 1000),
            ],
        ),
    )
    provider = ControllerLivenessProvider(
        worker=fake_worker,
        mapping=[
            InstanceMapping("inst-fresh", cxl_node_id=2),
            InstanceMapping("inst-stale", cxl_node_id=3),
        ],
        stale_after_s=120.0,
    )
    assert provider() == frozenset({2})


def test_liveness_ignores_unmapped_instances(fake_worker):
    """An instance the registry knows but the mapping doesn't -> ignored.

    We only manage CXL state for nodes we configured at boot. An
    LMCache instance that isn't in the mapping isn't using CXL, so
    its liveness is irrelevant to GC.
    """
    now = time.time()
    fake_worker.install(
        QueryWorkerInfoMsg,
        lambda msg: QueryWorkerInfoRetMsg(
            event_id=msg.event_id,
            worker_infos=[
                _make_worker_info("inst-mapped", 0, now),
                _make_worker_info("inst-unmapped", 0, now),
            ],
        ),
    )
    provider = ControllerLivenessProvider(
        worker=fake_worker,
        mapping=[InstanceMapping("inst-mapped", cxl_node_id=4)],
    )
    assert provider() == frozenset({4})


def test_liveness_one_alive_worker_keeps_instance_alive(fake_worker):
    """Multiple workers per instance: any fresh worker -> instance alive."""
    now = time.time()
    fake_worker.install(
        QueryWorkerInfoMsg,
        lambda msg: QueryWorkerInfoRetMsg(
            event_id=msg.event_id,
            worker_infos=[
                _make_worker_info("inst-A", 0, now - 1000),  # stale
                _make_worker_info("inst-A", 1, now),         # fresh
                _make_worker_info("inst-A", 2, now - 1000),  # stale
            ],
        ),
    )
    provider = ControllerLivenessProvider(
        worker=fake_worker,
        mapping=[InstanceMapping("inst-A", cxl_node_id=5)],
        stale_after_s=120.0,
    )
    assert provider() == frozenset({5})


def test_liveness_failsafe_on_rpc_error(fake_worker):
    """Controller unreachable -> return ALL known node ids as alive.

    A transient outage must NOT trigger GC reclaiming everyone's
    regions. The cost of a false-positive (no GC this tick) is
    negligible; the cost of a false-negative (GC reclaims a live
    node's chunks) is catastrophic.
    """
    def _explode(_msg):
        raise RuntimeError("controller down")

    fake_worker.install(QueryWorkerInfoMsg, _explode)
    provider = ControllerLivenessProvider(
        worker=fake_worker,
        mapping=[
            InstanceMapping("inst-A", cxl_node_id=0),
            InstanceMapping("inst-B", cxl_node_id=1),
        ],
    )
    assert provider() == frozenset({0, 1})


def test_liveness_rejects_duplicate_node_ids():
    with pytest.raises(ValueError, match="duplicate cxl_node_id"):
        ControllerLivenessProvider(
            worker=None,  # type: ignore
            mapping=[
                InstanceMapping("inst-A", cxl_node_id=0),
                InstanceMapping("inst-B", cxl_node_id=0),  # duplicate
            ],
        )


def test_liveness_known_node_ids_property(fake_worker):
    """`known_node_ids` is the GC's "what's the world look like" anchor."""
    provider = ControllerLivenessProvider(
        worker=fake_worker,
        mapping=[
            InstanceMapping("inst-A", cxl_node_id=10),
            InstanceMapping("inst-B", cxl_node_id=20),
        ],
    )
    assert provider.known_node_ids == frozenset({10, 20})


# ---------- ControllerDonorRouter ----------


def test_router_translates_p2p_lookup_to_donor_route(fake_worker):
    fake_worker.install(
        BatchedP2PLookupMsg,
        lambda msg: BatchedP2PLookupRetMsg(
            layout_info=[("inst-A", "LocalCPUBackend", 3, "10.0.0.1:5555")]
        ),
    )
    router = ControllerDonorRouter(
        worker=fake_worker,
        mapping=[InstanceMapping("inst-A", cxl_node_id=7)],
    )
    keys = [_make_key(0xC001 + i) for i in range(3)]
    route = router.lookup(
        keys, requester_instance_id="inst-B", requester_worker_id=0
    )
    assert route is not None
    assert route.donor_node_id == 7
    assert route.num_hit == 3
    # Default URL derivation: tcp://<host>:8447
    assert route.donor_url == "tcp://10.0.0.1:8447"


def test_router_returns_none_on_zero_hits(fake_worker):
    fake_worker.install(
        BatchedP2PLookupMsg,
        lambda msg: BatchedP2PLookupRetMsg(
            layout_info=[("", "", 0, "")]
        ),
    )
    router = ControllerDonorRouter(
        worker=fake_worker,
        mapping=[InstanceMapping("inst-A", cxl_node_id=0)],
    )
    route = router.lookup(
        [_make_key(0)], requester_instance_id="inst-B", requester_worker_id=0
    )
    assert route is None


def test_router_returns_none_on_unmapped_donor(fake_worker):
    """Donor instance the controller named isn't in our CXL mapping."""
    fake_worker.install(
        BatchedP2PLookupMsg,
        lambda msg: BatchedP2PLookupRetMsg(
            layout_info=[("inst-strange", "LocalCPUBackend", 1, "host:1")]
        ),
    )
    router = ControllerDonorRouter(
        worker=fake_worker,
        mapping=[InstanceMapping("inst-A", cxl_node_id=0)],
    )
    route = router.lookup(
        [_make_key(0)], requester_instance_id="inst-A", requester_worker_id=0
    )
    assert route is None


def test_router_returns_none_on_rpc_error(fake_worker):
    fake_worker.install(
        BatchedP2PLookupMsg,
        lambda msg: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    router = ControllerDonorRouter(
        worker=fake_worker,
        mapping=[InstanceMapping("inst-A", cxl_node_id=0)],
    )
    route = router.lookup(
        [_make_key(0)], requester_instance_id="inst-B", requester_worker_id=0
    )
    assert route is None


def test_router_uses_custom_url_deriver(fake_worker):
    """A deployment with non-default ports passes its own deriver."""
    fake_worker.install(
        BatchedP2PLookupMsg,
        lambda msg: BatchedP2PLookupRetMsg(
            layout_info=[("inst-A", "LocalCPUBackend", 1, "node1:9999")]
        ),
    )
    router = ControllerDonorRouter(
        worker=fake_worker,
        mapping=[InstanceMapping("inst-A", cxl_node_id=0)],
        derive_donor_url=lambda peer: f"tcp://{peer.split(':')[0]}:12345",
    )
    route = router.lookup(
        [_make_key(0)], requester_instance_id="inst-B", requester_worker_id=0
    )
    assert route is not None
    assert route.donor_url == "tcp://node1:12345"


def test_router_caches_clients_per_url(fake_worker):
    """Two get_endpoint calls for the same URL return the same client."""
    router = ControllerDonorRouter(
        worker=fake_worker,
        mapping=[InstanceMapping("inst-A", cxl_node_id=0)],
    )
    try:
        c1 = router.get_endpoint("tcp://10.0.0.1:8447")
        c2 = router.get_endpoint("tcp://10.0.0.1:8447")
        c3 = router.get_endpoint("tcp://10.0.0.2:8447")
        assert c1 is c2
        assert c1 is not c3
    finally:
        router.close()


def test_router_close_is_idempotent(fake_worker):
    router = ControllerDonorRouter(
        worker=fake_worker,
        mapping=[InstanceMapping("inst-A", cxl_node_id=0)],
    )
    router.get_endpoint("tcp://10.0.0.1:8447")
    router.close()
    router.close()  # second call: no-op, no exception


# ---------- end-to-end with real CXLBackend pair ----------

POOL_SIZE = 64 * (1 << 20)
REGION_SIZE = 2 * (1 << 20)
CHUNK_SIZE = 64 * 1024


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _metadata():
    return LMCacheMetadata(
        model_name="ctrl-int-test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(4, 2, 16, 4, 64),
        chunk_size=16,
    )


def _make_key(int_hash: int) -> CacheEngineKey:
    md = _metadata()
    return CacheEngineKey(
        model_name=md.model_name,
        world_size=md.world_size,
        worker_id=md.worker_id,
        chunk_hash=int_hash,
        dtype=md.kv_dtype,
    )


class _LocalTier:
    def __init__(self):
        self._store: dict[str, tuple[bytes, MemoryFormat]] = {}

    def put(self, key, payload, fmt=MemoryFormat.KV_2LTD):
        self._store[key.to_string()] = (payload, fmt)

    def __call__(self, key_str):
        rec = self._store.get(key_str)
        if rec is None:
            return None
        payload, fmt = rec
        size = len(payload)
        data = torch.empty(size, dtype=torch.uint8)
        data.copy_(torch.frombuffer(bytearray(payload), dtype=torch.uint8))
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
def end_to_end_setup(fake_worker):
    """Two CXLBackends, A's CXLP2PServer running, fake controller wired."""
    with tempfile.NamedTemporaryFile(prefix="cxl-ctrl-", delete=False) as f:
        f.truncate(POOL_SIZE)
        path = f.name

    cfg_a = CXLBackendConfig(
        dev_path=path, node_id=0, chunk_size_bytes=CHUNK_SIZE,
        region_size=REGION_SIZE, initialize=True, run_lock_manager=True,
    )
    cfg_b = CXLBackendConfig(
        dev_path=path, node_id=1, chunk_size_bytes=CHUNK_SIZE,
        region_size=REGION_SIZE, initialize=False, run_lock_manager=False,
    )
    backend_a = CXLBackend(cfg_a, _metadata())
    backend_b = CXLBackend(cfg_b, _metadata())

    a_local = _LocalTier()
    a_donor = CXLDonor(
        handle=backend_a._pool,
        index_writer=backend_a._index_writer,
        heap=backend_a._heap,
        node_id=backend_a._node_id,
        local_copy_provider=a_local,
    )
    port = _free_port()
    bind_url = f"tcp://127.0.0.1:{port}"
    p2p_server = CXLP2PServer(donor=a_donor, bind_url=bind_url)
    p2p_server.start()

    try:
        yield {
            "a": backend_a,
            "b": backend_b,
            "a_local": a_local,
            "donor_url": bind_url,
            "donor_port": port,
            "fake_worker": fake_worker,
        }
    finally:
        p2p_server.stop()
        backend_b.close()
        backend_a.close()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def test_end_to_end_controller_backed_fetch(end_to_end_setup):
    """Full path: controller advertises donor, router builds endpoint,
    remote_fetch publishes bytes, requester reads from CXL."""
    a = end_to_end_setup["a"]
    b = end_to_end_setup["b"]
    a_local = end_to_end_setup["a_local"]
    fake_worker = end_to_end_setup["fake_worker"]
    donor_port = end_to_end_setup["donor_port"]
    donor_url = end_to_end_setup["donor_url"]

    # Stage payloads on A's local tier.
    keys = [_make_key(0xE000 + i) for i in range(3)]
    for i, k in enumerate(keys):
        a_local.put(k, bytes([(0x40 + i) & 0xFF] * 256))

    # Fake controller routes inst-A -> donor URL we created.
    fake_worker.install(
        BatchedP2PLookupMsg,
        lambda msg: BatchedP2PLookupRetMsg(
            layout_info=[("inst-A", "LocalCPUBackend", 3, f"127.0.0.1:99")]
        ),
    )

    router = ControllerDonorRouter(
        worker=fake_worker,
        mapping=[InstanceMapping("inst-A", cxl_node_id=a._node_id)],
        derive_donor_url=lambda _peer: donor_url,
    )
    fetch = ControllerBackedFetch(
        router=router,
        index_writer=b._index_writer,
        requester_node_id=b._node_id,
        requester_instance_id="inst-B",
        requester_worker_id=0,
        epoch_provider=lambda: int(b._pool.header.gen),
        sender_id="node-b",
    )

    try:
        result = fetch.fetch(keys)
        assert result is not None
        assert result.num_satisfied == 3
        assert result.status == PushStatus.OK
        for i, k in enumerate(keys):
            got = b.get_blocking(k)
            assert got is not None
            assert int(got.raw_data[0]) == (0x40 + i) & 0xFF, (
                f"byte mismatch for key #{i}"
            )
            got.ref_count_down()
    finally:
        router.close()


def test_end_to_end_no_donor_returns_none(end_to_end_setup):
    """Controller has no donor for these keys -> fetch returns None."""
    b = end_to_end_setup["b"]
    fake_worker = end_to_end_setup["fake_worker"]

    fake_worker.install(
        BatchedP2PLookupMsg,
        lambda msg: BatchedP2PLookupRetMsg(layout_info=[("", "", 0, "")]),
    )

    router = ControllerDonorRouter(
        worker=fake_worker,
        mapping=[InstanceMapping("inst-A", cxl_node_id=0)],
        derive_donor_url=lambda peer: "tcp://nope:1",
    )
    fetch = ControllerBackedFetch(
        router=router,
        index_writer=b._index_writer,
        requester_node_id=b._node_id,
        requester_instance_id="inst-B",
        requester_worker_id=0,
        epoch_provider=lambda: int(b._pool.header.gen),
        sender_id="node-b",
    )
    try:
        assert fetch.fetch([_make_key(0xFF00)]) is None
    finally:
        router.close()
