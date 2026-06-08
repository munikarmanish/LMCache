# SPDX-License-Identifier: Apache-2.0
"""RDMA/NIXL peer L2 adapter for the MP-mode StorageManager.

Design doc:
[`docs/design/v1/distributed/l2_adapters/nixl_rdma_peer.md`](docs/design/v1/distributed/l2_adapters/nixl_rdma_peer.md).

A read-only (pull-only) L2 tier. On an L1 miss the ``StorageManager``
hands the missing keys to this adapter, which looks them up on a static
list of remote peers and, for the hits, pulls the chunk bytes straight
into this node's L1 over a one-sided NIXL RDMA READ. The subsequent
``retrieve`` is then a plain L1 read — unchanged.

It is the network analogue of the CXL adapter: same "L1-first, L2 fetches
the misses from peers and prefetches them into L1" shape, but peers are
on other hosts reachable by RDMA rather than a shared CXL pool.

Operation mapping (driven by the ``PrefetchController``):
- ``submit_lookup_and_lock_task`` — fan out ``RemoteLookupReq`` to peers;
  each peer ``reserve_read``s its matching L1 chunks (a *remote*
  read-lock) and returns their page indices, which we record in a remote
  pin table. The returned bitmap marks the keys we can satisfy.
- ``submit_load_task`` — one-sided NIXL RDMA READ of each pinned chunk
  from the holding peer's L1 buffer into the caller-provided L1
  ``MemoryObj``.
- ``submit_unlock`` — the controller calls this for every loaded key once
  load completes; we send ``RemoteUnlockReq`` so the peer drops its
  read-lock. This is the hook that releases remote pins.
- ``submit_store_task`` — inert no-op (STORE never propagates to peers;
  relies on ``store_policy="lazy"``, belt-and-suspenders here).

``supports_l2_resident_retrieve`` stays ``False``: chunks land in L1 and
retrieve is unchanged, exactly as the requirements specify.

Threading model (matches CXL / Mock adapters): a single asyncio loop on a
daemon thread runs all task handlers; ``submit_*`` allocate a task id and
schedule onto the loop, ``pop``/``query`` drain result dicts under a lock,
and each completion writes 1 to the relevant eventfd.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Protocol
import asyncio
import itertools
import os
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.distributed.l2_adapters.nixl_peer_donor import (
    object_key_to_wire_key,
)
from lmcache.v1.distributed.l2_adapters.nixl_peer_messages import (
    RemoteLookupReq,
    RemoteUnlockReq,
)
from lmcache.v1.distributed.l2_adapters.nixl_peer_transport import (
    NixlPeerControlClient,
)
from lmcache.v1.memory_management import MemoryObj

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.internal_api import L1MemoryDesc
    from lmcache.v1.distributed.l1_manager import L1Manager

logger = init_logger(__name__)


def _strip_tcp_scheme(url: str) -> str:
    """Return ``url`` without a leading ``tcp://`` scheme.

    ``NixlChannel`` builds its ZMQ endpoint as ``tcp://{url}`` internally
    (via ``get_zmq_socket``), so the init bind/peer URLs it receives must
    be bare ``host:port``. We accept either form in config — a user
    writing ``tcp://host:8501`` for consistency with the ``control_*``
    URLs would otherwise produce an invalid ``tcp://tcp://...`` endpoint.

    Args:
        url: A ZMQ endpoint, with or without a ``tcp://`` scheme.

    Returns:
        ``host:port`` with any single leading ``tcp://`` removed.
    """
    prefix = "tcp://"
    return url[len(prefix) :] if url.startswith(prefix) else url


class PeerDataChannel(Protocol):
    """The data-plane surface the adapter needs from a transfer channel.

    The ``_NixlReadChannel`` wrapper over ``NixlChannel`` satisfies this:
    ``read_chunks`` does a one-sided RDMA READ from a peer's registered
    buffer into local buffers, handling the byte-offset → descriptor-index
    conversion. Declaring it as a Protocol lets tests inject an in-process
    fake without real RDMA hardware.
    """

    def lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
    ) -> object:
        """Establish the NIXL connection to a peer (metadata + descriptor
        handshake), registering it under ``peer_id`` for later reads.

        Blocks on a ZMQ round-trip to the peer's init side-channel, so
        the adapter calls it off the request path (eagerly in the
        background, or lazily on first hit) and under a timeout.

        Args:
            local_id: This node's id for the handshake.
            peer_id: The key to register the peer's transfer handler
                under (used as ``transfer_spec["sender_id"]`` later).
            peer_init_url: The peer's init side-channel as ``host:port``.
        """
        ...

    def set_on_peer_registered(self, callback: object) -> None:
        """Install a callback fired when a peer handshakes US (inbound).

        The callback is invoked with the connecting peer's id once its
        transfer handler is registered, letting the adapter mark that peer
        connected without an outbound handshake.

        Args:
            callback: A callable taking the peer's id (a ``str``).
        """
        ...

    def read_chunks(
        self,
        buffers: list[MemoryObj],
        remote_page_indices: list[int],
        peer_id: str,
    ) -> int:
        """One-sided READ of ``len(buffers)`` chunks from a remote peer.

        The implementation converts each local ``buffer.meta.address``
        (a byte offset into the registered L1 buffer) into the NIXL
        descriptor index it needs, and pairs it with the matching
        ``remote_page_indices[i]`` (already a descriptor index supplied
        by the donor) for the transfer.

        Args:
            buffers: Local destination ``MemoryObj``s.
            remote_page_indices: The peer's descriptor index per chunk,
                aligned with ``buffers``.
            peer_id: The source peer's data-channel registration key.

        Returns:
            The number of chunks transferred.
        """
        ...

    def close(self) -> None:
        """Release transfer-channel resources."""
        ...


@dataclass
class _Peer:
    """One configured remote peer's control + data handles.

    The NIXL data-plane connection is established lazily (on the first
    lookup that finds a hit on this peer), NOT at startup — so the server
    comes up immediately even if no peer is reachable yet. Until
    ``connected`` is set, the peer behaves as absent: lookups against it
    return MISS rather than blocking on a handshake.

    Attributes:
        node_id: The peer's rack node id.
        peer_id: The data-channel registration key for this peer (the
            ``sender_id`` for a READ from it); equals ``f"node-{node_id}"``.
        control: ZMQ control client for lookup/unlock RPCs (carries its
            own timeout, so a dead peer fails fast).
        init_url: The peer's NIXL handshake side-channel (bare
            ``host:port``), used the first time we connect to it.
        local_id: This node's id passed to the handshake.
        connected: Whether the NIXL handshake with this peer has
            completed. Guarded by ``connect_lock``.
        connect_lock: Serializes the lazy handshake so it runs at most
            once even under concurrent lookups.
    """

    node_id: int
    peer_id: str
    control: NixlPeerControlClient
    init_url: str
    local_id: str
    connected: bool = False
    connect_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class _RemotePin:
    """A held remote read-lock plus the info needed to READ and release it.

    Attributes:
        peer_index: Index into the adapter's ``_peers`` list of the peer
            that holds the read-lock.
        remote_index: The chunk's page index in the peer's L1 buffer
            (the ``remote_index`` for the NIXL READ).
        size: The chunk's byte size, for logging/validation.
        lease_id: The lease the pin was granted under; echoed back on
            ``RemoteUnlockReq`` so the peer releases the right pin.
    """

    peer_index: int
    remote_index: int
    size: int
    lease_id: str


class _NixlReadChannel:
    """Data-plane wrapper over a ``NixlChannel`` for one-sided READs.

    Owns the index translation the bare ``NixlChannel.batched_read`` gets
    wrong for a byte-offset L1 allocator. A NIXL prepped transfer is
    addressed by *descriptor index* (the channel registers one descriptor
    per ``page_size`` bytes of the L1 buffer), but ``MemoryObj.meta.address``
    is a byte offset. So a local destination buffer's descriptor index is
    ``meta.address // page_size``; the remote indices arrive already
    converted by the donor. This wrapper does that conversion and drives
    ``make_prepped_xfer`` directly, leaving ``NixlChannel`` as the
    registered-agent + handshake holder.
    """

    def __init__(self, channel: object, page_size: int):
        """Initialize the read channel.

        Args:
            channel: The live ``NixlChannel`` (holds the NIXL agent, the
                local transfer handler, and the per-peer remote handlers).
            page_size: The L1 NIXL page size (``align_bytes`` = one chunk).
        """
        self._channel = channel
        self._page_size = page_size

    def lazy_init_peer_connection(
        self, local_id: str, peer_id: str, peer_init_url: str
    ) -> object:
        """Delegate the NIXL handshake to the underlying channel."""
        return self._channel.lazy_init_peer_connection(
            local_id=local_id, peer_id=peer_id, peer_init_url=peer_init_url
        )

    def set_on_peer_registered(self, callback: object) -> None:
        """Install a callback the channel fires on an inbound handshake.

        Args:
            callback: Called with the peer's id (the requester's
                ``local_id``) once that peer's transfer handler is
                registered by an incoming handshake.
        """
        self._channel.on_peer_registered = callback

    def read_chunks(
        self,
        buffers: list[MemoryObj],
        remote_page_indices: list[int],
        peer_id: str,
    ) -> int:
        """One-sided READ ``buffers`` from ``peer_id`` by descriptor index.

        Converts each local buffer's byte offset to a descriptor index and
        issues a single prepped READ paired with ``remote_page_indices``.

        Args:
            buffers: Local destination ``MemoryObj``s.
            remote_page_indices: Donor-supplied descriptor index per chunk.
            peer_id: The source peer's registration key.

        Returns:
            The number of chunks transferred.

        Raises:
            ValueError: If a local buffer is not page-aligned.
            RuntimeError: If the NIXL transfer reports an error.
        """
        agent = self._channel.nixl_agent
        local_indices: list[int] = []
        for buf in buffers:
            addr = buf.meta.address
            if addr % self._page_size != 0:
                raise ValueError(
                    f"local L1 address {addr} not aligned to page_size "
                    f"{self._page_size}"
                )
            local_indices.append(addr // self._page_size)

        handle = agent.make_prepped_xfer(
            "READ",
            self._channel.nixl_wrapper.xfer_handler,
            local_indices,
            self._channel.remote_xfer_handlers_dict[peer_id],
            list(remote_page_indices),
        )
        agent.transfer(handle)
        while True:
            status = agent.check_xfer_state(handle)
            if status == "ERR":
                raise RuntimeError("NIXL one-sided READ failed")
            if status == "DONE":
                break
            time.sleep(0.001)
        return len(buffers)

    def close(self) -> None:
        """Delegate channel teardown."""
        self._channel.close()


class NixlPeerL2AdapterConfig(L2AdapterConfigBase):
    """Config for the RDMA/NIXL peer L2 adapter.

    Fields:
    - node_id: this node's id within the rack; must be distinct per rack.
    - peers: static list of ``{"node_id": int, "control_url": str,
      "init_url": str}``. ``control_url`` is the peer's
      ``NixlPeerControlServer``; ``init_url`` is the peer's NIXL handshake
      side-channel. Both accept ``host:port`` or ``tcp://host:port`` —
      the scheme is optional and normalized either way. Must not list
      this node.
    - control_bind_url: bind URL for this node's ``NixlPeerControlServer``
      (``host:port`` or ``tcp://host:port``).
    - init_bind_url: bind URL for this node's NIXL handshake side-channel
      (``host:port`` or ``tcp://host:port``).
    - nixl_backends: NIXL data-plane backend(s), e.g. ``["UCX"]``.
    - control_timeout_ms: REQ socket timeout for control RPCs.
    - lease_ms: how long a granted remote read-lock survives without an
      explicit unlock before the donor reclaims it.
    - device: device type of the L1 buffer registered with NIXL
      (``"cpu"`` for the DRAM L1 tier).

    Geometry inputs (must match across the rack — they fix the page size
    and dtype both sides register with NIXL):
    - model_name, world_size, kv_dtype_str, kv_shape, use_mla,
      cluster_chunk_size.
    - worker_id, local_world_size, local_worker_id: worker identity.
    """

    def __init__(
        self,
        node_id: int,
        peers: list[dict] | None = None,
        control_bind_url: str = "tcp://0.0.0.0:8500",
        init_bind_url: str = "0.0.0.0:8501",
        nixl_backends: list[str] | None = None,
        control_timeout_ms: int = 30000,
        lease_ms: int = 60000,
        device: str = "cpu",
        model_name: str = "",
        world_size: int = 1,
        kv_dtype_str: str = "torch.bfloat16",
        kv_shape: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0),
        use_mla: bool = False,
        cluster_chunk_size: int = 256,
        worker_id: int = 0,
        local_world_size: int = 1,
        local_worker_id: int = 0,
    ):
        if node_id < 0:
            raise ValueError("node_id must be non-negative")

        peers_list: list[dict] = []
        for i, p in enumerate(peers or []):
            if not isinstance(p, dict):
                raise ValueError(f"peers[{i}] must be a dict")
            if not isinstance(p.get("node_id"), int) or p["node_id"] < 0:
                raise ValueError(f"peers[{i}].node_id must be a non-negative int")
            if p["node_id"] == node_id:
                raise ValueError(
                    f"peers[{i}].node_id ({node_id}) must differ from this "
                    f"node's node_id"
                )
            for field_name in ("control_url", "init_url"):
                if not isinstance(p.get(field_name), str) or not p[field_name]:
                    raise ValueError(
                        f"peers[{i}].{field_name} must be a non-empty string"
                    )
            peers_list.append(
                {
                    "node_id": int(p["node_id"]),
                    "control_url": p["control_url"],
                    "init_url": p["init_url"],
                }
            )

        if not isinstance(control_bind_url, str) or not control_bind_url:
            raise ValueError("control_bind_url must be a non-empty string")
        if not isinstance(init_bind_url, str) or not init_bind_url:
            raise ValueError("init_bind_url must be a non-empty string")
        if not isinstance(control_timeout_ms, int) or control_timeout_ms <= 0:
            raise ValueError("control_timeout_ms must be a positive integer")
        if not isinstance(lease_ms, int) or lease_ms <= 0:
            raise ValueError("lease_ms must be a positive integer")

        self.node_id = node_id
        self.peers = peers_list
        self.control_bind_url = control_bind_url
        self.init_bind_url = init_bind_url
        self.nixl_backends = list(nixl_backends) if nixl_backends else ["UCX"]
        self.control_timeout_ms = control_timeout_ms
        self.lease_ms = lease_ms
        self.device = device
        self.model_name = model_name
        self.world_size = world_size
        self.kv_dtype_str = kv_dtype_str
        self.kv_shape = tuple(kv_shape)
        self.use_mla = use_mla
        self.cluster_chunk_size = cluster_chunk_size
        self.worker_id = worker_id
        self.local_world_size = local_world_size
        self.local_worker_id = local_worker_id

    @classmethod
    def from_dict(cls, d: dict) -> "NixlPeerL2AdapterConfig":
        if not isinstance(d.get("node_id"), int):
            raise ValueError("'node_id' (int) is required")

        kv_shape = d.get("kv_shape", (0, 0, 0, 0, 0))
        if not isinstance(kv_shape, (list, tuple)) or len(kv_shape) != 5:
            raise ValueError("kv_shape must be a 5-element list/tuple")

        peers = d.get("peers", [])
        if peers is not None and not isinstance(peers, list):
            raise ValueError("peers must be a list")

        nixl_backends = d.get("nixl_backends")
        if nixl_backends is not None and not isinstance(nixl_backends, list):
            raise ValueError("nixl_backends must be a list of strings")

        return cls(
            node_id=d["node_id"],
            peers=peers,
            control_bind_url=d.get("control_bind_url", "tcp://0.0.0.0:8500"),
            init_bind_url=d.get("init_bind_url", "0.0.0.0:8501"),
            nixl_backends=nixl_backends,
            control_timeout_ms=int(d.get("control_timeout_ms", 30000)),
            lease_ms=int(d.get("lease_ms", 60000)),
            device=d.get("device", "cpu"),
            model_name=d.get("model_name", ""),
            world_size=int(d.get("world_size", 1)),
            kv_dtype_str=d.get("kv_dtype_str", "torch.bfloat16"),
            kv_shape=tuple(kv_shape),
            use_mla=bool(d.get("use_mla", False)),
            cluster_chunk_size=int(d.get("cluster_chunk_size", 256)),
            worker_id=int(d.get("worker_id", 0)),
            local_world_size=int(d.get("local_world_size", 1)),
            local_worker_id=int(d.get("local_worker_id", 0)),
        )

    @classmethod
    def help(cls) -> str:
        return (
            "NIXL peer (RDMA) L2 adapter config fields:\n"
            "- node_id (int): this node's id; distinct per rack (required)\n"
            "- peers (list of {node_id, control_url, init_url}): static peer "
            "list. control_url is the peer's NixlPeerControlServer; init_url "
            "is the peer's NIXL handshake side-channel. Both accept "
            "host:port or tcp://host:port (scheme optional).\n"
            "- control_bind_url (str): this node's control REP bind URL "
            "(default tcp://0.0.0.0:8500)\n"
            "- init_bind_url (str): this node's NIXL handshake bind URL "
            "(default 0.0.0.0:8501)\n"
            "- nixl_backends (list[str]): NIXL data-plane backends "
            "(default ['UCX'])\n"
            "- control_timeout_ms (int): control RPC timeout (default 30000)\n"
            "- lease_ms (int): remote read-lock lease before donor reclaim "
            "(default 60000)\n"
            "- device (str): L1 buffer device registered with NIXL "
            "(default 'cpu')\n"
            "- model_name, world_size, kv_dtype_str, kv_shape, use_mla, "
            "cluster_chunk_size: rack geometry; must match across peers\n"
            "- worker_id, local_world_size, local_worker_id: worker identity"
        )


class NixlPeerL2Adapter(L2AdapterInterface):
    """L2 adapter that pulls KV chunks from static remote peers over RDMA.

    Peer connection is kept OUT of the lookup/load critical path:

    - At startup the adapter kicks off a background, bounded *eager*
      outbound handshake to each peer (a daemon thread). If the peer is
      not up yet the attempt times out and the peer stays unconnected;
      lookups against it return MISS, and the server runs fine alone.
    - When a peer later comes up and handshakes US (its inbound
      ``NixlMemRegRequest`` registers its transfer handler on our
      channel), the channel's ``on_peer_registered`` callback marks that
      peer connected — so we can READ from it WITHOUT ever doing our own
      outbound handshake. This is the steady state once both nodes are up.
    - A lazy on-first-hit handshake remains as a bounded fallback for the
      window where a lookup hits a peer the eager/inbound paths haven't
      connected yet.

    Runs a single asyncio loop on a daemon thread; all task handlers run
    there.
    """

    def __init__(
        self,
        peers: list[_Peer],
        data_channel: PeerDataChannel,
        node_id: int,
        *,
        control_server: object | None = None,
        connect_timeout_s: float = 10.0,
        register_peer_callback: bool = True,
        eager_connect: bool = True,
    ):
        """Initialize the adapter.

        Args:
            peers: Configured peers with control clients and peer ids.
            data_channel: The data-plane channel for one-sided RDMA READ.
            node_id: This node's rack id (the ``sender_id`` on RPCs).
            control_server: This node's control server (donor side), held
                so ``close`` can stop it. ``None`` when no donor side is
                configured (e.g. pure-client tests).
            connect_timeout_s: Upper bound on a single NIXL handshake
                attempt (eager or lazy). If it elapses the peer stays
                unconnected and lookups against it MISS until it connects.
            register_peer_callback: If ``True`` (and the channel supports
                it), install ``on_peer_registered`` on the channel so an
                inbound handshake from a peer marks it connected. Tests
                may pass ``False`` to isolate the lazy path.
            eager_connect: If ``True``, kick off a background outbound
                handshake to each peer at startup. Tests may pass
                ``False`` to isolate the lazy-on-hit path.
        """
        # Pull-only tier: no aggregate capacity, no global eviction.
        super().__init__(max_capacity_bytes=0)

        self._peers = peers
        self._channel = data_channel
        self._node_id = node_id
        self._sender_id = f"node-{node_id}"
        self._control_server = control_server
        self._connect_timeout_s = connect_timeout_s

        # Index peers by their data-channel id so the inbound-handshake
        # callback can mark the right one connected.
        self._peer_by_id: dict[str, _Peer] = {p.peer_id: p for p in peers}

        # Distinct event fds per kind (per the base class invariant).
        self._store_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._lookup_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._load_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)

        # Task bookkeeping + the remote pin table. ``_pins`` maps a key we
        # locked + can READ to where it lives; load consumes it, unlock
        # releases the remote lock and removes it.
        self._lock = threading.Lock()
        self._next_task_id: L2TaskId = 0
        self._lease_counter = itertools.count()
        self._completed_store: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookup: dict[L2TaskId, Bitmap] = {}
        self._completed_load: dict[L2TaskId, Bitmap] = {}
        self._pins: dict[ObjectKey, _RemotePin] = {}

        # Bg loop.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop, name="nixl-peer-l2-loop", daemon=True
        )
        self._loop_thread.start()

        # Learn of inbound handshakes so a peer that connects to US marks
        # itself connected here (no outbound handshake needed thereafter).
        if register_peer_callback and hasattr(self._channel, "set_on_peer_registered"):
            self._channel.set_on_peer_registered(self._on_peer_registered)

        # Eagerly (and in the background) try to connect to each peer so
        # the handshake is paid off-path, before any request arrives.
        if eager_connect:
            for peer in self._peers:
                asyncio.run_coroutine_threadsafe(
                    self._ensure_peer_connected(peer), self._loop
                )

    # ---------------- event fds ----------------

    def get_store_event_fd(self) -> int:
        return self._store_efd

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd

    def get_load_event_fd(self) -> int:
        return self._load_efd

    # ---------------- store (inert) ----------------

    def submit_store_task(
        self, keys: list[ObjectKey], objects: list[MemoryObj]
    ) -> L2TaskId:
        """Record an inert successful store.

        This tier is pull-only: STORE never propagates to peers. With
        ``store_policy="lazy"`` the controller never calls this; we still
        implement it as a 0-byte success so a non-lazy misconfiguration
        cannot push KV to peers and cannot stall the store controller.
        """
        with self._lock:
            task_id = self._alloc_task_id()
            self._completed_store[task_id] = L2StoreResult(True, 0)
        os.eventfd_write(self._store_efd, 1)
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        with self._lock:
            done = self._completed_store
            self._completed_store = {}
        return done

    # ---------------- lookup-and-lock ----------------

    def submit_lookup_and_lock_task(self, keys: list[ObjectKey]) -> L2TaskId:
        """Look up the keys on peers and remote-readlock the hits.

        Unlike CXL's purely-synchronous lookup, this issues blocking ZMQ
        round-trips to peers, so the work runs as a coroutine on the bg
        loop rather than via ``call_soon`` — a slow peer cannot stall the
        loop's other tasks.
        """
        with self._lock:
            task_id = self._alloc_task_id()
        asyncio.run_coroutine_threadsafe(
            self._do_lookup(list(keys), task_id), self._loop
        )
        return task_id

    async def _do_lookup(self, keys: list[ObjectKey], task_id: L2TaskId) -> None:
        bitmap = Bitmap(len(keys))
        # Indices still unsatisfied; consult peers in config order, first
        # peer to claim a key wins.
        pending = list(range(len(keys)))

        for peer_index, peer in enumerate(self._peers):
            if not pending:
                break
            lease_id = f"{self._sender_id}-{next(self._lease_counter)}"
            wire_keys = [object_key_to_wire_key(keys[i]) for i in pending]
            req = RemoteLookupReq(
                sender_id=self._sender_id, lease_id=lease_id, keys=wire_keys
            )
            # The control RPC carries a timeout, so a peer that never came
            # up fails fast here and contributes no hits — the server is
            # never blocked waiting on it.
            try:
                resp = await self._loop.run_in_executor(None, peer.control.lookup, req)
            except Exception:
                logger.debug("RemoteLookup to peer node_id=%d failed", peer.node_id)
                continue

            hit_positions = [
                local_pos
                for local_pos in range(len(pending))
                if local_pos < len(resp.found) and resp.found[local_pos]
            ]

            # The peer answered and read-locked some chunks. We can only
            # serve them if the NIXL data-plane connection to it is up;
            # establish it lazily now (the peer just proved it's alive).
            # If the handshake can't be completed, drop the locks the peer
            # took and treat those keys as MISS for this task.
            if hit_positions and not await self._ensure_peer_connected(peer):
                logger.warning(
                    "RemoteLookup hit on peer node_id=%d but NIXL connect "
                    "failed; treating %d key(s) as MISS",
                    peer.node_id,
                    len(hit_positions),
                )
                await self._release_remote(
                    peer_index, lease_id, [keys[pending[p]] for p in hit_positions]
                )
                continue

            still_pending: list[int] = []
            for local_pos, key_index in enumerate(pending):
                if local_pos in hit_positions:
                    with self._lock:
                        self._pins[keys[key_index]] = _RemotePin(
                            peer_index=peer_index,
                            remote_index=resp.page_indices[local_pos],
                            size=resp.sizes[local_pos],
                            lease_id=lease_id,
                        )
                    bitmap.set(key_index)
                else:
                    still_pending.append(key_index)
            pending = still_pending

        with self._lock:
            self._completed_lookup[task_id] = bitmap
        os.eventfd_write(self._lookup_efd, 1)

    async def _ensure_peer_connected(self, peer: _Peer) -> bool:
        """Lazily establish the NIXL data-plane connection to ``peer``.

        Memoized: the handshake runs at most once per peer (guarded by
        ``peer.connect_lock``). Bounded by ``connect_timeout_s`` so a peer
        that died right after answering a control lookup cannot wedge the
        loop. The blocking handshake runs in the default executor.

        Args:
            peer: The peer to connect to.

        Returns:
            ``True`` if the peer is (now) connected, ``False`` if the
            handshake failed or timed out (caller should treat the peer
            as absent for this task).
        """
        if peer.connected:
            return True

        def _connect() -> None:
            # connect_lock makes the handshake idempotent under concurrent
            # lookups; the re-check inside avoids a second handshake once
            # one thread has succeeded.
            with peer.connect_lock:
                if peer.connected:
                    return
                self._channel.lazy_init_peer_connection(
                    local_id=peer.local_id,
                    peer_id=peer.peer_id,
                    peer_init_url=peer.init_url,
                )
                peer.connected = True

        try:
            # asyncio.TimeoutError is a subclass of Exception, so the
            # broad handler below covers both the timeout and any
            # handshake error.
            await asyncio.wait_for(
                self._loop.run_in_executor(None, _connect),
                timeout=self._connect_timeout_s,
            )
        except Exception as e:
            # Our outbound attempt failed (peer not up yet, or one-way
            # reachability). But the peer's INBOUND handshake may have
            # connected us in the meantime — if so, we're still good.
            if peer.connected:
                return True
            logger.debug(
                "NIXL connect to peer node_id=%d not ready (%s)",
                peer.node_id,
                type(e).__name__,
            )
            return False
        if peer.connected:
            logger.info("NIXL peer node_id=%d connected (outbound)", peer.node_id)
        return peer.connected

    def _on_peer_registered(self, peer_local_id: str) -> None:
        """Mark a peer connected after it handshook US (inbound).

        Called from the channel's init loop when an inbound
        ``NixlMemRegRequest`` registers that peer's transfer handler —
        which means we can now READ from it without our own outbound
        handshake. ``peer_local_id`` is the requester's ``local_id``,
        which equals the ``peer_id`` we use for it (``node-<id>``).

        Args:
            peer_local_id: The connecting peer's id.
        """
        peer = self._peer_by_id.get(peer_local_id)
        if peer is None:
            # An inbound handshake from a node we don't list as a peer
            # (e.g. it only pulls from us, or its local_id doesn't match
            # the f"node-{id}" we expect). Log loudly: a mismatch here is
            # why a peer would still try (and fail) an outbound handshake.
            logger.warning(
                "inbound handshake local_id=%r does not match any configured "
                "peer id %s; will not mark connected",
                peer_local_id,
                sorted(self._peer_by_id),
            )
            return
        # Do NOT hold connect_lock across anything blocking; just flip the
        # flag. A concurrent outbound _connect() may also be running, but
        # both only ever set connected=True, so the race is benign.
        already = peer.connected
        peer.connected = True
        if not already:
            logger.info(
                "NIXL peer node_id=%d connected (inbound handshake, local_id=%r)",
                peer.node_id,
                peer_local_id,
            )

    async def _release_remote(
        self, peer_index: int, lease_id: str, keys: list[ObjectKey]
    ) -> None:
        """Release remote read-locks the peer granted under ``lease_id``.

        Used when we got a lookup hit but can't use it (e.g. the NIXL
        connect failed), so the peer's pin isn't stranded until lease
        expiry. Best-effort: a failure is non-fatal (the lease sweep on
        the donor reclaims it).

        Args:
            peer_index: Index into ``self._peers``.
            lease_id: The lease the locks were granted under.
            keys: The keys to unlock.
        """
        if not keys:
            return
        peer = self._peers[peer_index]
        req = RemoteUnlockReq(
            sender_id=self._sender_id,
            lease_id=lease_id,
            keys=[object_key_to_wire_key(k) for k in keys],
        )
        try:
            await self._loop.run_in_executor(None, peer.control.unlock, req)
        except Exception:
            logger.debug(
                "release_remote to peer node_id=%d failed (lease %s will "
                "expire on the donor)",
                peer.node_id,
                lease_id,
            )

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Optional[Bitmap]:
        with self._lock:
            return self._completed_lookup.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        """Release the remote read-locks held for these keys.

        Fire-and-forget per the interface contract; the bg coroutine
        retries transient RPC failures internally. Keys with no held pin
        (already unlocked, or never a hit on this adapter) are ignored.
        """
        self._loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(self._do_unlock(list(keys)))
        )

    async def _do_unlock(self, keys: list[ObjectKey]) -> None:
        # The donor releases pins keyed by (lease_id, key), so unlocks
        # must be grouped by (peer, lease) — two lookup tasks can hit the
        # same peer under different leases, and one batch carries exactly
        # one lease id. Drop pin-table entries as we go so a key is
        # unlocked at most once.
        by_peer_lease: dict[tuple[int, str], list[ObjectKey]] = {}
        with self._lock:
            for key in keys:
                pin = self._pins.pop(key, None)
                if pin is None:
                    continue
                by_peer_lease.setdefault((pin.peer_index, pin.lease_id), []).append(key)

        for (peer_index, lease_id), batch in by_peer_lease.items():
            peer = self._peers[peer_index]
            req = RemoteUnlockReq(
                sender_id=self._sender_id,
                lease_id=lease_id,
                keys=[object_key_to_wire_key(k) for k in batch],
            )
            try:
                await self._loop.run_in_executor(None, peer.control.unlock, req)
            except Exception:
                # The donor's lease sweep will reclaim these pins even if
                # the unlock RPC never lands, so a failure here is
                # non-fatal — log and move on.
                logger.exception(
                    "RemoteUnlock to peer node_id=%d failed (lease %s will "
                    "expire on the donor)",
                    peer.node_id,
                    lease_id,
                )

    # ---------------- load ----------------

    def submit_load_task(
        self, keys: list[ObjectKey], objects: list[MemoryObj]
    ) -> L2TaskId:
        if len(keys) != len(objects):
            raise ValueError(
                f"submit_load_task: {len(keys)} keys vs {len(objects)} objects"
            )
        with self._lock:
            task_id = self._alloc_task_id()
        asyncio.run_coroutine_threadsafe(
            self._do_load(list(keys), list(objects), task_id), self._loop
        )
        return task_id

    async def _do_load(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        bitmap = Bitmap(len(keys))

        # Group destination buffers by the peer their chunk lives on, so
        # each peer is one batched one-sided READ.
        by_peer: dict[int, list[tuple[int, MemoryObj, int]]] = {}
        with self._lock:
            for i, key in enumerate(keys):
                pin = self._pins.get(key)
                if pin is None:
                    continue
                by_peer.setdefault(pin.peer_index, []).append(
                    (i, objects[i], pin.remote_index)
                )

        for peer_index, items in by_peer.items():
            peer = self._peers[peer_index]
            buffers = [obj for (_i, obj, _ri) in items]
            remote_indexes = [ri for (_i, _obj, ri) in items]
            try:
                await self._loop.run_in_executor(
                    None,
                    lambda b=buffers, r=remote_indexes, pid=peer.peer_id: (
                        self._channel.read_chunks(b, r, pid)
                    ),
                )
            except Exception:
                logger.exception(
                    "RDMA READ from peer node_id=%d failed for %d chunk(s)",
                    peer.node_id,
                    len(items),
                )
                continue
            for i, _obj, _ri in items:
                bitmap.set(i)

        logger.info(
            "NIXL peer L2 load task=%d hits=%d/%d",
            task_id,
            bitmap.popcount(),
            len(keys),
        )
        with self._lock:
            self._completed_load[task_id] = bitmap
        os.eventfd_write(self._load_efd, 1)

    def query_load_result(self, task_id: L2TaskId) -> Optional[Bitmap]:
        with self._lock:
            return self._completed_load.pop(task_id, None)

    # ---------------- close ----------------

    def close(self) -> None:
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5)

        os.close(self._store_efd)
        os.close(self._lookup_efd)
        os.close(self._load_efd)

        # Stop accepting peer requests before tearing down the data plane.
        if self._control_server is not None:
            try:
                self._control_server.stop()
            except Exception:
                logger.exception("NixlPeerControlServer.stop failed during shutdown")
            self._control_server = None

        for peer in self._peers:
            try:
                peer.control.close()
            except Exception:
                logger.exception("NixlPeerControlClient.close failed during shutdown")

        # NixlChannel.close() joins its init-loop thread, which is parked
        # in a blocking recv() with no timeout — that join can hang
        # forever. We're shutting down, so bound it: run the close in a
        # daemon thread, wait briefly, and move on if it doesn't finish.
        # Any leaked NIXL thread dies with the process.
        closer = threading.Thread(
            target=self._safe_channel_close, name="nixl-peer-chan-close", daemon=True
        )
        closer.start()
        closer.join(timeout=5)
        if closer.is_alive():
            logger.warning(
                "data channel close did not finish within 5s; abandoning it "
                "(NixlChannel init loop is blocked in recv). Process exit will "
                "reap the leaked thread."
            )

    def _safe_channel_close(self) -> None:
        try:
            self._channel.close()
        except Exception:
            logger.exception("data channel close failed during adapter shutdown")

    # ---------------- internals ----------------

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    def _alloc_task_id(self) -> L2TaskId:
        # Caller holds self._lock.
        tid = self._next_task_id
        self._next_task_id += 1
        return tid

    # ---------------- debug ----------------

    def debug_held_pin_count(self) -> int:
        """Test-only: number of remote pins currently held."""
        with self._lock:
            return len(self._pins)


# -----------------------------------------------------------------------------
# Factory: build a live adapter from a parsed NixlPeerL2AdapterConfig.
# -----------------------------------------------------------------------------


def build_nixl_peer_adapter_from_config(
    config: NixlPeerL2AdapterConfig,
    l1_memory_desc: "L1MemoryDesc",
    *,
    l1_manager: "L1Manager",
) -> NixlPeerL2Adapter:
    """Construct a ``NixlPeerL2Adapter`` from a parsed config.

    Builds the data-plane ``NixlChannel`` over this node's registered L1
    buffer, starts the donor-side control server, performs the NIXL
    handshake with each configured peer, and creates a control client per
    peer.

    Args:
        config: Parsed adapter config.
        l1_memory_desc: Descriptor of this node's L1 buffer to register
            with NIXL for RDMA.
        l1_manager: This node's L1 manager (donor side reads/locks it).

    Returns:
        A live ``NixlPeerL2Adapter``.

    Raises:
        ValueError: If ``l1_memory_desc`` or ``l1_manager`` is missing.
    """
    if l1_memory_desc is None:
        raise ValueError("nixl_peer adapter requires an L1 memory descriptor")
    if l1_manager is None:
        raise ValueError("nixl_peer adapter requires an L1Manager")

    # Lazy imports: keep NIXL / transfer-channel out of the import path
    # for deployments that don't use this adapter.
    # First Party
    from lmcache.v1.distributed.l2_adapters.nixl_peer_donor import NixlPeerDonor
    from lmcache.v1.distributed.l2_adapters.nixl_peer_transport import (
        NixlPeerControlServer,
    )
    from lmcache.v1.transfer_channel.nixl_channel import NixlChannel

    channel = NixlChannel(
        async_mode=False,
        device=config.device,
        role="both",
        buffer_ptr=l1_memory_desc.ptr,
        buffer_size=l1_memory_desc.size,
        align_bytes=l1_memory_desc.align_bytes,
        tp_rank=config.local_worker_id,
        # NixlChannel prepends tcp:// itself, so pass a bare host:port.
        peer_init_url=_strip_tcp_scheme(config.init_bind_url),
        backends=config.nixl_backends,
    )

    # Our NIXL agent id is the data channel's agent name; peers use it as
    # the ``sender_id`` when reading from us, and we echo it from the
    # donor so peers can sanity-check.
    our_peer_id = channel.nixl_agent.name

    # A MemoryObj.meta.address is a byte offset into the registered L1
    # buffer; the one-sided READ is addressed by descriptor index (one
    # per align_bytes). The donor converts offset -> index; the read
    # channel converts the local buffers the same way.
    page_size = l1_memory_desc.align_bytes
    read_channel = _NixlReadChannel(channel, page_size=page_size)

    donor = NixlPeerDonor(
        l1_manager=l1_manager,
        peer_agent_id=our_peer_id,
        lease_seconds=config.lease_ms / 1000.0,
        page_size=page_size,
    )
    control_server = NixlPeerControlServer(
        donor=donor, bind_url=config.control_bind_url
    )
    control_server.start()

    # NOTE: the NIXL handshake with each peer is NOT performed synchronously
    # here. Doing it inline would block this server until every peer's init
    # side-channel is up (a dead/late peer would hang the whole MP server).
    # The adapter instead connects peers OFF the request path: a background
    # eager attempt at startup, an inbound-handshake callback when a peer
    # connects to us, and a bounded lazy retry on first hit. The donor-side
    # control server is already listening, so a peer that starts later can
    # immediately look up and read from us.
    peers: list[_Peer] = []
    for p in config.peers:
        control = NixlPeerControlClient(
            control_url=p["control_url"],
            recv_timeout_ms=config.control_timeout_ms,
            send_timeout_ms=config.control_timeout_ms,
        )
        peers.append(
            _Peer(
                node_id=p["node_id"],
                peer_id=f"node-{p['node_id']}",
                control=control,
                init_url=_strip_tcp_scheme(p["init_url"]),
                local_id=f"node-{config.node_id}",
            )
        )

    logger.info(
        "NIXL peer adapter node_id=%d started: control bind=%s, %d peer(s) "
        "(connected off-path: eager + inbound + lazy), backends=%s",
        config.node_id,
        config.control_bind_url,
        len(peers),
        config.nixl_backends,
    )

    return NixlPeerL2Adapter(
        peers=peers,
        data_channel=read_channel,
        node_id=config.node_id,
        control_server=control_server,
        connect_timeout_s=config.control_timeout_ms / 1000.0,
    )


def _nixl_peer_l2_adapter_factory(
    config: NixlPeerL2AdapterConfig,
    l1_memory_desc: object | None = None,
    *,
    l1_manager: object | None = None,
) -> NixlPeerL2Adapter:
    """Registry factory for the NIXL peer L2 adapter.

    Opts into both ``l1_memory_desc`` (to register L1 with NIXL for RDMA)
    and ``l1_manager`` (for the donor side to read-lock/serve local
    chunks); ``create_l2_adapter_from_registry`` forwards ``l1_manager``
    only to factories that declare it.
    """
    return build_nixl_peer_adapter_from_config(
        config,
        l1_memory_desc,  # type: ignore[arg-type]
        l1_manager=l1_manager,  # type: ignore[arg-type]
    )


register_l2_adapter_type("nixl_peer", NixlPeerL2AdapterConfig)
register_l2_adapter_factory("nixl_peer", _nixl_peer_l2_adapter_factory)
