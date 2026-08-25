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

Two retrieve paths:

- **DRAM (default).** Chunks are pulled into L1 during the prefetch LOAD
  phase and retrieve is a plain L1 read.
- **GPUDirect (opt-in, ``enable_gpu_direct``).** The chunk is pulled
  straight into the GPU staging buffer at retrieve time, skipping the L1
  landing and the H2D bounce. ``supports_l2_resident_retrieve`` reports
  ``True`` only once ``register_gpu_staging_buffer`` has run, so it is
  **dynamic** rather than the constant the CXL adapter returns. See
  [`nixl_peer_gpudirect.md`](docs/design/v1/distributed/l2_adapters/nixl_peer_gpudirect.md).

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
from lmcache.v1.distributed.l2_adapters.peer_health import PeerHealthMonitor
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


class GpuPeerDataChannel(Protocol):
    """The data-plane surface needed for a GPUDirect READ into GPU memory.

    Satisfied by :class:`_NixlGpuReadChannel`. Kept separate from
    :class:`PeerDataChannel` because the destination is addressed by a raw
    device pointer into the registered GPU staging buffer rather than by a
    local ``MemoryObj``. Declaring it as a Protocol lets tests inject an
    in-process fake without RDMA hardware or a GPU.
    """

    def lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
    ) -> object:
        """Establish the NIXL connection to a peer's GPU-side agent.

        Args:
            local_id: This node's id for the handshake.
            peer_id: The key to register the peer's transfer handler under.
            peer_init_url: The peer's GPU init side-channel as ``host:port``.
        """
        ...

    def read_chunks_to_gpu(
        self,
        gpu_ptrs: list[int],
        sizes: list[int],
        remote_page_indices: list[int],
        peer_id: str,
    ) -> int:
        """One-sided READ of chunks from a peer straight into GPU memory.

        MUST NOT return until the transfer has completed — an RDMA READ is
        not ordered against the caller's CUDA stream, so the caller can only
        establish that ordering by this call being synchronous. See
        :meth:`NixlPeerL2Adapter.submit_h2d_batch`.

        Args:
            gpu_ptrs: Destination device pointer per chunk.
            sizes: Byte size per chunk.
            remote_page_indices: The peer's base descriptor index per chunk.
            peer_id: The source peer's data-channel registration key.

        Returns:
            The number of chunks transferred.

        Raises:
            RuntimeError: If the transfer reports an error.
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
        gpu_init_url: The peer's GPU-side NIXL handshake side-channel
            (bare ``host:port``). Empty when the peer advertises no GPU
            channel, which disables GPUDirect reads from it.
        gpu_connected: Whether the GPU-side NIXL handshake has completed.
            Tracked separately from ``connected`` because the two channels
            are distinct NIXL agents with independent handshakes: a peer can
            be usable for the DRAM path and not (yet) for the GPU path.
        gpu_connect_lock: Serializes the GPU handshake, as
            ``connect_lock`` does for the DRAM one.
    """

    node_id: int
    peer_id: str
    control: NixlPeerControlClient
    init_url: str
    local_id: str
    connected: bool = False
    connect_lock: threading.Lock = field(default_factory=threading.Lock)
    gpu_init_url: str = ""
    gpu_connected: bool = False
    gpu_connect_lock: threading.Lock = field(default_factory=threading.Lock)


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
            ValueError: If a local buffer is not page-aligned or its size
                is not a whole number of NIXL pages.
            RuntimeError: If the NIXL transfer reports an error.
        """
        agent = self._channel.nixl_agent
        # The NIXL xfer dlist has one descriptor per ``page_size`` bytes of
        # the registered L1 buffer (page_size = l1 align_bytes, e.g. 4 KiB).
        # A KV chunk is much larger (e.g. 32 MiB = pages_per_chunk pages), so
        # each chunk spans MANY consecutive descriptors. We must expand each
        # chunk's base index into all ``pages_per_chunk`` descriptor indices
        # — both local and remote, paired in order — or NIXL transfers only
        # the first page of each chunk (the bug that made a 3.3 GB READ
        # "complete" in ~1.6 ms with corrupt tails).
        local_indices: list[int] = []
        remote_indices: list[int] = []
        for buf, remote_base in zip(buffers, remote_page_indices, strict=True):
            addr = buf.meta.address
            if addr % self._page_size != 0:
                raise ValueError(
                    f"local L1 address {addr} not aligned to page_size "
                    f"{self._page_size}"
                )
            size = buf.get_size()
            if size % self._page_size != 0:
                raise ValueError(
                    f"chunk size {size} not a multiple of page_size {self._page_size}"
                )
            pages_per_chunk = size // self._page_size
            local_base = addr // self._page_size
            for p in range(pages_per_chunk):
                local_indices.append(local_base + p)
                remote_indices.append(remote_base + p)

        handle = agent.make_prepped_xfer(
            "READ",
            self._channel.nixl_wrapper.xfer_handler,
            local_indices,
            self._channel.remote_xfer_handlers_dict[peer_id],
            remote_indices,
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


class _NixlGpuReadChannel:
    """Data-plane wrapper for one-sided RDMA READs into GPU memory.

    The GPUDirect analogue of :class:`_NixlReadChannel`. Two differences,
    both forced by the destination being the GPU staging buffer rather than
    an L1 ``MemoryObj``:

    1. **Addressing.** The destination is a raw device pointer into the
       registered staging buffer, so the local base descriptor index is
       ``(gpu_ptr - buffer_base) // page_size`` instead of
       ``meta.address // page_size``.
    2. **Synchronous completion.** ``read_chunks_to_gpu`` polls the transfer
       to DONE before returning. See :meth:`NixlPeerL2Adapter.submit_h2d_batch`
       for why this is required rather than merely convenient.

    ``page_size`` MUST equal the DRAM channel's page size. NIXL rejects a
    transfer whose paired descriptors differ in length
    (``makeXferReq: length mismatch at index pair N``), and the remote
    descriptors are fixed at that size by the cross-node wire contract.
    """

    def __init__(self, channel: object, page_size: int, buffer_base: int):
        """Initialize the GPU read channel.

        Args:
            channel: The live ``NixlChannel`` registered over the GPU
                staging buffer (holds the NIXL agent and remote handlers).
            page_size: NIXL descriptor size; must match the donor's.
            buffer_base: Device pointer of the staging buffer's base, used
                to convert a destination pointer into a descriptor index.
        """
        self._channel = channel
        self._page_size = page_size
        self._buffer_base = buffer_base

    def lazy_init_peer_connection(
        self, local_id: str, peer_id: str, peer_init_url: str
    ) -> object:
        """Delegate the GPU-side NIXL handshake to the underlying channel."""
        return self._channel.lazy_init_peer_connection(
            local_id=local_id, peer_id=peer_id, peer_init_url=peer_init_url
        )

    def read_chunks_to_gpu(
        self,
        gpu_ptrs: list[int],
        sizes: list[int],
        remote_page_indices: list[int],
        peer_id: str,
    ) -> int:
        """One-sided READ from ``peer_id`` straight into GPU memory.

        Expands each chunk's base index into all ``size // page_size``
        consecutive descriptor indices — local and remote paired in order —
        exactly as the DRAM path must (a chunk spans many descriptors; pairing
        only the first transfers only the first page of each chunk).

        Blocks until the transfer reports DONE.

        Args:
            gpu_ptrs: Destination device pointer per chunk.
            sizes: Byte size per chunk.
            remote_page_indices: Donor-supplied base descriptor index per chunk.
            peer_id: The source peer's registration key.

        Returns:
            The number of chunks transferred.

        Raises:
            ValueError: If a destination pointer lies outside the registered
                staging buffer, is not page-aligned, or a size is not a whole
                number of pages.
            RuntimeError: If the NIXL transfer reports an error.
        """
        agent = self._channel.nixl_agent
        local_indices: list[int] = []
        remote_indices: list[int] = []
        for gpu_ptr, size, remote_base in zip(
            gpu_ptrs, sizes, remote_page_indices, strict=True
        ):
            offset = gpu_ptr - self._buffer_base
            if offset < 0:
                raise ValueError(
                    f"destination pointer {gpu_ptr:#x} is below the registered "
                    f"GPU staging buffer base {self._buffer_base:#x}"
                )
            if offset % self._page_size != 0:
                raise ValueError(
                    f"GPU staging offset {offset} not aligned to page_size "
                    f"{self._page_size}"
                )
            if size % self._page_size != 0:
                raise ValueError(
                    f"chunk size {size} not a multiple of page_size {self._page_size}"
                )
            pages_per_chunk = size // self._page_size
            local_base = offset // self._page_size
            for p in range(pages_per_chunk):
                local_indices.append(local_base + p)
                remote_indices.append(remote_base + p)

        handle = agent.make_prepped_xfer(
            "READ",
            self._channel.nixl_wrapper.xfer_handler,
            local_indices,
            self._channel.remote_xfer_handlers_dict[peer_id],
            remote_indices,
        )
        agent.transfer(handle)
        # Poll to completion: the caller is about to enqueue a CUDA kernel
        # that reads these bytes, and an RDMA READ carries no ordering
        # against that stream.
        while True:
            status = agent.check_xfer_state(handle)
            if status == "ERR":
                raise RuntimeError("NIXL one-sided GPUDirect READ failed")
            if status == "DONE":
                break
            time.sleep(0.0001)
        return len(gpu_ptrs)

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
        # Peer liveness / background reconnection (see PeerHealthMonitor).
        # A configured peer that is down is marked dead and SKIPPED on the
        # lookup path so a lookup never pays its control_timeout_ms; a
        # background thread pings dead peers every peer_probe_interval_ms
        # with a short peer_probe_timeout_ms and promotes any that answer,
        # so a node launched alone works standalone and picks up peers that
        # come online later.
        peer_probe_interval_ms: int = 5000,
        peer_probe_timeout_ms: int = 1000,
        # GPUDirect: pull a remote hit straight into the GPU staging buffer,
        # skipping the L1/DRAM landing and the H2D bounce. Opt-in (default
        # off) because it needs GPUDirect-capable RDMA hardware and moves
        # RDMA latency from the prefetch phase into the TTFT-critical
        # retrieve phase — a trade that must be measured per deployment.
        enable_gpu_direct: bool = False,
        gpu_init_bind_url: str = "0.0.0.0:8502",
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
            # gpu_init_url is optional: a peer that advertises none simply
            # cannot be read GPU-direct (we fall back to the DRAM path for
            # its chunks), so a mixed-capability rack still works.
            gpu_init_url = p.get("gpu_init_url", "")
            if not isinstance(gpu_init_url, str):
                raise ValueError(f"peers[{i}].gpu_init_url must be a string")
            peers_list.append(
                {
                    "node_id": int(p["node_id"]),
                    "control_url": p["control_url"],
                    "init_url": p["init_url"],
                    "gpu_init_url": gpu_init_url,
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
        if not isinstance(peer_probe_interval_ms, int) or peer_probe_interval_ms <= 0:
            raise ValueError("peer_probe_interval_ms must be a positive integer")
        if not isinstance(peer_probe_timeout_ms, int) or peer_probe_timeout_ms <= 0:
            raise ValueError("peer_probe_timeout_ms must be a positive integer")
        if not isinstance(enable_gpu_direct, bool):
            raise ValueError("enable_gpu_direct must be a bool")
        if enable_gpu_direct and (
            not isinstance(gpu_init_bind_url, str) or not gpu_init_bind_url
        ):
            raise ValueError(
                "gpu_init_bind_url must be a non-empty string when "
                "enable_gpu_direct is set"
            )

        self.enable_gpu_direct = enable_gpu_direct
        self.gpu_init_bind_url = gpu_init_bind_url
        self.node_id = node_id
        self.peers = peers_list
        self.control_bind_url = control_bind_url
        self.init_bind_url = init_bind_url
        self.nixl_backends = list(nixl_backends) if nixl_backends else ["UCX"]
        self.control_timeout_ms = control_timeout_ms
        self.lease_ms = lease_ms
        self.peer_probe_interval_ms = peer_probe_interval_ms
        self.peer_probe_timeout_ms = peer_probe_timeout_ms
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
            peer_probe_interval_ms=int(d.get("peer_probe_interval_ms", 5000)),
            peer_probe_timeout_ms=int(d.get("peer_probe_timeout_ms", 1000)),
            enable_gpu_direct=bool(d.get("enable_gpu_direct", False)),
            gpu_init_bind_url=d.get("gpu_init_bind_url", "0.0.0.0:8502"),
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
            "- peer_probe_interval_ms (int): background ping interval for dead "
            "peers (default 5000). Dead peers are skipped on the lookup path "
            "so a lone node never blocks on them.\n"
            "- peer_probe_timeout_ms (int): timeout for a single liveness ping "
            "(default 1000; far shorter than control_timeout_ms)\n"
            "- enable_gpu_direct (bool): pull remote hits straight into the "
            "GPU staging buffer, skipping the L1/DRAM landing and the H2D "
            "bounce (default False). Requires GPUDirect-capable RDMA and a "
            "per-chunk staging stride that is a multiple of the 2 MiB NIXL "
            "descriptor size.\n"
            "- gpu_init_bind_url (str): this node's GPU-side NIXL handshake "
            "bind URL, used only when enable_gpu_direct is set "
            "(default 0.0.0.0:8502)\n"
            "- peers[].gpu_init_url (str): a peer's GPU-side handshake "
            "side-channel; optional — a peer without one is read via the "
            "DRAM path\n"
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
        health_monitor: PeerHealthMonitor | None = None,
        gpu_channel_factory: object | None = None,
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
            health_monitor: Optional peer-liveness monitor. When present,
                ``_do_lookup`` skips peers it marks dead (so a lookup never
                pays a dead peer's ``control_timeout_ms``) and its
                background thread pings dead peers to pick them up when
                they come online. ``None`` in tests that drive liveness
                directly.
            gpu_channel_factory: Enables GPUDirect retrieve. A callable
                ``(gpu_ptr, size) -> GpuPeerDataChannel`` invoked by
                :meth:`register_gpu_staging_buffer` once the GPU context
                exists. ``None`` (the default) leaves the adapter on the
                DRAM path, with ``supports_l2_resident_retrieve()`` always
                ``False``. Deferred rather than taking a channel directly
                because the staging buffer is created by the GPU context,
                which does not exist when the adapter is built.
        """
        # Pull-only tier: no aggregate capacity, no global eviction.
        super().__init__(max_capacity_bytes=0)

        self._peers = peers
        self._channel = data_channel
        self._node_id = node_id
        self._sender_id = f"node-{node_id}"
        self._control_server = control_server
        self._connect_timeout_s = connect_timeout_s
        self._health_monitor = health_monitor

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

        # GPUDirect retrieve state. ``_gpu_channel`` stays None until
        # ``register_gpu_staging_buffer`` runs, which is what flips
        # ``supports_l2_resident_retrieve()`` to True.
        self._gpu_channel_factory = gpu_channel_factory
        self._gpu_channel: GpuPeerDataChannel | None = None
        # h2d token -> the key whose remote pin it holds, so
        # ``release_after_h2d`` can unlock exactly that key.
        self._h2d_token_to_key: dict[int, ObjectKey] = {}
        self._next_h2d_token: int = 0

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

        # Start the background peer prober (if configured). It pings dead
        # peers so a peer that is down at startup — or dies mid-run — is
        # skipped on the lookup path and picked back up when it recovers.
        if self._health_monitor is not None:
            self._health_monitor.start()

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
            # Skip peers the health monitor believes are dead: a live
            # lookup RPC to a down peer would block for the full
            # control_timeout_ms. A node running alone (all peers dead)
            # therefore returns an all-miss bitmap immediately, and the
            # background prober flips a peer back to alive once it answers
            # a ping.
            if self._health_monitor is not None and not self._health_monitor.is_alive(
                peer_index
            ):
                continue
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
                # A live peer just failed to answer: demote it so the next
                # lookup skips it instead of paying the timeout again.
                if self._health_monitor is not None:
                    self._health_monitor.record_failure(peer_index)
                logger.debug("RemoteLookup to peer node_id=%d failed", peer.node_id)
                continue

            # The peer answered — it is reachable.
            if self._health_monitor is not None:
                self._health_monitor.record_success(peer_index)

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

    # ---------------- l2-resident retrieve (GPUDirect) ----------------

    def register_gpu_staging_buffer(self, gpu_ptr: int, size: int) -> None:
        """Register the GPU staging buffer, enabling GPUDirect retrieve.

        Called once by the retrieve module after the GPU context has
        allocated its staging buffer (which does not exist when the adapter
        is constructed). Builds the GPU-side data channel via the configured
        factory and starts background handshakes to each peer that advertises
        a GPU init URL.

        After this returns successfully, ``supports_l2_resident_retrieve()``
        reports ``True`` and the ``PrefetchController`` begins leaving hits
        L2-resident for GPU-direct retrieve.

        No-op when the adapter was built without a ``gpu_channel_factory``
        (GPUDirect disabled), so the caller can invoke it unconditionally.

        Args:
            gpu_ptr: Device pointer of the staging buffer's base.
            size: Byte size of the staging buffer.

        Raises:
            ValueError: If ``gpu_ptr``/``size`` are unusable for NIXL
                registration (non-positive, or not tiling evenly at the
                NIXL descriptor size).
            RuntimeError: If called twice.
        """
        if self._gpu_channel_factory is None:
            return
        if self._gpu_channel is not None:
            raise RuntimeError("GPU staging buffer is already registered")
        if gpu_ptr <= 0:
            raise ValueError(f"gpu_ptr must be a positive pointer, got {gpu_ptr}")
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")

        channel = self._gpu_channel_factory(gpu_ptr, size)
        self._gpu_channel = channel
        logger.info(
            "NIXL peer adapter: GPU staging buffer registered "
            "(ptr=%#x, size=%d); GPUDirect retrieve enabled",
            gpu_ptr,
            size,
        )
        # Handshake off the request path, as the DRAM channel does, so the
        # first GPUDirect retrieve does not pay for it. A plain daemon
        # thread rather than the bg asyncio loop: the handshake is a
        # blocking ZMQ round-trip with no async component, and a pending
        # loop task would be stranded if the adapter closed meanwhile.
        gpu_peers = [p for p in self._peers if p.gpu_init_url]
        if gpu_peers:
            threading.Thread(
                target=lambda: [self._connect_peer_gpu_sync(p) for p in gpu_peers],
                name="nixl-peer-gpu-connect",
                daemon=True,
            ).start()

    def supports_l2_resident_retrieve(self) -> bool:
        """Whether hits can currently be served GPU-direct from a peer.

        Unlike the CXL adapter's constant ``True``, this is **dynamic**: it
        is ``False`` until :meth:`register_gpu_staging_buffer` has run, and
        ``False`` for the whole adapter when GPUDirect is disabled. The
        ``PrefetchController`` re-reads it per request, so flipping it at
        runtime is safe: requests before the flip take the DRAM path,
        requests after take the GPU path, and none observe a mix.

        Returns:
            ``True`` iff GPUDirect retrieve is enabled and registered.
        """
        return self._gpu_channel is not None

    def _connect_peer_gpu_sync(self, peer: _Peer) -> bool:
        """Establish the GPU-side NIXL connection to ``peer``, blocking.

        The GPU analogue of :meth:`_ensure_peer_connected`, tracked by its
        own flag and lock because the GPU channel is a distinct NIXL agent
        with an independent handshake. Memoized: the handshake runs at most
        once per peer even under concurrent callers.

        **GPUDirect is asymmetric, and this handshake reflects that.** Only
        the *local* side of the transfer is GPU memory: we READ from the
        peer's DRAM L1 into our GPU staging buffer. So this connects our GPU
        agent to the peer's **L1 (DRAM) init side-channel** — ``init_url``,
        the same endpoint the DRAM channel uses — NOT to the peer's own GPU
        agent.

        Connecting the two GPU agents instead would register the peer's
        *staging buffer* as the remote dlist (64 descriptors for a 128 MiB
        buffer), while the donor reports page indices into its 32 GiB L1
        (up to ~16k). Every index past the staging buffer is then rejected:
        ``makeXferReq: remote index out of range at index 0 with value 1872``
        -> ``NIXL_ERR_INVALID_PARAM``. The peer's ``gpu_init_url`` is
        therefore NOT a handshake target; it exists only so a peer can
        advertise that it, too, has GPUDirect enabled.

        Runs on the calling thread. Safe from either the bg loop's executor
        or the retrieve thread, and it never schedules onto the loop, so it
        cannot be stranded by a concurrent ``close``.

        Args:
            peer: The peer to connect to.

        Returns:
            ``True`` if the peer's GPU channel is (now) connected, ``False``
            if it advertises no GPU URL, GPUDirect is off, or the handshake
            failed (the caller should treat the peer as unavailable).
        """
        if peer.gpu_connected:
            return True
        channel = self._gpu_channel
        if channel is None or not peer.gpu_init_url:
            return False
        try:
            with peer.gpu_connect_lock:
                if peer.gpu_connected:
                    return True
                # peer.init_url (the peer's L1/DRAM agent), NOT
                # peer.gpu_init_url — see the docstring.
                channel.lazy_init_peer_connection(
                    local_id=peer.local_id,
                    peer_id=peer.peer_id,
                    peer_init_url=peer.init_url,
                )
                peer.gpu_connected = True
        except Exception as e:
            logger.debug(
                "NIXL GPU connect to peer node_id=%d not ready (%s)",
                peer.node_id,
                type(e).__name__,
            )
            return False
        logger.info("NIXL peer node_id=%d GPU channel connected", peer.node_id)
        return True

    def submit_h2d_batch(
        self,
        keys: list[ObjectKey],
        gpu_ptrs: list[int],
        dst_sizes: list[int],
    ) -> list[int]:
        """Pull a batch of pinned remote chunks straight into GPU memory.

        **This call blocks until the RDMA transfers complete**, which is a
        deliberate divergence from the CXL adapter's fire-and-forget
        ``cudaMemcpyAsync``. A ``cudaMemcpyAsync`` issued on the caller's
        stream is ordered against the scatter kernel the caller enqueues
        next; a one-sided RDMA READ is issued to the NIC and carries no such
        ordering. Returning early would let that kernel read a staging
        buffer the NIC has not finished writing — silent KV corruption. The
        only ordering primitive available is CPU-side completion, so we take
        it. (Overriding the base per-key loop matters for more than
        overhead: it collapses a batch into one RDMA per peer instead of one
        per chunk, so the blocking wait is paid once.)

        The interface contract is still honored: it does not synchronize the
        CUDA stream — it never touches the stream at all.

        Args:
            keys: Pinned keys to copy (from ``lookup_and_lock``).
            gpu_ptrs: Destination device pointer per key, into the
                registered staging buffer.
            dst_sizes: Destination capacity in bytes per key.

        Returns:
            One token per key in input order: a non-negative token for
            ``release_after_h2d``, or ``-1`` if the key had no pin, its peer
            has no GPU channel, or the transfer failed. A ``-1`` is
            unrecoverable for the caller — there is no L1 copy to fall back
            on — so the retrieve must fail rather than use that slot.

        Raises:
            ValueError: If the three lists do not have equal length.
            NotImplementedError: If GPUDirect retrieve is not enabled.
        """
        if not (len(keys) == len(gpu_ptrs) == len(dst_sizes)):
            raise ValueError(
                "submit_h2d_batch: keys, gpu_ptrs and dst_sizes must have equal length"
            )
        channel = self._gpu_channel
        if channel is None:
            raise NotImplementedError(
                "NixlPeerL2Adapter: GPUDirect retrieve is not enabled "
                "(register_gpu_staging_buffer has not run)"
            )
        if not keys:
            return []

        # Group by peer so each peer costs one RDMA (and one blocking wait).
        by_peer: dict[int, list[int]] = {}
        tokens: list[int] = [-1] * len(keys)
        with self._lock:
            for i, key in enumerate(keys):
                pin = self._pins.get(key)
                if pin is None:
                    logger.warning("submit_h2d_batch: no remote pin held for %s", key)
                    continue
                by_peer.setdefault(pin.peer_index, []).append(i)

        for peer_index, positions in by_peer.items():
            peer = self._peers[peer_index]
            # The GPU handshake normally completed in the background at
            # registration; this only covers the window before it lands.
            # Done inline on THIS thread rather than by scheduling onto the
            # bg loop and blocking on the result: we are already blocking
            # (see the docstring), and hopping to the loop would queue the
            # TTFT-critical retrieve behind whatever else it is running —
            # and strand the coroutine if the adapter closes meanwhile.
            if not peer.gpu_connected and not self._connect_peer_gpu_sync(peer):
                logger.warning(
                    "submit_h2d_batch: peer node_id=%d has no GPU channel; "
                    "%d key(s) cannot be served GPU-direct",
                    peer.node_id,
                    len(positions),
                )
                continue

            with self._lock:
                pins = [self._pins.get(keys[i]) for i in positions]
            # A pin dropped between grouping and here (e.g. a concurrent
            # unlock) makes the whole peer batch unsafe to address.
            if any(pin is None for pin in pins):
                logger.warning(
                    "submit_h2d_batch: pin vanished for peer node_id=%d; "
                    "skipping %d key(s)",
                    peer.node_id,
                    len(positions),
                )
                continue

            try:
                channel.read_chunks_to_gpu(
                    [gpu_ptrs[i] for i in positions],
                    [
                        min(pin.size, dst_sizes[i])
                        for i, pin in zip(positions, pins, strict=True)
                    ],
                    [pin.remote_index for pin in pins],
                    peer.peer_id,
                )
            except Exception:
                logger.exception(
                    "GPUDirect READ from peer node_id=%d failed for %d chunk(s)",
                    peer.node_id,
                    len(positions),
                )
                continue

            with self._lock:
                for i in positions:
                    token = self._next_h2d_token
                    self._next_h2d_token += 1
                    self._h2d_token_to_key[token] = keys[i]
                    tokens[i] = token

        return tokens

    def submit_h2d(self, key: ObjectKey, gpu_ptr: int, dst_size: int) -> int:
        """Pull one pinned remote chunk into GPU memory.

        Single-key form of :meth:`submit_h2d_batch`; see it for the blocking
        semantics. Prefer the batch form — one RDMA per peer rather than one
        per chunk.

        Args:
            key: The pinned key to copy.
            gpu_ptr: Destination device pointer.
            dst_size: Destination capacity in bytes.

        Returns:
            A token for ``release_after_h2d``, or ``-1`` on failure.

        Raises:
            NotImplementedError: If GPUDirect retrieve is not enabled.
        """
        return self.submit_h2d_batch([key], [gpu_ptr], [dst_size])[0]

    def release_after_h2d_batch(self, tokens: list[int]) -> None:
        """Release the remote read-locks for a batch of copied chunks.

        Resolves tokens to keys and routes them through the same
        ``_do_unlock`` path as a normal unlock, so the lease-grouping rules
        (a peer may hold pins for one key under several leases) are applied
        in exactly one place. ``-1`` tokens are ignored.

        Unlike CXL this need not wait for the stream — the RDMA already
        completed inside ``submit_h2d_batch`` — but the caller invokes it
        from a stream callback anyway, which is harmless: releasing late
        only delays the peer's ability to evict.

        Args:
            tokens: Tokens returned by ``submit_h2d_batch``.
        """
        with self._lock:
            keys = [
                key
                for token in tokens
                if token >= 0
                and (key := self._h2d_token_to_key.pop(token, None)) is not None
            ]
        if keys:
            self.submit_unlock(keys)

    def release_after_h2d(self, token: int) -> None:
        """Release the remote read-lock for one copied chunk.

        Args:
            token: A token returned by ``submit_h2d`` (``-1`` is a no-op).
        """
        self.release_after_h2d_batch([token])

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
        # Stop the background peer prober before closing the control
        # clients it probes through.
        if self._health_monitor is not None:
            try:
                self._health_monitor.stop()
            except Exception:
                logger.exception("PeerHealthMonitor.stop failed during shutdown")
            self._health_monitor = None

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
        gpu_channel = self._gpu_channel
        self._gpu_channel = None
        if gpu_channel is not None:
            try:
                gpu_channel.close()
            except Exception:
                logger.exception("GPU data channel close failed during shutdown")

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

    # NIXL descriptor (transfer-op) size. The one-sided READ moves one
    # descriptor per RDMA op, so this sets the op size. We register at 2 MiB
    # rather than the L1 allocator's small align_bytes (e.g. 4 KiB): a KV
    # chunk then spans only chunk_bytes/2MiB descriptors (e.g. 16 for a
    # 32 MiB chunk) instead of thousands, so each transfer is a handful of
    # large RDMA ops at ~line rate, and the per-chunk index expansion in
    # _NixlReadChannel stays tiny (no million-element Python list). 2 MiB is
    # the x86 hugepage size and divides typical KV chunk sizes.
    #
    # CRITICAL: this page size is part of the cross-node wire contract — the
    # donor returns ``meta.address // page_size`` as a chunk's base index and
    # the requester expands by ``chunk_bytes // page_size``. Both nodes must
    # use the SAME value or the index arithmetic desyncs and corrupts KV. It
    # is a fixed constant (not derived from per-node state) precisely so all
    # nodes agree. We require the L1 buffer to tile evenly at 2 MiB and fail
    # loudly otherwise rather than silently picking a different (desyncing)
    # size on one node.
    nixl_page_bytes = 2 * 1024 * 1024
    if l1_memory_desc.size % nixl_page_bytes != 0:
        raise ValueError(
            f"nixl_peer requires the L1 buffer size ({l1_memory_desc.size}) to "
            f"be a multiple of the {nixl_page_bytes}-byte NIXL descriptor size. "
            f"Adjust --l1-size-gb so the buffer tiles evenly at 2 MiB."
        )

    channel = NixlChannel(
        async_mode=False,
        device=config.device,
        role="both",
        buffer_ptr=l1_memory_desc.ptr,
        buffer_size=l1_memory_desc.size,
        align_bytes=nixl_page_bytes,
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
    # buffer; the one-sided READ is addressed by descriptor index (one per
    # nixl_page_bytes). The donor converts offset -> base index and the read
    # channel expands each chunk into its chunk_bytes/nixl_page_bytes
    # consecutive descriptors. Both MUST use the SAME page size or the index
    # arithmetic desyncs, so they share nixl_page_bytes here.
    page_size = nixl_page_bytes
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
        gpu_init_url = p.get("gpu_init_url", "")
        peers.append(
            _Peer(
                node_id=p["node_id"],
                peer_id=f"node-{p['node_id']}",
                control=control,
                init_url=_strip_tcp_scheme(p["init_url"]),
                local_id=f"node-{config.node_id}",
                gpu_init_url=(
                    _strip_tcp_scheme(gpu_init_url)
                    if (config.enable_gpu_direct and gpu_init_url)
                    else ""
                ),
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

    # Peer liveness monitor: probes dead peers' control servers off the
    # lookup path so a node launched alone (or before its peers) never
    # blocks on them, and picks a peer back up when it comes online.
    health_monitor: PeerHealthMonitor | None = None
    if peers:
        sender_id = f"node-{config.node_id}"
        probe_timeout_ms = config.peer_probe_timeout_ms

        def _probe_peer(peer_index: int) -> bool:
            return peers[peer_index].control.ping(sender_id, probe_timeout_ms)

        def _describe_peer(peer_index: int) -> str:
            return f"peer node_id={peers[peer_index].node_id}"

        health_monitor = PeerHealthMonitor(
            num_peers=len(peers),
            probe_fn=_probe_peer,
            probe_interval_s=config.peer_probe_interval_ms / 1000.0,
            name="nixl-peer-health",
            describe_fn=_describe_peer,
        )

    # GPUDirect: a SECOND NixlChannel over the GPU staging buffer. NIXL
    # registers one region with one memory type per agent, so VRAM needs its
    # own agent rather than an addition to the DRAM one. Deferred behind a
    # factory because the staging buffer belongs to the GPU context, which
    # does not exist yet.
    gpu_channel_factory = None
    if config.enable_gpu_direct:

        def gpu_channel_factory(  # noqa: F811
            gpu_ptr: int, size: int
        ) -> _NixlGpuReadChannel:
            """Build the GPU-side read channel over the staging buffer.

            Args:
                gpu_ptr: Base device pointer of the staging buffer.
                size: Its byte size.

            Returns:
                A ``_NixlGpuReadChannel`` registered over that buffer.

            Raises:
                ValueError: If the buffer is not addressable at the NIXL
                    descriptor size (see below).
            """
            # The local descriptors must be the SAME size as the donor's, or
            # NIXL rejects the transfer outright ("makeXferReq: length
            # mismatch at index pair N"). The donor's are nixl_page_bytes by
            # the cross-node wire contract, so ours must be too — which makes
            # both of these hard requirements on a buffer we don't allocate.
            # Fail loudly rather than silently desync, matching how the L1
            # buffer is validated above.
            if gpu_ptr % nixl_page_bytes != 0:
                raise ValueError(
                    f"GPUDirect requires the GPU staging buffer pointer "
                    f"({gpu_ptr:#x}) to be {nixl_page_bytes}-byte aligned."
                )
            if size % nixl_page_bytes != 0:
                raise ValueError(
                    f"GPUDirect requires the GPU staging buffer size ({size}) "
                    f"to be a multiple of the {nixl_page_bytes}-byte NIXL "
                    f"descriptor size. This is the per-chunk staging stride "
                    f"times max_batch_size; a KV geometry whose chunk bytes "
                    f"are not a 2 MiB multiple (e.g. an odd KV-head or layer "
                    f"count) cannot use GPUDirect."
                )
            gpu_channel = NixlChannel(
                async_mode=False,
                device="cuda",
                role="both",
                buffer_ptr=gpu_ptr,
                buffer_size=size,
                align_bytes=nixl_page_bytes,
                tp_rank=config.local_worker_id,
                peer_init_url=_strip_tcp_scheme(config.gpu_init_bind_url),
                backends=config.nixl_backends,
            )
            return _NixlGpuReadChannel(
                gpu_channel, page_size=nixl_page_bytes, buffer_base=gpu_ptr
            )

        logger.info(
            "NIXL peer adapter node_id=%d: GPUDirect enabled "
            "(gpu init bind=%s); awaiting GPU staging buffer registration",
            config.node_id,
            config.gpu_init_bind_url,
        )

    return NixlPeerL2Adapter(
        peers=peers,
        data_channel=read_channel,
        node_id=config.node_id,
        control_server=control_server,
        connect_timeout_s=config.control_timeout_ms / 1000.0,
        health_monitor=health_monitor,
        gpu_channel_factory=gpu_channel_factory,
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
