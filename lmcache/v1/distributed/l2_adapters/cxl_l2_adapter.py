# SPDX-License-Identifier: Apache-2.0
"""CXL-backed L2 adapter for the MP-mode StorageManager.

Plan reference: F10 (MP mode is the primary deployment), Integration
Points (the MP server's StorageManager owns L2 adapters; CXL is one
of them).

This is a thin async / eventfd shim over the synchronous `CXLBackend`
from step 5. The adapter's responsibilities:

- Translate ObjectKey ↔ CXL `chunk_hash` (u64).
- Provide store / lookup-and-lock / load tasks with task-id-keyed
  results and per-tier eventfd notifications (the L1 manager's
  controllers poll these fds).
- Map `lookup_and_lock` → `CXLBackend.pin`, `submit_unlock` →
  `CXLBackend.unpin`, and `submit_load` → memcpy from CXL into the
  caller-provided buffer. The caller owns the destination MemoryObj
  lifecycle — the adapter never frees it.

Single asyncio loop on a daemon thread, mirroring MockL2Adapter so
behavior under load is the same shape.
"""

# Standard
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional
import asyncio
import concurrent.futures
import ctypes
import os
import threading
import time

# Third Party
import torch

# First Party
import lmcache.c_ops as lmc_ops
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.utils import CacheEngineKey
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
from lmcache.v1.distributed.l2_adapters.peer_health import PeerHealthMonitor
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.mp_observability.profile import PROFILE_ENABLED
from lmcache.v1.storage_backend.cxl_backend import CXLBackend, CXLBackendConfig

logger = init_logger(__name__)


class CXLL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for a CXL-backed L2 adapter.

    The CXL adapter wraps a `CXLBackend` over a CXL 2.0 shared-memory
    pool exposed as a DAX device (or a regular file for testing).

    Fields:
    - dev_path: filesystem path to the CXL device (e.g. /dev/dax0.0).
    - node_id: this node's id within the rack-wide MAX_NODES space.
      Each participating node MUST use a distinct id.
    - chunk_size_bytes: size of one KV chunk on this CXL pool. Must
      evenly divide region_size.
    - region_size: coarse per-node allocation unit. Default 256 MiB
      per the plan's "Global allocator state" sizing.
    - initialize: if True, this adapter writes a fresh header (and
      bumps `generation`). If False, it attaches to an already-
      initialized pool. Exactly one node per rack should initialize.
    - generation: only used when initialize=True. Bump this on
      controller restart to invalidate stale slots (plan F8).
    - run_lock_manager: if True, this adapter starts the rack's
      lock-manager thread. Exactly one adapter per rack should run
      it; the rest pass False.

    Geom-hash inputs (must match across the rack — see plan F1 /
    geom_hash discussion):
    - model_name, world_size, kv_dtype_str, kv_shape, use_mla,
      cluster_chunk_size (the LMCache token-chunk size, distinct
      from `chunk_size_bytes` which is the byte size on CXL).
    """

    def __init__(
        self,
        dev_path: str,
        node_id: int,
        chunk_size_bytes: int,
        region_size: int = 256 * 1024 * 1024,
        initialize: bool = False,
        generation: int = 1,
        run_lock_manager: bool = False,
        pool_size_override: int | None = None,
        # Cap on participating nodes (pool-header field). Default in the
        # CXL layout is 64; setting this to the actual deployment size
        # (e.g. 2) shrinks the lock-table sweep by max_nodes/64×, which
        # is a big win for the arbiter on small clusters.
        max_nodes: int | None = None,
        # Static-peer cross-node fetch (Alternative A — no controller).
        # If empty, the adapter does no cross-node fetch on a CXL miss.
        # Each peer is {"node_id": int, "url": "tcp://host:port"}, where
        # `url` is the CXLP2PServer endpoint on that peer.
        peers: list[dict] | None = None,
        # Local CXLP2PServer bind URL for this node's donor side. The
        # adapter will start a server here in build_cxl_adapter_from_config
        # so peers can issue PushKVToCXLMsg to us.
        cxl_p2p_bind_url: str = "tcp://0.0.0.0:8447",
        # ZMQ socket recv/send timeouts on the requester (CXLP2PClient)
        # side. The donor's `handle_push` does an O(N_keys × chunk_bytes)
        # memmove + commit — for large prompts (lots of chunks × 32 MiB
        # each) this can run into many seconds. Default 30 s; set higher
        # for larger pools or larger prompts. The default 5 s in
        # CXLP2PClient is too tight for prompts > ~2k tokens on Llama 8B.
        cxl_p2p_timeout_ms: int = 30000,
        # Peer liveness / background reconnection (see PeerHealthMonitor).
        # A configured peer that is down (not launched yet, crashed, or
        # unreachable) is marked dead and SKIPPED on the request path, so a
        # lookup miss never blocks on it. A background thread pings dead
        # peers every `peer_probe_interval_ms` with a short
        # `peer_probe_timeout_ms` timeout and promotes any that answer — so
        # a node launched alone works standalone and picks up peers that
        # come online later. The probe timeout is intentionally far shorter
        # than `cxl_p2p_timeout_ms` (which sizes a real donor push): a probe
        # only needs a round-trip, not a multi-second copy.
        peer_probe_interval_ms: int = 5000,
        peer_probe_timeout_ms: int = 1000,
        # Geom-hash inputs:
        model_name: str = "",
        world_size: int = 1,
        kv_dtype_str: str = "torch.float16",
        kv_shape: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0),
        use_mla: bool = False,
        cluster_chunk_size: int = 256,
        # Worker identity (does not feed geom_hash):
        worker_id: int = 0,
        local_world_size: int = 1,
        local_worker_id: int = 0,
    ):
        if not dev_path:
            raise ValueError("dev_path must be a non-empty string")
        if node_id < 0:
            raise ValueError("node_id must be non-negative")
        if chunk_size_bytes <= 0:
            raise ValueError("chunk_size_bytes must be positive")
        if region_size <= 0 or region_size & (region_size - 1):
            raise ValueError("region_size must be a positive power of two")
        if region_size % chunk_size_bytes != 0:
            raise ValueError(
                f"region_size {region_size} must be divisible by chunk_size_bytes "
                f"{chunk_size_bytes}"
            )

        # Validate peers: must be list of {"node_id": int, "url": str}.
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
            if not isinstance(p.get("url"), str) or not p["url"]:
                raise ValueError(f"peers[{i}].url must be a non-empty string")
            peers_list.append({"node_id": int(p["node_id"]), "url": p["url"]})

        if not isinstance(cxl_p2p_bind_url, str) or not cxl_p2p_bind_url:
            raise ValueError("cxl_p2p_bind_url must be a non-empty string")
        if not isinstance(cxl_p2p_timeout_ms, int) or cxl_p2p_timeout_ms <= 0:
            raise ValueError("cxl_p2p_timeout_ms must be a positive integer")
        if not isinstance(peer_probe_interval_ms, int) or peer_probe_interval_ms <= 0:
            raise ValueError("peer_probe_interval_ms must be a positive integer")
        if not isinstance(peer_probe_timeout_ms, int) or peer_probe_timeout_ms <= 0:
            raise ValueError("peer_probe_timeout_ms must be a positive integer")

        self.dev_path = dev_path
        self.node_id = node_id
        self.chunk_size_bytes = chunk_size_bytes
        self.region_size = region_size
        self.initialize = initialize
        self.generation = generation
        self.run_lock_manager = run_lock_manager
        self.pool_size_override = pool_size_override
        if max_nodes is not None and (not isinstance(max_nodes, int) or max_nodes <= 0):
            raise ValueError("max_nodes must be a positive integer when set")
        self.max_nodes = max_nodes
        self.peers = peers_list
        self.cxl_p2p_bind_url = cxl_p2p_bind_url
        self.cxl_p2p_timeout_ms = cxl_p2p_timeout_ms
        self.peer_probe_interval_ms = peer_probe_interval_ms
        self.peer_probe_timeout_ms = peer_probe_timeout_ms
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
    def from_dict(cls, d: dict) -> "CXLL2AdapterConfig":
        required_str = ("dev_path",)
        for k in required_str:
            if not isinstance(d.get(k), str) or not d[k]:
                raise ValueError(f"{k!r} (str) is required")
        required_int = ("node_id", "chunk_size_bytes")
        for k in required_int:
            if not isinstance(d.get(k), int):
                raise ValueError(f"{k!r} (int) is required")

        kv_shape = d.get("kv_shape", (0, 0, 0, 0, 0))
        if not isinstance(kv_shape, (list, tuple)) or len(kv_shape) != 5:
            raise ValueError("kv_shape must be a 5-element list/tuple")

        pool_size_override = d.get("pool_size_override")
        if pool_size_override is not None:
            if not isinstance(pool_size_override, int) or pool_size_override <= 0:
                raise ValueError("pool_size_override must be a positive integer")

        peers = d.get("peers", [])
        if peers is not None and not isinstance(peers, list):
            raise ValueError("peers must be a list")

        return cls(
            dev_path=d["dev_path"],
            node_id=d["node_id"],
            chunk_size_bytes=d["chunk_size_bytes"],
            region_size=d.get("region_size", 256 * 1024 * 1024),
            initialize=bool(d.get("initialize", False)),
            generation=int(d.get("generation", 1)),
            run_lock_manager=bool(d.get("run_lock_manager", False)),
            pool_size_override=pool_size_override,
            max_nodes=d.get("max_nodes"),
            peers=peers,
            cxl_p2p_bind_url=d.get("cxl_p2p_bind_url", "tcp://0.0.0.0:8447"),
            cxl_p2p_timeout_ms=int(d.get("cxl_p2p_timeout_ms", 30000)),
            peer_probe_interval_ms=int(d.get("peer_probe_interval_ms", 5000)),
            peer_probe_timeout_ms=int(d.get("peer_probe_timeout_ms", 1000)),
            model_name=d.get("model_name", ""),
            world_size=int(d.get("world_size", 1)),
            kv_dtype_str=d.get("kv_dtype_str", "torch.float16"),
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
            "CXL L2 adapter config fields:\n"
            "- dev_path (str): path to the CXL pool, e.g. /dev/dax0.0 (required)\n"
            "- node_id (int): this node's id; must be distinct per rack (required)\n"
            "- chunk_size_bytes (int): bytes per KV chunk on CXL (required, >0)\n"
            "- region_size (int): coarse alloc unit; default 256 MiB; "
            "must be a power of two and a multiple of chunk_size_bytes\n"
            "- initialize (bool): True iff this node writes a fresh header; "
            "exactly one per rack (default false)\n"
            "- generation (int): bump on controller restart (default 1)\n"
            "- run_lock_manager (bool): True iff this node runs the rack-wide "
            "lock-manager thread (default false)\n"
            "- peers (list of {node_id, url}): static peer list for cross-node "
            "PushKVToCXL fallback (Alternative A — no controller). "
            "Each url is the peer's CXLP2PServer endpoint.\n"
            "- cxl_p2p_bind_url (str): bind URL for this node's CXLP2PServer "
            "(default tcp://0.0.0.0:8447)\n"
            "- peer_probe_interval_ms (int): background ping interval for dead "
            "peers (default 5000). Dead peers are skipped on the request path "
            "so a lone node never blocks on them.\n"
            "- peer_probe_timeout_ms (int): timeout for a single liveness ping "
            "(default 1000; far shorter than cxl_p2p_timeout_ms)\n"
            "- model_name, world_size, kv_dtype_str, kv_shape, use_mla, "
            "cluster_chunk_size: feed the geom_hash; must match across the rack\n"
            "- worker_id, local_world_size, local_worker_id: worker identity"
        )


def _object_key_to_chunk_hash(key: ObjectKey) -> int:
    """Stable mapping from `ObjectKey.chunk_hash` (bytes) to a u64.

    Uses the first 8 bytes little-endian, zero-padded if shorter.
    Two ObjectKeys with different `kv_rank`/`model_name` but the same
    bytes will collide here — that's intentional: this u64 is the
    *index* hash, not the equality key. The CXL index's geom_hash
    field guards against geometry mismatches (different model →
    different geom_hash → no false hit).
    """
    raw = key.chunk_hash
    if len(raw) >= 8:
        return int.from_bytes(raw[:8], "little") & 0xFFFFFFFFFFFFFFFF
    padded = raw + b"\x00" * (8 - len(raw))
    return int.from_bytes(padded, "little")


def _object_key_to_cache_engine_key(
    key: ObjectKey, metadata: LMCacheMetadata
) -> CacheEngineKey:
    """Bridge from L2's ObjectKey to the v1 CacheEngineKey CXLBackend uses.

    The CXL backend was built against `CacheEngineKey`. To avoid
    changing its surface, we synthesize a CacheEngineKey from the
    ObjectKey using `_object_key_to_chunk_hash`. The model_name and
    dtype come from metadata so the synthesized key compares equal
    across all calls within one cluster.
    """
    chunk_hash = _object_key_to_chunk_hash(key)
    return CacheEngineKey(
        model_name=metadata.model_name,
        world_size=metadata.world_size,
        worker_id=metadata.worker_id,
        chunk_hash=chunk_hash,
        dtype=metadata.kv_dtype,
    )


@dataclass
class CXLL2Adapter(L2AdapterInterface):
    """L2 adapter backed by CXL shared memory.

    Constructed with a fully-initialized `CXLBackend` (the caller
    handles the bootstrap config, including whether this node is the
    pool initializer or just attaches). The adapter starts a single
    asyncio loop in a daemon thread; all task handlers run there.

    Threading model (matches MockL2Adapter):
      - submit_*  → grabs a task_id under self._lock, schedules a
                    coroutine on the bg loop, returns immediately.
      - bg loop   → does the work synchronously against CXLBackend,
                    stores the result in the appropriate dict,
                    writes 1 to the relevant eventfd.
      - pop / query → drains the dict under self._lock.
    """

    def __init__(
        self,
        backend: CXLBackend,
        metadata: LMCacheMetadata,
        *,
        p2p_server: Optional["object"] = None,
        peer_clients: Optional[list["object"]] = None,
        health_monitor: Optional[PeerHealthMonitor] = None,
    ):
        # Initialize base-class listener list and byte accounting so
        # ``_notify_keys_stored`` (used by the eviction-spill path) and
        # ``get_usage`` work. ``max_capacity_bytes=0`` keeps the adapter out
        # of aggregate global eviction, matching prior behavior.
        super().__init__()

        self._backend = backend
        self._metadata = metadata

        # Cross-node fallback (Alternative A: static peer list, no controller).
        # `p2p_server` is a CXLP2PServer already started in the factory
        # so peers can issue PushKVToCXL to us. `peer_clients` is a list
        # of (donor_node_id, CXLP2PClient) tuples we try in turn on a
        # CXL miss. Both are optional — if absent the adapter behaves
        # exactly like before (no remote fetch).
        self._p2p_server = p2p_server
        self._peer_clients: list = peer_clients or []

        # Peer liveness: a dead peer (never launched, crashed, or
        # unreachable) is skipped on the request path so a lookup miss
        # never blocks on it, and a background thread pings dead peers to
        # pick them up when they come online. Absent (``None``) in tests
        # that inject no peers — the fetch path then treats every peer as
        # alive, preserving the pre-monitor behavior.
        self._health_monitor = health_monitor
        if self._health_monitor is not None:
            self._health_monitor.start()

        # Distinct event fds per kind (per the base class invariant).
        self._store_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._lookup_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._load_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)

        # Task bookkeeping.
        self._lock = threading.Lock()
        self._next_task_id: L2TaskId = 0
        self._completed_store: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookup: dict[L2TaskId, Bitmap] = {}
        self._completed_load: dict[L2TaskId, Bitmap] = {}

        # L2-resident retrieve: maps an opaque h2d token -> the pinned key,
        # so ``release_after_h2d`` can ``unpin`` the right key. Keyed by a
        # monotonic counter so a token is never reused across in-flight DMAs.
        self._h2d_token_to_key: dict[int, CacheEngineKey] = {}
        self._next_h2d_token: int = 0

        # Bg loop.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop, name="cxl-l2-loop", daemon=True
        )
        self._loop_thread.start()

        # Thread pool used to parallelize per-chunk read_into in _do_load.
        # Per-chunk work (index lookup → ref_count_up → CXL memcpy →
        # ref_count_down) is independent across chunks; the C calls
        # involved (ctypes.memmove, CXL line reads/writes) all release
        # the GIL, so threading scales well up to memory-bandwidth limits.
        load_workers = int(os.environ.get("LMCACHE_CXL_LOAD_WORKERS", "8"))
        self._load_executor = ThreadPoolExecutor(
            max_workers=max(1, load_workers),
            thread_name_prefix="cxl-l2-load",
        )

    # ---------------- event fds ----------------

    def get_store_event_fd(self) -> int:
        return self._store_efd

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd

    def get_load_event_fd(self) -> int:
        return self._load_efd

    # ---------------- store ----------------

    def submit_store_task(
        self, keys: list[ObjectKey], objects: list[MemoryObj]
    ) -> L2TaskId:
        with self._lock:
            task_id = self._alloc_task_id()
        asyncio.run_coroutine_threadsafe(
            self._do_store(keys, objects, task_id), self._loop
        )
        return task_id

    async def _do_store(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        success = True
        bytes_transferred = 0
        try:
            ce_keys = [_object_key_to_cache_engine_key(k, self._metadata) for k in keys]
            self._backend.batched_submit_put_task(ce_keys, list(objects))
            bytes_transferred = sum(obj.get_size() for obj in objects)
        except Exception:
            logger.exception("CXL L2 store task %d failed", task_id)
            success = False
        with self._lock:
            self._completed_store[task_id] = L2StoreResult(success, bytes_transferred)
        os.eventfd_write(self._store_efd, 1)

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        with self._lock:
            done = self._completed_store
            self._completed_store = {}
        return done

    def spill_store(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        timeout_s: float,
    ) -> L2StoreResult:
        """Synchronously copy ``objects`` into CXL for an L1-eviction spill.

        Runs the same backend write as :meth:`submit_store_task` on the
        adapter's asyncio loop but blocks on its own future and returns the
        result directly. It never allocates a shared ``L2TaskId``, never
        writes ``_completed_store``, and never signals ``_store_efd``, so it
        cannot race the async store controller for the shared completion
        channel (see :meth:`L2AdapterInterface.spill_store`).

        Args:
            keys (list[ObjectKey]): keys to store; same length as ``objects``.
            objects (list[MemoryObj]): caller-owned (read-locked) objects.
            timeout_s (float): max seconds to wait before reporting failure.

        Returns:
            L2StoreResult: success flag plus bytes actually transferred.
        """
        future = asyncio.run_coroutine_threadsafe(
            self._do_spill_store(keys, list(objects)), self._loop
        )
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            future.cancel()
            logger.warning(
                "CXL spill_store timed out after %.1fs (%d keys); "
                "the backend copy may still land.",
                timeout_s,
                len(keys),
            )
            return L2StoreResult(success=False, bytes_transferred=0)
        except Exception:
            logger.exception("CXL spill_store failed")
            return L2StoreResult(success=False, bytes_transferred=0)

    async def _do_spill_store(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2StoreResult:
        """Backend write for :meth:`spill_store`, returning the result inline.

        Mirrors :meth:`_do_store` but returns an ``L2StoreResult`` instead of
        depositing into ``_completed_store``, and notifies the adapter's own
        L2 eviction accounting of the newly resident bytes.
        """
        try:
            ce_keys = [_object_key_to_cache_engine_key(k, self._metadata) for k in keys]
            self._backend.batched_submit_put_task(ce_keys, objects)
            sizes = [obj.get_size() for obj in objects]
            self._notify_keys_stored(keys, sizes)
            return L2StoreResult(success=True, bytes_transferred=sum(sizes))
        except Exception:
            logger.exception("CXL _do_spill_store failed")
            return L2StoreResult(success=False, bytes_transferred=0)

    # ---------------- lookup-and-lock ----------------

    def submit_lookup_and_lock_task(self, keys: list[ObjectKey]) -> L2TaskId:
        with self._lock:
            task_id = self._alloc_task_id()
        # Lookup is cheap and synchronous; schedule via call_soon so
        # we don't pay the coroutine setup cost.
        self._loop.call_soon_threadsafe(self._do_lookup, keys, task_id)
        return task_id

    def _do_lookup(self, keys: list[ObjectKey], task_id: L2TaskId) -> None:
        ce_keys = [_object_key_to_cache_engine_key(k, self._metadata) for k in keys]
        bitmap = Bitmap(len(keys))

        prof = PROFILE_ENABLED
        t_local0 = time.perf_counter() if prof else 0.0

        # First pass: pure CXL hits. Pin all keys in one batched lock
        # acquisition (≈ one arbiter sweep) instead of one sweep per key —
        # this is the dominant warm-lookup cost at long prompts. A pin
        # succeeds only for a currently-VALID slot, so the True positions are
        # exactly the local CXL hits; the rest are misses to fetch remotely.
        pinned = self._backend.pin_batch(ce_keys)
        miss_indices: list[int] = []
        for i, ok in enumerate(pinned):
            if ok:
                bitmap.set(i)
            else:
                miss_indices.append(i)

        local_pin_s = (time.perf_counter() - t_local0) if prof else 0.0
        n_miss = len(miss_indices)
        remote_breakdown: dict[str, float] = {}

        # Second pass (Alternative A — static peers, no controller):
        # For each contiguous run of misses, ask each known peer in
        # turn via PushKVToCXL. After a successful donor commit, retry
        # contains(pin=True) so the bitmap reflects the new hits.
        if miss_indices and self._peer_clients:
            remote_breakdown = self._try_remote_fetch_misses(
                ce_keys, miss_indices, bitmap
            )

        with self._lock:
            self._completed_lookup[task_id] = bitmap
        os.eventfd_write(self._lookup_efd, 1)

        if prof:
            # Compact sub-breakdown of the L2 lookup/pin (``l2lk``) span,
            # measured on the adapter loop. ``local_pin`` is the local CXL
            # hit-scan + pin; ``reserve``/``rpc``/``repin`` are the cross-node
            # cold-fetch parts (per-key slot reservation, the one batched
            # donor RPC round-trip, and re-pinning satisfied keys). All in ms.
            # Correlate with the main PROFILE line by n (chunk count).
            logger.info(
                "PROFILE-L2LK n=%d miss=%d local_pin=%.1f reserve=%.1f "
                "rpc=%.1f repin=%.1f",
                len(keys),
                n_miss,
                local_pin_s * 1000.0,
                remote_breakdown.get("reserve", 0.0) * 1000.0,
                remote_breakdown.get("rpc", 0.0) * 1000.0,
                remote_breakdown.get("repin", 0.0) * 1000.0,
            )

    def _try_remote_fetch_misses(
        self,
        ce_keys: list,
        miss_indices: list[int],
        bitmap: Bitmap,
    ) -> dict[str, float]:
        """Attempt to fetch missed chunks from each peer in turn.

        We hand the donor the *contiguous prefix of misses starting from
        the first miss*. A donor that has only the head of the request
        will commit a contiguous prefix; the rest stay missed and get
        another peer asked next time around. We stop on the first
        successful prefix (any donor that commits ≥1 key).

        This is intentionally simple: O(peers * misses) in the worst
        case. With 2 nodes it's just "ask the other one once."

        Returns:
            A ``{stage: seconds}`` profiling breakdown (``reserve``, ``rpc``,
            ``repin``) when ``LMC_PROFILE`` is set, else an empty dict.
        """
        # First Party
        from lmcache.v1.storage_backend.cxl.cross_node import remote_fetch
        from lmcache.v1.storage_backend.cxl.p2p_messages import PushStatus

        # We only fetch the contiguous prefix of misses — gaps mean the
        # caller already has those keys via CXL hits, so chunk-by-chunk
        # recovery from peers is moot.
        first_miss_idx = miss_indices[0]
        miss_keys = [ce_keys[i] for i in miss_indices]

        epoch = int(self._backend._pool.header.gen)
        sender_id = f"node-{self._backend._node_id}"

        # Accumulated across all peers tried: remote_fetch fills reserve/rpc;
        # the re-pin loop below adds repin. Empty (and unused) unless profiling.
        timings: dict[str, float] = {} if PROFILE_ENABLED else None  # type: ignore[assignment]

        for peer_index, (donor_node_id, client) in enumerate(self._peer_clients):
            # Skip peers the health monitor currently believes are dead:
            # issuing remote_fetch to a down peer would block on the full
            # cxl_p2p_timeout_ms. A lone node (all peers dead) therefore
            # falls straight through to "all misses stay misses" with no
            # stall. The background prober flips a peer back to alive once
            # it answers a ping, so this recovers automatically.
            if self._health_monitor is not None and not self._health_monitor.is_alive(
                peer_index
            ):
                continue
            try:
                result = remote_fetch(
                    requester_node_id=self._backend._node_id,
                    keys=miss_keys,
                    index_writer=self._backend._index_writer,
                    donor_node_id=donor_node_id,
                    donor=client,
                    sender_id=sender_id,
                    epoch=epoch,
                    timings=timings,
                )
            except Exception:
                # A live peer just failed: demote it so the next lookup
                # skips it instead of paying the timeout again.
                if self._health_monitor is not None:
                    self._health_monitor.record_failure(peer_index)
                logger.exception(
                    "remote_fetch to peer node_id=%d failed", donor_node_id
                )
                continue

            # The peer answered (even a 0-key/NACK reply proves liveness).
            if self._health_monitor is not None:
                self._health_monitor.record_success(peer_index)

            num = result.num_satisfied
            if num <= 0:
                continue

            # Settle the satisfied prefix into the bitmap + local slot cache.
            # Freshly-committed slots are born-pinned by the donor (the pin is
            # already held on our behalf), so we only verify+cache them
            # (pin=False) and skip the redundant per-chunk pin round-trip on
            # the arbiter. ALREADY_PRESENT slots were not committed by this
            # fetch, so they still need a pin.
            t_repin0 = time.perf_counter() if PROFILE_ENABLED else 0.0
            satisfied_indices = miss_indices[:num]
            born_pinned = result.born_pinned
            for j, idx in enumerate(satisfied_indices):
                already_pinned = j < len(born_pinned) and born_pinned[j]
                if self._backend.contains(miss_keys[j], pin=not already_pinned):
                    bitmap.set(idx)
                elif already_pinned:
                    # The donor born-pinned this slot but it is no longer
                    # VALID (evicted between commit and our verify — pin should
                    # have prevented this, but guard against a protocol race).
                    # Drop the orphaned pin so it doesn't leak.
                    logger.warning(
                        "born-pinned slot for key %s not VALID at verify; "
                        "releasing orphaned pin",
                        miss_keys[j],
                    )
                    self._backend.unpin(miss_keys[j])
            if timings is not None:
                timings["repin"] = timings.get("repin", 0.0) + (
                    time.perf_counter() - t_repin0
                )

            logger.info(
                "remote_fetch from node %d satisfied %d/%d keys (status=%s)",
                donor_node_id,
                num,
                len(miss_keys),
                result.status.name,
            )

            # If the donor returned PARTIAL, the tail might be at
            # another peer — keep trying.
            if result.status == PushStatus.OK:
                return timings or {}
            # Otherwise, attempt remaining tail with next peer.
            miss_indices = miss_indices[num:]
            miss_keys = miss_keys[num:]
            if not miss_keys:
                return timings or {}

        return timings or {}

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Optional[Bitmap]:
        with self._lock:
            return self._completed_lookup.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        # Per the contract, no task_id; must always succeed eventually.
        # We retry on transient failures via call_soon_threadsafe.
        self._loop.call_soon_threadsafe(self._do_unlock, list(keys))

    def _do_unlock(self, keys: list[ObjectKey]) -> None:
        # Unpin all keys in one batched lock acquisition (≈ one arbiter
        # sweep) instead of one sweep per key.
        ce_keys = [_object_key_to_cache_engine_key(k, self._metadata) for k in keys]
        try:
            self._backend.unpin_batch(ce_keys)
        except Exception:
            logger.exception("CXL L2 batched unpin failed for %d keys", len(ce_keys))

    # ---------------- l2-resident retrieve (GPU-direct) ----------------

    def supports_l2_resident_retrieve(self) -> bool:
        """CXL serves hits straight to GPU from the registered pool."""
        return True

    def submit_h2d(self, key: ObjectKey, gpu_ptr: int, dst_size: int) -> int:
        """Issue an async H2D copy of ``key``'s chunk into ``gpu_ptr``.

        The CXL pool is ``cudaHostRegister``'d at bootstrap, so this is a
        real async DMA on the caller's current CUDA stream. No new lock is
        taken: the ``pin_count`` acquired by ``submit_lookup_and_lock_task``
        keeps the slot alive until ``release_after_h2d``.

        Args:
            key: The object key to copy.
            gpu_ptr: Destination device pointer.
            dst_size: Destination capacity in bytes.

        Returns:
            An opaque token for ``release_after_h2d`` on hit, or ``-1`` on
            miss / error.
        """
        ce_key = _object_key_to_cache_engine_key(key, self._metadata)
        try:
            res = self._backend.gpu_src_view(ce_key)
        except Exception:
            logger.exception("CXL gpu_src_view failed for %s", ce_key)
            return -1
        if res is None:
            return -1
        n_bytes, src_ptr = res
        n = min(n_bytes, dst_size)
        # The whole CXL pool is one contiguous host-registered region, so
        # there are no internal pin-chunk boundaries to respect; passing
        # offset=0 and an alignment >= n makes the native helper issue a
        # single full-size cudaMemcpyAsync on the current stream.
        try:
            lmc_ops.lmcache_memcpy_async(
                gpu_ptr,
                src_ptr,
                n,
                lmc_ops.TransferDirection.H2D,
                0,
                self._backend._chunk_size_bytes,
            )
        except Exception:
            logger.exception("CXL H2D memcpy failed for %s", ce_key)
            return -1
        with self._lock:
            token = self._next_h2d_token
            self._next_h2d_token += 1
            self._h2d_token_to_key[token] = ce_key
        return token

    def submit_h2d_batch(
        self,
        keys: list[ObjectKey],
        gpu_ptrs: list[int],
        dst_sizes: list[int],
    ) -> list[int]:
        """Issue H2D copies for a whole batch of CXL chunks in one call.

        Collapses the per-chunk Python/lock round-trips that dominate the
        L2-resident retrieve at long prompts: all chunk-index lookups are
        resolved lock-free up front, every chunk's ``cudaMemcpyAsync`` is
        enqueued on the current stream, and the token bookkeeping for the
        whole batch happens under a single critical section. The DMAs
        themselves pipeline on the stream exactly as before.

        Each chunk still issues its own ``lmcache_memcpy_async`` (one
        ``cudaMemcpyAsync``); only the Python-side overhead is batched. A
        native batched-launch kernel would remove the remaining per-chunk
        launches but is intentionally not required here.

        Args:
            keys: Object keys to copy.
            gpu_ptrs: Destination device pointer per key.
            dst_sizes: Destination capacity in bytes per key.

        Returns:
            One token per key in input order (``-1`` on miss / error).

        Raises:
            ValueError: If the three lists do not have equal length.
        """
        if not (len(keys) == len(gpu_ptrs) == len(dst_sizes)):
            raise ValueError(
                "submit_h2d_batch: keys, gpu_ptrs and dst_sizes must have equal length"
            )

        # Phase 1 (lock-free): resolve each chunk's host source view and
        # enqueue its copy. Defer token assignment so the lock is taken
        # once for the whole batch, not once per chunk.
        ce_keys = [_object_key_to_cache_engine_key(k, self._metadata) for k in keys]
        copied_ce_keys: list[CacheEngineKey] = []
        result_slots: list[int] = []  # index into copied_ce_keys, or -1 (miss)

        for ce_key, gpu_ptr, dst_size in zip(ce_keys, gpu_ptrs, dst_sizes, strict=True):
            try:
                res = self._backend.gpu_src_view(ce_key)
            except Exception:
                logger.exception("CXL gpu_src_view failed for %s", ce_key)
                res = None
            if res is None:
                result_slots.append(-1)
                continue
            n_bytes, src_ptr = res
            n = min(n_bytes, dst_size)
            try:
                lmc_ops.lmcache_memcpy_async(
                    gpu_ptr,
                    src_ptr,
                    n,
                    lmc_ops.TransferDirection.H2D,
                    0,
                    self._backend._chunk_size_bytes,
                )
            except Exception:
                logger.exception("CXL H2D memcpy failed for %s", ce_key)
                result_slots.append(-1)
                continue
            result_slots.append(len(copied_ce_keys))
            copied_ce_keys.append(ce_key)

        # Phase 2: assign tokens for the copied chunks under one lock.
        tokens: list[int] = [-1] * len(keys)
        if copied_ce_keys:
            with self._lock:
                base_token = self._next_h2d_token
                self._next_h2d_token += len(copied_ce_keys)
                for offset, ce_key in enumerate(copied_ce_keys):
                    self._h2d_token_to_key[base_token + offset] = ce_key
            for i, slot in enumerate(result_slots):
                if slot >= 0:
                    tokens[i] = base_token + slot
        return tokens

    def release_after_h2d(self, token: int) -> None:
        """Unpin the chunk whose H2D was issued under ``token``.

        Must be called only after the issuing stream has drained the copy
        (i.e. from a stream-ordered host callback). A ``token < 0`` (a
        ``submit_h2d`` miss) is a no-op.

        Args:
            token: A token returned by ``submit_h2d``.
        """
        if token < 0:
            return
        with self._lock:
            ce_key = self._h2d_token_to_key.pop(token, None)
        if ce_key is None:
            logger.warning("CXL release_after_h2d: unknown token %d", token)
            return
        try:
            self._backend.unpin(ce_key)
        except Exception:
            logger.exception("CXL release_after_h2d unpin failed for %s", ce_key)

    def release_after_h2d_batch(self, tokens: list[int]) -> None:
        """Unpin a whole batch of ``submit_h2d`` chunks in one lock pass.

        Overrides the base per-token loop: resolves every token to its key
        under a single critical section, then drops all the CXL pins via one
        batched lock acquisition (≈ one arbiter sweep) instead of one sweep
        per chunk — the dominant resident-retrieve teardown cost at long
        prompts. ``-1`` tokens (``submit_h2d`` misses) are ignored.

        Args:
            tokens: Tokens returned by ``submit_h2d_batch`` (``-1`` ignored).
        """
        with self._lock:
            ce_keys = [
                ce_key
                for token in tokens
                if token >= 0
                and (ce_key := self._h2d_token_to_key.pop(token, None)) is not None
            ]
        if not ce_keys:
            return
        try:
            self._backend.unpin_batch(ce_keys)
        except Exception:
            logger.exception(
                "CXL release_after_h2d batched unpin failed for %d keys",
                len(ce_keys),
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
            self._do_load(keys, list(objects), task_id), self._loop
        )
        return task_id

    async def _do_load(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        bitmap = Bitmap(len(keys))
        t0 = time.perf_counter()
        bytes_copied = 0
        # phase_ns[i] aggregates ns spent in phase i across all chunks:
        #   0 = first index lookup
        #   1 = ref_count_up (total)
        #   2 = second index lookup (post-pin re-verify)
        #   3 = memmove
        #   4 = ref_count_down (total)
        #   5..7  = ref_count_up sub-phases [acquire, body, fence]
        #   8..10 = ref_count_down sub-phases [acquire, body, fence]
        phase_ns = [0] * 11

        def _one(idx: int, key: ObjectKey, dst: MemoryObj):
            """Worker: run read_into for one chunk. Returns (idx, n, phases).

            On exception logs and returns (idx, 0, [0]*11) so the main
            thread can still aggregate bitmap state.
            """
            local_phase = [0] * 11
            try:
                ce_key = _object_key_to_cache_engine_key(key, self._metadata)
                dst_t = dst.raw_data
                if dst_t.dtype != torch.uint8:
                    dst_t = dst_t.view(torch.uint8)
                n = self._backend.read_into(
                    ce_key, dst_t.data_ptr(), dst.get_size(), phase_ns=local_phase
                )
                return idx, n, local_phase
            except Exception:
                logger.exception("CXL L2 load failed for key index %d", idx)
                return idx, 0, local_phase

        futures = [
            self._load_executor.submit(_one, i, k, d)
            for i, (k, d) in enumerate(zip(keys, objects, strict=True))
        ]
        for fut in futures:
            idx, n, local_phase = fut.result()
            if n > 0:
                bytes_copied += n
                bitmap.set(idx)
            for j in range(11):
                phase_ns[j] += local_phase[j]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        mm_ns = phase_ns[3]
        memmove_gbps = (bytes_copied / (mm_ns / 1e9)) / 1e9 if mm_ns > 0 else 0.0
        logger.info(
            "CXL L2 load task=%d hits=%d/%d bytes=%d elapsed_ms=%.3f "
            "lookup1_ms=%.3f refup_ms=%.3f lookup2_ms=%.3f memmove_ms=%.3f "
            "refdown_ms=%.3f memmove_gbps=%.3f "
            "refup[acq=%.3f body=%.3f fence=%.3f] "
            "refdown[acq=%.3f body=%.3f fence=%.3f]",
            task_id,
            bitmap.popcount(),
            len(keys),
            bytes_copied,
            elapsed_ms,
            phase_ns[0] / 1_000_000.0,
            phase_ns[1] / 1_000_000.0,
            phase_ns[2] / 1_000_000.0,
            phase_ns[3] / 1_000_000.0,
            phase_ns[4] / 1_000_000.0,
            memmove_gbps,
            phase_ns[5] / 1_000_000.0,
            phase_ns[6] / 1_000_000.0,
            phase_ns[7] / 1_000_000.0,
            phase_ns[8] / 1_000_000.0,
            phase_ns[9] / 1_000_000.0,
            phase_ns[10] / 1_000_000.0,
        )
        with self._lock:
            self._completed_load[task_id] = bitmap
        os.eventfd_write(self._load_efd, 1)

    def query_load_result(self, task_id: L2TaskId) -> Optional[Bitmap]:
        with self._lock:
            return self._completed_load.pop(task_id, None)

    @staticmethod
    def _copy_into(dst: MemoryObj, src: MemoryObj) -> tuple[int, int]:
        """Memcpy `src` bytes into `dst.raw_data`.

        Returns `(bytes_copied, memmove_ns)` — the second value is the
        time spent inside `ctypes.memmove` only, excluding dtype-view
        and size-min bookkeeping.

        Both must have a uint8-or-bigger flat byte view. The L1
        manager allocates `dst` to fit the cached object size; we
        copy `min(src_size, dst_size)` to be defensive.
        """
        dst_t = dst.raw_data
        src_t = src.raw_data
        if dst_t.dtype != torch.uint8:
            dst_t = dst_t.view(torch.uint8)
        if src_t.dtype != torch.uint8:
            src_t = src_t.view(torch.uint8)
        n = min(src.get_size(), dst.get_size())
        t0 = time.perf_counter_ns()
        ctypes.memmove(dst_t.data_ptr(), src_t.data_ptr(), n)
        return n, time.perf_counter_ns() - t0

    # ---------------- close ----------------

    def close(self) -> None:
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5)

        self._load_executor.shutdown(wait=True)

        os.close(self._store_efd)
        os.close(self._lookup_efd)
        os.close(self._load_efd)

        # Stop the background peer prober before tearing down the clients
        # it probes through.
        if self._health_monitor is not None:
            try:
                self._health_monitor.stop()
            except Exception:
                logger.exception("PeerHealthMonitor.stop failed during shutdown")
            self._health_monitor = None

        # Tear down P2P side first so we stop accepting requests before
        # the backend closes the pool.
        if self._p2p_server is not None:
            try:
                self._p2p_server.stop()
            except Exception:
                logger.exception("CXLP2PServer.stop failed during shutdown")
            self._p2p_server = None
        for entry in self._peer_clients:
            client = entry[1] if isinstance(entry, tuple) else entry
            try:
                client.close()
            except Exception:
                logger.exception("CXLP2PClient.close failed during shutdown")
        self._peer_clients = []

        try:
            self._backend.close()
        except Exception:
            logger.exception("CXLBackend close failed during adapter shutdown")

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

    def debug_get_backend(self) -> CXLBackend:
        """Test-only: peek at the backing CXLBackend."""
        return self._backend


# -----------------------------------------------------------------------------
# Factory: build a live adapter from a parsed CXLL2AdapterConfig.
# -----------------------------------------------------------------------------


_TORCH_DTYPE_FROM_STR = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.uint8": torch.uint8,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
}


def _resolve_dtype(name: str) -> torch.dtype:
    if name not in _TORCH_DTYPE_FROM_STR:
        raise ValueError(
            f"unsupported kv_dtype_str {name!r}; expected one of "
            f"{sorted(_TORCH_DTYPE_FROM_STR)}"
        )
    return _TORCH_DTYPE_FROM_STR[name]


def build_cxl_adapter_from_config(
    config: CXLL2AdapterConfig,
    *,
    l1_manager: object | None = None,
) -> CXLL2Adapter:
    """Construct a CXLL2Adapter from a parsed CXLL2AdapterConfig.

    Bootstraps the CXLBackend (which mmap's the pool, optionally
    cudaHostRegister's it, and starts the lock manager), wraps it in
    the eventfd-based adapter shim.

    If `config.peers` is non-empty AND `l1_manager` is provided, also
    starts a CXLP2PServer for the donor side and creates a CXLP2PClient
    per peer for the requester side. Both sides are required for the
    static-peer remote-fetch path (Alternative A).
    """
    backend_config = CXLBackendConfig(
        dev_path=config.dev_path,
        node_id=config.node_id,
        chunk_size_bytes=config.chunk_size_bytes,
        region_size=config.region_size,
        initialize=config.initialize,
        generation=config.generation,
        run_lock_manager=config.run_lock_manager,
        pool_size_override=config.pool_size_override,
        max_nodes=config.max_nodes,
    )
    metadata = LMCacheMetadata(
        model_name=config.model_name,
        world_size=config.world_size,
        local_world_size=config.local_world_size,
        worker_id=config.worker_id,
        local_worker_id=config.local_worker_id,
        kv_dtype=_resolve_dtype(config.kv_dtype_str),
        kv_shape=config.kv_shape,
        use_mla=config.use_mla,
        chunk_size=config.cluster_chunk_size,
    )
    backend = CXLBackend(backend_config, metadata)

    # Optional: cross-node fallback (donor server + peer clients).
    p2p_server = None
    peer_clients: list = []
    if config.peers and l1_manager is not None:
        # Lazy imports — keep CXL P2P transport / L1 donor out of the
        # import path for adapters that don't need them.
        # First Party
        from lmcache.v1.distributed.l2_adapters.cxl_l1_donor import (
            L1LocalCopyProvider,
        )
        from lmcache.v1.storage_backend.cxl.cross_node import CXLDonor
        from lmcache.v1.storage_backend.cxl.p2p_transport import (
            CXLP2PClient,
            CXLP2PServer,
        )

        local_copy = L1LocalCopyProvider(l1_manager=l1_manager, metadata=metadata)
        donor = CXLDonor(
            handle=backend._pool,
            index_writer=backend._index_writer,
            heap=backend._heap,
            node_id=backend._node_id,
            local_copy_provider=local_copy,
        )
        p2p_server = CXLP2PServer(donor=donor, bind_url=config.cxl_p2p_bind_url)
        p2p_server.start()

        for p in config.peers:
            peer_clients.append(
                (
                    p["node_id"],
                    CXLP2PClient(
                        donor_url=p["url"],
                        recv_timeout_ms=config.cxl_p2p_timeout_ms,
                        send_timeout_ms=config.cxl_p2p_timeout_ms,
                    ),
                )
            )
        logger.info(
            "CXL adapter started donor server at %s with %d peer client(s) "
            "(timeout=%d ms)",
            config.cxl_p2p_bind_url,
            len(peer_clients),
            config.cxl_p2p_timeout_ms,
        )
    elif config.peers and l1_manager is None:
        logger.warning(
            "CXL adapter has peers configured but no L1Manager was provided; "
            "skipping donor/peer setup. Cross-node fetch will be disabled."
        )

    # Peer liveness monitor: probes dead peers off the request path so a
    # node launched alone (or before its peers) never blocks on them. The
    # probe is a short-timeout ZMQ ping against the peer's CXLP2PServer.
    health_monitor: PeerHealthMonitor | None = None
    if peer_clients:
        sender_id = f"node-{backend._node_id}"
        probe_timeout_ms = config.peer_probe_timeout_ms

        def _probe_peer(peer_index: int) -> bool:
            return peer_clients[peer_index][1].ping(sender_id, probe_timeout_ms)

        def _describe_peer(peer_index: int) -> str:
            return f"peer node_id={peer_clients[peer_index][0]}"

        health_monitor = PeerHealthMonitor(
            num_peers=len(peer_clients),
            probe_fn=_probe_peer,
            probe_interval_s=config.peer_probe_interval_ms / 1000.0,
            name="cxl-peer-health",
            describe_fn=_describe_peer,
        )

    return CXLL2Adapter(
        backend=backend,
        metadata=metadata,
        p2p_server=p2p_server,
        peer_clients=peer_clients,
        health_monitor=health_monitor,
    )


def _cxl_l2_adapter_factory(
    config: CXLL2AdapterConfig,
    l1_memory_desc: object | None = None,
    *,
    l1_manager: object | None = None,
) -> CXLL2Adapter:
    """Registry factory for the CXL L2 adapter.

    Matches the registry call convention ``(config, l1_memory_desc)`` and
    additionally opts into the ``l1_manager`` keyword (forwarded by
    ``create_l2_adapter_from_registry`` only to factories that declare it).
    The CXL backend does not register L1 with an external backend, so
    ``l1_memory_desc`` is accepted for signature compatibility and ignored;
    ``l1_manager`` is what the donor/peer cross-node path needs.
    """
    return build_cxl_adapter_from_config(config, l1_manager=l1_manager)


register_l2_adapter_type("cxl", CXLL2AdapterConfig)
register_l2_adapter_factory("cxl", _cxl_l2_adapter_factory)
