# SPDX-License-Identifier: Apache-2.0
"""CXLBackend: AllocatorBackendInterface over a CXL shared-memory pool.

Single-process skeleton for step 5. This wires together:

- bootstrap.py (pool mmap + header + cudaHostRegister)
- locks.py + lock_manager.py (two-tier lock + arbiter thread)
- regions.py + heap.py (region and chunk allocators)
- index.py + index_writer.py (lock-free reads, lock-protected writes)
- allocator.py (MemoryAllocatorInterface wrapper)

Not yet wired: MP mode IPC (step 6), P2P PushKVToCXL (step 6), cache
controller notifications for local tiers. The backend deliberately
does NOT emit KVAdmitMsg/KVEvictMsg — CXL state is discovered via
CXL_LOOKUP.
"""

# Standard
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.cxl.allocator import CXLMemoryAllocator
from lmcache.v1.storage_backend.cxl.bootstrap import (
    CXLBootstrapConfig,
    PoolHandle,
    bootstrap_pool,
)
from lmcache.v1.storage_backend.cxl.heap import NodeHeap, OutOfChunks
from lmcache.v1.storage_backend.cxl.index import CXLIndex, SlotView
from lmcache.v1.storage_backend.cxl.index_writer import (
    CXLIndexWriter,
    ReserveOutcome,
    ReserveResult,
)
from lmcache.v1.storage_backend.cxl.layout import OWNER_FREE
from lmcache.v1.storage_backend.cxl.lock_manager import LockManager
from lmcache.v1.storage_backend.cxl.lock_manager_proc import (
    ProcessLockManager,
    ProcessLockManagerConfig,
)
from lmcache.v1.storage_backend.cxl.locks import TwoTierLock
from lmcache.v1.storage_backend.cxl.lru_tracker import NodeLRUTracker
from lmcache.v1.storage_backend.cxl.regions import (
    NoRegionAvailable,
    RegionAllocator,
)

logger = init_logger(__name__)


@dataclass
class CXLBackendConfig:
    """Construction-time config for CXLBackend.

    In the real deployment these come from LMCacheEngineConfig's
    `cxl_*` fields; the dataclass here is what the backend actually
    needs and keeps tests decoupled from the full config surface.
    """

    dev_path: str
    node_id: int
    chunk_size_bytes: int
    region_size: int = 256 * 1024 * 1024
    initialize: bool = False
    generation: int = 1
    run_lock_manager: bool = True
    pool_size_override: Optional[int] = None
    max_nodes: Optional[int] = None
    num_locks: Optional[int] = None
    # Node-local LRU eviction (see CXLBackend eviction ladder). When a store
    # cannot claim a new region because the *global* pool is exhausted, the
    # node evicts its own coldest chunks back into its heap free-list and
    # retries the store from there, capping its footprint instead of dropping
    # the put. `evict_low_watermark` is the occupancy floor eviction drains
    # toward (occupied owned-slots / total owned-slots) when there is headroom;
    # a batch that needs more space evicts past it, down to the last
    # non-pinned chunk. Range (0.0, 1.0]. There is no separate high watermark:
    # the trigger is region exhaustion (NoRegionAvailable), so eviction costs
    # nothing until the pool is actually full.
    evict_low_watermark: float = 0.8
    # When True (default) and run_lock_manager is also True, the lock
    # manager runs as a standalone C subprocess via ProcessLockManager.
    # The in-process Python LockManager is starved of GIL under donor
    # load (sweep mean balloons ~20×), which the subprocess avoids.
    # Set False to fall back to the Python thread (e.g. for tests or
    # environments without a C compiler).
    use_process_lock_manager: bool = True


@dataclass
class ClearResult:
    """Outcome of :meth:`CXLBackend.clear`.

    Attributes:
        chunks_deleted: Number of this node's VALID chunks tombstoned and
            freed back to the node heap.
        slots_skipped_busy: Number of this node's VALID chunks left in place
            because they were pinned or had a nonzero ref_count (in-flight
            read / GPU copy). Their regions cannot trim while they persist.
        regions_released: Number of now-empty regions returned to the global
            pool by ``heap.trim()``.
    """

    chunks_deleted: int
    slots_skipped_busy: int
    regions_released: int


class CXLBackend(AllocatorBackendInterface):
    """CXL shared-memory L2 tier.

    Single-process for now: one CXLBackend per MP server process
    (step 6 will add cross-process / cross-node wiring). Implements
    AllocatorBackendInterface so it plugs into StorageManager.

    Lifecycle:
      1. __init__ takes a CXLBackendConfig + LMCacheMetadata and runs
         the full bootstrap. After __init__, the pool is mmap'd,
         cudaHostRegister'd (best effort), the lock manager is running
         (if enabled), and the backend is ready to serve put/get.
      2. close() stops the lock manager, closes the pool. Does NOT
         initialize the pool on close — a restart of the same node
         should re-attach to the existing header.
    """

    def __init__(
        self,
        cxl_config: CXLBackendConfig,
        metadata: LMCacheMetadata,
        dst_device: str = "cuda",
    ):
        super().__init__(dst_device=dst_device)
        self._node_id = cxl_config.node_id
        self._chunk_size_bytes = cxl_config.chunk_size_bytes
        if not 0.0 < cxl_config.evict_low_watermark <= 1.0:
            raise ValueError(
                "evict_low_watermark must be in (0.0, 1.0], got "
                f"{cxl_config.evict_low_watermark}"
            )
        self._evict_low_watermark = cxl_config.evict_low_watermark

        # ``max_nodes`` / ``num_locks`` size the shared lock table, and the
        # arbiter sweeps every one of its ``num_locks * max_nodes`` cells each
        # pass — so both directly set the sweep cost. Pass them through only
        # when set, letting CXLBootstrapConfig's defaults apply otherwise.
        bootstrap_kwargs: dict[str, int] = {}
        if cxl_config.max_nodes is not None:
            bootstrap_kwargs["max_nodes"] = cxl_config.max_nodes
        if cxl_config.num_locks is not None:
            bootstrap_kwargs["num_locks"] = cxl_config.num_locks
        bootstrap_cfg = CXLBootstrapConfig(
            dev_path=cxl_config.dev_path,
            region_size=cxl_config.region_size,
            initialize=cxl_config.initialize,
            generation=cxl_config.generation,
            pool_size_override=cxl_config.pool_size_override,
            **bootstrap_kwargs,
        )
        self._pool: PoolHandle = bootstrap_pool(bootstrap_cfg, metadata)
        self._fence = None  # default StubFence; explicit None == use module default

        # Two-tier lock shared across all subsystems on this node.
        self._lock = TwoTierLock(self._pool, node_id=self._node_id)

        # Lock manager (single-writer arbiter). In multi-node runs this
        # should run on one elected node only. For the skeleton, every
        # backend starts one; later steps will add election.
        #
        # Two implementations:
        #   - ProcessLockManager: runs a small C binary as a sidecar
        #     process. Preferred for production — avoids GIL starvation
        #     by the donor's commit_slot loop (profile showed Python
        #     arbiter's sweep time inflating 20× under load).
        #   - LockManager: in-process Python thread. Used in tests and
        #     when the C compiler isn't available.
        self._lock_manager: Optional[LockManager] = None
        self._proc_lock_manager: Optional[ProcessLockManager] = None
        if cxl_config.run_lock_manager:
            if cxl_config.use_process_lock_manager:
                proc_cfg = ProcessLockManagerConfig(
                    dev_path=cxl_config.dev_path,
                )
                self._proc_lock_manager = ProcessLockManager(proc_cfg)
                self._proc_lock_manager.start()
            else:
                self._lock_manager = LockManager(self._pool)
                self._lock_manager.start()

        # Region + heap + allocator.
        self._region_allocator = RegionAllocator(self._pool, self._lock)
        self._heap = NodeHeap(
            self._region_allocator,
            node_id=self._node_id,
            chunk_size=self._chunk_size_bytes,
        )
        self._mem_allocator = CXLMemoryAllocator(
            heap=self._heap, pool_base=self._pool.base
        )

        # Index reader + writer.
        self._index = CXLIndex(self._pool)
        self._index_writer = CXLIndexWriter(
            self._pool, self._index, self._lock, self._node_id
        )

        # Map: chunk_hash -> slot_idx, for quick remove/pin by key.
        # The index itself is the source of truth; this is an O(1)
        # cache to skip probing when the caller already knows the key.
        self._key_to_slot: dict[int, int] = {}
        self._key_to_slot_lock = threading.Lock()

        # Node-local LRU recency over the slots this node owns. Touched on every
        # local read/commit (DRAM only, no CXL write) and consulted only when a
        # store must evict to reclaim free-list space after region exhaustion.
        self._lru = NodeLRUTracker()

        # In-flight put futures keyed by chunk_hash — tracked so
        # `exists_in_put_tasks` can answer correctly.
        self._inflight_puts: set[int] = set()
        self._inflight_lock = threading.Lock()

        logger.info(
            "CXLBackend ready: node_id=%d chunk_size=%d pool_size=%d regions=%d",
            self._node_id,
            self._chunk_size_bytes,
            self._pool.size,
            self._pool.layout.region_count,
        )

    # -------- read path --------------------------------------------------

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        view = self._index.lookup(key)
        if view is None:
            return False
        if pin and not self._index_writer.pin(view.slot_idx):
            # Slot flipped (evicted) between lookup and pin.
            return False
        self._cache_slot(key, view.slot_idx)
        return True

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self._inflight_lock:
            return key.chunk_hash in self._inflight_puts

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        view = self._index.lookup(key)
        if view is None:
            return None
        # Bump ref count under the slot lock and re-verify.
        if not self._index_writer.ref_count_up(view.slot_idx):
            return None
        # Re-fetch the slot to capture the canonical post-pin state.
        post = self._index.lookup(key)
        if post is None or post.slot_idx != view.slot_idx:
            # Slot was evicted concurrently; back out the ref_count.
            self._index_writer.ref_count_down(view.slot_idx)
            return None
        self._cache_slot(key, view.slot_idx)
        return self._materialize(post)

    def read_into(
        self,
        key: CacheEngineKey,
        dst_ptr: int,
        dst_size: int,
        phase_ns: Optional[List[int]] = None,
    ) -> int:
        """Fast hit path: memcpy a chunk directly into `dst_ptr`.

        Returns bytes copied, or 0 on miss. Avoids building a `MemoryObj`
        wrapper (numpy/torch view + ctypes array) for the source — that
        wrapping dominates the per-chunk cost in the load path. Callers
        that don't need a Python-level view of the chunk should use this
        instead of `get_blocking`.

        If `phase_ns` is provided (length 11), accumulates per-phase
        nanosecond timings:
            [0] lookup1 (first index lookup)
            [1] refcount_up (total, including sub-phases below)
            [2] lookup2 (post-pin re-verify)
            [3] memmove
            [4] refcount_down (total, including sub-phases below)
            [5..7] refcount_up sub-phases: [acquire, body, fence]
            [8..10] refcount_down sub-phases: [acquire, body, fence]
        """
        import ctypes

        # ref_count_up/down accumulate into a 3-slot list each
        # ([acquire, body, fence]); we fold the results back into the
        # caller's phase_ns at slots [5..7] and [8..10] below.
        refup_sub_list = [0, 0, 0] if phase_ns is not None else None
        refdown_sub_list = [0, 0, 0] if phase_ns is not None else None

        t0 = time.perf_counter_ns()
        view = self._index.lookup(key)
        t1 = time.perf_counter_ns()
        if phase_ns is not None:
            phase_ns[0] += t1 - t0
        if view is None:
            return 0
        if not self._index_writer.ref_count_up(view.slot_idx, phase_ns=refup_sub_list):
            if phase_ns is not None:
                phase_ns[1] += time.perf_counter_ns() - t1
                for j in range(3):
                    phase_ns[5 + j] += refup_sub_list[j]
            return 0
        t2 = time.perf_counter_ns()
        if phase_ns is not None:
            phase_ns[1] += t2 - t1
            for j in range(3):
                phase_ns[5 + j] += refup_sub_list[j]
        try:
            post = self._index.lookup(key)
            t3 = time.perf_counter_ns()
            if phase_ns is not None:
                phase_ns[2] += t3 - t2
            if post is None or post.slot_idx != view.slot_idx:
                return 0
            self._cache_slot(key, view.slot_idx)
            n = min(post.chunk_len, dst_size)
            ctypes.memmove(dst_ptr, self._pool.base + post.chunk_offset, n)
            t4 = time.perf_counter_ns()
            if phase_ns is not None:
                phase_ns[3] += t4 - t3
            return n
        finally:
            t_rd = time.perf_counter_ns()
            self._index_writer.ref_count_down(view.slot_idx, phase_ns=refdown_sub_list)
            if phase_ns is not None:
                phase_ns[4] += time.perf_counter_ns() - t_rd
                for j in range(3):
                    phase_ns[8 + j] += refdown_sub_list[j]

    def batched_contains(self, keys: List[CacheEngineKey], pin: bool = False) -> int:
        hit = 0
        for k in keys:
            if not self.contains(k, pin=pin):
                break
            hit += 1
        return hit

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        return self.batched_contains(keys, pin=pin)

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> List[MemoryObj]:
        out: List[MemoryObj] = []
        for k in keys:
            obj = self.get_blocking(k)
            if obj is None:
                break
            out.append(obj)
        return out

    # -------- write path -------------------------------------------------

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Optional[List[Future]]:
        """Synchronous batched put; returns None (completed inline).

        In the skeleton we copy bytes directly from the source obj to
        CXL. The future-returning async path is a later step when DMA
        is actually asynchronous.
        """
        if len(keys) != len(objs):
            raise ValueError(
                f"batched_submit_put_task: {len(keys)} keys vs {len(objs)} objs"
            )
        # Batch admission: when the pool is exhausted, evict this node's cold
        # chunks *once* for the whole batch (evict-as-much-as-possible). If
        # eviction still cannot free room for every chunk, drop the entire
        # batch rather than storing a partial prefix — per the eviction
        # contract, a store batch lands whole or not at all.
        if not self._ensure_batch_space(len(keys)):
            logger.warning(
                "CXL node %d: pool exhausted and local eviction could not free "
                "space for a %d-chunk store batch; dropping the batch",
                self._node_id,
                len(keys),
            )
            return None
        for key, obj in zip(keys, objs, strict=True):
            try:
                self._put_one(key, obj)
                if on_complete_callback is not None:
                    try:
                        on_complete_callback(key)
                    except Exception:
                        logger.exception("on_complete_callback raised for key %s", key)
            except Exception:
                logger.exception("failed to put key %s; skipping", key)
        return None

    async def async_batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        self.batched_submit_put_task(keys, objs, transfer_spec, on_complete_callback)

    def _put_one(self, key: CacheEngineKey, src: MemoryObj) -> None:
        """Full INSERT lifecycle for one key.

        Steps:
          1. Mark in-flight for exists_in_put_tasks.
          2. reserve_slot; if ALREADY_PRESENT, skip; if WAIT, poll.
          3. Allocate a chunk from the heap.
          4. Copy bytes from src.raw_data into the chunk.
          5. commit_slot (publish VALID).
        On any failure, release_slot and free the chunk.
        """
        with self._inflight_lock:
            self._inflight_puts.add(key.chunk_hash)
        try:
            result = self._reserve_with_retries(key)
            if result.outcome == ReserveOutcome.ALREADY_PRESENT:
                if result.slot_idx is not None:
                    self._cache_slot(key, result.slot_idx)
                return
            if result.outcome == ReserveOutcome.INDEX_FULL:
                logger.warning("CXL index full; put for key %s dropped", key)
                return
            if result.slot_idx is None:
                return
            slot_idx = result.slot_idx

            size_bytes = src.get_size()
            if size_bytes > self._chunk_size_bytes:
                self._index_writer.release_slot(slot_idx)
                raise ValueError(
                    f"payload {size_bytes} bytes > chunk size {self._chunk_size_bytes}"
                )

            try:
                chunk_offset = self._alloc_chunk_with_eviction()
            except Exception:
                self._index_writer.release_slot(slot_idx)
                raise

            try:
                self._copy_into_chunk(src, chunk_offset, size_bytes)
                self._index_writer.commit_slot(
                    slot_idx=slot_idx,
                    chunk_offset=chunk_offset,
                    chunk_len=size_bytes,
                    fmt=src.meta.fmt,
                )
                # _cache_slot also refreshes LRU recency, so the freshly
                # committed slot is most-recently-used.
                self._cache_slot(key, slot_idx)
            except Exception:
                self._heap.free(chunk_offset)
                self._index_writer.release_slot(slot_idx)
                raise
        finally:
            with self._inflight_lock:
                self._inflight_puts.discard(key.chunk_hash)

    def _alloc_chunk_with_eviction(self) -> int:
        """Allocate one chunk offset, evicting local cold chunks if needed.

        The store allocation ladder:

          1. ``heap.alloc()`` — serve from the free-list, else claim a new
             region from the global pool.
          2. On ``NoRegionAvailable`` (the global pool is exhausted, so this
             node's region count is now fixed), evict this node's coldest
             chunks back into the free-list via :meth:`_evict_cold_slots`, then
             retry with ``heap.alloc_no_claim()`` (free-list only — never
             claims).
          3. If eviction frees nothing (every cold slot is pinned or has an
             in-flight read), ``alloc_no_claim`` raises ``OutOfChunks`` and this
             method re-raises the original ``NoRegionAvailable`` so the caller
             drops the store.

        Returns:
            A pool-relative chunk offset.

        Raises:
            NoRegionAvailable: If the pool is exhausted and eviction could not
                free any local slot.
        """
        try:
            return self._heap.alloc()
        except NoRegionAvailable:
            self._evict_cold_slots(target=1)
            try:
                return self._heap.alloc_no_claim()
            except OutOfChunks:
                # Nothing evictable (all cold slots pinned / in-flight).
                logger.warning(
                    "CXL node %d: pool exhausted and no evictable local chunk; "
                    "dropping store",
                    self._node_id,
                )
                raise NoRegionAvailable(
                    "pool exhausted and local eviction freed no chunk"
                ) from None

    def _ensure_batch_space(self, n: int) -> bool:
        """Ensure ``n`` chunk slots can be allocated for a store batch.

        Runs once per batch, before the per-chunk store loop. Fast-path: if the
        node can already back ``n`` chunks — from its free-list plus regions it
        can still claim from a non-exhausted pool — nothing is evicted and this
        returns True immediately (the common case; eviction costs nothing until
        the pool is full).

        Only when the pool is exhausted (no free-list slack and no claimable
        region) does it evict this node's coldest chunks, asking
        :meth:`_evict_cold_slots` to free the batch's shortfall (and at least
        down to the low watermark). It returns whether, after that, the
        free-list holds enough slots for the whole batch.

        Args:
            n: Number of chunks the batch will store.

        Returns:
            True if the batch can proceed (each chunk will find a slot), False
            if the pool is exhausted and eviction could not free ``n`` slots —
            in which case the caller drops the whole batch.
        """
        if n <= 0:
            return True
        occupied, total = self._heap.occupancy()
        free_now = total - occupied
        if free_now >= n:
            return True
        # Free-list is short. If the pool still has claimable regions, the
        # per-chunk alloc() will grow the free-list on demand — no eviction
        # needed. Probe cheaply by asking the region layer for free capacity.
        if self._pool_has_claimable_region():
            return True
        # Pool exhausted: free the shortfall from local cold chunks.
        shortfall = n - free_now
        self._evict_cold_slots(target=shortfall)
        occupied_after, total_after = self._heap.occupancy()
        return (total_after - occupied_after) >= n

    def _pool_has_claimable_region(self) -> bool:
        """Return True if the global pool has at least one FREE region.

        A cheap read of the region descriptors (the same scan the GC does);
        used by :meth:`_ensure_batch_space` to skip eviction while the node can
        still grow its footprint by claiming.
        """
        for info in self._region_allocator.iter_regions():
            if info.owner_node_id == OWNER_FREE:
                return True
        return False

    def _evict_cold_slots(self, target: int) -> int:
        """Evict this node's coldest chunks to free at least ``target`` slots.

        Evicts oldest-first per the node-local LRU, draining toward the
        configured low watermark (``occupied / total`` owned slots) and, if the
        caller needs more than that yields, continuing past the floor until
        either ``target`` slots are freed or no evictable chunk remains. A slot
        that is pinned or has an in-flight read is refused by
        :meth:`CXLIndexWriter.evict` and skipped (dropped from the LRU so it is
        not retried this pass); its chunk stays resident.

        Freed chunks return to the node heap's free-list for immediate reuse by
        the retrying store; regions are never released here.

        Args:
            target: Minimum number of slots to free (the store needs this many;
                a single ``_put_one`` needs 1). Eviction also honors the low
                watermark, so it may free more than ``target`` when occupancy is
                above the floor.

        Returns:
            The number of chunks actually evicted this pass.
        """
        occupied, total = self._heap.occupancy()
        if total == 0:
            return 0
        floor = int(self._evict_low_watermark * total)
        freed = 0
        # Snapshot cold victims; ask for enough to both hit the watermark floor
        # and satisfy `target`, whichever is larger.
        want = max(target, occupied - floor)
        if want <= 0:
            return 0
        for slot_idx in self._lru.coldest(want):
            ok, view = self._index_writer.evict(slot_idx)
            if not ok:
                # Pinned / in-flight — leave it resident and stop tracking it
                # for this pass so it isn't reconsidered until re-touched.
                self._lru.forget(slot_idx)
                continue
            assert view is not None
            try:
                self._heap.free(view.chunk_offset)
            except Exception:
                logger.exception(
                    "CXL node %d: heap.free failed evicting slot %d offset %d; "
                    "leaking chunk",
                    self._node_id,
                    slot_idx,
                    view.chunk_offset,
                )
            self._lru.forget(slot_idx)
            self._forget_slot_by_hash(view.chunk_hash)
            freed += 1
        if freed:
            logger.info(
                "CXL node %d: evicted %d cold chunk(s) to reclaim free-list "
                "space (occupied=%d total=%d floor=%d)",
                self._node_id,
                freed,
                occupied,
                total,
                floor,
            )
        return freed

    def _reserve_with_retries(
        self, key: CacheEngineKey, max_waits: int = 100, wait_sleep_s: float = 0.001
    ) -> ReserveResult:
        for _ in range(max_waits):
            result = self._index_writer.reserve_slot(key)
            if result.outcome != ReserveOutcome.WAIT_FOR_OTHER:
                return result
            # Another writer is on this key; poll briefly and try again.
            time.sleep(wait_sleep_s)
        return ReserveResult(ReserveOutcome.WAIT_FOR_OTHER, result.slot_idx)

    def _copy_into_chunk(
        self, src: MemoryObj, chunk_offset: int, size_bytes: int
    ) -> None:
        """Copy `size_bytes` from src.raw_data into the CXL chunk.

        In the real GPU path this will be a DMA; for the single-process
        skeleton we just memcpy. `src.raw_data` is typically a uint8
        flat tensor per LocalCPU's allocator, but we handle arbitrary
        shape/dtype by flattening its byte view.
        """
        dst_addr = self._pool.base + chunk_offset
        # Flatten src to a contiguous uint8 view for a raw memcpy.
        src_tensor = src.raw_data
        if src_tensor.dtype != torch.uint8:
            src_tensor = src_tensor.view(torch.uint8)
        src_tensor = src_tensor.reshape(-1)[:size_bytes].contiguous()
        src_ptr = src_tensor.data_ptr()
        import ctypes

        ctypes.memmove(dst_addr, src_ptr, size_bytes)

    def _materialize(self, view: SlotView) -> MemoryObj:
        """Build a MemoryObj that views the CXL chunk for a HIT."""
        # Standard
        import ctypes

        # Third Party
        import numpy as np

        fmt = MemoryFormat(view.fmt) if view.fmt != 0 else MemoryFormat.UNDEFINED
        buf_type = ctypes.c_uint8 * self._chunk_size_bytes
        ctypes_buf = buf_type.from_address(self._pool.base + view.chunk_offset)
        np_buf = np.frombuffer(ctypes_buf, dtype=np.uint8, count=self._chunk_size_bytes)
        raw_data = torch.from_numpy(np_buf)

        # We don't know the caller's intended (shape, dtype) at get
        # time — the slot only carries the format and a byte length.
        # Return a 1-D uint8 MemoryObj sized to chunk_len; callers
        # reinterpret via a GPUConnector that knows the KV geometry.
        shape = torch.Size([view.chunk_len])
        dtype = torch.uint8
        # First Party
        from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj

        meta = MemoryObjMetadata(
            shape=shape,
            dtype=dtype,
            address=self._pool.base + view.chunk_offset,
            phy_size=self._chunk_size_bytes,
            ref_count=1,  # caller will ref_count_down to release the pin
            pin_count=0,
            fmt=fmt,
        )
        obj = TensorMemoryObj(
            raw_data=raw_data,
            metadata=meta,
            parent_allocator=_SlotRefcountAdapter(self, view.slot_idx, np_buf),
        )
        return obj

    def gpu_src_view(self, key: CacheEngineKey) -> Optional[Tuple[int, int]]:
        """Resolve a chunk's host source address for a GPU-direct H2D copy.

        Returns ``(n_bytes, src_host_ptr)`` where ``src_host_ptr`` points
        into the (``cudaHostRegister``'d) CXL pool, suitable as the source
        of an async ``cudaMemcpyAsync`` H2D. Returns ``None`` on miss.

        Unlike ``read_into`` / ``get_blocking`` this does NOT bump
        ``ref_count``: the L2-resident retrieve path relies on the
        ``pin_count`` held since ``lookup_and_lock`` to keep the slot alive
        for the duration of the DMA. The caller MUST hold that pin (i.e.
        have called ``pin(key)`` / ``contains(key, pin=True)``) and release
        it via ``unpin(key)`` only after the stream confirms the copy.

        Args:
            key: The chunk's cache key.

        Returns:
            ``(n_bytes, src_host_ptr)`` on hit, or ``None`` on miss.
        """
        view = self._index.lookup(key)
        if view is None:
            return None
        self._cache_slot(key, view.slot_idx)
        return view.chunk_len, self._pool.base + view.chunk_offset

    # -------- pin / unpin / remove --------------------------------------

    def pin(self, key: CacheEngineKey) -> bool:
        view = self._index.lookup(key)
        if view is None:
            return False
        ok = self._index_writer.pin(view.slot_idx)
        if ok:
            self._cache_slot(key, view.slot_idx)
        return ok

    def unpin(self, key: CacheEngineKey) -> bool:
        slot_idx = self._lookup_slot(key)
        if slot_idx is None:
            return False
        return self._index_writer.unpin(slot_idx)

    def pin_batch(self, keys: List[CacheEngineKey]) -> List[bool]:
        """Pin many keys with a single batched lock acquisition.

        Resolves each key to its slot, then pins all of them under one
        :meth:`CXLIndexWriter.pin_batch` (≈ one arbiter sweep total instead
        of one per key). Keys with no VALID slot report False.

        Args:
            keys: Keys to pin (one pin each).

        Returns:
            Per-key success flags, parallel to ``keys``.
        """
        results = [False] * len(keys)
        # Resolve keys to slots; track which input positions have a slot.
        positions: List[int] = []
        slot_idxs: List[int] = []
        for i, key in enumerate(keys):
            view = self._index.lookup(key)
            if view is None:
                continue
            positions.append(i)
            slot_idxs.append(view.slot_idx)
        if not slot_idxs:
            return results
        pinned = self._index_writer.pin_batch(slot_idxs)
        for pos, slot_idx, ok in zip(positions, slot_idxs, pinned, strict=True):
            if ok:
                self._cache_slot(keys[pos], slot_idx)
                results[pos] = True
        return results

    def unpin_batch(self, keys: List[CacheEngineKey]) -> List[bool]:
        """Unpin many keys with a single batched lock acquisition.

        Release counterpart to :meth:`pin_batch`. Keys with no known slot
        report False.

        Args:
            keys: Keys to unpin (one unpin each).

        Returns:
            Per-key success flags, parallel to ``keys``.
        """
        results = [False] * len(keys)
        positions: List[int] = []
        slot_idxs: List[int] = []
        for i, key in enumerate(keys):
            slot_idx = self._lookup_slot(key)
            if slot_idx is None:
                continue
            positions.append(i)
            slot_idxs.append(slot_idx)
        if not slot_idxs:
            return results
        unpinned = self._index_writer.unpin_batch(slot_idxs)
        for pos, ok in zip(positions, unpinned, strict=True):
            results[pos] = ok
        return results

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        slot_idx = self._lookup_slot(key)
        if slot_idx is None:
            return False
        ok, view = self._index_writer.evict(slot_idx)
        if not ok:
            return False
        assert view is not None
        try:
            self._heap.free(view.chunk_offset)
        except Exception:
            logger.exception(
                "heap.free failed for slot %d offset %d; leaking chunk",
                slot_idx,
                view.chunk_offset,
            )
        self._forget_slot(key)
        self._lru.forget(slot_idx)
        return True

    def clear(self) -> ClearResult:
        """Delete all of this node's chunks and trim its now-empty regions.

        Bulk teardown of this node's CXL residency, in three steps:

          1. ``index_writer.clear_owned_slots()`` tombstones every VALID slot
             this node owns (skipping busy ones) and returns their heap
             offsets.
          2. ``heap.free_batch()`` returns those offsets to the node heap's
             free-list, so the regions holding them become fully free.
          3. ``heap.trim()`` releases every fully-empty region back to the
             global pool.

        Busy chunks (pinned or mid-read) are left intact and reported in the
        result; any region still holding one of them will not be trimmed.
        Donor slots owned by other nodes in the shared index are untouched.

        Returns:
            ClearResult: counts of chunks deleted, chunks skipped because
            busy, and regions released to the pool.
        """
        freed_offsets, skipped_busy = self._index_writer.clear_owned_slots()
        if freed_offsets:
            try:
                self._heap.free_batch(freed_offsets)
            except Exception:
                logger.exception(
                    "heap.free_batch failed during clear() for %d offsets; "
                    "some chunks may leak",
                    len(freed_offsets),
                )
        # Every owned chunk is now tombstoned, so the whole hash->slot cache
        # and the LRU recency map are stale. Drop them wholesale.
        with self._key_to_slot_lock:
            self._key_to_slot.clear()
        self._lru.clear()
        released = self._heap.trim()
        logger.info(
            "CXLBackend.clear (node=%d): deleted %d chunks, skipped %d busy, "
            "released %d regions",
            self._node_id,
            len(freed_offsets),
            skipped_busy,
            len(released),
        )
        return ClearResult(
            chunks_deleted=len(freed_offsets),
            slots_skipped_busy=skipped_busy,
            regions_released=len(released),
        )

    # -------- AllocatorBackendInterface ---------------------------------

    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheMetadata
    ) -> MemoryAllocatorInterface:
        # Allocator was built in __init__; return the existing one.
        return self._mem_allocator

    def get_memory_allocator(self) -> MemoryAllocatorInterface:
        return self._mem_allocator

    def allocate(
        self,
        shapes,
        dtypes,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        obj = self._mem_allocator.allocate(shapes, dtypes, fmt)
        if obj is not None or not eviction:
            return obj
        # Eviction fallback is a step-5-scope simplification: we just
        # retry with eviction=False after a single LRU pass in later
        # slices. For now, return None on exhaustion.
        return None

    def batched_allocate(
        self,
        shapes,
        dtypes,
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[List[MemoryObj]]:
        return self._mem_allocator.batched_allocate(shapes, dtypes, batch_size, fmt)

    def calculate_chunk_budget(self) -> int:
        """Max in-flight chunks before we risk OOM on the CXL pool."""
        return self._heap.slots_per_region * self._pool.layout.region_count

    def get_allocator_backend(self) -> "AllocatorBackendInterface":
        # Per plan: peers borrow LocalCPUBackend, not us. But the
        # interface requires returning *something*. Returning self is
        # correct and peers should not be wired to use it.
        return self

    def close(self) -> None:
        if self._lock_manager is not None:
            self._lock_manager.stop()
            self._lock_manager = None
        if self._proc_lock_manager is not None:
            self._proc_lock_manager.stop()
            self._proc_lock_manager = None
        self._mem_allocator.close()
        self._pool.close()

    # -------- internals --------------------------------------------------

    def _cache_slot(self, key: CacheEngineKey, slot_idx: int) -> None:
        with self._key_to_slot_lock:
            self._key_to_slot[key.chunk_hash] = slot_idx
        # Every local resolution (read hit, pin, store) is an access: refresh
        # the slot's recency so warm chunks survive eviction. DRAM-only,
        # keyed by slot_idx so it is stable across key->slot re-caching.
        self._lru.touch(slot_idx)

    def _forget_slot(self, key: CacheEngineKey) -> None:
        with self._key_to_slot_lock:
            self._key_to_slot.pop(key.chunk_hash, None)

    def _forget_slot_by_hash(self, chunk_hash: int) -> None:
        """Drop the key->slot cache entry for a raw chunk_hash.

        Used by the eviction path, which learns the evicted slot's hash from
        the returned ``SlotView`` rather than a ``CacheEngineKey``. The stored
        hash is masked to u64 (matching ``_claim``), so mask here too.
        """
        with self._key_to_slot_lock:
            self._key_to_slot.pop(chunk_hash & 0xFFFFFFFFFFFFFFFF, None)

    def _lookup_slot(self, key: CacheEngineKey) -> Optional[int]:
        with self._key_to_slot_lock:
            slot_idx = self._key_to_slot.get(key.chunk_hash)
        if slot_idx is not None:
            return slot_idx
        # Miss in the cache — fall back to a probe.
        view = self._index.lookup(key)
        if view is None:
            return None
        self._cache_slot(key, view.slot_idx)
        return view.slot_idx


class _SlotRefcountAdapter(MemoryAllocatorInterface):
    """Parent-allocator shim for MemoryObjs returned from `get_blocking`.

    When the MemoryObj's ref_count drops to zero, TensorMemoryObj
    calls `parent_allocator.free(self)`. For a CXL HIT that means:
    drop our ref count on the slot, allowing a future EVICT. The
    numpy buffer is released as `self` is GC'd.
    """

    def __init__(
        self,
        backend: CXLBackend,
        slot_idx: int,
        np_buf,
    ):
        self._backend = backend
        self._slot_idx = slot_idx
        self._np_buf = np_buf  # keep alive

    # MemoryAllocatorInterface shims we don't use ----------------------

    def allocate(self, shapes, dtypes, fmt=MemoryFormat.UNDEFINED, allocator_type=None):
        raise NotImplementedError

    def batched_allocate(
        self,
        shapes,
        dtypes,
        batch_size,
        fmt=MemoryFormat.UNDEFINED,
        allocator_type=None,
    ):
        raise NotImplementedError

    def free(self, memory_obj: MemoryObj, allocator_type=None):
        memory_obj.invalidate()
        self._backend._index_writer.ref_count_down(self._slot_idx)

    def batched_free(self, memory_objs, allocator_type=None, update_stats=True):
        for obj in memory_objs:
            self.free(obj)
