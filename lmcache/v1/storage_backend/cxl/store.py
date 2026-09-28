# SPDX-License-Identifier: Apache-2.0
"""CXLStore: the synchronous object store over a CXL shared-memory pool.

Wires together the pool primitives:

- bootstrap.py (pool mmap + header + cudaHostRegister)
- locks.py + lock_manager.py (two-tier lock + arbiter)
- regions.py + heap.py (region and chunk allocators)
- index.py + index_writer.py (lock-free reads, lock-protected writes)

**Keyed on `ObjectKey`.** The store is an L2 tier only -- it is driven
exclusively by `CXLL2Adapter` and is not a `StorageBackendInterface`.
It therefore speaks the L2 key type directly rather than bridging
through `CacheEngineKey`, which previously discarded the `kv_rank` and
`cache_salt` that distinguish TP shards and tenants.

The shared CXL index addresses slots by a single u64
(``index.py``: ``line0.chunk_hash != needle`` -> keep probing), so the
full ObjectKey identity is folded into that u64 by
:func:`object_key_to_chunk_hash`. Anything folded out aliases in the
shared pool.

The store deliberately does NOT emit KVAdmitMsg/KVEvictMsg -- CXL state
is discovered via CXL_LOOKUP.
"""

# Standard
import threading
import time
from dataclasses import dataclass
from hashlib import blake2b
from typing import Dict, List, Optional, Sequence, Tuple

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.memory_management import (
    MemoryObj,
)
from lmcache.v1.storage_backend.cxl.bootstrap import (
    CXLBootstrapConfig,
    PoolHandle,
    bootstrap_pool,
)
from lmcache.v1.storage_backend.cxl.heap import OutOfChunks
from lmcache.v1.storage_backend.cxl.heap_set import HeapSet
from lmcache.v1.storage_backend.cxl.index import CXLIndex
from lmcache.v1.storage_backend.cxl.index_writer import (
    CXLIndexWriter,
    ReserveOutcome,
    ReserveResult,
)
from lmcache.v1.storage_backend.cxl.layout import GEOM_HASH_SIZE, OWNER_FREE
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


def object_key_to_chunk_hash(key: ObjectKey) -> int:
    """Derive the u64 CXL index hash from an ObjectKey's full identity.

    The CXL index addresses slots by a single u64. That u64 must
    distinguish every key that maps to a *different* KV payload,
    because the index probe compares only this value
    (``index.py``: ``line0.chunk_hash != needle`` -> keep probing).
    Anything folded out here aliases in the shared pool.

    Three ObjectKey fields beyond the content hash are therefore
    mixed in:

    - ``kv_rank``: under TP>1 the serving engine emits one ObjectKey
      per rank that differ *only* in this field -- the token hash
      carries no rank (see ``ipc_key_to_object_keys``). Folding it
      out would alias every rank's shard onto one slot, so a rank-1
      retrieve could be served rank-0's bytes.
    - ``model_name``: distinct models must not share a slot.
    - ``cache_salt``: per-user isolation (different users, same
      content, different keys).

    ``\\x00`` separators frame the variable-length fields.
    ``ObjectKey.__post_init__`` forbids ``@`` in ``model_name`` and
    ``@/\\`` plus NUL in ``cache_salt``, so NUL is an unambiguous
    delimiter that cannot appear inside a field.

    Args:
        key: The L2 object key to hash.

    Returns:
        A 64-bit unsigned index hash.
    """
    return int.from_bytes(_tenant_digest(key, digest_size=8), "little")


def object_key_to_tenant_digest(key: ObjectKey, geometry_salt: bytes = b"") -> bytes:
    """Derive the 16-byte tenant discriminator stored in a slot.

    The u64 index hash alone cannot safely identify a slot: two distinct
    tenants that collide on it would produce a false hit, because the
    probe compares only that value. This wider digest is stamped into
    ``line0.geom_hash`` at insert and compared on every probe, so a
    collision falls through to the next slot instead of returning
    another tenant's bytes.

    It covers the same identity as :func:`object_key_to_chunk_hash` --
    ``chunk_hash``, ``model_name``, ``kv_rank``, ``cache_salt`` -- at 16
    bytes rather than 8. At 2**20 resident chunks the residual
    false-hit probability across both fields is ~2**-88.

    ``geometry_salt``, when supplied, folds the KV geometry the bytes
    were written under into the digest. Two nodes running the same
    model under *different* geometry (e.g. a dtype mismatch) then
    produce different digests for the same key, so neither reads the
    other's chunks -- they miss instead of misinterpreting. The caller
    supplies it because geometry is known to the adapter (from KV-cache
    registration), not to the key.

    It defaults to empty so a caller with no geometry to declare still
    gets a well-defined digest; identity separation does not depend on
    it.

    Args:
        key: The L2 object key to digest.
        geometry_salt: Optional digest of the KV geometry these bytes
            were written under.

    Returns:
        A 16-byte tenant discriminator.
    """
    return _tenant_digest(key, digest_size=GEOM_HASH_SIZE, geometry_salt=geometry_salt)


def _tenant_digest(
    key: ObjectKey, digest_size: int, geometry_salt: bytes = b""
) -> bytes:
    """Digest an ObjectKey's tenant identity at the requested width.

    ``\\x00`` separators frame the variable-length fields.
    ``ObjectKey.__post_init__`` forbids ``@`` in ``model_name`` and
    ``@/\\`` plus NUL in ``cache_salt``, so NUL is an unambiguous
    delimiter that cannot appear inside a field.

    Args:
        key: The L2 object key to digest.
        digest_size: Output width in bytes.
        geometry_salt: Optional trailing field; see
            :func:`object_key_to_tenant_digest`.

    Returns:
        The digest bytes.
    """
    h = blake2b(digest_size=digest_size)
    h.update(key.chunk_hash)
    h.update(b"\x00")
    h.update(key.model_name.encode("utf-8"))
    h.update(b"\x00")
    h.update(key.kv_rank.to_bytes(8, "little"))
    h.update(b"\x00")
    h.update(key.cache_salt.encode("utf-8"))
    if geometry_salt:
        h.update(b"\x00")
        h.update(geometry_salt)
    return h.digest()


@dataclass
class CXLStoreConfig:
    """Construction-time config for CXLStore.

    In the real deployment these come from the ``--l2-adapter`` JSON via
    ``CXLL2AdapterConfig``; the dataclass here is what the store actually
    needs and keeps tests decoupled from the full config surface.
    """

    dev_path: str
    node_id: int
    region_size: int = 256 * 1024 * 1024
    # Optional upper bound on a single chunk, as a guard against a garbled
    # geometry asking for an absurd slab. Not a size to tune: heap classes
    # are created from the exact byte size of what is actually stored.
    # None disables the check (a class larger than region_size still fails
    # loudly when its heap is created).
    max_chunk_size_bytes: Optional[int] = None
    initialize: bool = False
    generation: int = 1
    run_lock_manager: bool = True
    pool_size_override: Optional[int] = None
    max_nodes: Optional[int] = None
    num_locks: Optional[int] = None
    # Node-local LRU eviction (see CXLStore eviction ladder). When a store
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
    """Outcome of :meth:`CXLStore.clear`.

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


class CXLStore:
    """CXL shared-memory L2 tier: a content-addressed KV object store.

    One store per MP server process, driven by :class:`CXLL2Adapter`.
    It is deliberately **not** a ``StorageBackendInterface``: nothing
    registers it as an in-process storage backend, and the read path
    copies into a caller-owned buffer (:meth:`read_into`) or hands the
    caller a source pointer (:meth:`gpu_src_view`) rather than
    allocating and owning ``MemoryObj``s.

    The store never interprets chunk bytes. A slot carries a byte
    length and a format tag; callers reinterpret via a GPUConnector
    that knows the KV geometry.

    Lifecycle:
      1. __init__ takes a CXLStoreConfig and runs the full bootstrap.
         After it returns, the pool is mmap'd, cudaHostRegister'd (best
         effort), the lock manager is running (if enabled), and the
         store is ready to serve put/read.
      2. close() stops the lock manager and closes the pool. It does NOT
         initialize the pool on close — a restart of the same node
         should re-attach to the existing header.
    """

    def __init__(self, cxl_config: CXLStoreConfig):
        self._node_id = cxl_config.node_id
        self._max_chunk_size_bytes = cxl_config.max_chunk_size_bytes
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
        self._pool: PoolHandle = bootstrap_pool(bootstrap_cfg)
        self._fence = None  # default StubFence; explicit None == use module default

        # Two-tier lock shared across all subsystems on this node.
        self._lock = TwoTierLock(self._pool, node_id=self._node_id)

        # Lock manager (single-writer arbiter). Exactly one node per rack
        # runs it, selected by config (`run_lock_manager`); there is no
        # election. A second arbiter on the same pool is a correctness
        # bug -- both would drive WAITING -> LOCKED transitions.
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

        # Region + per-geometry heap classes. A class is created on first
        # use from the exact byte size of the chunk being stored, so one
        # pool serves many models/TP degrees without any size declared up
        # front. ``max_chunk_size_bytes``, when set, is only a guard.
        self._region_allocator = RegionAllocator(self._pool, self._lock)
        self._heaps = HeapSet(self._region_allocator, node_id=self._node_id)

        # Index reader + writer.
        self._index = CXLIndex(self._pool)
        self._index_writer = CXLIndexWriter(
            self._pool, self._index, self._lock, self._node_id
        )

        # Map: chunk_hash -> slot_idx, for quick unpin/lookup by key.
        # The index itself is the source of truth; this is an O(1)
        # cache to skip probing when the caller already knows the key.
        # Keyed on the u64 (not the ObjectKey) because eviction
        # invalidates entries from the slot's own stored hash, which is
        # all it can read back from CXL.
        self._key_to_slot: dict[int, int] = {}
        self._key_to_slot_lock = threading.Lock()

        # Per-model KV geometry digests, declared via `set_geometry` when
        # the serving engine registers its KV caches. Folded into each
        # chunk's tenant digest so a node running the same model under a
        # different geometry misses rather than misreads. Empty until
        # declared, which is the correct default: a pool whose writers
        # never declare geometry behaves exactly as before.
        self._geometry: Dict[str, bytes] = {}
        self._geometry_lock = threading.Lock()

        # Node-local LRU recency over the slots this node owns. Touched on every
        # local read/commit (DRAM only, no CXL write) and consulted only when a
        # store must evict to reclaim free-list space after region exhaustion.
        self._lru = NodeLRUTracker()

        logger.info(
            "CXLStore ready: node_id=%d pool_size=%d regions=%d",
            self._node_id,
            self._pool.size,
            self._pool.layout.region_count,
        )

    # -------- accessors --------------------------------------------------
    #
    # The cross-node donor/requester machinery (CXLDonor, remote_fetch) is
    # built on the pool primitives directly rather than on the store, so the
    # adapter needs to hand them out. These are read-only views of state the
    # store owns; they exist so callers never reach into private members.

    @property
    def node_id(self) -> int:
        """This node's id within the rack-wide ``max_nodes`` space."""
        return self._node_id

    @property
    def max_chunk_size_bytes(self) -> Optional[int]:
        """Optional upper bound on one chunk's size, or None."""
        return self._max_chunk_size_bytes

    @property
    def pool(self) -> PoolHandle:
        """The mmap'd pool handle (header, layout, base address)."""
        return self._pool

    @property
    def index_writer(self) -> CXLIndexWriter:
        """The lock-protected index writer over this pool."""
        return self._index_writer

    @property
    def heaps(self) -> HeapSet:
        """This node's per-geometry chunk free-lists."""
        return self._heaps

    def set_geometry(self, model_name: str, geometry_salt: bytes) -> None:
        """Declare the KV geometry `model_name` writes under.

        The salt is folded into the tenant digest of every chunk for that
        model, so a peer that wrote the same model under a different
        geometry produces different digests and is simply missed rather
        than read back misinterpreted.

        Declaring a *different* salt for a model that already has one is
        a programming error, not a peer disagreement: it would silently
        orphan every chunk this node already wrote.

        Args:
            model_name: The model these bytes belong to.
            geometry_salt: A digest of its KV geometry.

        Raises:
            ValueError: If a conflicting salt was already declared for
                this model in this process.
        """
        with self._geometry_lock:
            existing = self._geometry.get(model_name)
            if existing is not None and existing != geometry_salt:
                raise ValueError(
                    f"geometry for model {model_name!r} already declared as "
                    f"{existing.hex()}; refusing to redeclare as "
                    f"{geometry_salt.hex()}"
                )
            self._geometry[model_name] = geometry_salt

    def tenant_digest_for(self, key: ObjectKey) -> bytes:
        """Return `key`'s tenant digest, salted by its model's geometry.

        Exposed so the cross-node requester stamps reserved slots with
        the same digest this store would, rather than the bare identity
        digest.

        Args:
            key: The object key to digest.

        Returns:
            The 16-byte discriminator for this key on this node.
        """
        return self._digest_for(key)

    def _digest_for(self, key: ObjectKey) -> bytes:
        """Return `key`'s tenant digest, salted by its model's geometry."""
        with self._geometry_lock:
            salt = self._geometry.get(key.model_name, b"")
        return object_key_to_tenant_digest(key, salt)

    def slot_index_of(self, key: ObjectKey) -> Optional[int]:
        """Return the index slot currently holding `key`, or None on miss.

        A read-only probe exposed for diagnostics and tests that need to
        inspect on-CXL slot state (pin/ref counts) for a key.

        Args:
            key: The object key to resolve.

        Returns:
            The slot index of the VALID slot for this key, else None.
        """
        view = self._index.lookup_by_hash(
            object_key_to_chunk_hash(key), self._digest_for(key)
        )
        return None if view is None else view.slot_idx

    @property
    def region_allocator(self) -> RegionAllocator:
        """The shared region allocator over this pool's bitmap."""
        return self._region_allocator

    @property
    def epoch(self) -> int:
        """The pool's current generation, as stamped in the header."""
        return int(self._pool.header.gen)

    # -------- read path --------------------------------------------------

    def contains(self, key: ObjectKey, pin: bool = False) -> bool:
        """Report whether the pool holds `key`, optionally pinning it.

        Args:
            key: The object key to look up.
            pin: When True, take a pin on a hit so the slot cannot be
                evicted until :meth:`unpin`. A pin that loses a race with
                a concurrent eviction reports a miss.

        Returns:
            True if a VALID slot holds this key (and, when ``pin``, the
            pin was taken).
        """
        chunk_hash = object_key_to_chunk_hash(key)
        view = self._index.lookup_by_hash(chunk_hash, self._digest_for(key))
        if view is None:
            return False
        if pin and not self._index_writer.pin(view.slot_idx):
            # Slot flipped (evicted) between lookup and pin.
            return False
        self._cache_slot(chunk_hash, view.slot_idx)
        return True

    def read_into(
        self,
        key: ObjectKey,
        dst_ptr: int,
        dst_size: int,
        phase_ns: Optional[List[int]] = None,
    ) -> int:
        """Fast hit path: memcpy a chunk directly into `dst_ptr`.

        The store never builds a Python-level view of the chunk: the
        caller owns the destination buffer and reinterprets the bytes
        via a GPUConnector that knows the KV geometry.

        Args:
            key: The object key to read.
            dst_ptr: Destination host address.
            dst_size: Destination capacity in bytes; the copy is capped
                at this and at the slot's own ``chunk_len``.
            phase_ns: Optional 11-slot list accumulating the timings below.

        Returns:
            Bytes copied, or 0 on miss.

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

        chunk_hash = object_key_to_chunk_hash(key)
        tenant_digest = self._digest_for(key)
        t0 = time.perf_counter_ns()
        view = self._index.lookup_by_hash(chunk_hash, tenant_digest)
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
            post = self._index.lookup_by_hash(chunk_hash, tenant_digest)
            t3 = time.perf_counter_ns()
            if phase_ns is not None:
                phase_ns[2] += t3 - t2
            if post is None or post.slot_idx != view.slot_idx:
                return 0
            self._cache_slot(chunk_hash, view.slot_idx)
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

    # -------- write path -------------------------------------------------

    def put_batch(
        self,
        keys: Sequence[ObjectKey],
        objs: List[MemoryObj],
    ) -> None:
        """Store a batch of chunks into the pool. Completes inline.

        The batch lands whole or not at all: when the pool is exhausted,
        this evicts the node's cold chunks *once* for the whole batch,
        and drops the entire batch if that still cannot free room. A
        per-key failure is logged and skipped, leaving the rest stored.

        Args:
            keys: Object keys to store, positionally matching ``objs``.
            objs: Source memory objects. The caller retains ownership;
                the store copies out of them and never frees them.

        Raises:
            ValueError: If ``keys`` and ``objs`` differ in length.
        """
        if len(keys) != len(objs):
            raise ValueError(f"put_batch: {len(keys)} keys vs {len(objs)} objs")
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
            return
        for key, obj in zip(keys, objs, strict=True):
            try:
                self._put_one(key, obj)
            except Exception:
                logger.exception("failed to put key %s; skipping", key)

    def _put_one(self, key: ObjectKey, src: MemoryObj) -> None:
        """Full INSERT lifecycle for one key.

        Steps:
          1. reserve_slot; if ALREADY_PRESENT, skip; if WAIT, poll.
          2. Allocate a chunk from the heap.
          3. Copy bytes from src.raw_data into the chunk.
          4. commit_slot (publish VALID).
        On any failure, release_slot and free the chunk.

        Args:
            key: The object key to store under.
            src: Source memory object; the caller retains ownership.

        Raises:
            ValueError: If the payload exceeds the pool's chunk size.
        """
        chunk_hash = object_key_to_chunk_hash(key)
        result = self._reserve_with_retries(chunk_hash, self._digest_for(key))
        if result.outcome == ReserveOutcome.ALREADY_PRESENT:
            if result.slot_idx is not None:
                self._cache_slot(chunk_hash, result.slot_idx)
            return
        if result.outcome == ReserveOutcome.INDEX_FULL:
            logger.warning("CXL index full; put for key %s dropped", key)
            return
        if result.slot_idx is None:
            return
        slot_idx = result.slot_idx

        size_bytes = src.get_size()
        if self._max_chunk_size_bytes and size_bytes > self._max_chunk_size_bytes:
            self._index_writer.release_slot(slot_idx)
            raise ValueError(
                f"payload {size_bytes} bytes exceeds max_chunk_size_bytes "
                f"{self._max_chunk_size_bytes}"
            )

        try:
            chunk_offset = self._alloc_chunk_with_eviction(size_bytes)
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
            self._cache_slot(chunk_hash, slot_idx)
        except Exception:
            self._heaps.free(chunk_offset)
            self._index_writer.release_slot(slot_idx)
            raise

    def _alloc_chunk_with_eviction(self, size_bytes: int) -> int:
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

        Args:
            size_bytes: Exact chunk size; selects the heap class.

        Returns:
            A pool-relative chunk offset.

        Raises:
            NoRegionAvailable: If the pool is exhausted and eviction could not
                free any local slot.
        """
        try:
            return self._heaps.alloc(size_bytes)
        except NoRegionAvailable:
            self._evict_cold_slots(target=1)
            try:
                return self._heaps.alloc_no_claim(size_bytes)
            except OutOfChunks:
                # Nothing evictable in this class (all cold slots pinned or
                # in-flight), or eviction freed slots of a different size.
                logger.warning(
                    "CXL node %d: pool exhausted and no evictable local chunk "
                    "for size %d; dropping store",
                    self._node_id,
                    size_bytes,
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
        occupied, total = self._heaps.occupancy()
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
        occupied_after, total_after = self._heaps.occupancy()
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

        Freed chunks return to their own class's free-list for immediate reuse;
        regions are never released here.

        **Eviction is class-agnostic**: it drops the globally coldest chunks by
        LRU, which may belong to a different size class than the store that
        triggered it. Those freed slots do not help the triggering store — its
        retry uses ``alloc_no_claim`` on its own class. This is deliberate:
        evicting by recency across the whole node is the right global policy,
        and the freed regions become claimable once a class trims them. The
        consequence is that a store can still fail after a successful eviction
        pass when the pool is full of *other* classes; the caller drops that
        store and a later one succeeds once ``trim`` returns the regions.

        Args:
            target: Minimum number of slots to free (the store needs this many;
                a single ``_put_one`` needs 1). Eviction also honors the low
                watermark, so it may free more than ``target`` when occupancy is
                above the floor.

        Returns:
            The number of chunks actually evicted this pass.
        """
        occupied, total = self._heaps.occupancy()
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
                self._heaps.free(view.chunk_offset)
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
        self,
        chunk_hash: int,
        tenant_digest: bytes,
        max_waits: int = 100,
        wait_sleep_s: float = 0.001,
    ) -> ReserveResult:
        for _ in range(max_waits):
            result = self._index_writer.reserve_slot(chunk_hash, tenant_digest)
            if result.outcome != ReserveOutcome.WAIT_FOR_OTHER:
                return result
            # Another writer is on this key; poll briefly and try again.
            time.sleep(wait_sleep_s)
        return ReserveResult(ReserveOutcome.WAIT_FOR_OTHER, result.slot_idx)

    def _copy_into_chunk(
        self, src: MemoryObj, chunk_offset: int, size_bytes: int
    ) -> None:
        """Copy `size_bytes` from src.raw_data into the CXL chunk.

        `src.raw_data` is typically a uint8 flat tensor per LocalCPU's
        allocator, but arbitrary shape/dtype is handled by flattening
        its byte view.

        Args:
            src: Source memory object to copy from.
            chunk_offset: Pool-relative destination offset.
            size_bytes: Number of bytes to copy.
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

    def gpu_src_view(self, key: ObjectKey) -> Optional[Tuple[int, int]]:
        """Resolve a chunk's host source address for a GPU-direct H2D copy.

        Returns ``(n_bytes, src_host_ptr)`` where ``src_host_ptr`` points
        into the (``cudaHostRegister``'d) CXL pool, suitable as the source
        of an async ``cudaMemcpyAsync`` H2D. Returns ``None`` on miss.

        Unlike :meth:`read_into` this does NOT bump
        ``ref_count``: the L2-resident retrieve path relies on the
        ``pin_count`` held since ``lookup_and_lock`` to keep the slot alive
        for the duration of the DMA. The caller MUST hold that pin (i.e.
        have called ``pin(key)`` / ``contains(key, pin=True)``) and release
        it via ``unpin(key)`` only after the stream confirms the copy.

        Args:
            key: The chunk's object key.

        Returns:
            ``(n_bytes, src_host_ptr)`` on hit, or ``None`` on miss.
        """
        chunk_hash = object_key_to_chunk_hash(key)
        view = self._index.lookup_by_hash(chunk_hash, self._digest_for(key))
        if view is None:
            return None
        self._cache_slot(chunk_hash, view.slot_idx)
        return view.chunk_len, self._pool.base + view.chunk_offset

    # -------- pin / unpin / remove --------------------------------------

    def pin(self, key: ObjectKey) -> bool:
        """Pin one key so its slot cannot be evicted until :meth:`unpin`.

        Args:
            key: The object key to pin.

        Returns:
            True if a VALID slot was found and pinned.
        """
        chunk_hash = object_key_to_chunk_hash(key)
        view = self._index.lookup_by_hash(chunk_hash, self._digest_for(key))
        if view is None:
            return False
        ok = self._index_writer.pin(view.slot_idx)
        if ok:
            self._cache_slot(chunk_hash, view.slot_idx)
        return ok

    def unpin(self, key: ObjectKey) -> bool:
        """Release one pin taken by :meth:`pin` or ``contains(pin=True)``.

        Args:
            key: The object key to unpin.

        Returns:
            True if a slot was found and its pin released.
        """
        slot_idx = self._lookup_slot(key)
        if slot_idx is None:
            return False
        return self._index_writer.unpin(slot_idx)

    def pin_batch(self, keys: List[ObjectKey]) -> List[bool]:
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
        hashes: List[int] = []
        for i, key in enumerate(keys):
            chunk_hash = object_key_to_chunk_hash(key)
            view = self._index.lookup_by_hash(chunk_hash, self._digest_for(key))
            if view is None:
                continue
            positions.append(i)
            slot_idxs.append(view.slot_idx)
            hashes.append(chunk_hash)
        if not slot_idxs:
            return results
        pinned = self._index_writer.pin_batch(slot_idxs)
        for pos, chunk_hash, slot_idx, ok in zip(
            positions, hashes, slot_idxs, pinned, strict=True
        ):
            if ok:
                self._cache_slot(chunk_hash, slot_idx)
                results[pos] = True
        return results

    def unpin_batch(self, keys: List[ObjectKey]) -> List[bool]:
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

    def remove(self, key: ObjectKey) -> bool:
        """Evict one key's slot and return its chunk to the node heap.

        Refuses a slot that is pinned or has an in-flight read
        (``ref_count != 0``); the caller sees ``False``.

        Args:
            key: The object key to evict.

        Returns:
            True if the slot was tombstoned and its chunk freed.
        """
        chunk_hash = object_key_to_chunk_hash(key)
        slot_idx = self._lookup_slot(key)
        if slot_idx is None:
            return False
        ok, view = self._index_writer.evict(slot_idx)
        if not ok:
            return False
        if view is None:
            raise RuntimeError(
                f"index_writer.evict reported success for slot {slot_idx} "
                "but returned no SlotView"
            )
        try:
            self._heaps.free(view.chunk_offset)
        except Exception:
            logger.exception(
                "heap.free failed for slot %d offset %d; leaking chunk",
                slot_idx,
                view.chunk_offset,
            )
        self._forget_slot_by_hash(chunk_hash)
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
                self._heaps.free_batch(freed_offsets)
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
        released = self._heaps.trim()
        logger.info(
            "CXLStore.clear (node=%d): deleted %d chunks, skipped %d busy, "
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

    def close(self) -> None:
        if self._lock_manager is not None:
            self._lock_manager.stop()
            self._lock_manager = None
        if self._proc_lock_manager is not None:
            self._proc_lock_manager.stop()
            self._proc_lock_manager = None
        self._pool.close()

    # -------- internals --------------------------------------------------

    def _cache_slot(self, chunk_hash: int, slot_idx: int) -> None:
        with self._key_to_slot_lock:
            self._key_to_slot[chunk_hash & 0xFFFFFFFFFFFFFFFF] = slot_idx
        # Every local resolution (read hit, pin, store) is an access: refresh
        # the slot's recency so warm chunks survive eviction. DRAM-only,
        # keyed by slot_idx so it is stable across key->slot re-caching.
        self._lru.touch(slot_idx)

    def _forget_slot_by_hash(self, chunk_hash: int) -> None:
        """Drop the key->slot cache entry for a raw chunk_hash.

        Used by the eviction path, which learns the evicted slot's hash from
        the returned ``SlotView`` rather than an ``ObjectKey``. The stored
        hash is masked to u64 (matching ``_claim``), so mask here too.
        """
        with self._key_to_slot_lock:
            self._key_to_slot.pop(chunk_hash & 0xFFFFFFFFFFFFFFFF, None)

    def _lookup_slot(self, key: ObjectKey) -> Optional[int]:
        """Resolve `key` to its slot, via the DRAM cache then a probe.

        Args:
            key: The object key to resolve.

        Returns:
            The slot index holding this key, or None on miss.
        """
        needle = object_key_to_chunk_hash(key) & 0xFFFFFFFFFFFFFFFF
        with self._key_to_slot_lock:
            slot_idx = self._key_to_slot.get(needle)
        if slot_idx is not None:
            return slot_idx
        # Miss in the cache — fall back to a probe.
        view = self._index.lookup_by_hash(needle, self._digest_for(key))
        if view is None:
            return None
        self._cache_slot(needle, view.slot_idx)
        return view.slot_idx
