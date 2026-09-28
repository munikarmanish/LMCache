# SPDX-License-Identifier: Apache-2.0
"""Per-node DRAM free-list heap over owned CXL regions.

Plan reference: F4 and "Global allocator state". This is the hot path
for every `INSERT` — the backend calls `heap.alloc(size)` to place a
new chunk. The heap only touches DRAM; it does not cross the region
lock except when it has to claim a new region from the global pool
(~once per region_size / chunk_size chunks).

Design:

- **Fixed chunk size per heap.** LMCache chunks within a model config
  are effectively uniform (kv_size × num_layers × chunk_tokens ×
  hidden_dim × dtype_size). The heap is initialized with one chunk
  size and rejects allocations for other sizes. Multiple sizes are
  served by multiple heaps.
- **Slab per region.** A claimed region is carved into
  `region_size // chunk_size` fixed slots. On claim, all slots are
  pushed onto a free-list. No coalescing is needed — all slots are
  the same size.
- **Eager region reclamation is OFF by default.** When a region
  becomes empty we keep it; returning to the pool would cause churn.
  Callers can invoke `trim()` to shed fully-empty regions if memory
  pressure rises.

The heap returns **pool-relative offsets** (bytes from `pool_base`).
These are what the index's `chunk_offset` field wants.
"""

# Standard
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.cxl.regions import (
    RegionAllocator,
)

logger = init_logger(__name__)


class OutOfChunks(Exception):
    """Raised when `alloc()` fails and eviction is disabled."""


@dataclass
class HeapStats:
    total_regions: int
    total_slots: int
    free_slots: int
    chunk_size: int


class NodeHeap:
    """DRAM free-list over one node's claimed regions, one chunk size.

    One heap serves exactly one chunk size. A pool serving several
    geometries owns several heaps, multiplexed by
    :class:`~lmcache.v1.storage_backend.cxl.heap_set.HeapSet`; a claimed
    region belongs to exactly one heap for its lifetime, so an offset
    identifies its chunk size unambiguously.

    ``chunk_size`` need not divide ``region_size``: the region is carved
    into ``region_size // chunk_size`` slots and the remainder is left
    unused. That tail is bounded by one slot per region (~1.6% for a
    6 MiB chunk in a 256 MiB region). Rounding the chunk size up to
    divide evenly would instead waste space in *every* chunk, which is
    strictly worse for the exact-fit sizes this heap is given.

    Thread-safe: all methods take an internal mutex. Contention is
    expected to be low because there is one heap per node per chunk
    size, and the critical sections are O(1).
    """

    def __init__(
        self,
        region_allocator: RegionAllocator,
        node_id: int,
        chunk_size: int,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        rsize = region_allocator.region_size()
        if chunk_size > rsize:
            raise ValueError(
                f"chunk_size {chunk_size} exceeds region_size {rsize}; a chunk "
                "cannot span regions. Increase cxl_region_size."
            )

        self._regions = region_allocator
        self._node_id = node_id
        self._chunk_size = chunk_size
        self._slots_per_region = rsize // chunk_size

        # DRAM state.
        self._lock = threading.Lock()
        self._free: Deque[int] = deque()  # pool-relative offsets
        # Map of region_id -> count of currently-free slots in that region.
        self._free_count_per_region: Dict[int, int] = {}
        # Regions we currently own, in claim order.
        self._owned: List[int] = []

    # -------- public API -------------------------------------------------

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def slots_per_region(self) -> int:
        return self._slots_per_region

    def alloc(self) -> int:
        """Return a pool-relative offset for one free chunk.

        Claims a new region from the pool if the free-list is empty.
        Raises `NoRegionAvailable` if the pool is out of regions.
        """
        with self._lock:
            if not self._free:
                self._claim_and_seed_region_locked()
            return self._pop_free_locked()

    def alloc_no_claim(self) -> int:
        """Return a pool-relative offset for one free chunk, never claiming.

        Free-list-only counterpart of :meth:`alloc`: it serves from the
        existing free-list and never claims a new region from the global pool.
        The backend uses it to retry a store after eviction has refilled the
        free-list, so a slot freed by evicting a cold chunk is reused in place
        rather than growing this node's region footprint.

        Returns:
            A pool-relative offset for a free chunk.

        Raises:
            OutOfChunks: If the free-list is empty (nothing was evicted, or the
                freed slots were consumed by a concurrent alloc).
        """
        with self._lock:
            if not self._free:
                raise OutOfChunks("no free chunk without claiming a new region")
            return self._pop_free_locked()

    def alloc_batch(self, n: int) -> List[int]:
        """Allocate `n` chunks in one lock acquisition.

        Matches the backend's batched INSERT path. Falls back to
        single `alloc()` semantics if the free-list can't be refilled
        enough; in that case, the first `NoRegionAvailable` propagates
        and any chunks already handed out in this call are NOT freed
        (caller should catch and back them out).
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        out: List[int] = []
        with self._lock:
            while len(out) < n:
                if not self._free:
                    self._claim_and_seed_region_locked()
                out.append(self._pop_free_locked())
        return out

    def alloc_batch_no_claim(self, n: int) -> List[int]:
        """Allocate `n` chunks from the free-list only, never claiming.

        Free-list-only counterpart of :meth:`alloc_batch`, used by the
        post-eviction store retry. Serves from the existing free-list and
        never claims a new region. If the free-list runs dry before `n`
        chunks are handed out, the offsets already popped in this call are
        returned to the free-list before raising, so the heap is left
        unchanged (unlike :meth:`alloc_batch`, whose partial hand-outs the
        caller must back out).

        Args:
            n: Number of chunks to allocate.

        Returns:
            A list of `n` pool-relative offsets.

        Raises:
            ValueError: If `n <= 0`.
            OutOfChunks: If the free-list holds fewer than `n` chunks. The
                heap is left unmodified in this case.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        out: List[int] = []
        with self._lock:
            while len(out) < n:
                if not self._free:
                    # Roll back this call's partial hand-outs so the heap is
                    # unchanged, then signal the caller to drop the batch.
                    self._return_free_locked(out)
                    raise OutOfChunks(
                        f"free-list has {len(out)} of {n} requested chunks "
                        "without claiming a new region"
                    )
                out.append(self._pop_free_locked())
        return out

    def _pop_free_locked(self) -> int:
        """Pop one offset off the free-list and update occupancy. Holds lock."""
        off = self._free.popleft()
        region_id = self._region_for_offset(off)
        self._free_count_per_region[region_id] -= 1
        return off

    def _return_free_locked(self, offsets: List[int]) -> None:
        """Return already-popped offsets to the free-list. Holds lock."""
        for off in offsets:
            region_id = self._region_for_offset(off)
            self._free.append(off)
            self._free_count_per_region[region_id] += 1

    def free(self, offset: int) -> None:
        """Return one chunk (by pool-relative offset) to the free-list.

        Caller is responsible for any fences on CXL side — the heap
        itself lives in DRAM and is unaware of the chunk's contents.
        """
        region_id = self._region_for_offset(offset)
        with self._lock:
            if region_id not in self._free_count_per_region:
                raise ValueError(
                    f"offset {offset} falls in region {region_id}, "
                    "which this heap does not own"
                )
            if self._free_count_per_region[region_id] >= self._slots_per_region:
                raise ValueError(
                    f"double-free detected: region {region_id} already fully free"
                )
            self._free.append(offset)
            self._free_count_per_region[region_id] += 1

    def free_batch(self, offsets: List[int]) -> None:
        with self._lock:
            for offset in offsets:
                region_id = self._region_for_offset(offset)
                if region_id not in self._free_count_per_region:
                    raise ValueError(
                        f"offset {offset} falls in region {region_id}, "
                        "which this heap does not own"
                    )
                if self._free_count_per_region[region_id] >= self._slots_per_region:
                    raise ValueError(
                        f"double-free detected in batch: region {region_id} "
                        "already fully free"
                    )
                self._free.append(offset)
                self._free_count_per_region[region_id] += 1

    def trim(self) -> List[int]:
        """Release fully-empty regions back to the global pool.

        Returns the list of region_ids released. Useful under memory
        pressure; not called automatically so steady-state workloads
        don't churn the global allocator.
        """
        released: List[int] = []
        with self._lock:
            # Identify fully-free regions.
            victims = [
                rid
                for rid in list(self._owned)
                if self._free_count_per_region.get(rid, 0) == self._slots_per_region
            ]
            for rid in victims:
                # Remove the region's offsets from the free deque.
                region_base = self._regions.region_address(rid) - self._pool_base()
                region_hi = region_base + self._regions.region_size()
                self._free = deque(
                    o for o in self._free if not (region_base <= o < region_hi)
                )
                self._free_count_per_region.pop(rid, None)
                self._owned.remove(rid)
                self._regions.release(rid, self._node_id)
                released.append(rid)
        return released

    def occupancy(self) -> tuple[int, int]:
        """Return ``(occupied_slots, total_slots)`` across owned regions.

        Both counts are DRAM-only and cover only this node's currently-claimed
        regions: ``total = len(owned) * slots_per_region`` and
        ``occupied = total - len(free_list)``. The backend's eviction loop uses
        the ratio as a node-local usage watermark that eviction can actually
        lower — freeing a chunk moves it from occupied to free without changing
        ``total`` (regions are never released on this path).

        Returns:
            A ``(occupied_slots, total_slots)`` pair. ``total_slots`` is 0 when
            this node owns no regions, in which case the caller treats
            occupancy as 0% (nothing local to evict).
        """
        with self._lock:
            total = len(self._owned) * self._slots_per_region
            occupied = total - len(self._free)
            return occupied, total

    def stats(self) -> HeapStats:
        with self._lock:
            total_slots = len(self._owned) * self._slots_per_region
            free_slots = len(self._free)
            return HeapStats(
                total_regions=len(self._owned),
                total_slots=total_slots,
                free_slots=free_slots,
                chunk_size=self._chunk_size,
            )

    def owned_regions(self) -> List[int]:
        with self._lock:
            return list(self._owned)

    # -------- internals --------------------------------------------------

    def _pool_base(self) -> int:
        return self._regions.handle.base

    def _region_for_offset(self, offset: int) -> int:
        off_regions = self._regions.handle.layout.off_regions
        region_size = self._regions.region_size()
        if offset < off_regions:
            raise ValueError(
                f"offset {offset} precedes regions section at {off_regions}"
            )
        return (offset - off_regions) // region_size

    def _claim_and_seed_region_locked(self) -> None:
        """Caller must hold self._lock."""
        region_id = self._regions.claim(self._node_id)
        self._owned.append(region_id)
        off_regions = self._regions.handle.layout.off_regions
        region_offset = off_regions + region_id * self._regions.region_size()
        for slot_idx in range(self._slots_per_region):
            self._free.append(region_offset + slot_idx * self._chunk_size)
        self._free_count_per_region[region_id] = self._slots_per_region
        logger.debug(
            "heap (node=%d chunk=%d) seeded region %d with %d slots",
            self._node_id,
            self._chunk_size,
            region_id,
            self._slots_per_region,
        )
