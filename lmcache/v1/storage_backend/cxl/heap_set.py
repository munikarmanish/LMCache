# SPDX-License-Identifier: Apache-2.0
"""Per-geometry heap classes over one node's regions.

A pool serves many tenants at once, and a tenant's chunk size is a pure
function of its KV geometry — `num_layers * kv_dim * chunk_tokens *
num_kv_heads * head_size * dtype_size`. The token chunk count is uniform
across every client of one MP server (the vLLM client *queries* the
server for it), so the set of live chunk sizes is exactly the set of
distinct registered geometries.

`HeapSet` multiplexes one :class:`NodeHeap` per distinct **byte size**,
created on first use. Keying on the computed size rather than on the
model means two models that happen to share a geometry share one class
safely — the tenant digest in the index keeps their contents distinct
(see `store.object_key_to_tenant_digest`).

Sizes are exact, not rounded to powers of two: rounding would waste
space in every chunk, where exact-fit wastes only a bounded tail per
region (see :class:`NodeHeap`).

**A region belongs to exactly one class for its lifetime.** That is what
makes `free(offset)` unambiguous: the owning heap is found by asking
each heap whether the offset falls in a region it owns, with no extra
per-chunk bookkeeping and nothing recorded in the shared index.
"""

# Standard
from typing import Dict, List, Optional, Tuple
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.cxl.heap import HeapStats, NodeHeap
from lmcache.v1.storage_backend.cxl.regions import RegionAllocator

logger = init_logger(__name__)

# Below this many slots per region, a class claims a new region every few
# allocations, so the DRAM free-list stops amortizing the shared region
# lock. Functional, just heavier on the arbiter — warn, don't fail.
_MIN_SLOTS_PER_REGION_WARN = 4


class HeapSet:
    """One :class:`NodeHeap` per distinct chunk size, created on demand.

    Thread-safe. The per-class heaps carry their own locks; this class's
    mutex covers only the size -> heap map, so allocation on an existing
    class does not serialize against allocation on another.
    """

    def __init__(self, region_allocator: RegionAllocator, node_id: int):
        """Initialize an empty set of heap classes.

        Args:
            region_allocator: Shared allocator the classes claim regions from.
            node_id: This node's id, stamped as region owner.
        """
        self._regions = region_allocator
        self._node_id = node_id
        self._heaps: Dict[int, NodeHeap] = {}
        self._lock = threading.Lock()

    # -------- class management -------------------------------------------

    def heap_for(self, chunk_size: int) -> NodeHeap:
        """Return the heap for `chunk_size`, creating it on first use.

        Args:
            chunk_size: Exact chunk size in bytes.

        Returns:
            The :class:`NodeHeap` serving that size.

        Raises:
            ValueError: If ``chunk_size`` is not positive, or exceeds
                ``region_size`` so that no slot could ever be carved.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        with self._lock:
            heap = self._heaps.get(chunk_size)
            if heap is not None:
                return heap
            # NodeHeap raises if the size cannot fit a region at all.
            heap = NodeHeap(self._regions, node_id=self._node_id, chunk_size=chunk_size)
            slots = heap.slots_per_region
            if slots < _MIN_SLOTS_PER_REGION_WARN:
                logger.warning(
                    "CXL node %d: chunk size %d yields only %d slot(s) per "
                    "%d-byte region, so this class claims a region every %d "
                    "allocation(s) and the DRAM free-list barely amortizes the "
                    "shared region lock. Consider a larger cxl_region_size.",
                    self._node_id,
                    chunk_size,
                    slots,
                    self._regions.region_size(),
                    slots,
                )
            self._heaps[chunk_size] = heap
            logger.info(
                "CXL node %d: created heap class chunk_size=%d (%d slots/region)",
                self._node_id,
                chunk_size,
                slots,
            )
            return heap

    def classes(self) -> List[int]:
        """Return the chunk sizes currently served, ascending."""
        with self._lock:
            return sorted(self._heaps)

    # -------- allocation --------------------------------------------------

    def alloc(self, chunk_size: int) -> int:
        """Allocate one chunk of `chunk_size`, claiming a region if needed.

        Args:
            chunk_size: Exact chunk size in bytes.

        Returns:
            A pool-relative chunk offset.
        """
        return self.heap_for(chunk_size).alloc()

    def alloc_no_claim(self, chunk_size: int) -> int:
        """Allocate from the class's free-list only, never claiming.

        Args:
            chunk_size: Exact chunk size in bytes.

        Returns:
            A pool-relative chunk offset.

        Raises:
            OutOfChunks: If that class's free-list is empty.
        """
        return self.heap_for(chunk_size).alloc_no_claim()

    def alloc_batch(self, chunk_size: int, n: int) -> List[int]:
        """Allocate `n` chunks of one size under a single lock hold.

        Args:
            chunk_size: Exact chunk size in bytes.
            n: Number of chunks.

        Returns:
            ``n`` pool-relative offsets.
        """
        return self.heap_for(chunk_size).alloc_batch(n)

    # -------- release -----------------------------------------------------

    def free(self, offset: int) -> None:
        """Return one chunk to whichever class owns its region.

        Args:
            offset: A pool-relative offset previously returned by this set.

        Raises:
            ValueError: If no class owns the region containing ``offset``.
        """
        heap = self._owner_of(offset)
        if heap is None:
            raise ValueError(
                f"offset {offset} does not fall in any region owned by this "
                "node's heap classes"
            )
        heap.free(offset)

    def free_batch(self, offsets: List[int]) -> None:
        """Return many chunks, grouping by owning class.

        Offsets are bucketed so each class takes its lock once, rather
        than once per offset.

        Args:
            offsets: Pool-relative offsets to free.

        Raises:
            ValueError: If any offset falls in no owned region. Offsets
                belonging to known classes are still freed first, so a
                bad entry does not strand the rest.
        """
        buckets: Dict[int, List[int]] = {}
        unknown: List[int] = []
        for off in offsets:
            heap = self._owner_of(off)
            if heap is None:
                unknown.append(off)
                continue
            buckets.setdefault(heap.chunk_size, []).append(off)
        for chunk_size, group in buckets.items():
            self._heaps[chunk_size].free_batch(group)
        if unknown:
            raise ValueError(
                f"{len(unknown)} offset(s) fall in no region owned by this "
                f"node's heap classes (first: {unknown[0]})"
            )

    def trim(self) -> List[int]:
        """Release every class's fully-empty regions to the global pool.

        Returns:
            The region ids released, across all classes.
        """
        released: List[int] = []
        for heap in self._snapshot():
            released.extend(heap.trim())
        return released

    # -------- accounting --------------------------------------------------

    def occupancy(self) -> Tuple[int, int]:
        """Return ``(occupied_slots, total_slots)`` summed over classes.

        Slot counts are not comparable in bytes across classes, but the
        eviction ladder uses only the *ratio* as a node-local watermark,
        and the ratio is meaningful under mixed sizes: it still answers
        "how full are the regions this node holds".

        Returns:
            A ``(occupied_slots, total_slots)`` pair; ``(0, 0)`` when the
            node owns no regions.
        """
        occupied = total = 0
        for heap in self._snapshot():
            o, t = heap.occupancy()
            occupied += o
            total += t
        return occupied, total

    def stats(self) -> Dict[int, HeapStats]:
        """Return per-class :class:`HeapStats`, keyed by chunk size."""
        return {heap.chunk_size: heap.stats() for heap in self._snapshot()}

    def total_stats(self) -> HeapStats:
        """Return slot counts summed across classes.

        Slot counts are not byte-comparable across classes; this is for
        callers that only need "how many chunks are resident", such as
        asserting that a batch consumed exactly N slots.

        Returns:
            A :class:`HeapStats` whose ``chunk_size`` is 0, marking it as
            an aggregate over heterogeneous classes rather than one
            class's geometry.
        """
        regions = slots = free = 0
        for st in self.stats().values():
            regions += st.total_regions
            slots += st.total_slots
            free += st.free_slots
        return HeapStats(
            total_regions=regions,
            total_slots=slots,
            free_slots=free,
            chunk_size=0,
        )

    def owned_regions(self) -> List[int]:
        """Return every region id owned across all classes."""
        out: List[int] = []
        for heap in self._snapshot():
            out.extend(heap.owned_regions())
        return out

    # -------- internals ---------------------------------------------------

    def _snapshot(self) -> List[NodeHeap]:
        """Return the current heaps without holding the map lock."""
        with self._lock:
            return list(self._heaps.values())

    def _owner_of(self, offset: int) -> Optional[NodeHeap]:
        """Return the heap whose owned regions contain `offset`, else None."""
        off_regions = self._regions.handle.layout.off_regions
        region_size = self._regions.region_size()
        if offset < off_regions:
            return None
        region_id = (offset - off_regions) // region_size
        for heap in self._snapshot():
            if region_id in heap.owned_regions():
                return heap
        return None
