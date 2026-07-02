# SPDX-License-Identifier: Apache-2.0
"""CXL pool layout: header, region bitmap, region descriptors, index slots.

The pool is one `/dev/dax0.0` mmap. All sections live at fixed offsets
recorded in the header so any attaching node can discover them.

Layout (see plan `Pool Layout` section):

    +-------------------+ off = 0
    | Header            | 4 KiB
    +-------------------+ off = header.off_global_locks
    | Global locks      | NUM_LOCKS * MAX_NODES * 64 B
    +-------------------+ off = header.off_region_bitmap
    | Region bitmap     | ceil(region_count / 8) B, cacheline-aligned
    +-------------------+ off = header.off_region_descs
    | Region descriptors| region_count * 64 B, cacheline-aligned
    +-------------------+ off = header.off_index
    | Hash index        | index_slot_count * 128 B
    +-------------------+ off = header.off_regions
    | Regions           | region_count * region_size
    +-------------------+ end

Endianness: little-endian throughout (x86 host assumption).
"""

# Standard
import ctypes
from dataclasses import dataclass

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# -------- Constants -----------------------------------------------------

CACHELINE_SIZE = 64
HEADER_SIZE = 4096
LAYOUT_VERSION = 1
MAGIC = 0x4C4D43584C504F4F  # "LMCXLPOO" little-endian

# Sentinel owner values.
OWNER_FREE = 0xFFFF
OWNER_ORPHANED = 0xFFFE
# Any other value is a real node_id.

# Slot states. Stored as u32 for easy CLFLUSH boundary alignment.
SLOT_STATE_EMPTY = 0
SLOT_STATE_ALLOCATING = 1
SLOT_STATE_VALID = 2
SLOT_STATE_TOMB = 3

# Two-tier lock states.
LOCK_STATE_IDLE = 0
LOCK_STATE_WAITING = 1
LOCK_STATE_LOCKED = 2

# Default sizing (tunable via config).
DEFAULT_NUM_LOCKS = 4096
DEFAULT_MAX_NODES = 64
DEFAULT_REGION_SIZE = 256 * 1024 * 1024  # 256 MiB
DEFAULT_INDEX_LOAD_FACTOR = 0.5  # index_slot_count >= chunks / load_factor

# Assumed smallest chunk a region can hold, used only to estimate the
# default ``index_slot_count`` when the caller does not pass one. The
# layout planner does not know the real chunk size, so it bounds the
# peak chunk count by ``capacity / DEFAULT_MIN_CHUNK_SIZE``. Set to 2 MiB:
# KV chunks in practice are far larger (tens of MiB), so this keeps the
# index comfortably sized without over-provisioning by orders of magnitude
# (the previous 4 KiB assumption inflated the index ~500x).
DEFAULT_MIN_CHUNK_SIZE = 2 * 1024 * 1024  # 2 MiB

GEOM_HASH_SIZE = 16


# -------- ctypes structs ------------------------------------------------


class Header(ctypes.Structure):
    """CXL pool header (4 KiB). Lives at offset 0.

    `gen` is bumped on every controller bootstrap; peers compare it to
    their cached generation to detect controller restarts (plan F8).
    `geom_hash` is a 16-byte digest of the cluster-wide chunk geometry;
    peers reject attach on mismatch.
    """

    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_uint64),
        ("layout_version", ctypes.c_uint32),
        ("_pad0", ctypes.c_uint32),
        ("gen", ctypes.c_uint64),
        ("geom_hash", ctypes.c_uint8 * GEOM_HASH_SIZE),
        # Sizing
        ("region_size", ctypes.c_uint64),
        ("region_count", ctypes.c_uint32),
        ("index_slot_count", ctypes.c_uint32),
        ("num_locks", ctypes.c_uint32),
        ("max_nodes", ctypes.c_uint32),
        # Section offsets (bytes from pool base)
        ("off_global_locks", ctypes.c_uint64),
        ("off_region_bitmap", ctypes.c_uint64),
        ("off_region_descs", ctypes.c_uint64),
        ("off_index", ctypes.c_uint64),
        ("off_regions", ctypes.c_uint64),
        ("pool_size", ctypes.c_uint64),
        # Region-allocator hint; single-writer under region_lock.
        ("search_hint", ctypes.c_uint32),
        ("_pad1", ctypes.c_uint32),
    ]


# Header must fit within HEADER_SIZE with room to spare for future fields.
assert ctypes.sizeof(Header) <= HEADER_SIZE, (
    f"Header {ctypes.sizeof(Header)} > {HEADER_SIZE}"
)


class LockSlot(ctypes.Structure):
    """One cell of the global_lock[NUM_LOCKS][MAX_NODES] array.

    One cacheline per slot to prevent false sharing between nodes
    contending on the same lock_id.
    """

    _pack_ = 1
    _fields_ = [
        ("state", ctypes.c_uint32),  # LOCK_STATE_{IDLE,WAITING,LOCKED}
        ("seq", ctypes.c_uint32),  # fairness counter
        ("_pad", ctypes.c_uint8 * (CACHELINE_SIZE - 8)),
    ]


assert ctypes.sizeof(LockSlot) == CACHELINE_SIZE


class RegionDesc(ctypes.Structure):
    """Per-region metadata. One cacheline per region.

    The region bitmap is the fast alloc/free path. This descriptor
    holds what GC needs: owner, epoch, heartbeat. Reading it is not
    on the per-chunk hot path.
    """

    _pack_ = 1
    _fields_ = [
        ("owner_node_id", ctypes.c_uint16),  # OWNER_FREE / OWNER_ORPHANED / id
        ("_pad0", ctypes.c_uint16),
        ("_pad1", ctypes.c_uint32),
        ("claim_epoch", ctypes.c_uint64),  # header.gen at claim time
        ("last_heartbeat_seen", ctypes.c_uint64),
        ("_pad", ctypes.c_uint8 * (CACHELINE_SIZE - 24)),
    ]


assert ctypes.sizeof(RegionDesc) == CACHELINE_SIZE


class SlotLine0(ctypes.Structure):
    """Read-mostly half of a hash-index slot (64 B).

    All fields a lookup consults fit in this cacheline, so a writer's
    final `CLFLUSH(&line0); MFENCE` publishes them as a unit.
    """

    _pack_ = 1
    _fields_ = [
        ("chunk_hash", ctypes.c_uint64),
        ("chunk_offset", ctypes.c_uint64),  # bytes from pool base
        ("chunk_len", ctypes.c_uint32),
        ("state", ctypes.c_uint32),  # SLOT_STATE_*
        ("fmt", ctypes.c_uint16),  # MemoryFormat enum value
        ("owner_node_id", ctypes.c_uint16),
        ("generation", ctypes.c_uint32),  # must equal header.gen
        ("geom_hash", ctypes.c_uint8 * GEOM_HASH_SIZE),
        ("_pad", ctypes.c_uint8 * (CACHELINE_SIZE - 48)),
    ]


assert ctypes.sizeof(SlotLine0) == CACHELINE_SIZE


class SlotLine1(ctypes.Structure):
    """Hot-mutable half of a hash-index slot (64 B).

    Ref counts and LRU links live here so they don't false-share
    with line 0's read-mostly fields.
    """

    _pack_ = 1
    _fields_ = [
        ("ref_count", ctypes.c_uint32),
        ("pin_count", ctypes.c_uint32),
        ("lru_prev", ctypes.c_uint64),  # slot index, not offset
        ("lru_next", ctypes.c_uint64),  # slot index, not offset
        ("lock_id", ctypes.c_uint16),
        ("_pad", ctypes.c_uint8 * (CACHELINE_SIZE - 26)),
    ]


assert ctypes.sizeof(SlotLine1) == CACHELINE_SIZE


class Slot(ctypes.Structure):
    """A full index slot: 128 B, two cachelines."""

    _pack_ = 1
    _fields_ = [
        ("line0", SlotLine0),
        ("line1", SlotLine1),
    ]


assert ctypes.sizeof(Slot) == 2 * CACHELINE_SIZE


# -------- Layout planner ------------------------------------------------


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


@dataclass
class PoolLayout:
    """Computed section offsets and sizes for a given pool size / config.

    Used both at bootstrap (to initialize the header) and on attach
    (to validate against the header already in the pool).
    """

    pool_size: int
    region_size: int
    region_count: int
    index_slot_count: int
    num_locks: int
    max_nodes: int

    off_header: int = 0
    off_global_locks: int = 0
    off_region_bitmap: int = 0
    off_region_descs: int = 0
    off_index: int = 0
    off_regions: int = 0

    global_locks_size: int = 0
    region_bitmap_size: int = 0
    region_descs_size: int = 0
    index_size: int = 0
    regions_size: int = 0

    @staticmethod
    def compute(
        pool_size: int,
        region_size: int,
        num_locks: int = DEFAULT_NUM_LOCKS,
        max_nodes: int = DEFAULT_MAX_NODES,
        index_slot_count: int | None = None,
    ) -> "PoolLayout":
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}")
        if region_size <= 0 or region_size & (region_size - 1):
            raise ValueError(
                f"region_size must be a positive power of two, got {region_size}"
            )
        if num_locks <= 0 or max_nodes <= 0:
            raise ValueError("num_locks and max_nodes must be positive")

        off_global_locks = _align_up(HEADER_SIZE, CACHELINE_SIZE)
        global_locks_size = num_locks * max_nodes * ctypes.sizeof(LockSlot)

        # We tentatively pick region_count from the remaining budget,
        # then iterate once because index_slot_count and the metadata
        # sections depend on region_count only weakly.
        overhead_lower_bound = (
            off_global_locks + global_locks_size + CACHELINE_SIZE  # bitmap slack
        )
        if pool_size <= overhead_lower_bound:
            raise ValueError(
                f"pool_size {pool_size} too small for header + locks "
                f"({overhead_lower_bound})"
            )

        # Upper bound on region count from raw capacity; real metadata
        # overhead shrinks it slightly.
        max_region_count_by_cap = (pool_size - overhead_lower_bound) // region_size
        if max_region_count_by_cap <= 0:
            raise ValueError(
                f"pool_size {pool_size} too small to fit any region of size "
                f"{region_size} after metadata overhead"
            )

        # A rough first pass — we allow the metadata sections to claim
        # space, then recompute the region count until it stabilizes.
        # In practice this converges in one iteration because metadata
        # overhead is O(region_count) but tiny compared to region_size.
        region_count = int(max_region_count_by_cap)
        for _ in range(4):
            if index_slot_count is None:
                # We don't know the real chunk size at the layout level,
                # so bound the peak chunk count by assuming the smallest
                # chunk a region could hold (DEFAULT_MIN_CHUNK_SIZE): one
                # slot per that many bytes of region capacity, floored at
                # 1024. Real chunks are larger, so this over-provisions a
                # little but no longer by orders of magnitude. Callers that
                # know the chunk size should pass index_slot_count.
                capacity = region_count * region_size
                index_slot_count_local = max(
                    1024, capacity // DEFAULT_MIN_CHUNK_SIZE
                )
            else:
                index_slot_count_local = index_slot_count

            region_bitmap_size = _align_up(
                max(1, (region_count + 7) // 8), CACHELINE_SIZE
            )
            region_descs_size = region_count * ctypes.sizeof(RegionDesc)
            index_size = index_slot_count_local * ctypes.sizeof(Slot)

            off_region_bitmap = off_global_locks + global_locks_size
            off_region_descs = off_region_bitmap + region_bitmap_size
            off_index = off_region_descs + region_descs_size
            off_regions = _align_up(off_index + index_size, region_size)

            available_for_regions = pool_size - off_regions
            new_region_count = available_for_regions // region_size
            if new_region_count <= 0:
                raise ValueError(
                    f"pool_size {pool_size} too small to fit even one region "
                    f"after metadata; try a smaller region_size"
                )
            if new_region_count == region_count:
                region_count = int(new_region_count)
                break
            region_count = int(new_region_count)
        else:
            raise RuntimeError(
                "layout planner did not converge; this indicates a bug"
            )

        regions_size = region_count * region_size

        return PoolLayout(
            pool_size=pool_size,
            region_size=region_size,
            region_count=region_count,
            index_slot_count=index_slot_count_local,
            num_locks=num_locks,
            max_nodes=max_nodes,
            off_header=0,
            off_global_locks=off_global_locks,
            off_region_bitmap=off_region_bitmap,
            off_region_descs=off_region_descs,
            off_index=off_index,
            off_regions=off_regions,
            global_locks_size=global_locks_size,
            region_bitmap_size=region_bitmap_size,
            region_descs_size=region_descs_size,
            index_size=index_size,
            regions_size=regions_size,
        )

    def total_used(self) -> int:
        return self.off_regions + self.regions_size

    def write_to_header(self, header: Header, geom_hash: bytes, gen: int) -> None:
        if len(geom_hash) != GEOM_HASH_SIZE:
            raise ValueError(
                f"geom_hash must be {GEOM_HASH_SIZE} bytes, got {len(geom_hash)}"
            )
        header.magic = MAGIC
        header.layout_version = LAYOUT_VERSION
        header.gen = gen
        ctypes.memmove(header.geom_hash, geom_hash, GEOM_HASH_SIZE)
        header.region_size = self.region_size
        header.region_count = self.region_count
        header.index_slot_count = self.index_slot_count
        header.num_locks = self.num_locks
        header.max_nodes = self.max_nodes
        header.off_global_locks = self.off_global_locks
        header.off_region_bitmap = self.off_region_bitmap
        header.off_region_descs = self.off_region_descs
        header.off_index = self.off_index
        header.off_regions = self.off_regions
        header.pool_size = self.pool_size
        header.search_hint = 0

    @staticmethod
    def from_header(header: Header) -> "PoolLayout":
        return PoolLayout(
            pool_size=header.pool_size,
            region_size=header.region_size,
            region_count=header.region_count,
            index_slot_count=header.index_slot_count,
            num_locks=header.num_locks,
            max_nodes=header.max_nodes,
            off_global_locks=header.off_global_locks,
            off_region_bitmap=header.off_region_bitmap,
            off_region_descs=header.off_region_descs,
            off_index=header.off_index,
            off_regions=header.off_regions,
            global_locks_size=header.num_locks
            * header.max_nodes
            * ctypes.sizeof(LockSlot),
            region_bitmap_size=header.off_region_descs - header.off_region_bitmap,
            region_descs_size=header.region_count * ctypes.sizeof(RegionDesc),
            index_size=header.index_slot_count * ctypes.sizeof(Slot),
            regions_size=header.region_count * header.region_size,
        )

    def validate_against(self, header: Header) -> None:
        """Raise if this layout is inconsistent with what the header claims."""
        if header.magic != MAGIC:
            raise ValueError(f"bad magic 0x{header.magic:x}, expected 0x{MAGIC:x}")
        if header.layout_version != LAYOUT_VERSION:
            raise ValueError(
                f"layout version mismatch: header has {header.layout_version}, "
                f"this build supports {LAYOUT_VERSION}"
            )
        if header.pool_size != self.pool_size:
            raise ValueError(
                f"pool_size mismatch: header {header.pool_size} vs mapped "
                f"{self.pool_size}"
            )
        if header.region_size != self.region_size:
            raise ValueError(
                f"region_size mismatch: header {header.region_size} vs "
                f"requested {self.region_size}"
            )
