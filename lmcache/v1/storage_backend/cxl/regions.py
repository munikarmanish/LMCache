# SPDX-License-Identifier: Apache-2.0
"""Global region allocator over the CXL-resident bitmap + descriptors.

Plan reference: F4 and "Global allocator state" under Pool Layout.

This tracks **region ownership only**: which node currently owns each
256 MiB (default) region. Chunk-granularity allocation within a region
is the job of `heap.py` and uses a DRAM-only free-list.

Operations:

- `claim(node_id)` — allocate a region for this node. Returns `region_id`
  or raises `NoRegionAvailable`. Writes the bitmap + descriptor under
  the region_lock and CLFLUSHes.
- `release(region_id, node_id)` — return a region to the pool. Caller
  is responsible for guaranteeing no live slots point into it.
- `gc_dead_node(dead_node_id)` — flip every region owned by the dead
  node to ORPHANED (not FREE). `VALID` slots pointing into those
  regions stay readable until LRU drains them; `PROMOTE_ORPHANED`
  recycles the region once drained.
- `promote_orphaned(region_id, has_no_live_slots_fn)` — after an
  ORPHANED region's last live slot is evicted, flip it to FREE.

We reserve `lock_id == 0` for the region allocator. The planner's
default `NUM_LOCKS == 4096` easily spares this.
"""

# Standard
import ctypes
import time
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.cxl.bootstrap import PoolHandle
from lmcache.v1.storage_backend.cxl.fence import Fence, default_fence
from lmcache.v1.storage_backend.cxl.layout import (
    OWNER_FREE,
    OWNER_ORPHANED,
    RegionDesc,
)
from lmcache.v1.storage_backend.cxl.locks import TwoTierLock

logger = init_logger(__name__)


# Reserved lock_id for the region allocator. All region alloc/free/GC
# serialize on this single lock; contention is rare (once per ~256 MiB
# of chunk churn).
REGION_LOCK_ID = 0


class NoRegionAvailable(Exception):
    """Raised when the pool has no FREE regions to claim."""


@dataclass(frozen=True)
class RegionInfo:
    region_id: int
    owner_node_id: int
    claim_epoch: int
    last_heartbeat_seen: int


class RegionAllocator:
    """Global region allocator. Holds no state of its own in DRAM.

    All "state" lives in the CXL pool's bitmap and descriptors; this
    object is just a typed view onto them plus the region_lock.
    """

    def __init__(
        self,
        handle: PoolHandle,
        region_lock: TwoTierLock,
        *,
        fence: Optional[Fence] = None,
    ):
        self._handle = handle
        self._region_lock = region_lock
        self._fence = fence or default_fence()
        self._region_count = handle.layout.region_count
        self._bitmap = handle.region_bitmap()
        self._descs = handle.region_descs()
        self._bitmap_base = ctypes.addressof(self._bitmap)
        self._descs_base = ctypes.addressof(self._descs)
        self._desc_size = ctypes.sizeof(RegionDesc)

    # -------- public API -------------------------------------------------

    @property
    def region_count(self) -> int:
        return self._region_count

    def region_size(self) -> int:
        return self._handle.layout.region_size

    def claim(self, node_id: int) -> int:
        """Claim a FREE region for `node_id`. Returns the region_id.

        Raises `NoRegionAvailable` if every region is allocated or
        orphaned. Takes the region_lock.
        """
        self._validate_node_id(node_id)
        with self._region_lock.acquire(REGION_LOCK_ID):
            self._flush_bitmap_and_descs()
            header = self._handle.header
            start = header.search_hint % self._region_count
            for step in range(self._region_count):
                i = (start + step) % self._region_count
                if self._bit_is_set(i):
                    continue
                # Bitmap says free; sanity-check the descriptor agrees.
                # If not, we have torn state from a crash — treat as
                # taken and skip.
                if self._descs[i].owner_node_id != OWNER_FREE:
                    logger.warning(
                        "region %d: bitmap=free but descriptor claims owner=%d; "
                        "skipping",
                        i,
                        self._descs[i].owner_node_id,
                    )
                    continue
                self._set_bit(i)
                self._write_desc(
                    i,
                    owner_node_id=node_id,
                    claim_epoch=header.gen,
                    last_heartbeat_seen=int(time.monotonic()),
                )
                header.search_hint = (i + 1) % self._region_count
                self._fence.fence_after_write(
                    ctypes.addressof(header), ctypes.sizeof(type(header))
                )
                logger.debug(
                    "node %d claimed region %d (epoch=%d)",
                    node_id,
                    i,
                    header.gen,
                )
                return i
            raise NoRegionAvailable(
                f"no FREE region in pool; region_count={self._region_count}"
            )

    def release(self, region_id: int, node_id: int) -> None:
        """Return `region_id` to the FREE pool. Caller owns it.

        Does NOT check slot liveness — caller is responsible for
        ensuring no `VALID` slots point into this region before calling.
        """
        self._validate_region_id(region_id)
        self._validate_node_id(node_id)
        with self._region_lock.acquire(REGION_LOCK_ID):
            self._flush_bitmap_and_descs()
            owner = self._descs[region_id].owner_node_id
            if owner != node_id:
                raise RuntimeError(
                    f"region {region_id} owner is {owner}, not {node_id}; "
                    "refusing to release"
                )
            self._clear_bit(region_id)
            self._write_desc(
                region_id,
                owner_node_id=OWNER_FREE,
                claim_epoch=0,
                last_heartbeat_seen=0,
            )

    def gc_dead_node(self, dead_node_id: int) -> List[int]:
        """Flip all regions owned by `dead_node_id` to ORPHANED.

        Returns the list of region_ids that were orphaned. Does NOT
        flip the bitmap — live `VALID` slots may still point into
        these regions; `promote_orphaned` handles final reclamation.
        """
        self._validate_node_id(dead_node_id)
        orphaned: List[int] = []
        with self._region_lock.acquire(REGION_LOCK_ID):
            self._flush_bitmap_and_descs()
            for i in range(self._region_count):
                if self._descs[i].owner_node_id == dead_node_id:
                    self._write_desc(
                        i,
                        owner_node_id=OWNER_ORPHANED,
                        # Keep claim_epoch for audit; zero the heartbeat.
                        claim_epoch=self._descs[i].claim_epoch,
                        last_heartbeat_seen=0,
                    )
                    orphaned.append(i)
        if orphaned:
            logger.info(
                "GC: node %d orphaned %d region(s): %s",
                dead_node_id,
                len(orphaned),
                orphaned,
            )
        return orphaned

    def promote_orphaned(
        self,
        region_id: int,
        is_drained: Callable[[int], bool],
    ) -> bool:
        """If `region_id` is ORPHANED and `is_drained(region_id)` is True,
        flip its bitmap bit to 0 and descriptor to FREE.

        Returns True if promotion occurred, False otherwise.

        `is_drained` is a caller-provided predicate. It is called
        UNDER the region_lock, so it must be cheap and non-blocking.
        Typical implementation: scan the hash index for any slot whose
        offset falls within this region and is still `VALID`.
        """
        self._validate_region_id(region_id)
        with self._region_lock.acquire(REGION_LOCK_ID):
            self._flush_bitmap_and_descs()
            if self._descs[region_id].owner_node_id != OWNER_ORPHANED:
                return False
            if not is_drained(region_id):
                return False
            self._clear_bit(region_id)
            self._write_desc(
                region_id,
                owner_node_id=OWNER_FREE,
                claim_epoch=0,
                last_heartbeat_seen=0,
            )
            logger.info("promoted orphaned region %d → FREE", region_id)
            return True

    def regions_owned_by(self, node_id: int) -> List[int]:
        """Return the region_ids currently owned by `node_id` (snapshot).

        Not thread-safe against concurrent claim/release on other
        threads; callers hold the region_lock if they need consistency.
        """
        self._flush_bitmap_and_descs()
        return [
            i
            for i in range(self._region_count)
            if self._descs[i].owner_node_id == node_id
        ]

    def info(self, region_id: int) -> RegionInfo:
        self._validate_region_id(region_id)
        self._flush_bitmap_and_descs()
        d = self._descs[region_id]
        return RegionInfo(
            region_id=region_id,
            owner_node_id=int(d.owner_node_id),
            claim_epoch=int(d.claim_epoch),
            last_heartbeat_seen=int(d.last_heartbeat_seen),
        )

    def iter_regions(self) -> Iterator[RegionInfo]:
        for i in range(self._region_count):
            yield self.info(i)

    def region_address(self, region_id: int) -> int:
        """Return the starting address of a region's payload bytes."""
        self._validate_region_id(region_id)
        return self._handle.region_address(region_id)

    # -------- bitmap primitives -----------------------------------------

    def _bit_is_set(self, i: int) -> bool:
        return (self._bitmap[i >> 3] >> (i & 7)) & 1 == 1

    def _set_bit(self, i: int) -> None:
        byte_idx = i >> 3
        self._bitmap[byte_idx] = self._bitmap[byte_idx] | (1 << (i & 7))
        self._fence.fence_after_write(
            self._bitmap_base + byte_idx, 1
        )

    def _clear_bit(self, i: int) -> None:
        byte_idx = i >> 3
        self._bitmap[byte_idx] = self._bitmap[byte_idx] & ~(1 << (i & 7))
        self._fence.fence_after_write(
            self._bitmap_base + byte_idx, 1
        )

    def _write_desc(
        self,
        i: int,
        *,
        owner_node_id: int,
        claim_epoch: int,
        last_heartbeat_seen: int,
    ) -> None:
        d = self._descs[i]
        d.owner_node_id = owner_node_id
        d.claim_epoch = claim_epoch
        d.last_heartbeat_seen = last_heartbeat_seen
        self._fence.fence_after_write(
            self._descs_base + i * self._desc_size, self._desc_size
        )

    def _flush_bitmap_and_descs(self) -> None:
        """Fence both structures before a read; callers run under lock."""
        self._fence.flush_before_read(
            self._bitmap_base, self._handle.layout.region_bitmap_size
        )
        self._fence.flush_before_read(
            self._descs_base, self._region_count * self._desc_size
        )

    # -------- validation ------------------------------------------------

    def _validate_region_id(self, region_id: int) -> None:
        if not 0 <= region_id < self._region_count:
            raise IndexError(region_id)

    def _validate_node_id(self, node_id: int) -> None:
        # OWNER_FREE / OWNER_ORPHANED are reserved sentinels, not ids.
        if node_id in (OWNER_FREE, OWNER_ORPHANED):
            raise ValueError(f"node_id {node_id:#x} is a reserved sentinel")
        if not 0 <= node_id < self._handle.layout.max_nodes:
            raise ValueError(
                f"node_id {node_id} out of range [0, {self._handle.layout.max_nodes})"
            )
