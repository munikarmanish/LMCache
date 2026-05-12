# SPDX-License-Identifier: Apache-2.0
"""Periodic garbage collector for the CXL pool.

Runs as a rack-wide singleton, co-located with the lock manager (which
is also a per-rack singleton). On each tick:

  1. Query the cluster's liveness oracle for the current set of alive
     node ids.
  2. For each region descriptor whose owner is not in that set, mark
     it ORPHANED via `RegionAllocator.gc_dead_node`.
  3. Sweep the slot index for ALLOCATING slots owned by dead nodes,
     flipping them to TOMB (those writes never committed; bytes are
     junk).
  4. Promote ORPHANED regions to FREE once their last VALID slot has
     drained via normal LRU.

All steps are idempotent. If a second GC instance ever runs briefly
during failover, it sees the same state we did and converges to the
same result.

Why polling, not push: the cluster controller's `RegistryTree` is the
authority on liveness, but coupling GC to a controller push event
means the controller has to know about CXL — a dependency we want
to avoid (CXL is not in `RegistryTree`). A periodic scan is cheap at
realistic scales: 16 nodes × 512 regions × 64 B descriptor =
~512 KiB of cacheline reads per tick.
"""

# Standard
import threading
import time
from dataclasses import dataclass
from typing import Callable, FrozenSet, Optional

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.cxl.bootstrap import PoolHandle
from lmcache.v1.storage_backend.cxl.index_writer import CXLIndexWriter
from lmcache.v1.storage_backend.cxl.layout import (
    OWNER_FREE,
    OWNER_ORPHANED,
)
from lmcache.v1.storage_backend.cxl.regions import RegionAllocator

logger = init_logger(__name__)


# Caller-injected: returns the set of currently-alive node ids.
# In production: queries the cluster controller's RegistryTree.
# In tests: a closure over a mutable set.
LivenessProvider = Callable[[], FrozenSet[int]]


@dataclass
class CXLGCConfig:
    # Interval between full sweeps. At 30s and 16 nodes, scanning the
    # bitmap + descriptors costs negligible CPU. Tune lower if dead
    # nodes need faster reclamation.
    sweep_interval_s: float = 30.0
    # Hard cap on GC threads per pool. The class itself enforces this
    # via a class-level lock.
    enforce_singleton: bool = True


class CXLGarbageCollector:
    """Per-pool background sweeper. Run exactly one per rack.

    Public lifecycle:
      gc = CXLGarbageCollector(allocator, index_writer, pool, liveness, config)
      gc.start()
      ...
      gc.stop()

    For deterministic tests, call `sweep_once()` directly without
    starting the background thread.
    """

    # Class-level guard against accidental multi-instantiation against
    # the same PoolHandle. This is a *defensive* check — the algorithm
    # is idempotent, but two GCs racing on bitmap writes does extra
    # work and complicates log analysis.
    _active_pools_lock = threading.Lock()
    _active_pools: set[int] = set()  # set of pool base addresses

    def __init__(
        self,
        allocator: RegionAllocator,
        index_writer: CXLIndexWriter,
        pool: PoolHandle,
        liveness: LivenessProvider,
        config: Optional[CXLGCConfig] = None,
    ):
        self._allocator = allocator
        self._index_writer = index_writer
        self._pool = pool
        self._liveness = liveness
        self._config = config or CXLGCConfig()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._sweeps = 0
        self._slots_swept = 0
        self._regions_orphaned = 0
        self._regions_freed = 0
        self._last_dead_set: FrozenSet[int] = frozenset()

    # -------- lifecycle --------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("CXLGarbageCollector already started")
        if self._config.enforce_singleton:
            with self._active_pools_lock:
                if self._pool.base in self._active_pools:
                    raise RuntimeError(
                        "another CXLGarbageCollector is already running on "
                        f"pool base 0x{self._pool.base:x}; refusing to "
                        "start a second"
                    )
                self._active_pools.add(self._pool.base)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="cxl-gc", daemon=True
        )
        self._thread.start()
        logger.info(
            "CXL GC started (interval=%.1fs, singleton=%s)",
            self._config.sweep_interval_s,
            self._config.enforce_singleton,
        )

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop_event.set()
        t = self._thread
        if t is not None:
            t.join(timeout_s)
            if t.is_alive():
                logger.warning("CXL GC did not exit within %.1fs", timeout_s)
            self._thread = None
        if self._config.enforce_singleton:
            with self._active_pools_lock:
                self._active_pools.discard(self._pool.base)

    # -------- single-step (tests + manual cron) -------------------------

    def sweep_once(self) -> dict:
        """Run one full GC pass. Returns a stats dict for logging/tests."""
        alive = self._liveness()
        # Bound dead-id detection to actually-claimed regions; scanning
        # `range(MAX_NODES)` would emit GC for IDs that never existed.
        dead = self._discover_dead_nodes(alive)

        regions_orphaned = 0
        for node_id in dead:
            orphaned = self._allocator.gc_dead_node(node_id)
            regions_orphaned += len(orphaned)

        slots_swept = 0
        for node_id in dead:
            freed = self._index_writer.sweep_dead_owner_allocating(node_id)
            slots_swept += len(freed)

        regions_freed = self._promote_drained_orphans()

        self._sweeps += 1
        self._slots_swept += slots_swept
        self._regions_orphaned += regions_orphaned
        self._regions_freed += regions_freed
        self._last_dead_set = dead

        if dead or regions_freed:
            logger.info(
                "CXL GC sweep: dead_nodes=%s regions_orphaned=%d "
                "slots_swept=%d regions_freed=%d",
                sorted(dead) if dead else "{}",
                regions_orphaned,
                slots_swept,
                regions_freed,
            )

        return {
            "alive": sorted(alive),
            "dead": sorted(dead),
            "regions_orphaned": regions_orphaned,
            "slots_swept": slots_swept,
            "regions_freed": regions_freed,
        }

    @property
    def stats(self) -> dict:
        return {
            "sweeps": self._sweeps,
            "slots_swept": self._slots_swept,
            "regions_orphaned": self._regions_orphaned,
            "regions_freed": self._regions_freed,
            "last_dead_set": sorted(self._last_dead_set),
        }

    # -------- internals --------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.sweep_once()
            except Exception:
                logger.exception("CXL GC sweep raised; continuing")
            if self._stop_event.is_set():
                break
            self._stop_event.wait(self._config.sweep_interval_s)

    def _discover_dead_nodes(self, alive: FrozenSet[int]) -> FrozenSet[int]:
        """Find node ids that own regions but are not in `alive`.

        Crucially, we don't compare against `range(max_nodes)` — only
        against owners we actually observe in descriptors. This avoids
        churning GC on every sweep for empty MAX_NODES slots that
        never had an owner.
        """
        observed_owners: set[int] = set()
        for info in self._allocator.iter_regions():
            owner = info.owner_node_id
            if owner == OWNER_FREE or owner == OWNER_ORPHANED:
                continue
            observed_owners.add(owner)
        return frozenset(observed_owners - alive)

    def _promote_drained_orphans(self) -> int:
        """Sweep ORPHANED regions; promote any with no live VALID slots."""
        layout = self._pool.layout
        off_regions = layout.off_regions
        region_size = layout.region_size

        promoted = 0
        for info in self._allocator.iter_regions():
            if info.owner_node_id != OWNER_ORPHANED:
                continue
            lo = off_regions + info.region_id * region_size
            hi = lo + region_size

            def _is_drained(_region_id: int, _lo=lo, _hi=hi) -> bool:
                return self._index_writer.region_has_no_live_slots(
                    _region_id, _lo, _hi
                )

            if self._allocator.promote_orphaned(info.region_id, _is_drained):
                promoted += 1
        return promoted
