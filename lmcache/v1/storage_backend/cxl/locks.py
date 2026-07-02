# SPDX-License-Identifier: Apache-2.0
"""Two-tier lock for cross-host mutual exclusion on the CXL pool.

Plan reference: F2 and the "Locking Summary" section. The short version:

- `global_lock[NUM_LOCKS][MAX_NODES]` lives in CXL shared memory. Each
  cell is a `LockSlot` (one cacheline) with a `state` in {IDLE, WAITING,
  LOCKED} and a fairness counter `seq`.
- No host CASes these cells. Workers only *store* state transitions
  (WAITING when joining the line; IDLE when releasing). One **lock
  manager** thread (per-rack, elected; the local node in single-host
  tests) does all WAITING→LOCKED flips.
- A per-node `threading.Lock` in DRAM serializes intra-node waiters
  for the same lock_id so only one of them contends for the CXL row.

The subsystem exposes `TwoTierLock` as a context manager:

    with two_tier_lock.acquire(lock_id):
        ... critical section ...

Acquisition path (TraCT §3.3):

    1. Take the local pthread_mutex (threading.Lock).
    2. Store WAITING + my_seq into my CXL row; fence-after-write.
    3. Poll my row's state == LOCKED (fence-before-read each loop).
    4. Run the critical section.
    5. Store IDLE into my row; fence-after-write.
    6. Release the local pthread_mutex.
"""

# Standard
import contextlib
import ctypes
import threading
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.cxl.bootstrap import PoolHandle
from lmcache.v1.storage_backend.cxl.fence import Fence, default_fence
from lmcache.v1.storage_backend.cxl.layout import (
    LOCK_STATE_IDLE,
    LOCK_STATE_LOCKED,
    LOCK_STATE_WAITING,
    LockSlot,
)

logger = init_logger(__name__)


class LockAcquisitionTimeout(Exception):
    """Raised when a waiter gives up because the manager appears dead."""


@dataclass
class LockConfig:
    # Seconds to wait for WAITING → LOCKED before raising. Should be
    # many manager sweeps (see manager_sweep_interval). Set to 0 to wait
    # indefinitely (not recommended outside tests).
    acquire_timeout_s: float = 30.0
    # How long to sleep between poll iterations on the waiter side.
    # Keep small: the critical sections here are tens of nanoseconds
    # in the real deployment, so fast polling is cheap.
    waiter_poll_interval_s: float = 0.0001  # 100 µs


class TwoTierLock:
    """Cross-host, sharded lock backed by a CXL global_lock array.

    Instances are bound to a specific `node_id`. `lock_id` identifies
    which of NUM_LOCKS logical locks we want.

    Correctness invariants:

    - Two threads on the same node contending for the same lock_id
      serialize via the local threading.Lock; only the winner
      publishes WAITING to CXL.
    - At most one row across the `MAX_NODES` slots for a given lock_id
      is ever in the LOCKED state, enforced by the single-writer
      lock-manager.
    - A waiter never observes LOCKED on its row unless the manager
      put it there.
    """

    def __init__(
        self,
        handle: PoolHandle,
        node_id: int,
        *,
        fence: Optional[Fence] = None,
        config: Optional[LockConfig] = None,
    ):
        num_locks = handle.layout.num_locks
        max_nodes = handle.layout.max_nodes
        if not 0 <= node_id < max_nodes:
            raise ValueError(f"node_id {node_id} out of range [0, {max_nodes})")

        self._handle = handle
        self._node_id = node_id
        self._fence = fence or default_fence()
        self._config = config or LockConfig()
        self._num_locks = num_locks
        self._max_nodes = max_nodes

        # Per-lock_id, per-node DRAM mutex. One threading.Lock per
        # (node, lock_id) combination — we only use this node's row.
        self._local_mutexes = [threading.Lock() for _ in range(num_locks)]

        # Monotonic per-lock_id sequence number. Writer stamps it on
        # every WAITING; manager prefers the smallest seq among waiters
        # for FIFO-ish fairness.
        self._next_seq = [1] * num_locks  # 0 reserved as "no sequence"
        self._next_seq_lock = threading.Lock()

        # Cached view of the CXL lock table for quick access.
        self._lock_array = handle.global_locks()
        self._lock_slot_size = ctypes.sizeof(LockSlot)
        self._lock_array_base = ctypes.addressof(self._lock_array)

    # -------- public API -------------------------------------------------

    @property
    def node_id(self) -> int:
        return self._node_id

    @property
    def num_locks(self) -> int:
        return self._num_locks

    @property
    def max_nodes(self) -> int:
        return self._max_nodes

    @contextlib.contextmanager
    def acquire(self, lock_id: int) -> Iterator[None]:
        """Acquire the lock; release on context exit (including exceptions)."""
        self._check_lock_id(lock_id)
        local_mutex = self._local_mutexes[lock_id]
        local_mutex.acquire()
        try:
            self._publish_waiting(lock_id)
            try:
                self._wait_for_locked(lock_id)
                yield
            finally:
                # Always drop the CXL row back to IDLE, even on exception.
                self._publish_idle(lock_id)
        finally:
            local_mutex.release()

    @contextlib.contextmanager
    def acquire_batch(self, lock_ids: "Iterable[int]") -> Iterator[None]:
        """Acquire many locks together in a single arbiter sweep.

        The serial pattern ``for id in ids: with acquire(id): ...`` pays one
        arbiter round-trip (~one sweep) *per* lock, because each ``acquire``
        publishes WAITING and then blocks before the next one is even
        requested. This batches the three phases so all locks are pending at
        once:

        1. Take every distinct lock's local mutex in ascending ``lock_id``
           order (a global order, so two concurrent batch callers cannot
           deadlock).
        2. Publish WAITING to all rows — cheap CXL stores, no waiting.
        3. Spin until *every* row is granted LOCKED. Because distinct
           ``lock_id``s are arbitrated independently within one
           ``sweep_once``, the manager grants the whole batch in ~one sweep
           rather than N.

        On exit (including exceptions) all rows are dropped to IDLE and the
        local mutexes released in reverse order.

        Duplicate ``lock_id``s are de-duplicated: the caller holds each
        distinct lock once and runs all of its critical-section work under
        that single hold. An empty ``lock_ids`` is a valid no-op.

        Args:
            lock_ids: The lock ids to acquire together (duplicates allowed).

        Yields:
            None — with all requested locks held.
        """
        # Dedup + sort for a deadlock-free global acquisition order.
        ordered = sorted(set(lock_ids))
        for lock_id in ordered:
            self._check_lock_id(lock_id)

        acquired_mutexes: list[int] = []
        published: list[int] = []
        try:
            # Phase 1: take all local mutexes in ascending order.
            for lock_id in ordered:
                self._local_mutexes[lock_id].acquire()
                acquired_mutexes.append(lock_id)

            # Phase 2: publish WAITING to every row (no blocking).
            for lock_id in ordered:
                self._publish_waiting(lock_id)
                published.append(lock_id)

            # Phase 3: spin until all rows are LOCKED (≈ one arbiter sweep).
            self._wait_for_locked_batch(ordered)

            yield
        finally:
            # Drop every published row back to IDLE, then release mutexes
            # in reverse acquisition order.
            for lock_id in published:
                self._publish_idle(lock_id)
            for lock_id in reversed(acquired_mutexes):
                self._local_mutexes[lock_id].release()

    def try_acquire(self, lock_id: int, timeout_s: float) -> Optional["_Held"]:
        """Non-blocking / bounded-wait variant. Returns None on timeout.

        Mostly useful for tests; production callers use `acquire()`.
        """
        self._check_lock_id(lock_id)
        local_mutex = self._local_mutexes[lock_id]
        if not local_mutex.acquire(timeout=timeout_s):
            return None
        try:
            self._publish_waiting(lock_id)
        except Exception:
            local_mutex.release()
            raise
        try:
            self._wait_for_locked(lock_id, timeout_override_s=timeout_s)
        except LockAcquisitionTimeout:
            self._publish_idle(lock_id)
            local_mutex.release()
            return None
        return _Held(self, lock_id)

    # -------- CXL row accessors -----------------------------------------

    def _row(self, lock_id: int, node_id: int) -> LockSlot:
        """Return the CXL-resident LockSlot for (lock_id, node_id)."""
        # Layout: row-major over (lock_id, node_id).
        row_idx = lock_id * self._max_nodes + node_id
        return self._lock_array[row_idx]

    def _row_address(self, lock_id: int, node_id: int) -> int:
        row_idx = lock_id * self._max_nodes + node_id
        return self._lock_array_base + row_idx * self._lock_slot_size

    def _check_lock_id(self, lock_id: int) -> None:
        if not 0 <= lock_id < self._num_locks:
            raise ValueError(f"lock_id {lock_id} out of range [0, {self._num_locks})")

    # -------- transitions ------------------------------------------------

    def _allocate_seq(self, lock_id: int) -> int:
        with self._next_seq_lock:
            s = self._next_seq[lock_id]
            # Wrap at u32 max; skip 0 since that's our sentinel.
            self._next_seq[lock_id] = s + 1 if s < 0xFFFFFFFE else 1
            return s

    def _publish_waiting(self, lock_id: int) -> None:
        row = self._row(lock_id, self._node_id)
        row.seq = self._allocate_seq(lock_id)
        row.state = LOCK_STATE_WAITING
        self._fence.fence_after_write(
            self._row_address(lock_id, self._node_id), self._lock_slot_size
        )

    def _publish_idle(self, lock_id: int) -> None:
        row = self._row(lock_id, self._node_id)
        row.state = LOCK_STATE_IDLE
        row.seq = 0
        self._fence.fence_after_write(
            self._row_address(lock_id, self._node_id), self._lock_slot_size
        )

    def _wait_for_locked(
        self, lock_id: int, timeout_override_s: Optional[float] = None
    ) -> None:
        row_addr = self._row_address(lock_id, self._node_id)
        timeout_s = (
            timeout_override_s
            if timeout_override_s is not None
            else self._config.acquire_timeout_s
        )
        deadline = time.monotonic() + timeout_s if timeout_s > 0 else None

        # Busy-spin the read loop. `time.sleep` even at 100 µs overshoots
        # to multi-ms on stock Linux schedulers, which dominated waiter
        # latency. The arbiter targets ~500 µs sweeps so the wait window
        # is short; spinning trades a hot CPU loop for ~50× lower latency
        # on the hot path.
        while True:
            self._fence.flush_before_read(row_addr, self._lock_slot_size)
            state = self._row(lock_id, self._node_id).state
            if state == LOCK_STATE_LOCKED:
                return
            if deadline is not None and time.monotonic() >= deadline:
                raise LockAcquisitionTimeout(
                    f"lock_id {lock_id} not granted within {timeout_s}s; "
                    "lock-manager may be stalled or dead"
                )

    def _wait_for_locked_batch(self, lock_ids: "list[int]") -> None:
        """Spin until every row in ``lock_ids`` is granted LOCKED.

        Tracks the still-pending set and rechecks only those each pass, so
        the batch completes as soon as the last grant lands (typically within
        one arbiter sweep, since distinct lock_ids are granted in the same
        sweep). Uses the same global ``acquire_timeout_s`` as the single-lock
        path; on timeout the caller's ``finally`` drops any rows already
        published to IDLE.

        Args:
            lock_ids: Distinct, sorted lock ids whose rows to await.

        Raises:
            LockAcquisitionTimeout: If any row is not granted in time.
        """
        timeout_s = self._config.acquire_timeout_s
        deadline = time.monotonic() + timeout_s if timeout_s > 0 else None
        pending = list(lock_ids)
        while pending:
            still_pending: list[int] = []
            for lock_id in pending:
                row_addr = self._row_address(lock_id, self._node_id)
                self._fence.flush_before_read(row_addr, self._lock_slot_size)
                if self._row(lock_id, self._node_id).state != LOCK_STATE_LOCKED:
                    still_pending.append(lock_id)
            if not still_pending:
                return
            pending = still_pending
            if deadline is not None and time.monotonic() >= deadline:
                raise LockAcquisitionTimeout(
                    f"batch lock_ids {pending} not all granted within "
                    f"{timeout_s}s; lock-manager may be stalled or dead"
                )


class _Held:
    """Handle returned by try_acquire; release() drops the lock."""

    def __init__(self, lock: TwoTierLock, lock_id: int):
        self._lock = lock
        self._lock_id = lock_id
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._lock._publish_idle(self._lock_id)
        self._lock._local_mutexes[self._lock_id].release()

    def __enter__(self) -> "_Held":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
