# SPDX-License-Identifier: Apache-2.0
"""Lock-manager: single-writer arbiter for the CXL global_lock array.

One `LockManager` runs per rack (in production, elected; in single-node
tests, on node 0). It continuously sweeps the global_lock table and,
for each lock_id that has no current LOCKED holder but at least one
WAITING waiter, grants the lock to exactly one waiter by flipping its
state to LOCKED.

Why a separate thread (plan F2 / TraCT §3.3): CXL 2.0 lacks hardware
atomics across hosts, so workers cannot CAS the lock cells safely.
The lock-manager is a single writer to the LOCKED field, which is
what makes the protocol atomic without HW atomics.

Fairness: the manager grants the WAITING row with the smallest `seq`.
Since waiters assign `seq` monotonically at publish time, this
approximates FIFO within each lock_id. Ties (`seq == seq`, rare) are
broken by node_id — deterministic but slightly biased toward low ids;
acceptable given our expected contention levels.
"""

# Standard
import ctypes
import threading
import time
from dataclasses import dataclass
from typing import Optional

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


@dataclass
class LockManagerConfig:
    # Interval between full scans of the lock table. Keep low (<= 1 ms)
    # in production; tests can raise it to verify waiter behaviour under
    # deliberately-slow manager.
    sweep_interval_s: float = 0.0005  # 500 µs


class LockManager:
    """Single-writer arbiter for `global_lock[NUM_LOCKS][MAX_NODES]`.

    Not thread-safe to instantiate multiple of these against the same
    PoolHandle — by design, there must be exactly one live manager per
    rack. A second manager would race on the LOCKED writes. Tests that
    want to step the scan manually use `sweep_once()` instead of
    starting the background thread.
    """

    def __init__(
        self,
        handle: PoolHandle,
        *,
        fence: Optional[Fence] = None,
        config: Optional[LockManagerConfig] = None,
    ):
        self._handle = handle
        self._fence = fence or default_fence()
        self._config = config or LockManagerConfig()
        self._num_locks = handle.layout.num_locks
        self._max_nodes = handle.layout.max_nodes
        self._lock_array = handle.global_locks()
        self._lock_slot_size = ctypes.sizeof(LockSlot)
        self._lock_array_base = ctypes.addressof(self._lock_array)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Bookkeeping for observability / tests.
        self._grants = 0
        self._sweeps = 0

    # -------- lifecycle --------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("LockManager already started")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="cxl-lock-manager", daemon=True
        )
        self._thread.start()
        logger.info(
            "CXL lock-manager started (num_locks=%d, max_nodes=%d, busy-loop)",
            self._num_locks,
            self._max_nodes,
        )

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop_event.set()
        t = self._thread
        if t is not None:
            t.join(timeout_s)
            if t.is_alive():
                logger.warning("CXL lock-manager did not exit within %ss", timeout_s)
            self._thread = None

    # -------- single-step (tests) ---------------------------------------

    def sweep_once(self) -> int:
        """Run one full scan; return the number of grants issued.

        Exposed for deterministic tests. Real deployments use `start()`.
        """
        grants = 0
        for lock_id in range(self._num_locks):
            grants += self._arbitrate_lock_id(lock_id)
        self._sweeps += 1
        self._grants += grants
        return grants

    @property
    def stats(self) -> dict:
        return {"grants": self._grants, "sweeps": self._sweeps}

    # -------- internals --------------------------------------------------

    def _run(self) -> None:
        # Busy-loop the sweep. `Event.wait` with a sub-millisecond
        # timeout overshoots to multi-ms on stock Linux schedulers,
        # which inflated waiter latency by 30-50×. Check the stop
        # event each iteration so shutdown stays prompt.
        report_interval_s = 10.0
        last_report = time.perf_counter()
        # All per-window state — reset after each report.
        window_samples: list[int] = []
        sweep_ns_sum = 0
        sweep_ns_max = 0
        grants_in_window = 0
        while not self._stop_event.is_set():
            t0 = time.perf_counter_ns()
            g = self.sweep_once()
            dt = time.perf_counter_ns() - t0
            sweep_ns_sum += dt
            if dt > sweep_ns_max:
                sweep_ns_max = dt
            window_samples.append(dt)
            grants_in_window += g

            now = time.perf_counter()
            if now - last_report >= report_interval_s:
                n = len(window_samples)
                mean_us = (sweep_ns_sum / n) / 1000.0
                max_us = sweep_ns_max / 1000.0
                window_samples.sort()
                p50_us = window_samples[n // 2] / 1000.0
                p99_us = window_samples[max(0, int(n * 0.99) - 1)] / 1000.0
                sweeps_per_s = n / (now - last_report)
                grants_per_s = grants_in_window / (now - last_report)
                logger.info(
                    "CXL lock-manager: sweeps=%d sweeps/s=%.0f grants/s=%.1f "
                    "sweep_us[mean=%.1f p50=%.1f p99=%.1f max=%.1f]",
                    n,
                    sweeps_per_s,
                    grants_per_s,
                    mean_us,
                    p50_us,
                    p99_us,
                    max_us,
                )
                last_report = now
                window_samples = []
                sweep_ns_sum = 0
                sweep_ns_max = 0
                grants_in_window = 0

    def _arbitrate_lock_id(self, lock_id: int) -> int:
        """Examine one lock_id's row-of-nodes. Grant if possible.

        Returns 1 if a grant was issued, 0 otherwise.
        """
        # Flush all rows for this lock_id before reading.
        row_base = lock_id * self._max_nodes
        self._fence.flush_before_read(
            self._lock_array_base + row_base * self._lock_slot_size,
            self._max_nodes * self._lock_slot_size,
        )

        # First pass: check if anyone is currently LOCKED. If so,
        # there's nothing to do for this lock_id — the holder must
        # release before we can grant the next waiter.
        best_waiter_idx = -1
        best_waiter_seq = None
        for n in range(self._max_nodes):
            state = self._lock_array[row_base + n].state
            if state == LOCK_STATE_LOCKED:
                return 0  # someone already holds it; move on
            if state == LOCK_STATE_WAITING:
                seq = self._lock_array[row_base + n].seq
                if best_waiter_seq is None or seq < best_waiter_seq:
                    best_waiter_seq = seq
                    best_waiter_idx = n

        if best_waiter_idx < 0:
            return 0  # no waiters

        # Grant the winner.
        grant_row = self._lock_array[row_base + best_waiter_idx]
        grant_row.state = LOCK_STATE_LOCKED
        self._fence.fence_after_write(
            self._lock_array_base
            + (row_base + best_waiter_idx) * self._lock_slot_size,
            self._lock_slot_size,
        )
        return 1

    # -------- context-manager convenience -------------------------------

    def __enter__(self) -> "LockManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
