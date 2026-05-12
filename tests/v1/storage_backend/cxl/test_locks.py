# SPDX-License-Identifier: Apache-2.0
"""Two-tier lock + lock-manager tests (plan F2).

These run entirely in-process with a tmpfile-backed pool. Two TwoTierLock
instances pointing at the same PoolHandle with different node_ids
simulate two hosts contending for a CXL-resident lock row.
"""

# Standard
import os
import tempfile
import threading
import time
from contextlib import contextmanager

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.cxl.bootstrap import (
    CXLBootstrapConfig,
    bootstrap_pool,
)
from lmcache.v1.storage_backend.cxl.layout import (
    LOCK_STATE_IDLE,
    LOCK_STATE_LOCKED,
    LOCK_STATE_WAITING,
)
from lmcache.v1.storage_backend.cxl.lock_manager import (
    LockManager,
    LockManagerConfig,
)
from lmcache.v1.storage_backend.cxl.locks import (
    LockAcquisitionTimeout,
    LockConfig,
    TwoTierLock,
)


POOL_SIZE = 64 * (1 << 20)
REGION_SIZE = 2 * (1 << 20)


def _metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="lock-test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(4, 2, 16, 4, 64),
        chunk_size=16,
    )


@pytest.fixture
def handle():
    with tempfile.NamedTemporaryFile(prefix="cxl-lock-", delete=False) as f:
        f.truncate(POOL_SIZE)
        path = f.name
    cfg = CXLBootstrapConfig(
        dev_path=path, region_size=REGION_SIZE, initialize=True
    )
    h = bootstrap_pool(cfg, _metadata())
    try:
        yield h
    finally:
        h.close()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


@contextmanager
def _running_manager(handle, sweep_interval_s=0.0005):
    cfg = LockManagerConfig(sweep_interval_s=sweep_interval_s)
    mgr = LockManager(handle, config=cfg)
    mgr.start()
    try:
        yield mgr
    finally:
        mgr.stop()


# ---------- basics ----------


def test_acquire_and_release_single_node(handle):
    lock = TwoTierLock(handle, node_id=0)
    with _running_manager(handle):
        with lock.acquire(lock_id=0):
            # Held — no assertion needed beyond "we got here".
            pass
        # After release, the row is back to IDLE.
        row = lock._row(0, 0)
        assert row.state == LOCK_STATE_IDLE


def test_rejects_invalid_node_id(handle):
    with pytest.raises(ValueError):
        TwoTierLock(handle, node_id=handle.layout.max_nodes)
    with pytest.raises(ValueError):
        TwoTierLock(handle, node_id=-1)


def test_rejects_invalid_lock_id(handle):
    lock = TwoTierLock(handle, node_id=0)
    with _running_manager(handle):
        with pytest.raises(ValueError):
            with lock.acquire(lock_id=handle.layout.num_locks):
                pass


# ---------- exclusion within one node ----------


def test_two_threads_same_node_same_lock_id_serialize(handle):
    lock = TwoTierLock(handle, node_id=0)
    held = threading.Event()
    releasing = threading.Event()
    observations = []

    def first():
        with lock.acquire(lock_id=3):
            held.set()
            observations.append("first-in")
            # Hold long enough for second thread to attempt.
            time.sleep(0.05)
            observations.append("first-out")

    def second():
        held.wait()
        # Should block until first releases.
        t0 = time.monotonic()
        with lock.acquire(lock_id=3):
            elapsed = time.monotonic() - t0
            observations.append(f"second-in-after-{elapsed:.3f}")

    with _running_manager(handle):
        t1 = threading.Thread(target=first)
        t2 = threading.Thread(target=second)
        t1.start()
        t2.start()
        t1.join(2)
        t2.join(2)

    # The critical sections must not interleave.
    assert observations[0] == "first-in"
    assert observations[1] == "first-out"
    assert observations[2].startswith("second-in-after-")


# ---------- exclusion across "nodes" ----------


def test_two_nodes_same_lock_id_serialize(handle):
    lock_a = TwoTierLock(handle, node_id=0)
    lock_b = TwoTierLock(handle, node_id=1)

    a_in = threading.Event()
    observations = []

    def node_a():
        with lock_a.acquire(lock_id=5):
            a_in.set()
            observations.append("A-in")
            time.sleep(0.05)
            observations.append("A-out")

    def node_b():
        a_in.wait()
        # Should block while A holds.
        t0 = time.monotonic()
        with lock_b.acquire(lock_id=5):
            elapsed = time.monotonic() - t0
            observations.append(f"B-in-after-{elapsed:.3f}")

    with _running_manager(handle):
        ta = threading.Thread(target=node_a)
        tb = threading.Thread(target=node_b)
        ta.start()
        tb.start()
        ta.join(2)
        tb.join(2)

    assert observations[0] == "A-in"
    assert observations[1] == "A-out"
    assert observations[2].startswith("B-in-after-")
    # B waited at least ~50 ms.
    waited = float(observations[2].split("-")[-1])
    assert waited >= 0.03, f"B did not wait long enough: {waited}"


# ---------- parallelism across lock_ids ----------


def test_different_lock_ids_do_not_block_each_other(handle):
    lock = TwoTierLock(handle, node_id=0)
    barrier = threading.Barrier(2)

    def hold(lock_id):
        with lock.acquire(lock_id=lock_id):
            barrier.wait(timeout=1)  # both must be inside simultaneously

    with _running_manager(handle):
        t1 = threading.Thread(target=hold, args=(1,))
        t2 = threading.Thread(target=hold, args=(2,))
        t1.start()
        t2.start()
        t1.join(2)
        t2.join(2)
    # No exception from the barrier means both were concurrently inside.


# ---------- manager behaviour ----------


def test_sweep_once_grants_a_single_waiter(handle):
    lock_a = TwoTierLock(handle, node_id=0, config=LockConfig(acquire_timeout_s=5))
    mgr = LockManager(handle)

    # Make node 0 ask for lock_id=7 but don't yet start the background
    # manager; we'll step manually.
    entered = threading.Event()

    def waiter():
        with lock_a.acquire(lock_id=7):
            entered.set()

    t = threading.Thread(target=waiter)
    t.start()
    # Spin until the waiter has published WAITING.
    deadline = time.time() + 1.0
    while time.time() < deadline:
        if lock_a._row(7, 0).state == LOCK_STATE_WAITING:
            break
        time.sleep(0.001)
    assert lock_a._row(7, 0).state == LOCK_STATE_WAITING

    # Manual sweep: should grant exactly one lock.
    grants = mgr.sweep_once()
    assert grants == 1
    t.join(2)
    assert entered.is_set()


def test_sweep_does_not_grant_when_someone_locked(handle):
    lock_a = TwoTierLock(handle, node_id=0)
    lock_b = TwoTierLock(handle, node_id=1, config=LockConfig(acquire_timeout_s=5))
    mgr = LockManager(handle)

    # Node A gets the lock via the background manager.
    with _running_manager(handle):
        a_in = threading.Event()
        a_release = threading.Event()

        def holder():
            with lock_a.acquire(lock_id=9):
                a_in.set()
                a_release.wait(1)

        t_a = threading.Thread(target=holder)
        t_a.start()
        a_in.wait(1)
        # Now publish B's WAITING while A still holds LOCKED.
        # (mgr from the `with` block is running; we'll still sanity-check
        # via sweep_once() after stopping it.)

    # Manager is stopped now, but A already released (end of with-block
    # blocked on a_release). Let's restructure: instead, do this with
    # the background manager off entirely.

    # Reset the pool state for a cleaner scenario.
    for i in range(handle.layout.max_nodes):
        row = lock_a._row(9, i)
        row.state = LOCK_STATE_IDLE
        row.seq = 0

    # Manually put node 0 into LOCKED and node 1 into WAITING.
    row_a = lock_a._row(9, 0)
    row_a.state = LOCK_STATE_LOCKED
    row_a.seq = 1
    row_b = lock_b._row(9, 1)
    row_b.state = LOCK_STATE_WAITING
    row_b.seq = 2

    # Sweep must NOT flip row_b while row_a is LOCKED.
    grants = mgr.sweep_once()
    assert grants == 0
    assert row_b.state == LOCK_STATE_WAITING

    # After A releases, sweep grants B.
    row_a.state = LOCK_STATE_IDLE
    row_a.seq = 0
    assert mgr.sweep_once() == 1
    assert row_b.state == LOCK_STATE_LOCKED


def test_fairness_smallest_seq_wins(handle):
    """Manager prefers the WAITING row with the smallest seq."""
    lock_a = TwoTierLock(handle, node_id=0)
    lock_b = TwoTierLock(handle, node_id=1)
    lock_c = TwoTierLock(handle, node_id=2)
    mgr = LockManager(handle)

    # Three simultaneous waiters on the same lock_id, ordered B, A, C.
    row_b = lock_b._row(11, 1)
    row_b.state = LOCK_STATE_WAITING
    row_b.seq = 1
    row_a = lock_a._row(11, 0)
    row_a.state = LOCK_STATE_WAITING
    row_a.seq = 2
    row_c = lock_c._row(11, 2)
    row_c.state = LOCK_STATE_WAITING
    row_c.seq = 3

    # Grant 1: B wins (seq=1).
    assert mgr.sweep_once() == 1
    assert row_b.state == LOCK_STATE_LOCKED
    assert row_a.state == LOCK_STATE_WAITING
    assert row_c.state == LOCK_STATE_WAITING

    # Another sweep with B still locked: no grant.
    assert mgr.sweep_once() == 0

    # B releases → A (seq=2) wins next.
    row_b.state = LOCK_STATE_IDLE
    assert mgr.sweep_once() == 1
    assert row_a.state == LOCK_STATE_LOCKED
    assert row_c.state == LOCK_STATE_WAITING

    # A releases → C wins.
    row_a.state = LOCK_STATE_IDLE
    assert mgr.sweep_once() == 1
    assert row_c.state == LOCK_STATE_LOCKED


# ---------- timeouts and cleanup ----------


def test_waiter_times_out_if_no_manager(handle):
    """Without a manager to grant, acquire() eventually raises."""
    lock = TwoTierLock(
        handle, node_id=0, config=LockConfig(acquire_timeout_s=0.1)
    )
    with pytest.raises(LockAcquisitionTimeout):
        with lock.acquire(lock_id=13):
            pytest.fail("should not enter critical section")
    # And the row must be IDLE now — timeout must have called _publish_idle.
    assert lock._row(13, 0).state == LOCK_STATE_IDLE


def test_exception_in_critical_section_still_releases(handle):
    lock = TwoTierLock(handle, node_id=0)
    with _running_manager(handle):
        with pytest.raises(RuntimeError):
            with lock.acquire(lock_id=15):
                raise RuntimeError("boom")
        # The row is IDLE again.
        assert lock._row(15, 0).state == LOCK_STATE_IDLE
        # We can acquire again without deadlocking.
        with lock.acquire(lock_id=15):
            pass


def test_try_acquire_returns_none_on_timeout(handle):
    lock = TwoTierLock(handle, node_id=0)
    # No manager running → try_acquire should return None after timeout.
    held = lock.try_acquire(lock_id=17, timeout_s=0.1)
    assert held is None
    assert lock._row(17, 0).state == LOCK_STATE_IDLE


def test_try_acquire_succeeds_with_manager(handle):
    lock = TwoTierLock(handle, node_id=0)
    with _running_manager(handle):
        held = lock.try_acquire(lock_id=19, timeout_s=1.0)
        assert held is not None
        try:
            # While held, a second try_acquire from the same node times out
            # (local mutex blocks it).
            held2 = lock.try_acquire(lock_id=19, timeout_s=0.05)
            assert held2 is None
        finally:
            held.release()
        # Now free; try_acquire works again.
        held3 = lock.try_acquire(lock_id=19, timeout_s=1.0)
        assert held3 is not None
        held3.release()


# ---------- soak ----------


def test_concurrent_soak_has_no_overlap(handle):
    """Many threads across multiple simulated nodes → mutual exclusion holds.

    Each critical section bumps a counter and verifies that it was 1
    before the bump (no one else is inside). At the end, the counter
    should equal the exact number of acquires issued.
    """
    nodes = 4
    acquires_per_thread = 50
    lock_id = 23
    locks = [
        TwoTierLock(handle, node_id=n, config=LockConfig(acquire_timeout_s=10))
        for n in range(nodes)
    ]

    counter = {"n": 0}
    inside = {"n": 0}
    inside_lock = threading.Lock()  # only to protect the *assertions*
    overlaps = []

    def worker(lock):
        for _ in range(acquires_per_thread):
            with lock.acquire(lock_id=lock_id):
                with inside_lock:
                    inside["n"] += 1
                    if inside["n"] != 1:
                        overlaps.append(inside["n"])
                # Tiny work to widen the window.
                time.sleep(0.0005)
                with inside_lock:
                    counter["n"] += 1
                    inside["n"] -= 1

    with _running_manager(handle, sweep_interval_s=0.0001):
        threads = [threading.Thread(target=worker, args=(locks[n],)) for n in range(nodes)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)

    assert overlaps == [], f"overlapping critical sections: {overlaps[:5]}"
    assert counter["n"] == nodes * acquires_per_thread
