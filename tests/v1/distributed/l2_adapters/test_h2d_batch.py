# SPDX-License-Identifier: Apache-2.0
"""Tests for the batched L2-resident H2D path.

Covers the provider-agnostic batching contract added to collapse the
per-chunk ``submit_h2d`` round-trips of the L2-resident retrieve:

  - ``L2AdapterInterface.submit_h2d_batch`` default (loops ``submit_h2d``),
    its length validation, and miss (``-1``) propagation.
  - An adapter that overrides ``submit_h2d_batch`` is honored.

These run on CPU with a tiny stub adapter — no GPU / CXL needed (the real
``cudaMemcpyAsync`` is exercised by the CXL e2e path). The storage-manager
grouping/scatter is verified in ``test_storage_manager`` tests.
"""

# Standard

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface


def _key(int_hash: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(int_hash),
        model_name="h2d-batch-test",
        kv_rank=0,
    )


class _StubH2DAdapter(L2AdapterInterface):
    """Minimal adapter exposing only the h2d surface for these tests.

    ``submit_h2d`` returns a fresh token per known key and records the
    call args; an unknown key returns ``-1`` (miss). Everything else the
    abstract base requires is stubbed to satisfy instantiation.
    """

    def __init__(self, known: set[ObjectKey]):
        super().__init__(max_capacity_bytes=0)
        self._known = known
        self._next = 0
        self.calls: list[tuple[ObjectKey, int, int]] = []

    def submit_h2d(self, key: ObjectKey, gpu_ptr: int, dst_size: int) -> int:
        self.calls.append((key, gpu_ptr, dst_size))
        if key not in self._known:
            return -1
        token = self._next
        self._next += 1
        return token

    # --- abstract members: not exercised here ---
    def get_store_event_fd(self) -> int:
        return -1

    def get_lookup_and_lock_event_fd(self) -> int:
        return -1

    def get_load_event_fd(self) -> int:
        return -1

    def submit_store_task(self, keys, objects):
        raise NotImplementedError

    def pop_completed_store_tasks(self):
        raise NotImplementedError

    def submit_lookup_and_lock_task(self, keys):
        raise NotImplementedError

    def query_lookup_and_lock_result(self, task_id):
        raise NotImplementedError

    def submit_unlock(self, keys):
        raise NotImplementedError

    def submit_load_task(self, keys, objects):
        raise NotImplementedError

    def query_load_result(self, task_id):
        raise NotImplementedError

    def close(self):
        pass


def test_default_batch_loops_submit_h2d():
    k0, k1, k2 = _key(0), _key(1), _key(2)
    adapter = _StubH2DAdapter(known={k0, k1, k2})
    tokens = adapter.submit_h2d_batch([k0, k1, k2], [100, 200, 300], [64, 64, 64])
    # One token per key, in order; the default fans out to submit_h2d.
    assert tokens == [0, 1, 2]
    assert adapter.calls == [(k0, 100, 64), (k1, 200, 64), (k2, 300, 64)]


def test_default_batch_propagates_misses():
    k_hit, k_miss = _key(0), _key(99)
    adapter = _StubH2DAdapter(known={k_hit})
    tokens = adapter.submit_h2d_batch([k_hit, k_miss], [10, 20], [64, 64])
    # Miss -> -1, hit -> a real token, positions preserved.
    assert tokens[1] == -1
    assert tokens[0] >= 0


def test_default_batch_rejects_unequal_lengths():
    adapter = _StubH2DAdapter(known=set())
    with pytest.raises(ValueError):
        adapter.submit_h2d_batch([_key(0)], [1, 2], [64])


def test_override_is_honored():
    class _BatchAdapter(_StubH2DAdapter):
        def submit_h2d_batch(self, keys, gpu_ptrs, dst_sizes):
            self.batch_called = True
            return [42] * len(keys)

    adapter = _BatchAdapter(known={_key(0)})
    tokens = adapter.submit_h2d_batch([_key(0), _key(1)], [1, 2], [64, 64])
    assert tokens == [42, 42]
    assert adapter.batch_called
    # The override did NOT fall through to per-key submit_h2d.
    assert adapter.calls == []
