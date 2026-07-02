# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the L2-resident GPU-direct retrieve path.

These exercise the PrefetchController plan-split and the tier-info sidecar
using a ``ResidentMockL2Adapter`` that opts into
``supports_l2_resident_retrieve()``. No GPU / CXL device is required: the
mock stands in for an adapter that keeps its lookup pin and serves hits
GPU-direct, so the controller's "skip L1 reserve, keep the pin" behavior is
verified without real H2D.

The CXL-specific pin-vs-evict invariant (``evict`` refuses while
``pin_count > 0``) is covered separately under
``tests/v1/storage_backend/cxl/`` and on the CXL testbed.
"""

# Standard
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    L2ResidentTierInfo,
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
)
from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)


# =============================================================================
# Resident-capable mock adapter
# =============================================================================


class ResidentMockL2Adapter(MockL2Adapter):
    """MockL2Adapter that opts into the L2-resident retrieve path.

    ``submit_h2d`` returns a synthetic non-negative token on hit and records
    the call; ``release_after_h2d`` records released tokens. The base class's
    ``_locked_keys`` counter stands in for the CXL ``pin_count`` so tests can
    assert the lookup pin is held across the resident path and released by
    ``submit_unlock``.
    """

    def __init__(self, config: MockL2AdapterConfig):
        super().__init__(config)
        self.submit_h2d_calls: list[tuple[ObjectKey, int, int]] = []
        self.release_tokens: list[int] = []
        self._next_token = 0

    def supports_l2_resident_retrieve(self) -> bool:
        return True

    def submit_h2d(self, key: ObjectKey, gpu_ptr: int, dst_size: int) -> int:
        self.submit_h2d_calls.append((key, gpu_ptr, dst_size))
        if not self.debug_has_key(key):
            return -1
        token = self._next_token
        self._next_token += 1
        return token

    def release_after_h2d(self, token: int) -> None:
        if token < 0:
            return
        self.release_tokens.append(token)


# =============================================================================
# Helpers
# =============================================================================


def make_object_key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def make_layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


def make_resident_adapter() -> ResidentMockL2Adapter:
    return ResidentMockL2Adapter(
        MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)
    )


def make_plain_adapter() -> MockL2Adapter:
    return MockL2Adapter(MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0))


def make_descriptor(index: int) -> AdapterDescriptor:
    return AdapterDescriptor(
        index=index,
        config=MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0),
    )


def wait_for_condition(predicate, timeout: float = 5.0, poll: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return False


def store_keys_in_l2(adapter, keys, layout) -> None:
    if not keys:
        return
    objs = []
    for _ in keys:
        tensor = torch.randn(layout.shapes[0], dtype=layout.dtypes[0])
        metadata = MemoryObjMetadata(
            shape=layout.shapes[0],
            dtype=layout.dtypes[0],
            address=0,
            phy_size=tensor.nelement() * tensor.element_size(),
            ref_count=0,
        )
        objs.append(
            TensorMemoryObj(raw_data=tensor, metadata=metadata, parent_allocator=None)
        )
    adapter.submit_store_task(keys, objs)
    assert wait_for_condition(
        lambda: all(adapter.debug_has_key(k) for k in keys), timeout=5.0
    ), "Failed to store test data in L2 adapter"


def wait_for_result(ctrl, req_id, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = ctrl.query_prefetch_result(req_id)
        if result is not None:
            return result
        time.sleep(0.02)
    return None


@pytest.fixture
def l1_manager():
    mgr = L1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=128 * 1024 * 1024,
                use_lazy=torch.cuda.is_available(),
                init_size_in_bytes=64 * 1024 * 1024,
                align_bytes=0x1000,
            ),
            write_ttl_seconds=600,
            read_ttl_seconds=300,
        )
    )
    yield mgr
    mgr.close()


# =============================================================================
# Tier-info dataclass
# =============================================================================


class TestTierInfoShape:
    def test_empty_default(self):
        info = L2ResidentTierInfo()
        assert info.keys == ()
        assert info.adapter_indices == ()

    def test_populated(self):
        keys = (make_object_key(0), make_object_key(1))
        info = L2ResidentTierInfo(keys=keys, adapter_indices=(0, 0))
        assert info.keys == keys
        assert info.adapter_indices == (0, 0)


# =============================================================================
# Controller plan-split
# =============================================================================


class TestResidentPlanSplit:
    def test_all_resident_keeps_l1_empty(self, l1_manager):
        """All hits on a resident adapter -> nothing reserved/loaded in L1,
        pins kept, tier info reports every key."""
        adapter = make_resident_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(4)]
        store_keys_in_l2(adapter, keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()
        try:
            l1_before, _ = l1_manager.get_memory_usage()

            req_id = ctrl.submit_prefetch_request(keys, layout)
            result = wait_for_result(ctrl, req_id)
            assert result is not None
            assert result.count_leading_ones() == 4

            # L1 was never used as a staging buffer for resident keys.
            l1_after, _ = l1_manager.get_memory_usage()
            assert l1_after == l1_before

            # None of the keys are read-locked in L1 (they live on L2 only).
            for key in keys:
                assert not l1_manager.unsafe_read([key])[key][1]

            # The lookup pins are still held on the adapter.
            assert adapter.debug_get_locked_key_count() == 4

            # Tier info reports all four keys on adapter 0.
            tier = ctrl.query_prefetch_tier_info(req_id)
            assert tuple(tier.keys) == tuple(keys)
            assert tier.adapter_indices == (0, 0, 0, 0)
        finally:
            ctrl.stop()
            adapter.close()

    def test_resident_prefix_gap(self, l1_manager):
        """A gap in the resident hits trims the resident plan to the prefix."""
        adapter = make_resident_adapter()
        layout = make_layout()
        all_keys = [make_object_key(i) for i in range(5)]
        # Store 0,1,3,4 — gap at index 2.
        store_keys_in_l2(adapter, [all_keys[i] for i in (0, 1, 3, 4)], layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()
        try:
            req_id = ctrl.submit_prefetch_request(all_keys, layout)
            result = wait_for_result(ctrl, req_id)
            assert result is not None
            assert result.count_leading_ones() == 2

            tier = ctrl.query_prefetch_tier_info(req_id)
            assert tuple(tier.keys) == tuple(all_keys[:2])
            # Pins kept only for the retained prefix; the rest were unlocked.
            assert wait_for_condition(
                lambda: adapter.debug_get_locked_key_count() == 2, timeout=2.0
            )
        finally:
            ctrl.stop()
            adapter.close()

    def test_no_resident_tier_info_for_plain_adapter(self, l1_manager):
        """A non-resident adapter follows the L1-load path; tier info empty."""
        adapter = make_plain_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()
        try:
            req_id = ctrl.submit_prefetch_request(keys, layout)
            result = wait_for_result(ctrl, req_id)
            assert result is not None
            assert result.count_leading_ones() == 3

            # Keys were loaded into L1 (legacy path).
            for key in keys:
                assert l1_manager.unsafe_read([key])[key][1] is not None

            tier = ctrl.query_prefetch_tier_info(req_id)
            assert tier.keys == ()

            l1_manager.finish_read(keys)
        finally:
            ctrl.stop()
            adapter.close()


# =============================================================================
# Adapter primitives
# =============================================================================


class TestResidentAdapterPrimitives:
    def test_submit_and_release_round_trip(self):
        adapter = make_resident_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, keys, layout)

        tokens = [adapter.submit_h2d(k, 0, 0) for k in keys]
        assert all(t >= 0 for t in tokens)
        assert len(adapter.submit_h2d_calls) == 3

        adapter.release_after_h2d_batch(tokens)
        assert sorted(adapter.release_tokens) == sorted(tokens)
        adapter.close()

    def test_submit_h2d_miss_returns_negative(self):
        adapter = make_resident_adapter()
        token = adapter.submit_h2d(make_object_key(99), 0, 0)
        assert token == -1
        # Releasing a miss token is a no-op.
        adapter.release_after_h2d(token)
        assert adapter.release_tokens == []
        adapter.close()

    def test_default_adapter_does_not_support_resident(self):
        adapter = make_plain_adapter()
        assert adapter.supports_l2_resident_retrieve() is False
        with pytest.raises(NotImplementedError):
            adapter.submit_h2d(make_object_key(0), 0, 0)
        adapter.close()

    def test_submit_h2d_batch_matches_per_key(self):
        # The batched path (default loop over submit_h2d) must produce the
        # same tokens, one per key, with misses as -1 — and record one
        # submit_h2d call per chunk.
        adapter = make_resident_adapter()
        layout = make_layout()
        hit_keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, hit_keys, layout)
        miss = make_object_key(99)
        keys = [hit_keys[0], miss, hit_keys[1], hit_keys[2]]

        tokens = adapter.submit_h2d_batch(keys, [0, 0, 0, 0], [0, 0, 0, 0])
        assert len(tokens) == len(keys)
        assert tokens[1] == -1  # the miss
        assert all(t >= 0 for t in (tokens[0], tokens[2], tokens[3]))
        assert len(adapter.submit_h2d_calls) == 4

        adapter.release_after_h2d_batch(tokens)
        assert sorted(adapter.release_tokens) == sorted(t for t in tokens if t >= 0)
        adapter.close()

    def test_submit_h2d_batch_rejects_unequal_lengths(self):
        adapter = make_resident_adapter()
        with pytest.raises(ValueError):
            adapter.submit_h2d_batch([make_object_key(0)], [0, 0], [0])
        adapter.close()


# =============================================================================
# MLA multi-reader: pin counter held N times, released N times
# =============================================================================


class TestResidentMultiReader:
    def test_pin_counter_held_and_released_n_times(self):
        """The resident pin is a counter: pinning N times needs N unlocks
        before the key is fully released (mirrors CXL pin_count for MLA)."""
        adapter = make_resident_adapter()
        layout = make_layout()
        key = make_object_key(0)
        store_keys_in_l2(adapter, [key], layout)

        # Three readers each acquire the lookup lock for the same key.
        for _ in range(3):
            task = adapter.submit_lookup_and_lock_task([key])
            assert wait_for_condition(
                lambda t=task: adapter.query_lookup_and_lock_result(t) is not None
            )
        assert adapter.debug_get_locked_key_count() == 1  # one distinct key
        assert adapter._locked_keys[key] == 3  # held three times

        # Two releases still leave it pinned; the third frees it.
        adapter.submit_unlock([key])
        adapter.submit_unlock([key])
        assert wait_for_condition(lambda: adapter._locked_keys.get(key, 0) == 1)
        adapter.submit_unlock([key])
        assert wait_for_condition(
            lambda: adapter.debug_get_locked_key_count() == 0, timeout=2.0
        )
        adapter.close()
