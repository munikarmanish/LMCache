# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the L1 (DRAM) -> L2 (CXL) eviction spill path.

Exercises ``L1EvictionController.execute_eviction_action`` for
``EvictionDestination.L2_CACHE``: keys are copied into the spill adapter
(before being deleted from L1) and then removed from L1. Failure/timeout
discard the keys anyway. Uses a real ``L1Manager`` and a lightweight fake
spill adapter that records what it received, so no real CXL backend is needed.
"""

# Standard
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
    L2StoreResult,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L1EvictionController,
)

# Skip all tests in this module if CUDA is not available (L1Manager needs it).
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)


# =============================================================================
# Helpers
# =============================================================================


def make_object_key(chunk_id: int) -> ObjectKey:
    """Create a test ObjectKey with the given chunk ID."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def make_layout() -> MemoryLayoutDesc:
    """Create a small MemoryLayoutDesc for testing."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


def write_keys_to_l1(
    l1_manager: L1Manager,
    keys: list[ObjectKey],
    layout: MemoryLayoutDesc,
) -> list[ObjectKey]:
    """Write keys to L1 (reserve_write + finish_write) and return written keys."""
    results = l1_manager.reserve_write(
        keys=keys,
        is_temporary=[False] * len(keys),
        layout_desc=layout,
        mode="new",
    )
    written = [k for k, (e, m) in results.items() if m is not None]
    if written:
        l1_manager.finish_write(written)
    return written


class FakeSpillAdapter:
    """Minimal stand-in for a spill-capable L2 adapter.

    Only ``spill_store`` is exercised by ``L1EvictionController``. Records the
    keys/objects it received and returns a configurable result. Optionally
    blocks to simulate a slow store for timeout testing.
    """

    def __init__(
        self,
        succeed: bool = True,
        block_s: float = 0.0,
    ):
        self.succeed = succeed
        self.block_s = block_s
        self.received_keys: list[ObjectKey] = []
        self.received_objects: list[object] = []
        self.call_count = 0

    def spill_store(self, keys, objects, timeout_s) -> L2StoreResult:
        self.call_count += 1
        self.received_keys = list(keys)
        self.received_objects = list(objects)
        if self.block_s > 0.0:
            # Simulate a store that overruns the caller's timeout. The real
            # adapter enforces timeout itself; the controller trusts the
            # returned result, so we model a timeout as a failure result.
            time.sleep(min(self.block_s, timeout_s))
            return L2StoreResult(success=False, bytes_transferred=0)
        if not self.succeed:
            return L2StoreResult(success=False, bytes_transferred=0)
        nbytes = sum(o.get_size() for o in objects)
        return L2StoreResult(success=True, bytes_transferred=nbytes)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def l1_manager():
    """Create an L1Manager with a reasonable memory config."""
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=128 * 1024 * 1024,
            use_lazy=torch.cuda.is_available(),
            init_size_in_bytes=64 * 1024 * 1024,
            align_bytes=0x1000,
        ),
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )
    mgr = L1Manager(config)
    yield mgr
    mgr.close()


def make_controller(
    l1_manager: L1Manager,
    spill_adapter,
    destination: str = "L2_CACHE",
    spill_timeout_s: float = 5.0,
) -> L1EvictionController:
    """Build an L1EvictionController without starting its background thread."""
    cfg = EvictionConfig(
        eviction_policy="LRU",
        eviction_destination=destination,
        spill_timeout_s=spill_timeout_s,
    )
    return L1EvictionController(
        l1_manager=l1_manager,
        eviction_config=cfg,
        spill_adapter=spill_adapter,
    )


def key_in_l1(l1_manager: L1Manager, key: ObjectKey) -> bool:
    """Return True iff ``key`` is currently resident (readable) in L1."""
    results = l1_manager.reserve_read([key])
    err, obj = results[key]
    if err == L1Error.SUCCESS:
        l1_manager.finish_read([key])
        return True
    return False


# =============================================================================
# Tests
# =============================================================================


class TestSpillStoreContract:
    """The spill path must not touch the async completion channel."""

    def test_fake_spill_store_returns_bytes(self, l1_manager):
        layout = make_layout()
        keys = [make_object_key(i) for i in range(2)]
        write_keys_to_l1(l1_manager, keys, layout)
        read = l1_manager.reserve_read(keys)
        objs = [read[k][1] for k in keys]

        adapter = FakeSpillAdapter(succeed=True)
        result = adapter.spill_store(keys, objs, 5.0)

        assert result.is_successful()
        assert result.bytes_transferred() == sum(o.get_size() for o in objs)
        l1_manager.finish_read(keys)


class TestExecuteEvictionActionSpill:
    """L2_CACHE branch of execute_eviction_action."""

    def test_happy_path_spills_then_deletes(self, l1_manager):
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        write_keys_to_l1(l1_manager, keys, layout)

        adapter = FakeSpillAdapter(succeed=True)
        controller = make_controller(l1_manager, adapter)

        controller.execute_eviction_action(
            EvictionAction(destination=EvictionDestination.L2_CACHE, keys=keys)
        )

        # Adapter received exactly the evicted keys.
        assert adapter.call_count == 1
        assert adapter.received_keys == keys
        # Keys are gone from L1.
        for k in keys:
            assert not key_in_l1(l1_manager, k)
        # No leaked read locks: the keys are deletable/evictable had they
        # remained (they're gone, so this is just a sanity check that
        # finish_read ran — a leaked lock would have blocked delete()).

    def test_store_failure_discards_keys(self, l1_manager):
        layout = make_layout()
        keys = [make_object_key(i) for i in range(2)]
        write_keys_to_l1(l1_manager, keys, layout)

        adapter = FakeSpillAdapter(succeed=False)
        controller = make_controller(l1_manager, adapter)

        controller.execute_eviction_action(
            EvictionAction(destination=EvictionDestination.L2_CACHE, keys=keys)
        )

        # Discard-on-failure: keys still leave L1 so memory is reclaimed.
        for k in keys:
            assert not key_in_l1(l1_manager, k)

    def test_timeout_discards_keys(self, l1_manager):
        layout = make_layout()
        keys = [make_object_key(i) for i in range(2)]
        write_keys_to_l1(l1_manager, keys, layout)

        # block longer than the timeout -> failure result -> discard.
        adapter = FakeSpillAdapter(succeed=True, block_s=1.0)
        controller = make_controller(l1_manager, adapter, spill_timeout_s=0.05)

        controller.execute_eviction_action(
            EvictionAction(destination=EvictionDestination.L2_CACHE, keys=keys)
        )

        for k in keys:
            assert not key_in_l1(l1_manager, k)

    def test_no_spill_adapter_discards(self, l1_manager):
        layout = make_layout()
        keys = [make_object_key(i) for i in range(2)]
        write_keys_to_l1(l1_manager, keys, layout)

        controller = make_controller(l1_manager, spill_adapter=None)

        controller.execute_eviction_action(
            EvictionAction(destination=EvictionDestination.L2_CACHE, keys=keys)
        )

        for k in keys:
            assert not key_in_l1(l1_manager, k)

    def test_write_locked_key_survives(self, l1_manager):
        """A write-locked key can't be read, so it is neither spilled nor
        deleted; the rest of the batch is spilled and evicted normally."""
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        write_keys_to_l1(l1_manager, keys[1:], layout)

        # keys[0] is reserved-for-write but never finished -> not readable.
        l1_manager.reserve_write(
            keys=[keys[0]],
            is_temporary=[False],
            layout_desc=layout,
            mode="new",
        )

        adapter = FakeSpillAdapter(succeed=True)
        controller = make_controller(l1_manager, adapter)

        controller.execute_eviction_action(
            EvictionAction(destination=EvictionDestination.L2_CACHE, keys=keys)
        )

        # reserve_read fails for the write-locked key, so it is not spilled.
        assert keys[0] not in adapter.received_keys
        assert set(adapter.received_keys) == {keys[1], keys[2]}
        # The two readable keys were spilled and evicted from L1.
        assert not key_in_l1(l1_manager, keys[1])
        assert not key_in_l1(l1_manager, keys[2])

    def test_read_locked_key_retained_not_deleted(self, l1_manager):
        """An externally read-locked key is copied to L2 (harmless,
        idempotent) but survives in L1 because delete() refuses locked keys.
        In the real loop such keys are pre-filtered by is_key_evictable."""
        layout = make_layout()
        keys = [make_object_key(i) for i in range(2)]
        write_keys_to_l1(l1_manager, keys, layout)

        locked = keys[0]
        l1_manager.reserve_read([locked])  # external read lock

        adapter = FakeSpillAdapter(succeed=True)
        controller = make_controller(l1_manager, adapter)

        controller.execute_eviction_action(
            EvictionAction(destination=EvictionDestination.L2_CACHE, keys=keys)
        )

        # The unlocked key is evicted; the externally-locked one survives.
        assert not key_in_l1(l1_manager, keys[1])
        assert key_in_l1(l1_manager, locked)
        l1_manager.finish_read([locked])

    def test_discard_destination_ignores_adapter(self, l1_manager):
        layout = make_layout()
        keys = [make_object_key(i) for i in range(2)]
        write_keys_to_l1(l1_manager, keys, layout)

        adapter = FakeSpillAdapter(succeed=True)
        controller = make_controller(l1_manager, adapter, destination="DISCARD")

        controller.execute_eviction_action(
            EvictionAction(destination=EvictionDestination.DISCARD, keys=keys)
        )

        # DISCARD path never calls the spill adapter.
        assert adapter.call_count == 0
        for k in keys:
            assert not key_in_l1(l1_manager, k)


class TestDestinationRegistrationGate:
    """L2_CACHE is only registered on the policy when a spill target exists."""

    def test_registered_when_adapter_present(self, l1_manager):
        adapter = FakeSpillAdapter(succeed=True)
        controller = make_controller(l1_manager, adapter, destination="L2_CACHE")
        # With the destination registered, the policy tags actions L2_CACHE.
        actions = controller._eviction_policy.get_eviction_actions(1.0)
        # No keys tracked yet -> no actions, but destinations list is set.
        assert EvictionDestination.L2_CACHE in controller._eviction_policy._destinations
        assert actions == []

    def test_not_registered_without_adapter(self, l1_manager):
        controller = make_controller(l1_manager, spill_adapter=None)
        assert (
            EvictionDestination.L2_CACHE
            not in controller._eviction_policy._destinations
        )

    def test_report_status_exposes_spill_fields(self, l1_manager):
        adapter = FakeSpillAdapter(succeed=True)
        controller = make_controller(l1_manager, adapter)
        status = controller.report_status()
        assert status["eviction_destination"] == "L2_CACHE"
        assert status["spill_adapter"] == "FakeSpillAdapter"
