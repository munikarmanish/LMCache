# SPDX-License-Identifier: Apache-2.0
"""Tests for the CXL L2 adapter (MP-mode integration).

Exercises:
  - Config parsing + factory dispatch via create_l2_adapter.
  - The async surface: submit_store / submit_lookup_and_lock /
    submit_unlock / submit_load with eventfd notifications and
    bitmap result polling.
  - Round-trip integrity: store bytes, load them, compare.
"""

# Standard
import os
import select
import tempfile
import threading
import time
from typing import Optional

# Third Party
import pytest
import torch

# First Party
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.cxl_l2_adapter import (
    CXLL2Adapter,
    CXLL2AdapterConfig,
    build_cxl_adapter_from_config,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)


POOL_SIZE = 64 * (1 << 20)
REGION_SIZE = 2 * (1 << 20)
CHUNK_SIZE = 64 * 1024


def _make_config(path: str, *, initialize: bool, run_lock_manager: bool) -> CXLL2AdapterConfig:
    return CXLL2AdapterConfig(
        dev_path=path,
        node_id=0,
        chunk_size_bytes=CHUNK_SIZE,
        region_size=REGION_SIZE,
        initialize=initialize,
        generation=1,
        run_lock_manager=run_lock_manager,
        model_name="cxl-l2-test",
        world_size=1,
        kv_dtype_str="torch.float16",
        kv_shape=(4, 2, 16, 4, 64),
        use_mla=False,
        cluster_chunk_size=16,
    )


def _make_object_key(int_hash: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(int_hash),
        model_name="cxl-l2-test",
        kv_rank=0,
    )


def _make_payload_obj(size_bytes: int, fill_byte: int) -> TensorMemoryObj:
    data = torch.full((size_bytes,), fill_byte, dtype=torch.uint8)
    meta = MemoryObjMetadata(
        shape=torch.Size([size_bytes]),
        dtype=torch.uint8,
        address=data.data_ptr(),
        phy_size=size_bytes,
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.KV_2LTD,
    )
    return TensorMemoryObj(raw_data=data, metadata=meta, parent_allocator=None)


def _empty_dst_obj(size_bytes: int) -> TensorMemoryObj:
    """A zero-filled buffer the adapter will copy into."""
    data = torch.zeros((size_bytes,), dtype=torch.uint8)
    meta = MemoryObjMetadata(
        shape=torch.Size([size_bytes]),
        dtype=torch.uint8,
        address=data.data_ptr(),
        phy_size=size_bytes,
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.KV_2LTD,
    )
    return TensorMemoryObj(raw_data=data, metadata=meta, parent_allocator=None)


def _wait_efd(efd: int, timeout_s: float = 5.0) -> None:
    """Block until the eventfd is signalled (and consume the count)."""
    poller = select.poll()
    poller.register(efd, select.POLLIN)
    events = poller.poll(int(timeout_s * 1000))
    assert events, f"eventfd not signalled within {timeout_s}s"
    # Drain so subsequent waits don't return immediately.
    os.eventfd_read(efd)


@pytest.fixture
def adapter():
    with tempfile.NamedTemporaryFile(prefix="cxl-l2-", delete=False) as f:
        f.truncate(POOL_SIZE)
        path = f.name
    cfg = _make_config(path, initialize=True, run_lock_manager=True)
    a = build_cxl_adapter_from_config(cfg)
    try:
        yield a
    finally:
        a.close()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


# ---------- factory wiring ----------


def test_factory_dispatches_cxl_config(tmp_path):
    pool_path = tmp_path / "pool"
    pool_path.touch()
    os.truncate(pool_path, POOL_SIZE)
    cfg = _make_config(str(pool_path), initialize=True, run_lock_manager=True)

    a = create_l2_adapter(cfg, l1_memory_desc=None)
    try:
        assert isinstance(a, CXLL2Adapter)
        # Distinct event fds (per base-class invariant).
        fds = {a.get_store_event_fd(), a.get_lookup_and_lock_event_fd(), a.get_load_event_fd()}
        assert len(fds) == 3
    finally:
        a.close()


def test_config_from_dict_parses_required_fields(tmp_path):
    pool_path = str(tmp_path / "pool")
    spec = {
        "type": "cxl",
        "dev_path": pool_path,
        "node_id": 3,
        "chunk_size_bytes": 64 * 1024,
        "region_size": 2 * (1 << 20),
        "model_name": "m",
        "world_size": 2,
        "kv_dtype_str": "torch.float16",
        "kv_shape": [4, 2, 16, 4, 64],
        "cluster_chunk_size": 16,
    }
    cfg = CXLL2AdapterConfig.from_dict(spec)
    assert cfg.dev_path == pool_path
    assert cfg.node_id == 3
    assert cfg.kv_shape == (4, 2, 16, 4, 64)
    # Optional fields defaulted.
    assert cfg.use_mla is False
    assert cfg.run_lock_manager is False


def test_config_from_dict_rejects_missing_fields():
    with pytest.raises(ValueError, match="dev_path"):
        CXLL2AdapterConfig.from_dict({"type": "cxl"})


def test_config_help_string_mentions_required_fields():
    h = CXLL2AdapterConfig.help()
    assert "dev_path" in h
    assert "node_id" in h
    assert "chunk_size_bytes" in h


# ---------- store / lookup / load round-trip ----------


def test_store_then_lookup_then_load_round_trip(adapter):
    keys = [_make_object_key(0xC001 + i) for i in range(3)]
    payloads = [_make_payload_obj(1024, fill_byte=0x40 + i) for i in range(3)]

    # Store
    store_id = adapter.submit_store_task(keys, payloads)
    _wait_efd(adapter.get_store_event_fd())
    store_results = adapter.pop_completed_store_tasks()
    assert store_id in store_results
    assert store_results[store_id] is True

    # Lookup-and-lock — bitmap should be all 1s.
    lookup_id = adapter.submit_lookup_and_lock_task(keys)
    _wait_efd(adapter.get_lookup_and_lock_event_fd())
    bitmap = adapter.query_lookup_and_lock_result(lookup_id)
    assert bitmap is not None
    assert bitmap.popcount() == 3

    # Load into fresh dst buffers.
    dst_objs = [_empty_dst_obj(1024) for _ in range(3)]
    load_id = adapter.submit_load_task(keys, dst_objs)
    _wait_efd(adapter.get_load_event_fd())
    load_bitmap = adapter.query_load_result(load_id)
    assert load_bitmap is not None
    assert load_bitmap.popcount() == 3

    # Bytes match the original fill.
    for i, dst in enumerate(dst_objs):
        assert int(dst.raw_data[0]) == 0x40 + i, f"dst[{i}] wrong fill"

    # Unlock so subsequent eviction can proceed.
    adapter.submit_unlock(keys)


def test_lookup_misses_for_unstored_keys(adapter):
    keys = [_make_object_key(0xD001 + i) for i in range(4)]
    lookup_id = adapter.submit_lookup_and_lock_task(keys)
    _wait_efd(adapter.get_lookup_and_lock_event_fd())
    bitmap = adapter.query_lookup_and_lock_result(lookup_id)
    assert bitmap is not None
    assert bitmap.popcount() == 0


def test_lookup_hits_only_for_stored_keys(adapter):
    keys = [_make_object_key(0xE001 + i) for i in range(4)]
    # Store only the first two.
    store_id = adapter.submit_store_task(
        keys[:2], [_make_payload_obj(512, 0x11), _make_payload_obj(512, 0x22)]
    )
    _wait_efd(adapter.get_store_event_fd())
    adapter.pop_completed_store_tasks()

    lookup_id = adapter.submit_lookup_and_lock_task(keys)
    _wait_efd(adapter.get_lookup_and_lock_event_fd())
    bitmap = adapter.query_lookup_and_lock_result(lookup_id)
    assert bitmap is not None
    # First two hit, last two miss.
    assert bitmap.test(0)
    assert bitmap.test(1)
    assert not bitmap.test(2)
    assert not bitmap.test(3)
    adapter.submit_unlock(keys[:2])


def test_load_misses_return_zero_bits(adapter):
    keys = [_make_object_key(0xF001 + i) for i in range(2)]
    # Store only the first.
    store_id = adapter.submit_store_task(keys[:1], [_make_payload_obj(256, 0xAA)])
    _wait_efd(adapter.get_store_event_fd())
    adapter.pop_completed_store_tasks()

    dst_objs = [_empty_dst_obj(256) for _ in range(2)]
    load_id = adapter.submit_load_task(keys, dst_objs)
    _wait_efd(adapter.get_load_event_fd())
    bitmap = adapter.query_load_result(load_id)
    assert bitmap is not None
    assert bitmap.test(0)
    assert not bitmap.test(1)
    # Loaded byte for the hit matches; the missed dst remains zero.
    assert int(dst_objs[0].raw_data[0]) == 0xAA
    assert int(dst_objs[1].raw_data[0]) == 0


def test_unlock_releases_pin_so_eviction_can_proceed(adapter):
    """After unlock, the backend's remove() succeeds."""
    keys = [_make_object_key(0xAB10)]
    store_id = adapter.submit_store_task(keys, [_make_payload_obj(256, 0x33)])
    _wait_efd(adapter.get_store_event_fd())
    adapter.pop_completed_store_tasks()

    # Lock pins the slot.
    lookup_id = adapter.submit_lookup_and_lock_task(keys)
    _wait_efd(adapter.get_lookup_and_lock_event_fd())
    bitmap = adapter.query_lookup_and_lock_result(lookup_id)
    assert bitmap is not None and bitmap.test(0)

    backend = adapter.debug_get_backend()
    # First Party
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.distributed.l2_adapters.cxl_l2_adapter import (
        _object_key_to_cache_engine_key,
    )

    ce_key = _object_key_to_cache_engine_key(keys[0], adapter._metadata)
    # Pinned: remove must refuse.
    assert not backend.remove(ce_key)

    # Unlock and wait for the bg loop to process.
    adapter.submit_unlock(keys)
    deadline = time.time() + 1.0
    while time.time() < deadline:
        if backend.remove(ce_key):
            break
        time.sleep(0.01)
    else:
        pytest.fail("remove() did not succeed after unlock within 1s")


# ---------- task id semantics ----------


def test_query_returns_none_after_first_pop(adapter):
    keys = [_make_object_key(0xAB20)]
    lookup_id = adapter.submit_lookup_and_lock_task(keys)
    _wait_efd(adapter.get_lookup_and_lock_event_fd())
    first = adapter.query_lookup_and_lock_result(lookup_id)
    assert first is not None
    second = adapter.query_lookup_and_lock_result(lookup_id)
    assert second is None  # contract: not idempotent


def test_pop_completed_store_drains(adapter):
    keys = [_make_object_key(0xAB30 + i) for i in range(2)]
    payloads = [_make_payload_obj(256, 0x55) for _ in range(2)]
    id1 = adapter.submit_store_task(keys[:1], payloads[:1])
    id2 = adapter.submit_store_task(keys[1:], payloads[1:])

    # Drain the eventfd as we go, polling pop_completed_store_tasks
    # until both task ids are accounted for. This is the single-poller
    # idiom (the store controller polls the fd, drains the dict).
    collected: dict[int, bool] = {}
    deadline = time.time() + 5.0
    while time.time() < deadline and not (id1 in collected and id2 in collected):
        # Drain whatever's in the fd (don't block on EAGAIN).
        try:
            os.eventfd_read(adapter.get_store_event_fd())
        except BlockingIOError:
            pass
        collected.update(adapter.pop_completed_store_tasks())
        if id1 not in collected or id2 not in collected:
            time.sleep(0.005)

    assert id1 in collected and id2 in collected
    # Second pop returns empty.
    assert adapter.pop_completed_store_tasks() == {}


def test_distinct_event_fds(adapter):
    fds = {
        adapter.get_store_event_fd(),
        adapter.get_lookup_and_lock_event_fd(),
        adapter.get_load_event_fd(),
    }
    assert len(fds) == 3


# ---------- concurrency ----------


def _poll_until(fn, timeout_s: float = 5.0, interval_s: float = 0.001):
    """Poll `fn()` until it returns a truthy value or the deadline expires."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        v = fn()
        if v is not None:
            return v
        time.sleep(interval_s)
    raise TimeoutError(f"polling timed out after {timeout_s}s")


def test_concurrent_store_and_load(adapter):
    """Stress: many store+load round-trips run concurrently. Verifies
    no torn data, no task-id collisions, and that the adapter handles
    overlapping submissions correctly.

    Realistic threading model: a single dispatcher thread drains
    `pop_completed_store_tasks` (this matches the L1 manager's store
    controller), and worker threads drive lookup/load directly via
    per-task `query_*` calls (which DO support multiple readers).
    """
    num_threads = 4
    per_thread = 20
    errors = []

    # Single dispatcher: collects store completions into a shared dict
    # keyed by task_id so workers can poll their own.
    store_results: dict[int, bool] = {}
    store_results_lock = threading.Lock()
    stop_dispatcher = threading.Event()

    def store_dispatcher():
        while not stop_dispatcher.is_set():
            done = adapter.pop_completed_store_tasks()
            if done:
                with store_results_lock:
                    store_results.update(done)
            time.sleep(0.001)

    def my_store_result(task_id: int):
        with store_results_lock:
            return store_results.pop(task_id, None)

    def worker(tid: int):
        try:
            for i in range(per_thread):
                base = 0x10000 * (tid + 1) + i
                key = _make_object_key(base)
                payload = _make_payload_obj(256, fill_byte=base & 0xFF)

                store_id = adapter.submit_store_task([key], [payload])
                _poll_until(lambda: my_store_result(store_id))

                lookup_id = adapter.submit_lookup_and_lock_task([key])
                bitmap = _poll_until(
                    lambda: adapter.query_lookup_and_lock_result(lookup_id)
                )
                assert bitmap.test(0)

                dst = _empty_dst_obj(256)
                load_id = adapter.submit_load_task([key], [dst])
                load_bm = _poll_until(
                    lambda: adapter.query_load_result(load_id)
                )
                assert load_bm.test(0)
                assert int(dst.raw_data[0]) == base & 0xFF

                adapter.submit_unlock([key])
        except Exception as e:
            errors.append((tid, e))

    dispatcher = threading.Thread(target=store_dispatcher, daemon=True)
    dispatcher.start()
    try:
        threads = [
            threading.Thread(target=worker, args=(t,)) for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        assert errors == [], f"thread errors: {errors[:3]}"
    finally:
        stop_dispatcher.set()
        dispatcher.join(2)
