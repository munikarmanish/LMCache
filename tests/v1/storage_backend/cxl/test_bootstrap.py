# SPDX-License-Identifier: Apache-2.0
"""Bootstrap tests for the CXL pool.

These exercise mmap + header init + re-attach semantics on a tmpfile;
no CUDA and no real CXL device are required. cudaHostRegister is
best-effort inside the bootstrap path and logged-not-raised when CUDA
is unavailable, so tests can run anywhere.
"""

# Standard
import os
import tempfile

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.cxl.bootstrap import (
    CXLBootstrapConfig,
    bootstrap_pool,
    compute_geom_hash,
)
from lmcache.v1.storage_backend.cxl.layout import (
    MAGIC,
    OWNER_FREE,
    SLOT_STATE_EMPTY,
)


POOL_SIZE = 64 * (1 << 20)  # 64 MiB — big enough for meaningful region_size
REGION_SIZE = 2 * (1 << 20)  # 2 MiB regions


def _metadata(model_name="test-model", chunk_size=16) -> LMCacheMetadata:
    # (num_layers, kv_size, chunk_size, num_heads, head_size)
    return LMCacheMetadata(
        model_name=model_name,
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(4, 2, chunk_size, 4, 64),
        chunk_size=chunk_size,
    )


@pytest.fixture
def pool_path():
    # Regular tmpfile is a valid stand-in for /dev/dax0.0 for the
    # purpose of mmap-based shared-memory tests.
    with tempfile.NamedTemporaryFile(prefix="cxl-pool-", delete=False) as f:
        f.truncate(POOL_SIZE)
        path = f.name
    try:
        yield path
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


# ---------- geom hash ----------


def test_geom_hash_is_deterministic():
    m1 = _metadata()
    m2 = _metadata()
    assert compute_geom_hash(m1) == compute_geom_hash(m2)


def test_geom_hash_changes_on_chunk_size_change():
    m1 = _metadata(chunk_size=16)
    m2 = _metadata(chunk_size=32)
    assert compute_geom_hash(m1) != compute_geom_hash(m2)


def test_geom_hash_changes_on_model_name_change():
    m1 = _metadata(model_name="A")
    m2 = _metadata(model_name="B")
    assert compute_geom_hash(m1) != compute_geom_hash(m2)


def test_geom_hash_is_sixteen_bytes():
    assert len(compute_geom_hash(_metadata())) == 16


# ---------- initialize ----------


def test_initialize_writes_header_and_clears_metadata(pool_path):
    cfg = CXLBootstrapConfig(
        dev_path=pool_path,
        region_size=REGION_SIZE,
        initialize=True,
        generation=42,
    )
    md = _metadata()
    handle = bootstrap_pool(cfg, md)
    try:
        # Header looks right.
        assert handle.header.magic == MAGIC
        assert handle.header.gen == 42
        assert bytes(handle.header.geom_hash) == compute_geom_hash(md)
        assert handle.header.region_size == REGION_SIZE
        assert handle.header.region_count >= 1

        # Region descriptors are all FREE.
        descs = handle.region_descs()
        for i in range(handle.layout.region_count):
            assert descs[i].owner_node_id == OWNER_FREE, i
            assert descs[i].claim_epoch == 0

        # All slots are EMPTY.
        slots = handle.slots()
        for i in range(handle.layout.index_slot_count):
            assert slots[i].line0.state == SLOT_STATE_EMPTY, i
            assert slots[i].line0.chunk_hash == 0

        # Bitmap is zero-initialized.
        bitmap = bytes(handle.region_bitmap())
        assert all(b == 0 for b in bitmap)
    finally:
        handle.close()


def test_attach_after_initialize_succeeds(pool_path):
    md = _metadata()

    # First run: initialize.
    cfg_init = CXLBootstrapConfig(
        dev_path=pool_path,
        region_size=REGION_SIZE,
        initialize=True,
        generation=3,
    )
    h1 = bootstrap_pool(cfg_init, md)
    region_count = h1.layout.region_count
    index_slot_count = h1.layout.index_slot_count
    h1.close()

    # Second run: attach without initialize. Should see the same layout.
    cfg_attach = CXLBootstrapConfig(
        dev_path=pool_path,
        region_size=REGION_SIZE,
        initialize=False,
    )
    h2 = bootstrap_pool(cfg_attach, md)
    try:
        assert h2.header.gen == 3
        assert h2.layout.region_count == region_count
        assert h2.layout.index_slot_count == index_slot_count
    finally:
        h2.close()


def test_attach_rejects_geom_hash_mismatch(pool_path):
    md_init = _metadata(model_name="alpha")
    md_attach = _metadata(model_name="beta")

    cfg_init = CXLBootstrapConfig(
        dev_path=pool_path, region_size=REGION_SIZE, initialize=True
    )
    bootstrap_pool(cfg_init, md_init).close()

    cfg_attach = CXLBootstrapConfig(
        dev_path=pool_path, region_size=REGION_SIZE, initialize=False
    )
    with pytest.raises(RuntimeError, match="geom_hash mismatch"):
        bootstrap_pool(cfg_attach, md_attach)


def test_attach_rejects_bad_magic(pool_path):
    # Simulate a freshly-truncated file that was never initialized.
    cfg_attach = CXLBootstrapConfig(
        dev_path=pool_path, region_size=REGION_SIZE, initialize=False
    )
    with pytest.raises(ValueError, match="bad magic"):
        bootstrap_pool(cfg_attach, _metadata())


def test_initialize_then_reinitialize_bumps_generation(pool_path):
    md = _metadata()
    cfg1 = CXLBootstrapConfig(
        dev_path=pool_path, region_size=REGION_SIZE, initialize=True, generation=1
    )
    bootstrap_pool(cfg1, md).close()

    cfg2 = CXLBootstrapConfig(
        dev_path=pool_path, region_size=REGION_SIZE, initialize=True, generation=2
    )
    h = bootstrap_pool(cfg2, md)
    try:
        assert h.header.gen == 2
    finally:
        h.close()


def test_region_address_is_within_mapping(pool_path):
    cfg = CXLBootstrapConfig(
        dev_path=pool_path, region_size=REGION_SIZE, initialize=True
    )
    h = bootstrap_pool(cfg, _metadata())
    try:
        for i in range(h.layout.region_count):
            addr = h.region_address(i)
            assert addr >= h.base + h.layout.off_regions
            assert addr + REGION_SIZE <= h.base + h.size
    finally:
        h.close()


def test_region_address_rejects_bad_index(pool_path):
    cfg = CXLBootstrapConfig(
        dev_path=pool_path, region_size=REGION_SIZE, initialize=True
    )
    h = bootstrap_pool(cfg, _metadata())
    try:
        with pytest.raises(IndexError):
            h.region_address(-1)
        with pytest.raises(IndexError):
            h.region_address(h.layout.region_count)
    finally:
        h.close()


def test_two_handles_share_same_mapping(pool_path):
    """Two bootstraps on the same path see each other's writes.

    This is the core property the CXL tier needs — one node writes a
    slot, another node reads it back via its own mmap. With a regular
    file this degenerates to a single-process sanity check.
    """
    md = _metadata()
    cfg_init = CXLBootstrapConfig(
        dev_path=pool_path, region_size=REGION_SIZE, initialize=True
    )
    h_writer = bootstrap_pool(cfg_init, md)

    cfg_attach = CXLBootstrapConfig(
        dev_path=pool_path, region_size=REGION_SIZE, initialize=False
    )
    h_reader = bootstrap_pool(cfg_attach, md)

    try:
        # Writer mutates region descriptor.
        descs_w = h_writer.region_descs()
        descs_w[0].owner_node_id = 5
        descs_w[0].claim_epoch = 99

        # Reader sees the mutation (mmap'd shared).
        descs_r = h_reader.region_descs()
        assert descs_r[0].owner_node_id == 5
        assert descs_r[0].claim_epoch == 99
    finally:
        h_reader.close()
        h_writer.close()
