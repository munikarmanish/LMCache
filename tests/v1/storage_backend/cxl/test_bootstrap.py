# SPDX-License-Identifier: Apache-2.0
"""Bootstrap tests for the CXL pool.

These exercise mmap + header init + re-attach semantics on a tmpfile;
no CUDA and no real CXL device are required. cudaHostRegister is
best-effort inside the bootstrap path and logged-not-raised when CUDA
is unavailable, so tests can run anywhere.
"""

# Standard
import os
import struct
import tempfile

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.cxl.bootstrap import (
    INTERLEAVED_DAX_DEVICE_NAME,
    CXLBootstrapConfig,
    PagePopulatePolicy,
    bootstrap_pool,
    interleaved_dax_capacity_bytes,
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


# ---------- initialize ----------


def test_initialize_writes_header_and_clears_metadata(pool_path):
    cfg = CXLBootstrapConfig(
        dev_path=pool_path,
        region_size=REGION_SIZE,
        initialize=True,
        generation=42,
    )
    handle = bootstrap_pool(cfg)
    try:
        # Header looks right.
        assert handle.header.magic == MAGIC
        assert handle.header.gen == 42
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

    # First run: initialize.
    cfg_init = CXLBootstrapConfig(
        dev_path=pool_path,
        region_size=REGION_SIZE,
        initialize=True,
        generation=3,
    )
    h1 = bootstrap_pool(cfg_init)
    region_count = h1.layout.region_count
    index_slot_count = h1.layout.index_slot_count
    h1.close()

    # Second run: attach without initialize. Should see the same layout.
    cfg_attach = CXLBootstrapConfig(
        dev_path=pool_path,
        region_size=REGION_SIZE,
        initialize=False,
    )
    h2 = bootstrap_pool(cfg_attach)
    try:
        assert h2.header.gen == 3
        assert h2.layout.region_count == region_count
        assert h2.layout.index_slot_count == index_slot_count
    finally:
        h2.close()


def test_attach_succeeds_regardless_of_model(pool_path):
    """A pool carries no model identity, so any node may attach.

    Multi-tenancy: one pool serves many models and TP degrees at once.
    Tenant separation is per slot (``line0.geom_hash``), not pool-wide,
    so attach checks only compatibility — magic, version, sizing.
    """
    cfg_init = CXLBootstrapConfig(
        dev_path=pool_path, region_size=REGION_SIZE, initialize=True
    )
    bootstrap_pool(cfg_init).close()

    cfg_attach = CXLBootstrapConfig(
        dev_path=pool_path, region_size=REGION_SIZE, initialize=False
    )
    h = bootstrap_pool(cfg_attach)
    try:
        assert h.header.magic != 0
    finally:
        h.close()


def test_attach_rejects_bad_magic(pool_path):
    # Simulate a freshly-truncated file that was never initialized.
    cfg_attach = CXLBootstrapConfig(
        dev_path=pool_path, region_size=REGION_SIZE, initialize=False
    )
    with pytest.raises(ValueError, match="bad magic"):
        bootstrap_pool(cfg_attach)


def test_initialize_then_reinitialize_bumps_generation(pool_path):
    cfg1 = CXLBootstrapConfig(
        dev_path=pool_path, region_size=REGION_SIZE, initialize=True, generation=1
    )
    bootstrap_pool(cfg1).close()

    cfg2 = CXLBootstrapConfig(
        dev_path=pool_path, region_size=REGION_SIZE, initialize=True, generation=2
    )
    h = bootstrap_pool(cfg2)
    try:
        assert h.header.gen == 2
    finally:
        h.close()


def test_region_address_is_within_mapping(pool_path):
    cfg = CXLBootstrapConfig(
        dev_path=pool_path, region_size=REGION_SIZE, initialize=True
    )
    h = bootstrap_pool(cfg)
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
    h = bootstrap_pool(cfg)
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
    cfg_init = CXLBootstrapConfig(
        dev_path=pool_path, region_size=REGION_SIZE, initialize=True
    )
    h_writer = bootstrap_pool(cfg_init)

    cfg_attach = CXLBootstrapConfig(
        dev_path=pool_path, region_size=REGION_SIZE, initialize=False
    )
    h_reader = bootstrap_pool(cfg_attach)

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


# -------- interleaved_dax capacity ----------------------------------------

_GIB = 1 << 30
_PAGE = 4096


def test_interleave_capacity_equal_weights_uses_both_ranges():
    # Two 128 GiB modules at 1:1 -> the full 256 GiB is addressable.
    assert interleaved_dax_capacity_bytes("0-128:1,128-256:1", _PAGE) == 256 * _GIB


def test_interleave_capacity_tolerates_sysfs_trailing_newline():
    assert interleaved_dax_capacity_bytes("0-128:1,128-256:1\n", _PAGE) == 256 * _GIB


def test_interleave_capacity_bounded_by_first_exhausted_range():
    # 1:1 over a 64 GiB and a 128 GiB range stops when the small one runs out.
    assert interleaved_dax_capacity_bytes("0-64:1,128-256:1", _PAGE) == 128 * _GIB


def test_interleave_capacity_honours_weights():
    # 2:1 over equal 128 GiB ranges: the weight-2 range drains first after
    # 64 GiB-worth of rounds, each round covering 3 pages.
    expected = (128 * _GIB // _PAGE // 2) * 3 * _PAGE
    assert interleaved_dax_capacity_bytes("0-128:2,128-256:1", _PAGE) == expected


def test_interleave_capacity_accepts_hex_like_the_kernel_parser():
    assert (
        interleaved_dax_capacity_bytes("0x0-0x80:1,0x80-0x100:1", _PAGE) == 256 * _GIB
    )


@pytest.mark.parametrize(
    "config",
    [
        "",  # nothing to parse
        "0-128",  # missing weight
        "128:1",  # missing end
        "128-0:1",  # empty range
        "0-128:0",  # zero weight
        "a-b:c",  # not numbers
    ],
)
def test_interleave_capacity_rejects_malformed_config(config):
    with pytest.raises(ValueError):
        interleaved_dax_capacity_bytes(config, _PAGE)


def test_interleave_capacity_rejects_nonpositive_page_size():
    with pytest.raises(ValueError):
        interleaved_dax_capacity_bytes("0-128:1", 0)


# -------- page-table pre-population --------------------------------------


def _present_page_fraction(base: int, size: int) -> float:
    """Fraction of the mapping's pages that have a present PTE.

    Reads bit 63 of each ``/proc/self/pagemap`` entry, which is readable
    without privileges (only the PFN bits are hidden).
    """
    page = os.sysconf("SC_PAGE_SIZE")
    n_pages = size // page
    with open("/proc/self/pagemap", "rb") as f:
        f.seek((base // page) * 8)
        raw = f.read(n_pages * 8)
    entries = struct.unpack(f"<{n_pages}Q", raw)
    return sum(1 for e in entries if e >> 63) / n_pages


@pytest.fixture
def no_cuda(monkeypatch):
    # cudaHostRegister faults pages in by itself, which would hide whether
    # the populate step ran. Take it out of the picture.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _bootstrap(path: str, policy: PagePopulatePolicy):
    return bootstrap_pool(
        CXLBootstrapConfig(
            dev_path=path,
            region_size=REGION_SIZE,
            initialize=True,
            populate_policy=policy,
        )
    )


def test_populate_always_maps_every_page(pool_path, no_cuda):
    handle = _bootstrap(pool_path, PagePopulatePolicy.ALWAYS)
    try:
        assert _present_page_fraction(handle.base, handle.size) == 1.0
    finally:
        handle.close()


def test_populate_never_leaves_payload_pages_unmapped(pool_path, no_cuda):
    handle = _bootstrap(pool_path, PagePopulatePolicy.NEVER)
    try:
        # Initialization touches only the metadata at the head of the pool.
        assert _present_page_fraction(handle.base, handle.size) < 0.5
    finally:
        handle.close()


def test_populate_auto_skips_ordinary_pools(pool_path, no_cuda):
    handle = _bootstrap(pool_path, PagePopulatePolicy.AUTO)
    try:
        assert _present_page_fraction(handle.base, handle.size) < 0.5
    finally:
        handle.close()


def test_populate_auto_covers_interleaved_dax(tmp_path, no_cuda):
    # AUTO keys on the device name; a regular file with that name stands in
    # for the misc device, whose 4 KiB-only mappings are what need populating.
    path = tmp_path / INTERLEAVED_DAX_DEVICE_NAME
    with open(path, "wb") as f:
        f.truncate(POOL_SIZE)
    handle = _bootstrap(str(path), PagePopulatePolicy.AUTO)
    try:
        assert _present_page_fraction(handle.base, handle.size) == 1.0
    finally:
        handle.close()


def test_populate_preserves_pool_contents(pool_path, no_cuda):
    # MADV_POPULATE_WRITE write-faults pages without storing to them, so
    # attaching with ALWAYS must not disturb an initialized pool.
    first = _bootstrap(pool_path, PagePopulatePolicy.NEVER)
    try:
        first.region_descs()[3].owner_node_id = 7
    finally:
        first.close()

    second = bootstrap_pool(
        CXLBootstrapConfig(
            dev_path=pool_path,
            region_size=REGION_SIZE,
            initialize=False,
            populate_policy=PagePopulatePolicy.ALWAYS,
        )
    )
    try:
        assert second.header.magic == MAGIC
        assert second.region_descs()[3].owner_node_id == 7
    finally:
        second.close()
