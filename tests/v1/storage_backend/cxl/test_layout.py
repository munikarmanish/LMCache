# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the CXL pool layout structs and planner."""

# Standard
import ctypes

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.cxl.layout import (
    CACHELINE_SIZE,
    GEOM_HASH_SIZE,
    HEADER_SIZE,
    Header,
    LockSlot,
    MAGIC,
    OWNER_FREE,
    PoolLayout,
    RegionDesc,
    SLOT_STATE_EMPTY,
    Slot,
    SlotLine0,
    SlotLine1,
)


# ---------- struct sizing invariants ----------


def test_struct_sizes_are_fixed():
    # Plan's slot design: line0 + line1 each exactly one cacheline.
    assert ctypes.sizeof(SlotLine0) == CACHELINE_SIZE
    assert ctypes.sizeof(SlotLine1) == CACHELINE_SIZE
    assert ctypes.sizeof(Slot) == 2 * CACHELINE_SIZE
    assert ctypes.sizeof(RegionDesc) == CACHELINE_SIZE
    assert ctypes.sizeof(LockSlot) == CACHELINE_SIZE
    assert ctypes.sizeof(Header) <= HEADER_SIZE


def test_slot_line0_field_layout_matches_plan():
    # chunk_hash first so readers see the identity before anything else.
    assert SlotLine0.chunk_hash.offset == 0
    # state must precede fmt/owner so a store to state publishes them.
    assert SlotLine0.chunk_offset.offset == 8
    assert SlotLine0.chunk_len.offset == 16
    assert SlotLine0.state.offset == 20
    # geom_hash is 16 bytes, used for defensive validation on hit.
    assert SlotLine0.geom_hash.size == GEOM_HASH_SIZE


# ---------- PoolLayout.compute ----------


def _compute(pool_size, region_size=1 << 20, **kw):
    """Small helper: most tests use 1 MiB regions."""
    return PoolLayout.compute(
        pool_size=pool_size, region_size=region_size, **kw
    )


def test_planner_rejects_nonpow2_region_size():
    with pytest.raises(ValueError):
        PoolLayout.compute(pool_size=1 << 30, region_size=3 * (1 << 20))


def test_planner_rejects_zero_pool_size():
    with pytest.raises(ValueError):
        PoolLayout.compute(pool_size=0, region_size=1 << 20)


def test_planner_rejects_too_small_pool():
    # Not enough room for header + locks + at least one region.
    with pytest.raises(ValueError):
        PoolLayout.compute(pool_size=1 << 12, region_size=1 << 20)


def test_planner_fits_within_pool_size():
    layout = _compute(pool_size=256 * (1 << 20))
    # Metadata must precede regions; regions must fit within pool.
    assert layout.off_regions >= layout.off_index + layout.index_size
    assert layout.total_used() <= layout.pool_size
    # Regions are aligned to region_size so kernels can DMA cleanly.
    assert layout.off_regions % layout.region_size == 0


def test_planner_produces_at_least_one_region():
    layout = _compute(pool_size=64 * (1 << 20))
    assert layout.region_count >= 1
    assert layout.region_size == 1 << 20


def test_planner_sized_for_realistic_64gib_pool():
    # Plan's reference sizing table: 64 GiB pool, 256 MiB regions,
    # should yield hundreds of regions.
    region_size = 256 * (1 << 20)
    layout = PoolLayout.compute(
        pool_size=64 * (1 << 30), region_size=region_size
    )
    # Expect ~255 regions (not quite 256 because of metadata overhead).
    assert 200 <= layout.region_count <= 256


def test_planner_respects_explicit_index_slot_count():
    layout = PoolLayout.compute(
        pool_size=256 * (1 << 20),
        region_size=1 << 20,
        index_slot_count=4096,
    )
    assert layout.index_slot_count == 4096
    assert layout.index_size == 4096 * ctypes.sizeof(Slot)


# ---------- header round-trip ----------


def _make_buffer(pool_size):
    buf = (ctypes.c_uint8 * pool_size)()
    base = ctypes.addressof(buf)
    return buf, base


def test_write_and_read_back_header():
    layout = _compute(pool_size=64 * (1 << 20))
    buf, base = _make_buffer(layout.pool_size)
    header = Header.from_address(base)

    geom = bytes(range(GEOM_HASH_SIZE))
    layout.write_to_header(header, geom, gen=7)

    # Header reads back exactly.
    assert header.magic == MAGIC
    assert header.gen == 7
    assert bytes(header.geom_hash) == geom
    assert header.region_size == layout.region_size
    assert header.region_count == layout.region_count
    assert header.pool_size == layout.pool_size
    assert header.off_regions % layout.region_size == 0

    # from_header reconstructs an equivalent layout.
    layout2 = PoolLayout.from_header(header)
    assert layout2.region_count == layout.region_count
    assert layout2.off_regions == layout.off_regions
    assert layout2.index_slot_count == layout.index_slot_count


def test_write_header_rejects_wrong_geom_hash_size():
    layout = _compute(pool_size=64 * (1 << 20))
    buf, base = _make_buffer(layout.pool_size)
    header = Header.from_address(base)
    with pytest.raises(ValueError):
        layout.write_to_header(header, b"too-short", gen=1)


def test_validate_against_detects_magic_mismatch():
    layout = _compute(pool_size=64 * (1 << 20))
    buf, base = _make_buffer(layout.pool_size)
    header = Header.from_address(base)
    layout.write_to_header(header, b"\x00" * GEOM_HASH_SIZE, gen=1)
    header.magic = 0xDEADBEEF
    with pytest.raises(ValueError, match="bad magic"):
        layout.validate_against(header)


def test_validate_against_detects_region_size_mismatch():
    layout = _compute(pool_size=64 * (1 << 20))
    buf, base = _make_buffer(layout.pool_size)
    header = Header.from_address(base)
    layout.write_to_header(header, b"\x00" * GEOM_HASH_SIZE, gen=1)
    header.region_size = layout.region_size * 2
    with pytest.raises(ValueError, match="region_size mismatch"):
        layout.validate_against(header)


# ---------- sentinels ----------


def test_owner_free_distinct_from_orphaned():
    # If these ever collide with a real node id someone will be very sad.
    from lmcache.v1.storage_backend.cxl.layout import OWNER_ORPHANED

    assert OWNER_FREE != OWNER_ORPHANED
    assert SLOT_STATE_EMPTY == 0  # enables ctypes.memset(0) as init
