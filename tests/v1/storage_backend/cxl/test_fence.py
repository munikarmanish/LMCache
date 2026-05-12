# SPDX-License-Identifier: Apache-2.0
"""Tests for the Fence abstractions and the CLFlushFence loader."""

# Standard
import ctypes
import os
import platform

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.cxl.fence import (
    CLFlushFence,
    Fence,
    StubFence,
    _try_build_clflush_lib,
    _try_load_clflush_lib,
    default_fence,
    reset_default_fence,
    set_default_fence,
)


IS_X86 = platform.machine() in ("x86_64", "i686", "i386", "AMD64")


def test_stub_fence_is_a_noop():
    """StubFence accepts any addr/size and never raises."""
    f = StubFence()
    f.flush_before_read(0, 0)
    f.fence_after_write(0, 0)
    # Even bogus addresses with size==0 are fine.
    f.flush_before_read(0xDEADBEEF, 0)


@pytest.mark.skipif(not IS_X86, reason="CLFLUSH is x86-specific")
def test_clflush_fence_loads_on_x86():
    """The .so should compile-and-load on this x86 host."""
    lib = _try_load_clflush_lib() or _try_build_clflush_lib()
    assert lib is not None, (
        "expected to be able to build cxl_fence.so on x86; "
        "check that gcc/cc is available"
    )
    fence = CLFlushFence(lib)
    assert isinstance(fence, Fence)


@pytest.mark.skipif(not IS_X86, reason="CLFLUSH is x86-specific")
def test_clflush_fence_executes_on_real_buffer():
    """CLFLUSH+MFENCE on a heap buffer must not crash."""
    lib = _try_load_clflush_lib() or _try_build_clflush_lib()
    assert lib is not None
    fence = CLFlushFence(lib)
    buf = (ctypes.c_uint8 * 4096)()
    addr = ctypes.addressof(buf)
    fence.flush_before_read(addr, 4096)
    fence.fence_after_write(addr, 4096)


def test_clflush_fence_handles_zero_size():
    """Edge case: zero size early-returns without touching the address."""
    lib = _try_load_clflush_lib() or _try_build_clflush_lib()
    if lib is None:
        pytest.skip("CLFLUSH .so not available on this host")
    fence = CLFlushFence(lib)
    # Bogus address is fine because we early-return on size==0.
    fence.flush_before_read(0xDEADBEEF, 0)
    fence.fence_after_write(0xDEADBEEF, 0)


@pytest.mark.skipif(not IS_X86, reason="auto-select picks CLFlushFence on x86")
def test_default_fence_picks_clflush_on_x86():
    reset_default_fence()
    try:
        f = default_fence()
        assert isinstance(f, CLFlushFence), (
            f"expected CLFlushFence on x86, got {type(f).__name__}"
        )
    finally:
        reset_default_fence()


def test_set_default_fence_overrides_auto_selection():
    reset_default_fence()
    try:
        my_stub = StubFence()
        set_default_fence(my_stub)
        assert default_fence() is my_stub
    finally:
        reset_default_fence()


def test_default_fence_is_cached():
    reset_default_fence()
    try:
        a = default_fence()
        b = default_fence()
        assert a is b
    finally:
        reset_default_fence()


def test_reset_clears_default():
    reset_default_fence()
    try:
        first = default_fence()
        reset_default_fence()
        second = default_fence()
        # Same type; auto-select runs again. Identity is not guaranteed.
        assert type(first) is type(second)
    finally:
        reset_default_fence()


def test_clflush_unaligned_addr_is_safe():
    """Flushes that span misaligned ranges hit every covered cacheline.

    The C implementation rounds the start down to the cacheline. We
    ask it to flush 1 byte at offset 33 (mid-cacheline); it should
    flush the cacheline at [0, 64) and not crash.
    """
    lib = _try_load_clflush_lib() or _try_build_clflush_lib()
    if lib is None:
        pytest.skip("CLFLUSH .so not available")
    fence = CLFlushFence(lib)
    buf = (ctypes.c_uint8 * 256)()
    addr = ctypes.addressof(buf)
    fence.flush_before_read(addr + 33, 1)
    fence.fence_after_write(addr + 65, 1)  # spans into next line
