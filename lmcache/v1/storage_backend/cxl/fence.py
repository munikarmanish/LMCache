# SPDX-License-Identifier: Apache-2.0
"""Cross-host cache-line visibility primitives for CXL shared memory.

On CXL 2.0 hardware that does NOT enforce cross-host cache coherence
(the common case — verified against our 2-node test rack), a writer
on host A must CLFLUSH its updated cacheline so the value reaches
the CXL device, and a reader on host B must CLFLUSH-invalidate its
local line so the next load fetches from the device.

This module provides a `Fence` interface with two concrete impls:

- `StubFence`: no-op. Correct on a single cache-coherent host (MP-mode
  with one MP server and N vLLM processes; tests; in-process
  deployments).
- `CLFlushFence`: emits real CLFLUSH+MFENCE via a small C shared
  object loaded with ctypes. Required for cross-host CXL.

The default fence is auto-selected at first use:

  - If we're on x86 AND the cxl_fence.so can be built or loaded,
    `CLFlushFence` is the default.
  - Otherwise we fall back to `StubFence` so non-x86 dev environments
    and CI containers still work.

Override at bootstrap:
  set_default_fence(StubFence())     # force no-op (e.g. single-host)
  set_default_fence(CLFlushFence())  # force CLFLUSH (e.g. cross-host)
"""

# Standard
import abc
import ctypes
import os
import platform
import subprocess
import threading
from typing import Optional

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_NATIVE_DIR = os.path.join(_HERE, "_native")
_SO_NAME = "cxl_fence.so"


class Fence(abc.ABC):
    """Cross-host visibility primitive for CXL shared memory."""

    @abc.abstractmethod
    def flush_before_read(self, addr: int, size: int) -> None:
        """Ensure the next load from [addr, addr+size) sees latest device state."""

    @abc.abstractmethod
    def fence_after_write(self, addr: int, size: int) -> None:
        """Ensure prior stores to [addr, addr+size) are visible to other hosts."""


class StubFence(Fence):
    """No-op fence. Correct on a single cache-coherent host.

    Used for tests and for deployments where all "peers" actually live
    on the same machine. The OS + CPU give us coherence for free.
    """

    _barrier_lock = threading.Lock()

    def flush_before_read(self, addr: int, size: int) -> None:
        with self._barrier_lock:
            pass

    def fence_after_write(self, addr: int, size: int) -> None:
        with self._barrier_lock:
            pass


# -------- CLFlushFence -------------------------------------------------


class CLFlushFence(Fence):
    """Real CLFLUSH-based fence for x86 cross-host CXL.

    Loads a small C shared object that exposes:
      - cxl_flush_range(addr, size)       — CLFLUSH every line in [addr, addr+size)
      - cxl_mfence()                       — full memory barrier
      - cxl_flush_range_and_fence(addr,size)  — combined fast path

    Both the read-side and write-side use the same primitive: a
    cacheline flush plus an MFENCE. The semantic difference is
    *when* the read happens — before the load (invalidate) or after
    the store (publish) — not what the C function does.
    """

    def __init__(self, lib: ctypes.CDLL):
        self._lib = lib
        # Configure ctypes signatures once.
        lib.cxl_flush_range.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.cxl_flush_range.restype = None
        lib.cxl_mfence.argtypes = []
        lib.cxl_mfence.restype = None
        lib.cxl_flush_range_and_fence.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.cxl_flush_range_and_fence.restype = None
        lib.cxl_fence_has_clflush.argtypes = []
        lib.cxl_fence_has_clflush.restype = ctypes.c_int
        if not lib.cxl_fence_has_clflush():
            raise RuntimeError(
                "cxl_fence.so was built without CLFLUSH support; "
                "this should not happen on x86. Refusing to use it."
            )

    def flush_before_read(self, addr: int, size: int) -> None:
        if size <= 0:
            return
        # Invalidate stale local cachelines. The MFENCE ensures the
        # invalidate is visible before the load that follows in
        # caller code.
        self._lib.cxl_flush_range_and_fence(ctypes.c_void_p(addr), ctypes.c_size_t(size))

    def fence_after_write(self, addr: int, size: int) -> None:
        if size <= 0:
            return
        # Drain pending stores out of the cache hierarchy and order
        # them with respect to subsequent operations.
        self._lib.cxl_flush_range_and_fence(ctypes.c_void_p(addr), ctypes.c_size_t(size))


# -------- loader -------------------------------------------------------


def _so_candidate_paths() -> list:
    """Plausible locations for cxl_fence.so, in priority order."""
    return [
        # Pre-built and shipped in the source tree.
        os.path.join(_NATIVE_DIR, _SO_NAME),
        # Caller-overridable via env var.
        os.environ.get("LMCACHE_CXL_FENCE_SO", ""),
    ]


def _try_load_clflush_lib() -> Optional[ctypes.CDLL]:
    """Try loading the prebuilt .so. None if not available."""
    for path in _so_candidate_paths():
        if not path:
            continue
        if not os.path.isfile(path):
            continue
        try:
            return ctypes.CDLL(path)
        except OSError as e:
            logger.warning("Found %s but failed to load: %s", path, e)
    return None


def _try_build_clflush_lib() -> Optional[ctypes.CDLL]:
    """Compile cxl_fence.so on demand using cc. Best-effort.

    Compiles with -O2 -march=native -shared -fPIC. Skipped on non-x86
    hosts and when cc is not available. Build artifacts go alongside
    the source in `_native/`.
    """
    if platform.machine() not in ("x86_64", "i686", "i386", "AMD64"):
        return None
    src = os.path.join(_NATIVE_DIR, "cxl_fence.c")
    if not os.path.isfile(src):
        return None
    out = os.path.join(_NATIVE_DIR, _SO_NAME)
    cc = os.environ.get("CC", "cc")
    cmd = [cc, "-O2", "-march=native", "-shared", "-fPIC", src, "-o", out]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.info(
            "Could not compile %s (%s); falling back to StubFence.", _SO_NAME, e
        )
        return None
    if result.returncode != 0:
        logger.info(
            "Compile of %s failed (rc=%d): %s",
            _SO_NAME,
            result.returncode,
            result.stderr.strip(),
        )
        return None
    try:
        return ctypes.CDLL(out)
    except OSError as e:
        logger.warning("Compiled %s but cannot load it: %s", out, e)
        return None


def _auto_select_fence() -> Fence:
    """Pick CLFlushFence if available, else StubFence."""
    lib = _try_load_clflush_lib() or _try_build_clflush_lib()
    if lib is None:
        logger.info(
            "Using StubFence: no CLFLUSH .so available. This is "
            "correct for single-host deployments and in-process "
            "tests. For cross-host CXL, build cxl_fence.so or call "
            "set_default_fence(CLFlushFence(...))."
        )
        return StubFence()
    try:
        return CLFlushFence(lib)
    except Exception as e:
        logger.warning(
            "Loaded cxl_fence.so but CLFlushFence init failed: %s. "
            "Falling back to StubFence.",
            e,
        )
        return StubFence()


_default_fence: Optional[Fence] = None
_default_fence_lock = threading.Lock()


def default_fence() -> Fence:
    """Return the process-wide default fence.

    Auto-selects CLFlushFence on x86 if the .so is available; else
    StubFence. Override with `set_default_fence`.
    """
    global _default_fence
    if _default_fence is None:
        with _default_fence_lock:
            if _default_fence is None:
                _default_fence = _auto_select_fence()
    return _default_fence


def set_default_fence(fence: Fence) -> None:
    """Override the default fence (used by tests and CXL bootstrap)."""
    global _default_fence
    with _default_fence_lock:
        _default_fence = fence


def reset_default_fence() -> None:
    """Drop the cached default. Next `default_fence()` re-selects.

    Test-only: useful when tests want to re-trigger auto-selection
    after manipulating the environment.
    """
    global _default_fence
    with _default_fence_lock:
        _default_fence = None
