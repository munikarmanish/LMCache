# SPDX-License-Identifier: Apache-2.0
"""Fast DRAM -> CXL copy for the cross-node donor path.

The cross-node `PushKVToCXL` donor copies a chunk's KV bytes from local
DRAM into the shared CXL pool. A plain ``ctypes.memmove`` does this with
regular CPU stores, which pull each destination cacheline into cache
before writing (read-for-ownership) — wasteful for write-only traffic
into device memory. On a real CXL device that measured ~2 GB/s.

Non-temporal streaming stores (``_mm256_stream``) bypass the cache and
write straight through; the same device measured ~10.6 GB/s (a ~5x
speedup, matching its DMA write bandwidth). This module compiles a tiny
C helper (``_native/cxl_copy.c``) and exposes :func:`fast_copy_to_cxl`,
which uses it when available and falls back to ``ctypes.memmove`` on
non-x86 hosts or when a C compiler is unavailable — so correctness never
depends on the optimization.

The loader mirrors ``fence.py``: try a prebuilt ``.so``, else build it
on demand with ``cc``, else fall back.
"""

# Standard
from typing import Callable, Optional
import ctypes
import os
import platform
import subprocess
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_NATIVE_DIR = os.path.join(_HERE, "_native")
_SO_NAME = "cxl_copy.so"

# Resolved once, lazily, under this lock. ``_nt_copy`` is the loaded
# C function (or None if unavailable); ``_resolved`` guards one-time init.
_lock = threading.Lock()
_resolved = False
_nt_copy: Optional[Callable[[ctypes.c_void_p, ctypes.c_void_p, int], None]] = None


def _so_candidate_paths() -> list[str]:
    """Plausible locations for ``cxl_copy.so``, in priority order."""
    return [
        os.path.join(_NATIVE_DIR, _SO_NAME),
        os.environ.get("LMCACHE_CXL_COPY_SO", ""),
    ]


def _try_load_lib() -> Optional[ctypes.CDLL]:
    """Try loading a prebuilt ``cxl_copy.so``. None if not present."""
    for path in _so_candidate_paths():
        if not path or not os.path.isfile(path):
            continue
        try:
            return ctypes.CDLL(path)
        except OSError as e:
            logger.warning("Found %s but failed to load: %s", path, e)
    return None


def _try_build_lib() -> Optional[ctypes.CDLL]:
    """Compile ``cxl_copy.so`` on demand with ``cc``. Best-effort.

    Skipped on non-x86 hosts and when ``cc`` is unavailable. Build
    artifacts go alongside the source in ``_native/``.
    """
    if platform.machine() not in ("x86_64", "i686", "i386", "AMD64"):
        return None
    src = os.path.join(_NATIVE_DIR, "cxl_copy.c")
    if not os.path.isfile(src):
        return None
    out = os.path.join(_NATIVE_DIR, _SO_NAME)
    cc = os.environ.get("CC", "cc")
    cmd = [cc, "-O3", "-march=native", "-shared", "-fPIC", src, "-o", out]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.info("Could not compile %s (%s); falling back to memmove.", _SO_NAME, e)
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


def _resolve() -> None:
    """One-time: load or build the helper and bind ``_nt_copy``."""
    global _resolved, _nt_copy
    if _resolved:
        return
    with _lock:
        if _resolved:
            return
        lib = _try_load_lib() or _try_build_lib()
        if lib is not None:
            try:
                if int(lib.cxl_copy_has_nt()) == 1:
                    lib.cxl_nt_copy.argtypes = [
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                        ctypes.c_size_t,
                    ]
                    lib.cxl_nt_copy.restype = None
                    _nt_copy = lib.cxl_nt_copy
                    logger.info(
                        "CXL donor copy: using non-temporal streaming stores (%s).",
                        _SO_NAME,
                    )
                else:
                    logger.info(
                        "CXL donor copy: %s built without NT support; using memmove.",
                        _SO_NAME,
                    )
            except Exception as e:
                logger.warning(
                    "Loaded %s but cxl_nt_copy unusable (%s); using memmove.",
                    _SO_NAME,
                    e,
                )
                _nt_copy = None
        else:
            logger.info(
                "CXL donor copy: no %s available; using memmove. Build it or "
                "ensure a C compiler is present for ~5x faster DRAM->CXL "
                "writes.",
                _SO_NAME,
            )
        _resolved = True


def fast_copy_to_cxl(dst_addr: int, src_addr: int, n_bytes: int) -> None:
    """Copy ``n_bytes`` from DRAM ``src_addr`` to CXL ``dst_addr``.

    Uses non-temporal streaming stores when the native helper is
    available (~5x faster than ``memmove`` writing into the CXL pool),
    else falls back to ``ctypes.memmove``. Any ``(dst, src, n)`` is safe;
    alignment only affects speed.

    Args:
        dst_addr: Destination address in the CXL pool (as an int).
        src_addr: Source address in DRAM (as an int).
        n_bytes: Number of bytes to copy.
    """
    if not _resolved:
        _resolve()
    if _nt_copy is not None:
        _nt_copy(
            ctypes.c_void_p(dst_addr),
            ctypes.c_void_p(src_addr),
            ctypes.c_size_t(n_bytes),
        )
    else:
        ctypes.memmove(dst_addr, src_addr, n_bytes)
