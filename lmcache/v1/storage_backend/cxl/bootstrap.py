# SPDX-License-Identifier: Apache-2.0
"""CXL pool bootstrap: mmap, header init/verify, cudaHostRegister.

Responsibilities:
- Open /dev/dax0.0 (or /dev/interleaved_dax, or any file) and mmap it.
- If the pool is uninitialized (or the caller is the bootstrap node),
  write the header; otherwise, validate the existing header and reject
  on a magic, layout-version, or sizing mismatch. The pool carries no
  model identity, so any model may attach.
- Pin the mapping with cudaHostRegister so CUDA treats it as
  page-locked host memory (TraCT §4.4) — this is what avoids the
  DRAM bounce buffer on GPU transfers.
- Expose accessor views for other modules (index, bitmap, descriptors,
  locks, region payload base).

Concurrency and lock-manager bootstrap live in cxl/locks.py (next slice).
"""

# Standard
from dataclasses import dataclass
from typing import Optional
import ctypes
import enum
import json
import mmap
import os
import stat
import subprocess
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.cxl.layout import (
    DEFAULT_MAX_NODES,
    DEFAULT_NUM_LOCKS,
    DEFAULT_REGION_SIZE,
    MAGIC,
    OWNER_FREE,
    SLOT_STATE_EMPTY,
    Header,
    LockSlot,
    PoolLayout,
    RegionDesc,
    Slot,
)

logger = init_logger(__name__)

# Name of the misc device registered by the interleaved_dax kernel module
# (page-granular weighted interleave over several CXL ranges).
INTERLEAVED_DAX_DEVICE_NAME = "interleaved_dax"
_INTERLEAVED_DAX_CONFIG_PARAM = "/sys/module/interleaved_dax/parameters/config"

# madvise(2) advice value; Linux >= 5.14. Not exposed by the ``mmap`` module
# before Python 3.13, so it is spelled out here.
_MADV_POPULATE_WRITE = 23
# Populate in steps so a multi-second pass over a large pool stays
# responsive to signals. Populating does not scale across threads (measured),
# so the steps run sequentially.
_POPULATE_STEP_BYTES = 1 << 30


class PagePopulatePolicy(enum.Enum):
    """When `bootstrap_pool` pre-populates the pool's page-table entries.

    Populating means ``madvise(MADV_POPULATE_WRITE)`` over the whole mapping
    before it is pinned, which installs every PTE already marked accessed and
    dirty.

    This matters on ``/dev/interleaved_dax``. That device can only map 4 KiB
    pages, and ``cudaHostRegister`` pins them without marking the PTEs
    accessed/dirty, so the first CPU write to each page costs ~0.9 us with no
    page fault. A cross-node donor push always writes never-touched chunks, so
    it ran at ~9 GB/s instead of ~41 GB/s. DAX devices map 2 MiB pages and do
    not show the effect.

    Populating is startup-neutral on interleaved_dax: the populate pass costs
    about what ``cudaHostRegister`` then saves by finding the PTEs present.
    """

    # Populate only devices that need it (currently /dev/interleaved_dax).
    AUTO = "auto"
    # Populate every pool. On a sparse regular file this allocates all of it.
    ALWAYS = "always"
    # Never populate.
    NEVER = "never"


@dataclass(frozen=True)
class CXLBootstrapConfig:
    """Inputs to `bootstrap_pool`. Produced from LMCacheEngineConfig."""

    dev_path: str
    region_size: int = DEFAULT_REGION_SIZE
    num_locks: int = DEFAULT_NUM_LOCKS
    max_nodes: int = DEFAULT_MAX_NODES
    # If True, caller is the bootstrap node and will (re)initialize the
    # pool unconditionally. If False, caller refuses to write a header
    # and fails if the pool is unrecognized.
    initialize: bool = False
    # Only used when initialize=True.
    generation: int = 1
    # Cap the usable pool size in bytes, regardless of what the device
    # advertises. None = use the full device size. Use this to stay
    # under cudaHostRegister's per-call cap on systems where the GPU
    # driver refuses to pin the whole device in one call.
    pool_size_override: Optional[int] = None
    # Optional NUMA node id to bind backend threads to. If None, the
    # caller's default binding is used.
    numa_node: Optional[int] = None
    # Whether to pre-populate the mapping's page-table entries before
    # pinning it. See PagePopulatePolicy.
    populate_policy: PagePopulatePolicy = PagePopulatePolicy.AUTO


@dataclass
class PoolHandle:
    """Live handle to an mmap'd, optionally CUDA-registered CXL pool."""

    cfg: CXLBootstrapConfig
    fd: int
    base: int  # address of the mmap, as an int
    size: int
    mmap_obj: mmap.mmap
    layout: PoolLayout
    header: Header
    cuda_registered: bool = False

    # -- accessor views -----------------------------------------------

    def _view(self, offset: int, size: int) -> ctypes.Array:
        """Return a ctypes byte array over [offset, offset+size)."""
        if offset < 0 or size < 0 or offset + size > self.size:
            raise ValueError(
                f"view out of range: off={offset} size={size} pool_size={self.size}"
            )
        ArrayT = ctypes.c_uint8 * size
        return ArrayT.from_address(self.base + offset)

    def region_bitmap(self) -> ctypes.Array:
        return self._view(self.layout.off_region_bitmap, self.layout.region_bitmap_size)

    def region_descs(self) -> ctypes.Array:
        DescArray = RegionDesc * self.layout.region_count
        return DescArray.from_address(self.base + self.layout.off_region_descs)

    def global_locks(self) -> ctypes.Array:
        LockArray = LockSlot * (self.layout.num_locks * self.layout.max_nodes)
        return LockArray.from_address(self.base + self.layout.off_global_locks)

    def slots(self) -> ctypes.Array:
        SlotArray = Slot * self.layout.index_slot_count
        return SlotArray.from_address(self.base + self.layout.off_index)

    def region_address(self, region_id: int) -> int:
        if not 0 <= region_id < self.layout.region_count:
            raise IndexError(region_id)
        return self.base + self.layout.off_regions + region_id * self.layout.region_size

    def close(self) -> None:
        if self.cuda_registered:
            try:
                _cuda_host_unregister(self.base)
            except Exception as exc:  # defensive: unregister is best-effort
                logger.warning("cudaHostUnregister failed: %s", exc)
            self.cuda_registered = False
        try:
            self.mmap_obj.close()
        finally:
            try:
                os.close(self.fd)
            except OSError:
                pass


# -------- top-level entry points ----------------------------------------


def bootstrap_pool(cfg: CXLBootstrapConfig) -> PoolHandle:
    """Open the pool and return a live PoolHandle.

    If `cfg.initialize` is True, the pool is (re)initialized: header is
    written, region bitmap cleared, descriptors reset, index zeroed.
    Otherwise the existing header is checked for *compatibility* — magic,
    layout version, and sizing — and the caller attaches without mutating
    the pool.

    The pool carries no model identity. A slot's tenant is recorded per
    slot (``line0.geom_hash``, see
    :func:`~lmcache.v1.storage_backend.cxl.store.object_key_to_tenant_digest`),
    so one pool serves many models, TP degrees, and tenants concurrently.

    Args:
        cfg: Device path, sizing, and initialize/generation flags.
            ``cfg.populate_policy`` controls whether the mapping's page-table
            entries are pre-populated before it is pinned (see
            :class:`PagePopulatePolicy`); by default only
            ``/dev/interleaved_dax`` pools are.

    Returns:
        A live PoolHandle over the mapped pool.
    """
    fd, size = _open_pool(cfg.dev_path)
    if cfg.pool_size_override is not None:
        if cfg.pool_size_override <= 0:
            os.close(fd)
            raise ValueError(
                f"pool_size_override must be positive, got {cfg.pool_size_override}"
            )
        if cfg.pool_size_override > size:
            os.close(fd)
            raise ValueError(
                f"pool_size_override {cfg.pool_size_override} exceeds device "
                f"size {size} for {cfg.dev_path}"
            )
        size = cfg.pool_size_override
    try:
        mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    except Exception:
        os.close(fd)
        raise

    base = _mmap_base_address(mm)

    try:
        if cfg.initialize:
            layout = PoolLayout.compute(
                pool_size=size,
                region_size=cfg.region_size,
                num_locks=cfg.num_locks,
                max_nodes=cfg.max_nodes,
            )
            header = _initialize_pool(base, layout, cfg.generation)
            logger.info(
                "CXL pool initialized: path=%s size=%d regions=%d region_size=%d "
                "index_slots=%d gen=%d",
                cfg.dev_path,
                size,
                layout.region_count,
                layout.region_size,
                layout.index_slot_count,
                cfg.generation,
            )
        else:
            header, layout = _attach_existing(base, size)
            logger.info(
                "CXL pool attached: path=%s size=%d regions=%d gen=%d",
                cfg.dev_path,
                size,
                layout.region_count,
                header.gen,
            )

        handle = PoolHandle(
            cfg=cfg,
            fd=fd,
            base=base,
            size=size,
            mmap_obj=mm,
            layout=layout,
            header=header,
        )

        # Populate BEFORE pinning: cudaHostRegister then finds the PTEs
        # present (so it is much faster) and they are already accessed/dirty.
        _populate_pages_if_needed(handle)
        _cuda_host_register_if_available(handle)
        _numa_bind_if_requested(cfg.numa_node)
        return handle
    except Exception:
        mm.close()
        os.close(fd)
        raise


# -------- internals -----------------------------------------------------


def _open_pool(path: str) -> tuple[int, int]:
    """Open the pool file/device and return (fd, size_bytes).

    Size discovery rules, in priority order:

    1. Regular file: use ``fstat.st_size``.
    2. DAX char device (`/dev/dax*.*`): query
       ``/sys/bus/dax/devices/<name>/size``, which is the kernel's
       authoritative source and does not require root. We avoid
       ``lseek(SEEK_END)`` because DAX char devices return 0 from it
       (they are not seekable in the file-position sense).
    3. ``daxctl list -d <name> -j``: fallback if sysfs isn't readable
       (e.g. unusual setups). Parses the JSON ``size`` field.
    4. ``/dev/interleaved_dax`` (the interleaved_dax kernel module's misc
       device): it is not on the DAX bus, so rules 2-3 do not apply. Its
       capacity is derived from the module's ``config`` parameter, see
       :func:`interleaved_dax_capacity_bytes`.

    Raises RuntimeError with a diagnostic message if none of these
    work — this previously masked the DAX case as "stat reported 0,
    lseek reported 0" which was unhelpful.
    """
    fd = os.open(path, os.O_RDWR)
    try:
        st = os.fstat(fd)
        size = st.st_size

        if size <= 0 and stat.S_ISCHR(st.st_mode):
            if os.path.basename(path) == INTERLEAVED_DAX_DEVICE_NAME:
                size = _interleaved_dax_device_size()
            else:
                size = _dax_device_size(path)

        if size <= 0:
            raise RuntimeError(
                f"could not determine size of {path}; "
                f"fstat.st_size={st.st_size}, sysfs/daxctl probes failed. "
                "If this is a DAX device, verify it is enabled "
                "(`daxctl list`) and that "
                "/sys/bus/dax/devices/<name>/size is readable. If this is "
                f"/dev/{INTERLEAVED_DAX_DEVICE_NAME}, verify the module is "
                f"loaded and {_INTERLEAVED_DAX_CONFIG_PARAM} is readable."
            )
        return fd, size
    except Exception:
        os.close(fd)
        raise


def _dax_device_size(path: str) -> int:
    """Return the size in bytes of the DAX char device at `path`.

    Tries `/sys/bus/dax/devices/<name>/size` first (no root needed),
    falls back to `daxctl list -d <name> -j` if sysfs is unavailable.
    Returns 0 if both fail; the caller turns that into a useful error.
    """
    name = os.path.basename(path)  # e.g. "dax0.0"

    # Path 1: sysfs.
    sysfs_size_path = f"/sys/bus/dax/devices/{name}/size"
    try:
        with open(sysfs_size_path, "r") as f:
            value = f.read().strip()
            if value:
                return int(value)
    except OSError:
        pass

    # Path 2: daxctl JSON.
    try:
        result = subprocess.run(
            ["daxctl", "list", "-d", name, "-j"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0
    if result.returncode != 0:
        return 0
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError:
        return 0
    # daxctl returns either a single object or a list.
    if isinstance(parsed, list):
        for entry in parsed:
            if entry.get("chardev") == name and isinstance(entry.get("size"), int):
                return int(entry["size"])
    elif isinstance(parsed, dict):
        if parsed.get("chardev") == name and isinstance(parsed.get("size"), int):
            return int(parsed["size"])
    return 0


def interleaved_dax_capacity_bytes(config: str, page_size: int) -> int:
    """Compute the usable size of an interleaved_dax device from its config.

    Mirrors ``il_capacity_pages()`` in the interleaved_dax kernel module. The
    device round-robins pages over its ranges by weight and runs out as soon
    as the first range is exhausted, so::

        capacity_pages = min_i(len_pages[i] // weight[i]) * sum_i(weight[i])

    Args:
        config: The module's ``config`` parameter: comma-separated
            ``<start_gib>-<end_gib>:<weight>`` entries, e.g.
            ``"0-128:1,128-256:1"``. Numbers use C ``strtoul`` base-0
            syntax (decimal, ``0x`` hex, or ``0`` octal prefix).
        page_size: The system page size in bytes.

    Returns:
        The device capacity in bytes. Mapping past it is rejected by the
        module.

    Raises:
        ValueError: If ``config`` is empty or malformed, a range is empty,
            a weight is zero, or ``page_size`` is not positive.
    """
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")

    min_rounds = -1
    total_weight = 0
    for entry in (e.strip() for e in config.strip().split(",")):
        if not entry:
            continue
        span, sep, weight_str = entry.partition(":")
        start_str, dash, end_str = span.partition("-")
        if not sep or not dash:
            raise ValueError(
                f"malformed interleaved_dax range {entry!r}; "
                "expected <start_gib>-<end_gib>:<weight>"
            )
        try:
            start_gib = _parse_c_ulong(start_str)
            end_gib = _parse_c_ulong(end_str)
            weight = _parse_c_ulong(weight_str)
        except ValueError as exc:
            raise ValueError(f"malformed interleaved_dax range {entry!r}") from exc
        if end_gib <= start_gib or weight == 0:
            raise ValueError(f"invalid interleaved_dax range {entry!r}")

        len_pages = ((end_gib - start_gib) << 30) // page_size
        rounds = len_pages // weight
        min_rounds = rounds if min_rounds < 0 else min(min_rounds, rounds)
        total_weight += weight

    if min_rounds < 0:
        raise ValueError(f"no ranges in interleaved_dax config {config!r}")
    return min_rounds * total_weight * page_size


def _parse_c_ulong(text: str) -> int:
    """Parse an unsigned integer the way the kernel's base-0 ``kstrtoul`` does.

    Python's ``int(x, 0)`` rejects the C octal form (``"010"``), so that one
    prefix is handled explicitly.
    """
    text = text.strip()
    if len(text) > 1 and text[0] == "0" and text[1] not in "xX":
        return int(text, 8)
    value = int(text, 0)
    if value < 0:
        raise ValueError(f"negative value {text!r}")
    return value


def _interleaved_dax_device_size() -> int:
    """Return the size in bytes of ``/dev/interleaved_dax``.

    The interleaved_dax misc device exposes no size attribute; the only
    userspace-visible source is the module's read-only ``config``
    parameter. Returns 0 if the module is not loaded or its config cannot
    be parsed; the caller turns that into a useful error.
    """
    try:
        with open(_INTERLEAVED_DAX_CONFIG_PARAM, "r") as f:
            config = f.read()
    except OSError:
        return 0
    try:
        return interleaved_dax_capacity_bytes(config, os.sysconf("SC_PAGE_SIZE"))
    except ValueError as exc:
        logger.warning("could not parse %s: %s", _INTERLEAVED_DAX_CONFIG_PARAM, exc)
        return 0


def _mmap_base_address(mm: mmap.mmap) -> int:
    """Return the virtual address of the start of an mmap."""
    # ctypes.c_char.from_buffer(mm) grabs a pointer to the mapping.
    buf = (ctypes.c_char * 1).from_buffer(mm)
    return ctypes.addressof(buf)


def _struct_at(base: int, offset: int, struct_type):
    return struct_type.from_address(base + offset)


def _initialize_pool(base: int, layout: PoolLayout, gen: int) -> Header:
    """Write the header and zero out all metadata sections."""
    # Zero the metadata region (header + locks + bitmap + descs + index).
    # We don't zero the region payload area — that's the bulk of the pool
    # and clobbering it is unnecessary; slots start EMPTY so no reader
    # can mistake leftover bytes for valid KV.
    metadata_end = layout.off_regions
    ctypes.memset(base, 0, metadata_end)

    # Initialize region descriptors to FREE. Memset already set owner to
    # 0, so we explicitly stamp OWNER_FREE.
    descs = (RegionDesc * layout.region_count).from_address(
        base + layout.off_region_descs
    )
    for i in range(layout.region_count):
        descs[i].owner_node_id = OWNER_FREE

    # Initialize all index slots to EMPTY. memset covered this since
    # SLOT_STATE_EMPTY == 0, but double-check for the assertion crowd.
    slots = (Slot * layout.index_slot_count).from_address(base + layout.off_index)
    assert slots[0].line0.state == SLOT_STATE_EMPTY

    # Write header last so a partially-initialized pool can't be attached.
    header = _struct_at(base, layout.off_header, Header)
    layout.write_to_header(header, gen)
    return header


def _attach_existing(base: int, size: int) -> tuple[Header, PoolLayout]:
    header = _struct_at(base, 0, Header)

    # Check magic/version FIRST — other fields can't be trusted before this.
    # Building a PoolLayout from a zeroed header would yield nonsense sizes.
    if header.magic != MAGIC:
        raise ValueError(
            f"bad magic 0x{header.magic:x}, expected 0x{MAGIC:x}; pool is "
            "uninitialized or corrupted"
        )

    layout = PoolLayout.from_header(header)
    if header.pool_size != size:
        raise RuntimeError(
            f"CXL pool size mismatch: header says {header.pool_size}, mapping is {size}"
        )

    # Remaining header sanity (version, region_size, pool_size).
    layout.validate_against(header)

    return header, layout


def _should_populate(cfg: CXLBootstrapConfig) -> bool:
    """Resolve `cfg.populate_policy` against the device being mapped."""
    if cfg.populate_policy is PagePopulatePolicy.ALWAYS:
        return True
    if cfg.populate_policy is PagePopulatePolicy.NEVER:
        return False
    return os.path.basename(cfg.dev_path) == INTERLEAVED_DAX_DEVICE_NAME


def _populate_pages_if_needed(handle: PoolHandle) -> None:
    """Pre-populate the pool mapping's PTEs as accessed+dirty.

    See :class:`PagePopulatePolicy` for why. Best-effort: a kernel without
    ``MADV_POPULATE_WRITE`` (< 5.14) or any other madvise failure only costs
    first-write performance, so it is logged and the pool still comes up.
    """
    if not _should_populate(handle.cfg):
        return

    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.madvise.restype = ctypes.c_int

    t0 = time.perf_counter()
    for offset in range(0, handle.size, _POPULATE_STEP_BYTES):
        length = min(_POPULATE_STEP_BYTES, handle.size - offset)
        if libc.madvise(handle.base + offset, length, _MADV_POPULATE_WRITE) != 0:
            err = ctypes.get_errno()
            logger.warning(
                "MADV_POPULATE_WRITE failed on CXL pool at offset %d: %s. "
                "Continuing; the first CPU write to each page will be slow.",
                offset,
                os.strerror(err),
            )
            return
    logger.info(
        "CXL pool page tables populated (size=%d) in %.1f s",
        handle.size,
        time.perf_counter() - t0,
    )


def _cuda_host_register_if_available(handle: PoolHandle) -> None:
    """Pin the whole pool with cudaHostRegister so GPU DMAs skip the bounce buffer.

    Best-effort: on systems without CUDA we simply skip.
    """
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable; skipping cudaHostRegister of CXL pool")
        return
    try:
        _cuda_host_register(handle.base, handle.size)
        handle.cuda_registered = True
        logger.info(
            "cudaHostRegister succeeded on CXL pool (base=0x%x, size=%d)",
            handle.base,
            handle.size,
        )
    except Exception as exc:
        # Do NOT raise: we want the backend to come up even if pinning
        # fails, just with a slower DMA path. The user is warned.
        logger.warning(
            "cudaHostRegister failed on CXL pool: %s. "
            "GPU transfers will fall back to the bounce-buffer path.",
            exc,
        )


def _cuda_host_register(addr: int, size: int) -> None:
    cudart = _load_cudart()
    # cudaHostRegisterDefault == 0
    rc = cudart.cudaHostRegister(ctypes.c_void_p(addr), ctypes.c_size_t(size), 0)
    if rc != 0:
        raise RuntimeError(f"cudaHostRegister returned {rc}")


def _cuda_host_unregister(addr: int) -> None:
    cudart = _load_cudart()
    rc = cudart.cudaHostUnregister(ctypes.c_void_p(addr))
    if rc != 0:
        raise RuntimeError(f"cudaHostUnregister returned {rc}")


_cudart = None


def _load_cudart() -> ctypes.CDLL:
    global _cudart
    if _cudart is not None:
        return _cudart
    for candidate in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            _cudart = ctypes.CDLL(candidate)
            return _cudart
        except OSError:
            continue
    raise RuntimeError(
        "could not load libcudart.so; install CUDA runtime or unset CXL pinning"
    )


def _numa_bind_if_requested(numa_node: Optional[int]) -> None:
    """Pin the current thread to the given NUMA node, if libnuma is present.

    Matches TraCT §4.4: the CXL device attaches to one CPU socket via
    PCIe; remote-NUMA access pays an inter-socket hop. We nudge the
    bootstrap thread; the backend binds its worker threads separately.
    """
    if numa_node is None:
        return
    try:
        libnuma = ctypes.CDLL("libnuma.so.1")
    except OSError:
        logger.warning(
            "libnuma not available; skipping NUMA bind to node %d", numa_node
        )
        return
    if libnuma.numa_available() < 0:
        logger.warning("numa_available() returned <0; skipping bind to %d", numa_node)
        return
    if libnuma.numa_run_on_node(ctypes.c_int(numa_node)) < 0:
        logger.warning("numa_run_on_node(%d) failed", numa_node)
    else:
        logger.info("NUMA-bound CXL bootstrap thread to node %d", numa_node)
