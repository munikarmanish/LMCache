# SPDX-License-Identifier: Apache-2.0
"""CXL pool bootstrap: mmap, header init/verify, cudaHostRegister.

Responsibilities:
- Open /dev/dax0.0 (or any file/device) and mmap it.
- Compute the geom_hash for the current cluster config.
- If the pool is uninitialized (or the caller is the bootstrap node),
  write the header; otherwise, validate the existing header and reject
  on generation/geom mismatch.
- Pin the mapping with cudaHostRegister so CUDA treats it as
  page-locked host memory (TraCT §4.4) — this is what avoids the
  DRAM bounce buffer on GPU transfers.
- Expose accessor views for other modules (index, bitmap, descriptors,
  locks, region payload base).

Concurrency and lock-manager bootstrap live in cxl/locks.py (next slice).
"""

# Standard
from dataclasses import dataclass
from hashlib import blake2b
import ctypes
import json
import mmap
import os
import stat
import subprocess
from typing import Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.cxl.layout import (
    CACHELINE_SIZE,
    DEFAULT_MAX_NODES,
    DEFAULT_NUM_LOCKS,
    DEFAULT_REGION_SIZE,
    GEOM_HASH_SIZE,
    Header,
    LockSlot,
    MAGIC,
    OWNER_FREE,
    PoolLayout,
    RegionDesc,
    SLOT_STATE_EMPTY,
    Slot,
)

logger = init_logger(__name__)


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
    # Compile-time assertion min: we need at least this many chunks of
    # capacity or the backend would be pointless. Tuned upward by
    # callers via config; the default is intentionally tiny for tests.
    min_chunks_per_region: int = 1


def compute_geom_hash(metadata: LMCacheMetadata) -> bytes:
    """Digest the fields that must match bit-for-bit across peers.

    If two peers disagree on any of these, raw bytes on CXL cannot be
    safely DMA'd: the layout of a KV chunk, its dtype, or the chunk
    hash scheme differs. The 16-byte digest is stored in the pool
    header at bootstrap and in every slot at insert.
    """
    payload = {
        "model_name": metadata.model_name,
        "world_size": metadata.world_size,
        "kv_dtype": str(metadata.kv_dtype),
        "kv_shape": list(metadata.kv_shape),
        "use_mla": metadata.use_mla,
        "chunk_size": metadata.chunk_size,
        # We don't hash worker_id: different TP ranks share geometry.
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return blake2b(encoded, digest_size=GEOM_HASH_SIZE).digest()


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
    geom_hash: bytes
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
        return (
            self.base
            + self.layout.off_regions
            + region_id * self.layout.region_size
        )

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


def bootstrap_pool(
    cfg: CXLBootstrapConfig, metadata: LMCacheMetadata
) -> PoolHandle:
    """Open the pool and return a live PoolHandle.

    If `cfg.initialize` is True, the pool is (re)initialized: header is
    written, region bitmap cleared, descriptors reset, index zeroed.
    Otherwise the existing header is validated against `cfg` and
    `metadata`, and the caller attaches without mutating the pool.
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
    geom_hash = compute_geom_hash(metadata)

    try:
        if cfg.initialize:
            layout = PoolLayout.compute(
                pool_size=size,
                region_size=cfg.region_size,
                num_locks=cfg.num_locks,
                max_nodes=cfg.max_nodes,
            )
            _validate_sizing(layout, cfg, metadata)
            header = _initialize_pool(base, layout, geom_hash, cfg.generation)
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
            header, layout = _attach_existing(base, size, geom_hash)
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
            geom_hash=geom_hash,
        )

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

    Raises RuntimeError with a diagnostic message if none of these
    work — this previously masked the DAX case as "stat reported 0,
    lseek reported 0" which was unhelpful.
    """
    fd = os.open(path, os.O_RDWR)
    try:
        st = os.fstat(fd)
        size = st.st_size

        if size <= 0 and stat.S_ISCHR(st.st_mode):
            size = _dax_device_size(path)

        if size <= 0:
            raise RuntimeError(
                f"could not determine size of {path}; "
                f"fstat.st_size={st.st_size}, sysfs/daxctl probes failed. "
                "If this is a DAX device, verify it is enabled "
                "(`daxctl list`) and that "
                "/sys/bus/dax/devices/<name>/size is readable."
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


def _mmap_base_address(mm: mmap.mmap) -> int:
    """Return the virtual address of the start of an mmap."""
    # ctypes.c_char.from_buffer(mm) grabs a pointer to the mapping.
    buf = (ctypes.c_char * 1).from_buffer(mm)
    return ctypes.addressof(buf)


def _struct_at(base: int, offset: int, struct_type):
    return struct_type.from_address(base + offset)


def _initialize_pool(
    base: int, layout: PoolLayout, geom_hash: bytes, gen: int
) -> Header:
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
    layout.write_to_header(header, geom_hash, gen)
    return header


def _attach_existing(
    base: int, size: int, geom_hash: bytes
) -> tuple[Header, PoolLayout]:
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

    observed_geom = bytes(header.geom_hash)
    if observed_geom != geom_hash:
        raise RuntimeError(
            "CXL pool geom_hash mismatch: pool was initialized with a different "
            "model/chunk geometry. Refusing to attach. "
            f"pool={observed_geom.hex()} local={geom_hash.hex()}"
        )
    return header, layout


def _validate_sizing(
    layout: PoolLayout, cfg: CXLBootstrapConfig, metadata: LMCacheMetadata
) -> None:
    """Assert that regions are big enough to hold a useful number of chunks."""
    # Rough estimate of a chunk's bytes: use the default KV_2LTD layout.
    # We deliberately keep this a loose lower bound — the backend
    # computes the real chunk size lazily once allocators are wired.
    dtype_size = torch.tensor([], dtype=metadata.kv_dtype).element_size()
    shapes = metadata.get_shapes(num_tokens=metadata.chunk_size)
    if not shapes:
        return
    chunk_bytes = int(shapes[0].numel()) * dtype_size
    if chunk_bytes <= 0:
        return
    if layout.region_size < cfg.min_chunks_per_region * chunk_bytes:
        raise ValueError(
            f"region_size {layout.region_size} < {cfg.min_chunks_per_region} x "
            f"estimated chunk bytes {chunk_bytes}; increase cxl_region_size"
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
