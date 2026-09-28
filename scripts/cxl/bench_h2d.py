#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark: host <-> GPU cudaMemcpyAsync bandwidth vs payload size.

Measures how H2D (host->GPU read) and D2H (GPU->host write) bandwidth scale
with the per-copy payload. Small copies are dominated by per-op launch/DMA-
setup overhead; large copies approach the PCIe/NVLink ceiling. The knee tells
you the smallest payload that still streams efficiently (e.g. the staging/
transfer size to use on the retrieve fill / store eviction paths). Use --dir
to pick a direction (default both).

Hosts compared (all pinned, so all are true async DMA — no bounce buffer):
  - DRAM : page-locked host DRAM (a torch pinned tensor; what the L1 KV
           path's staging uses).
  - CXL  : an mmap of a CXL DAX device, cudaHostRegister'd exactly like the
           CXL pool. Only measured when --cxl-dev is given. Lets you compare
           CXL<->GPU vs DRAM<->GPU DMA bandwidth head to head.

Timing uses CUDA events on the copy stream (device-side time, excludes
Python overhead). Each size is warmed up, then timed over many iterations;
bandwidth is reported from the median per-copy time.

Cold by default (DDIO-immune): reusing one host buffer would let the CPU keep
it in the LLC, so the DMA is served from cache (Intel DDIO) instead of the
real host media -- which inflates CXL bandwidth to the DRAM ceiling for any
payload under the DDIO window (~a couple of LLC ways). To measure the true
media bandwidth, the host offset rotates over a region larger than the LLC, so
a line has been evicted before it is touched again. Pass --no-rotate to
reproduce the cache-warm (DDIO-inflated) numbers. (For D2H, even rotation
cannot fully expose media bandwidth at small payloads -- the write completes
into the LLC and is written back to media asynchronously; see _time_copy.)

With --gpus, several GPUs drive the SAME host buffer at once (one CUDA stream
each) and the table reports solo vs aggregate bandwidth plus scaling
efficiency. That answers whether the host side is a shared budget: ~100%
efficiency means the GPUs have independent paths, while ~1/n means they are
splitting one ceiling (expected when the limit is the host media or a link
they share rather than each GPU's own PCIe link).

Run on a CUDA box (lmcache venv). For the CXL source, run where the DAX
device is mappable (not in a sandbox that SIGBUSes on DAX mmap):
  ~/.virtualenvs/lmcache/bin/python scripts/cxl/bench_h2d.py
  ~/.virtualenvs/lmcache/bin/python scripts/cxl/bench_h2d.py \
      --cxl-dev /dev/dax0.0
  # both GPUs pulling from one shared host buffer:
  ~/.virtualenvs/lmcache/bin/python scripts/cxl/bench_h2d.py \
      --cxl-dev /dev/dax0.0 --gpus all --dir h2d
"""

# Future
from __future__ import annotations

# Standard
import argparse
import ctypes
import dataclasses
import mmap
import os
import statistics
import sys
import time

# Third Party
import torch


@dataclasses.dataclass
class GpuContext:
    """One GPU's participation in a transfer: the device, its own copy stream,
    the device buffer it copies into/out of, and that buffer's address.

    `buf` is held only to keep the allocation alive for the lifetime of the
    context; `ptr` is what the copy primitive is handed."""

    device: torch.device
    stream: torch.cuda.Stream
    buf: torch.Tensor
    ptr: int


def _sizes(min_kib: int, max_kib: int) -> list[int]:
    """Power-of-two byte sizes from min_kib to max_kib inclusive."""
    sizes = []
    s = min_kib * 1024
    cap = max_kib * 1024
    while s <= cap:
        sizes.append(s)
        s *= 2
    return sizes


def _llc_bytes() -> int:
    """Largest CPU cache size (the LLC, usually L3) in bytes, read from
    sysfs. Falls back to 256 MiB if sysfs is unavailable. Used to size the
    cold rotation region so a revisited source line has been evicted from the
    LLC before the GPU DMA reads it again (defeating DDIO cache residency)."""
    best = 0
    cache_dir = "/sys/devices/system/cpu/cpu0/cache"
    units = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    try:
        entries = os.listdir(cache_dir)
    except OSError:
        entries = []
    for entry in entries:
        try:
            with open(os.path.join(cache_dir, entry, "size")) as f:
                raw = f.read().strip()  # e.g. "300M", "2M", "48K"
        except OSError:
            continue
        if not raw:
            continue
        mult = units.get(raw[-1].upper(), 1)
        digits = raw[:-1] if raw[-1].upper() in units else raw
        try:
            best = max(best, int(digits) * mult)
        except ValueError:
            continue
    return best if best > 0 else (256 << 20)


def _prefault(addr: int, nbytes: int) -> None:
    """CPU-read [addr, addr+nbytes) once so its page-table entries are
    populated. DAX mmaps fault PTEs lazily, so an unfaulted page would add a
    one-time fault to the first DMA that touches it; with rotation every timed
    copy reads fresh pages, which would fold that fault into the median. The
    read is sequential, so it leaves only the LLC-sized tail of the range warm
    -- the rotating DMA reads that follow stay cache-cold."""
    chunk = 1 << 20
    scratch = (ctypes.c_char * chunk)()
    off = 0
    while off < nbytes:
        n = min(chunk, nbytes - off)
        ctypes.memmove(scratch, addr + off, n)
        off += n


# cudaHostRegister flags (driver_types.h).
_CUDA_HOST_REGISTER_DEFAULT = 0
_CUDA_HOST_REGISTER_PORTABLE = 1
_CUDA_HOST_REGISTER_MAPPED = 2
_CUDA_HOST_REGISTER_IO_MEMORY = 4


def _cudart() -> ctypes.CDLL:
    lib = None
    for cand in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            lib = ctypes.CDLL(cand)
            break
        except OSError:
            continue
    if lib is None:
        raise RuntimeError("could not load libcudart.so")
    # Declare signatures so 64-bit pointers/sizes are not truncated.
    lib.cudaHostRegister.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    lib.cudaHostRegister.restype = ctypes.c_int
    lib.cudaHostUnregister.argtypes = [ctypes.c_void_p]
    lib.cudaHostUnregister.restype = ctypes.c_int
    lib.cudaHostGetDevicePointer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    lib.cudaHostGetDevicePointer.restype = ctypes.c_int
    return lib


def _verify_pinned(lib: ctypes.CDLL, addr: int, size: int) -> bool:
    """Return True iff CUDA can resolve a device pointer for the registered
    host range — i.e. the copy from it is a true pinned DMA, not a pageable
    bounce. ``cudaHostGetDevicePointer`` succeeds only for registered memory.
    """
    devptr = ctypes.c_void_p()
    rc = lib.cudaHostGetDevicePointer(ctypes.byref(devptr), ctypes.c_void_p(addr), 0)
    return rc == 0 and devptr.value is not None


def _dax_device_size(dev_path: str) -> int:
    """Size in bytes of a DAX character device, read from sysfs, or 0 if it
    cannot be determined. ``fstat`` reports 0 for a DAX chardev, so sysfs is
    the only way to learn the real extent without mapping it."""
    name = os.path.basename(dev_path)  # e.g. "dax0.0"
    for path in (
        f"/sys/bus/dax/devices/{name}/size",
        f"/sys/class/dax/{name}/size",
    ):
        try:
            with open(path) as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            continue
    return 0


def _open_cxl_source(
    dev_path: str,
    want_bytes: int,
    flags: int,
    windows: tuple[tuple[int, int], ...] = (),
) -> tuple[int, mmap.mmap, int]:
    """mmap a CXL DAX device and cudaHostRegister it with ``flags``. Returns
    (ptr, mmap, mapped_size). Verifies the pinning actually took (loud error if
    cudaHostRegister fails or the range isn't device-resolvable — which would
    silently fall back to a DRAM bounce buffer).

    The whole device is mapped so any offset within it is addressable, but only
    `windows` — a list of (offset, length) byte ranges relative to the mapping
    — is pinned. Pinning is a scarce resource: registering a whole 256 GiB pool
    fails with cudaErrorMemoryAllocation, so callers pass just the ranges they
    will actually touch. An empty list pins [0, want_bytes), the single-window
    default."""
    fd = os.open(dev_path, os.O_RDWR)
    try:
        size = os.fstat(fd).st_size
        if size == 0:  # DAX char device reports 0 via fstat
            # sysfs knows the real extent; fall back to the caller's request
            # only if it is unreadable. Mapping the whole device (rather than
            # just the rotation span) is what lets --offset-gap-gib reach a
            # second backing module further up the address range.
            size = _dax_device_size(dev_path) or want_bytes
        mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)
    base = ctypes.addressof(ctypes.c_char.from_buffer(mm))

    lib = _cudart()
    pin = windows if windows else [(0, min(want_bytes, size))]
    rc = 0
    for off, length in pin:
        rc = lib.cudaHostRegister(
            ctypes.c_void_p(base + off), ctypes.c_size_t(length), ctypes.c_uint(flags)
        )
        print(
            f"[bench] cudaHostRegister(base=0x{base + off:x}, size={length}, "
            f"flags={flags}) -> rc={rc} ({'OK' if rc == 0 else 'FAILED'})",
            flush=True,
        )
        if rc != 0:
            break
    if rc != 0:
        raise RuntimeError(
            f"cudaHostRegister failed rc={rc}; the CXL source would NOT be "
            f"pinned (copies would bounce through DRAM). Try --cxl-register-flags "
            f"4 (cudaHostRegisterIoMemory) for device memory."
        )
    pinned = _verify_pinned(lib, base + pin[0][0], pin[0][1])
    verdict = (
        "OK (true pinned DMA)"
        if pinned
        else "FAILED (NOT device-resolvable -> bounces through DRAM!)"
    )
    print(f"[bench] cudaHostGetDevicePointer -> {verdict}", flush=True)
    if not pinned:
        print(
            "[bench] WARNING: the CXL range registered but is not device-"
            "resolvable; the H2D copy is going through a DRAM bounce buffer, "
            "so the measured 'CXL' bandwidth is really the bounce path. Try "
            "--cxl-register-flags 4.",
            file=sys.stderr,
            flush=True,
        )
    return base, mm, size


def _cuda_host_unregister(addr: int) -> None:
    _cudart().cudaHostUnregister(ctypes.c_void_p(addr))


def _time_copy_parallel(
    gpus: list[GpuContext],
    host_base: int,
    nbytes: int,
    iters: int,
    rotate_span: int,
    direction: str,
    offset_gap: int = 0,
) -> tuple[float, list[float]]:
    """Drive every GPU in `gpus` against the SAME host buffer at once and
    return (aggregate_GB_per_s, [per_gpu_GB_per_s, ...]).

    `offset_gap` spaces the GPUs apart in the host mapping: GPU i works at
    ``host_base + i * offset_gap``, each rotating over its own `rotate_span`
    window. With the default 0 every GPU hammers the same window, which is what
    measures contention on one region. A gap large enough to land the GPUs in
    different backing devices instead measures whether those devices are
    independent -- see ``--offset-gap-gib``.

    Each GPU issues `iters` back-to-back `nbytes` copies on its own stream, so
    all of them are in flight over the shared host buffer simultaneously. This
    is what reveals whether the host-side media/link is a shared budget: if two
    GPUs each sustain their solo bandwidth the path scales, and if each falls
    to about half then they are contending for one ceiling.

    Timing is wall-clock across the whole overlapping batch rather than the
    per-copy CUDA-event timing used by the single-GPU path: event pairs on
    different devices cannot be compared, and the quantity of interest here is
    aggregate throughput while every GPU is busy, not the latency of one copy.
    Every stream is synchronized before the clock starts and again before it
    stops, so the measured window covers only fully overlapped transfers.

    Per-GPU GB/s is computed from that same shared window (bytes moved by that
    GPU divided by the wall time), so the per-GPU figures sum to the aggregate.

    `direction`, `rotate_span` and the DDIO caveats are exactly as described in
    ``_time_copy``. Each GPU walks the rotation region from a different
    starting slot, so concurrent GPUs read disjoint lines rather than sharing
    cache-resident ones."""
    # First Party
    import lmcache.c_ops as lmc_ops

    nslots = max(1, rotate_span // nbytes)
    is_h2d = direction == "h2d"
    xfer = lmc_ops.TransferDirection.H2D if is_h2d else lmc_ops.TransferDirection.D2H

    def _issue(gpu: GpuContext, gpu_idx: int, slot: int) -> None:
        host = host_base + gpu_idx * offset_gap + (slot % nslots) * nbytes
        dst, src = (gpu.ptr, host) if is_h2d else (host, gpu.ptr)
        lmc_ops.lmcache_memcpy_async(dst, src, nbytes, xfer, 0, nbytes)

    # Within a shared window, stagger the starting slots so concurrent GPUs
    # touch different lines. With an offset gap the GPUs are already in
    # separate windows, so each starts at the top of its own.
    stride = 0 if offset_gap else max(1, nslots // len(gpus))

    # Warm up: first copy on each device pays one-time setup.
    for idx, gpu in enumerate(gpus):
        torch.cuda.set_device(gpu.device)
        with torch.cuda.stream(gpu.stream):
            for j in range(5):
                _issue(gpu, idx, idx * stride + j)
    for gpu in gpus:
        gpu.stream.synchronize()

    # Enqueue round-robin across GPUs rather than draining one GPU's whole
    # batch before starting the next. Each _issue is a Python call, so filling
    # N copies takes real CPU time; enqueueing per-GPU would let the first GPU
    # run (and finish) while the last was still being fed, shrinking the
    # overlap window as `iters` grows and understating aggregate bandwidth.
    # Round-robin keeps every stream fed from the start.
    handles = [
        (idx, gpu, torch.cuda.stream(gpu.stream)) for idx, gpu in enumerate(gpus)
    ]
    start = time.perf_counter()
    for i in range(iters):
        for idx, gpu, ctx in handles:
            torch.cuda.set_device(gpu.device)
            with ctx:
                _issue(gpu, idx, idx * stride + i)
    for gpu in gpus:
        gpu.stream.synchronize()
    elapsed = time.perf_counter() - start

    if elapsed <= 0.0:
        return 0.0, [0.0 for _ in gpus]
    per_gpu = [(iters * nbytes) / elapsed / 1e9 for _ in gpus]
    return sum(per_gpu), per_gpu


def _time_copy(
    gpu_ptr: int,
    host_base: int,
    nbytes: int,
    stream: torch.cuda.Stream,
    iters: int,
    rotate_span: int,
    direction: str,
) -> float:
    """Return median per-copy seconds for an `nbytes` copy between the GPU
    buffer and the host buffer via the production primitive
    (lmc_ops.lmcache_memcpy_async), timed with CUDA events on `stream`.

    `direction` is "h2d" (host->GPU read) or "d2h" (GPU->host write). The host
    side carries the rotating offset in both cases (it is the source for h2d,
    the destination for d2h); the GPU side is fixed at offset 0.

    The host offset rotates over [0, rotate_span) in `nbytes` strides, so a
    line is not re-touched until rotate_span bytes later. With rotate_span
    larger than the LLC this keeps every access cache-cold, so the DMA hits the
    real host media instead of the CPU's LLC (DDIO). Pass rotate_span == nbytes
    to disable rotation and reuse a single buffer (the cache-warm / DDIO path).

    Note for d2h: the CUDA event fires when the write is accepted into the
    coherency domain, which for payloads under the DDIO window is the LLC, not
    the host media -- so small d2h numbers reflect write-into-LLC acceptance,
    and only payloads exceeding that window reflect true media writeback."""
    # First Party
    import lmcache.c_ops as lmc_ops

    nslots = max(1, rotate_span // nbytes)
    is_h2d = direction == "h2d"
    xfer = lmc_ops.TransferDirection.H2D if is_h2d else lmc_ops.TransferDirection.D2H

    def _do(offset: int):
        # alignment>=nbytes -> one full cudaMemcpyAsync on the current stream
        # (the same call the CXL / L2-resident path makes). `offset` rotates
        # the host side so consecutive copies touch distinct, cache-cold lines.
        host = host_base + offset
        dst, src = (gpu_ptr, host) if is_h2d else (host, gpu_ptr)
        lmc_ops.lmcache_memcpy_async(dst, src, nbytes, xfer, 0, nbytes)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(stream):
        for j in range(5):  # warm up (first copy pays one-time setup)
            _do((j % nslots) * nbytes)
        stream.synchronize()

        samples_ms = []
        for i in range(iters):
            offset = (i % nslots) * nbytes
            start.record(stream)
            _do(offset)
            end.record(stream)
            end.synchronize()
            samples_ms.append(start.elapsed_time(end))  # ms

    return statistics.median(samples_ms) / 1000.0  # seconds


def _run_direction_parallel(
    direction: str,
    sizes: list[int],
    gpus: list[GpuContext],
    hosts: list[tuple[str, int]],
    iters: int,
    rotate: bool,
    rotate_span: int,
    offset_gap: int = 0,
) -> None:
    """For each host buffer in `hosts` (label, base pointer), measure one
    `direction` across every payload in `sizes` twice: once with a single GPU
    (the solo baseline) and once with every GPU in `gpus` driving the host
    buffer concurrently. Print solo, aggregate, per-GPU and scaling efficiency.

    Efficiency is aggregate / (solo * n_gpus): near 100% means the GPUs have
    independent paths to the host buffer, while near 1/n_gpus means they are
    splitting one shared ceiling.

    `offset_gap` spaces the concurrent GPUs apart in the host mapping (GPU i at
    ``base + i * offset_gap``) so they can be aimed at different backing
    devices; the solo baseline always runs at the base, so the comparison is
    against the same region in both cases."""
    arrow = "host->GPU (H2D read)" if direction == "h2d" else "GPU->host (D2H write)"
    ngpu = len(gpus)

    for label, host_ptr in hosts:
        if not host_ptr:
            continue
        print(f"\n=== {direction.upper()}  {arrow}  [{label}] ===")
        print(
            f"{'payload':>10} {'solo GB/s':>11} {'aggr GB/s':>11} "
            f"{'speedup':>9} {'effic':>7}   per-GPU GB/s"
        )
        for n in sizes:
            span = rotate_span if rotate else n
            solo, _ = _time_copy_parallel(gpus[:1], host_ptr, n, iters, span, direction)
            aggr, per_gpu = _time_copy_parallel(
                gpus, host_ptr, n, iters, span, direction, offset_gap
            )
            speedup = aggr / solo if solo > 0 else 0.0
            effic = speedup / ngpu if ngpu else 0.0
            label_n = (
                f"{n // 1024} KiB" if n < 1024 * 1024 else f"{n // (1024 * 1024)} MiB"
            )
            per_txt = "  ".join(f"{g:.2f}" for g in per_gpu)
            print(
                f"{label_n:>10} {solo:>11.2f} {aggr:>11.2f} "
                f"{speedup:>8.2f}x {effic * 100:>6.0f}%   {per_txt}",
                flush=True,
            )


def _run_direction(
    direction: str,
    sizes: list[int],
    gpu_ptr: int,
    dram_ptr: int,
    cxl_ptr: int,
    stream: torch.cuda.Stream,
    iters: int,
    rotate: bool,
    rotate_span: int,
) -> None:
    """Time DRAM (and, if cxl_ptr != 0, CXL) copies for one `direction`
    ("h2d" or "d2h") across every payload in `sizes`, and print a table of
    GB/s and per-copy microseconds with the CXL/DRAM ratio. The host side
    rotates over [0, rotate_span) when `rotate` is set, else reuses offset 0."""
    arrow = "host->GPU (H2D read)" if direction == "h2d" else "GPU->host (D2H write)"
    print(f"\n=== {direction.upper()}  {arrow} ===")
    header = f"{'payload':>10} {'DRAM GB/s':>12} {'DRAM us':>10}"
    if cxl_ptr:
        header += f" {'CXL GB/s':>12} {'CXL us':>10} {'CXL/DRAM':>9}"
    print(header)

    for n in sizes:
        # rotate over the whole span when cold; reuse offset 0 when warm.
        span = rotate_span if rotate else n
        sec_d = _time_copy(gpu_ptr, dram_ptr, n, stream, iters, span, direction)
        gbps_d = (n / sec_d) / 1e9
        label = f"{n // 1024} KiB" if n < 1024 * 1024 else f"{n // (1024 * 1024)} MiB"
        line = f"{label:>10} {gbps_d:>12.2f} {sec_d * 1e6:>10.2f}"
        if cxl_ptr:
            sec_c = _time_copy(gpu_ptr, cxl_ptr, n, stream, iters, span, direction)
            gbps_c = (n / sec_c) / 1e9
            ratio = gbps_c / gbps_d if gbps_d > 0 else 0.0
            line += f" {gbps_c:>12.2f} {sec_c * 1e6:>10.2f} {ratio:>9.2f}"
        print(line, flush=True)


def main() -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--min-kib", type=int, default=1, help="smallest payload in KiB")
    p.add_argument(
        "--max-kib",
        type=int,
        default=64 * 1024,
        help="largest payload in KiB (default 65536 = 64 MiB)",
    )
    p.add_argument("--iters", type=int, default=200, help="timed copies per size")
    p.add_argument("--device", default="cuda:0", help="CUDA device")
    p.add_argument(
        "--gpus",
        default="",
        help="comma-separated CUDA device indices to drive in parallel against "
        "the SAME host buffer (e.g. '0,1', or 'all'). Each GPU gets its own "
        "stream and they copy concurrently, so the table reports solo vs "
        "aggregate bandwidth and the scaling efficiency -- this is what shows "
        "whether the host media/link is a shared budget. Omit for the "
        "original single-GPU per-payload table.",
    )
    p.add_argument(
        "--offset-gap-gib",
        type=float,
        default=0.0,
        help="with --gpus, place GPU i at base + i*GAP in the host mapping "
        "(default 0 = every GPU on the same region). Use it to aim concurrent "
        "GPUs at DIFFERENT backing devices: a pool built by concatenating two "
        "modules puts the second module's address range after the first, so a "
        "gap of half the pool size lands GPU 1 on the other module. Comparing "
        "gap=0 (same module) against gap=half-the-pool (different modules) "
        "shows whether the modules are independent -- if aggregate roughly "
        "doubles, the hardware has the bandwidth and the pool is concatenated "
        "rather than interleaved.",
    )
    p.add_argument(
        "--dir",
        choices=("h2d", "d2h", "both"),
        default="both",
        help="direction(s) to measure: h2d (host->GPU read), d2h (GPU->host "
        "write), or both",
    )
    p.add_argument(
        "--cxl-dev",
        default=None,
        help="CXL DAX device to also measure as a pinned host buffer "
        "(e.g. /dev/dax0.0); omit to measure DRAM only",
    )
    p.add_argument(
        "--cxl-register-flags",
        type=int,
        default=_CUDA_HOST_REGISTER_PORTABLE,
        help="cudaHostRegister flags for the CXL mapping: 0=Default, "
        "1=Portable, 2=Mapped, 4=IoMemory (values OR together). Defaults to "
        "Portable so the mapping is pinned for EVERY CUDA context -- required "
        "by --gpus, since a non-portable registration is true pinned DMA only "
        "for the registering device and silently bounces for the others. Use "
        "4 if the default registers but isn't device-resolvable.",
    )
    p.add_argument(
        "--no-rotate",
        action="store_true",
        help="reuse a single host buffer for every timed copy (cache-warm). "
        "By default the host offset rotates over a region larger than the LLC "
        "so accesses stay cache-cold and are not served from the CPU's LLC "
        "(DDIO); this flag restores the warm behavior for A/B comparison.",
    )
    p.add_argument(
        "--cold-region-mib",
        type=int,
        default=0,
        help="size in MiB of the rotation region (0 = auto: LLC size + max "
        "payload). Must exceed the LLC to defeat DDIO. Ignored with --no-rotate.",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="register + verify the CXL pinning and exit (no timing)",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 1

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    sizes = _sizes(args.min_kib, args.max_kib)
    max_bytes = sizes[-1]
    stream = torch.cuda.Stream(device=dev)

    # Parallel mode: which devices drive the shared host buffer together.
    if args.gpus.strip().lower() == "all":
        gpu_indices = list(range(torch.cuda.device_count()))
    elif args.gpus.strip():
        gpu_indices = [int(x) for x in args.gpus.split(",") if x.strip()]
    else:
        gpu_indices = []
    for idx in gpu_indices:
        if idx < 0 or idx >= torch.cuda.device_count():
            print(
                f"--gpus: device {idx} out of range (have {torch.cuda.device_count()})",
                file=sys.stderr,
            )
            return 1

    # Rotation region: with --no-rotate every copy reuses offset 0 (one
    # payload's worth, cache-warm); otherwise the host buffer spans a region
    # larger than the LLC so rotating accesses stay cache-cold (DDIO-immune).
    rotate = not args.no_rotate
    if not rotate:
        region_bytes = max_bytes
    elif args.cold_region_mib > 0:
        region_bytes = max(args.cold_region_mib << 20, max_bytes)
    else:
        region_bytes = _llc_bytes() + max_bytes

    # GPU buffer, reused (sliced) for every size and host: the destination for
    # H2D, the source for D2H.
    gpu_buf = torch.empty(max_bytes, dtype=torch.uint8, device=dev)
    gpu_ptr = gpu_buf.data_ptr()

    # Pinned DRAM host buffer spanning the whole rotation region.
    host_dram = torch.ones(region_bytes, dtype=torch.uint8, pin_memory=True)
    dram_ptr = host_dram.data_ptr()

    # Optional pinned CXL host buffer.
    cxl_ptr = None
    cxl_mm = None
    cxl_size = 0
    offset_gap = int(args.offset_gap_gib * (1 << 30))
    ngpu_par = len(gpu_indices)
    if offset_gap and not ngpu_par:
        print("--offset-gap-gib has no effect without --gpus.", file=sys.stderr)
        return 1

    # Each participating GPU works in its own window when a gap is set, so pin
    # one region_bytes range per GPU. Pinning the entire pool would fail: CUDA
    # cannot register 256 GiB (cudaErrorMemoryAllocation).
    cxl_windows = tuple(
        (i * offset_gap, region_bytes) for i in range(ngpu_par if offset_gap else 1)
    )
    if args.cxl_dev is not None:
        cxl_ptr, cxl_mm, cxl_size = _open_cxl_source(
            args.cxl_dev, region_bytes, args.cxl_register_flags, cxl_windows
        )
        if args.verify_only:
            for off, _ in cxl_windows:
                _cuda_host_unregister(cxl_ptr + off)
            return 0
        if cxl_size < max_bytes:
            print(
                f"[bench] WARNING: CXL device {cxl_size / (1 << 20):.0f} MiB < "
                f"max payload {max_bytes / (1 << 20):.0f} MiB; capping sizes.",
                file=sys.stderr,
            )
            sizes = [n for n in sizes if n <= cxl_size]
            max_bytes = sizes[-1]
        if offset_gap:
            needed = (ngpu_par - 1) * offset_gap + region_bytes
            if needed > cxl_size:
                print(
                    f"--offset-gap-gib {args.offset_gap_gib:g} needs "
                    f"{needed / (1 << 30):.1f} GiB but the CXL device is only "
                    f"{cxl_size / (1 << 30):.1f} GiB.",
                    file=sys.stderr,
                )
                return 1
    elif offset_gap:
        print(
            "--offset-gap-gib applies to the CXL mapping; pass --cxl-dev.",
            file=sys.stderr,
        )
        return 1

    # The rotation span is the region each GPU walks. With a gap the windows
    # are separate ranges of region_bytes each, so the span is region_bytes
    # rather than the whole (much larger) device.
    rotate_span = region_bytes
    if cxl_ptr is not None and not offset_gap:
        rotate_span = min(rotate_span, cxl_size)

    # Populate page-table entries over the span so rotating accesses do not
    # fold a first-touch fault into the median (DAX mmaps fault lazily). The
    # sequential read leaves only the LLC-sized tail warm, so accesses stay
    # cold for both directions. With an offset gap each GPU's CXL window is a
    # separate range, so every one of them is faulted.
    if rotate:
        _prefault(dram_ptr, rotate_span)
        if cxl_ptr is not None:
            nwin = ngpu_par if (offset_gap and ngpu_par) else 1
            for i in range(nwin):
                _prefault(cxl_ptr + i * offset_gap, rotate_span)

    mode = (
        f"cold (rotating over {rotate_span / (1 << 20):.0f} MiB, "
        f"LLC={_llc_bytes() / (1 << 20):.0f} MiB)"
        if rotate
        else "warm (single buffer, DDIO-cached)"
    )
    print(
        f"[bench] device={torch.cuda.get_device_name(dev)} "
        f"primitive=lmc_ops.lmcache_memcpy_async iters={args.iters} mode={mode}",
        flush=True,
    )

    directions = ("h2d", "d2h") if args.dir == "both" else (args.dir,)
    cxl_arg = cxl_ptr if cxl_ptr is not None else 0
    try:
        if gpu_indices:
            # One stream and one device buffer per participating GPU; they all
            # copy against the single shared host buffer.
            gpus = []
            for idx in gpu_indices:
                gdev = torch.device(f"cuda:{idx}")
                torch.cuda.set_device(gdev)
                gbuf = torch.empty(max_bytes, dtype=torch.uint8, device=gdev)
                gpus.append(
                    GpuContext(
                        device=gdev,
                        stream=torch.cuda.Stream(device=gdev),
                        buf=gbuf,
                        ptr=gbuf.data_ptr(),
                    )
                )
            gap_note = (
                f", CXL offset gap {args.offset_gap_gib:g} GiB/GPU (DRAM always 0)"
                if offset_gap
                else " sharing one host buffer"
            )
            print(
                f"[bench] parallel mode: {len(gpus)} GPU(s) "
                f"{[g.device.index for g in gpus]}{gap_note}",
                flush=True,
            )
            # The gap is a property of the CXL pool's layout, so DRAM keeps
            # gap 0 and remains the "this is what scaling looks like" row.
            for direction in directions:
                _run_direction_parallel(
                    direction,
                    sizes,
                    gpus,
                    [("DRAM", dram_ptr)],
                    args.iters,
                    rotate,
                    rotate_span,
                    0,
                )
                _run_direction_parallel(
                    direction,
                    sizes,
                    gpus,
                    [("CXL", cxl_arg)],
                    args.iters,
                    rotate,
                    rotate_span,
                    offset_gap,
                )
        else:
            for direction in directions:
                _run_direction(
                    direction,
                    sizes,
                    gpu_ptr,
                    dram_ptr,
                    cxl_arg,
                    stream,
                    args.iters,
                    rotate,
                    rotate_span,
                )
    finally:
        if cxl_ptr is not None:
            for off, _ in cxl_windows:
                try:
                    _cuda_host_unregister(cxl_ptr + off)
                except Exception:
                    pass
            try:
                cxl_mm.close()
            except BufferError:
                pass  # ctypes view still exported; OS reclaims on exit

    note = (
        "\n[bench] Small payloads are per-op-overhead bound (low GB/s); "
        "bandwidth rises toward the link ceiling as payload grows. The knee is "
        "the smallest payload that streams efficiently. CXL/DRAM < 1 means the "
        "CXL pool's DMA is slower than DRAM's at that size. D2H (write) to CXL "
        "is typically slower than H2D (read) -- CXL.mem writes cost more than "
        "reads."
    )
    if rotate:
        note += (
            " Cold mode: the host buffer rotates over a region larger than the "
            "LLC, so CXL H2D numbers reflect real media read bandwidth (not "
            "DDIO-cached L3). For D2H, small payloads still complete into the "
            "LLC (writeback to media is async), so only large-payload D2H "
            "reflects media write bandwidth. Re-run with --no-rotate for the "
            "cache-warm/DDIO-inflated numbers."
        )
    else:
        note += (
            " WARNING: --no-rotate reuses one buffer, so payloads under the "
            "DDIO window are served from the CPU's LLC -- the CXL numbers there "
            "are L3 bandwidth, not CXL. Drop the flag for true media bandwidth."
        )
    print(note, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
