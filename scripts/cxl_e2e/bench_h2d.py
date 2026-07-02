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

Run on a CUDA box (lmcache venv). For the CXL source, run where the DAX
device is mappable (not in a sandbox that SIGBUSes on DAX mmap):
  ~/.virtualenvs/lmcache/bin/python scripts/cxl_e2e/bench_h2d.py
  ~/.virtualenvs/lmcache/bin/python scripts/cxl_e2e/bench_h2d.py \
      --cxl-dev /dev/dax0.0
"""

# Standard
from __future__ import annotations
import argparse
import ctypes
import mmap
import os
import statistics
import sys

# Third Party
import torch


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


def _open_cxl_source(
    dev_path: str, want_bytes: int, flags: int
) -> tuple[int, mmap.mmap, int]:
    """mmap a CXL DAX device and cudaHostRegister it with ``flags``. Returns
    (ptr, mmap, registered_size). Verifies the pinning actually took (loud
    error if cudaHostRegister fails or the range isn't device-resolvable —
    which would silently fall back to a DRAM bounce buffer)."""
    fd = os.open(dev_path, os.O_RDWR)
    try:
        size = os.fstat(fd).st_size
        if size == 0:  # DAX char device reports 0 via fstat
            size = want_bytes
        mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)
    base = ctypes.addressof(ctypes.c_char.from_buffer(mm))

    lib = _cudart()
    rc = lib.cudaHostRegister(
        ctypes.c_void_p(base), ctypes.c_size_t(size), ctypes.c_uint(flags)
    )
    print(
        f"[bench] cudaHostRegister(base=0x{base:x}, size={size}, flags={flags}) "
        f"-> rc={rc} ({'OK' if rc == 0 else 'FAILED'})",
        flush=True,
    )
    if rc != 0:
        raise RuntimeError(
            f"cudaHostRegister failed rc={rc}; the CXL source would NOT be "
            f"pinned (copies would bounce through DRAM). Try --cxl-register-flags "
            f"4 (cudaHostRegisterIoMemory) for device memory."
        )
    pinned = _verify_pinned(lib, base, size)
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
        default=_CUDA_HOST_REGISTER_DEFAULT,
        help="cudaHostRegister flags for the CXL mapping: 0=Default, "
        "1=Portable, 2=Mapped, 4=IoMemory. Use 4 if the default registers "
        "but isn't device-resolvable (DMA bounces).",
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
    if args.cxl_dev is not None:
        cxl_ptr, cxl_mm, cxl_size = _open_cxl_source(
            args.cxl_dev, region_bytes, args.cxl_register_flags
        )
        if args.verify_only:
            _cuda_host_unregister(cxl_ptr)
            return 0
        if cxl_size < max_bytes:
            print(
                f"[bench] WARNING: CXL device {cxl_size / (1 << 20):.0f} MiB < "
                f"max payload {max_bytes / (1 << 20):.0f} MiB; capping sizes.",
                file=sys.stderr,
            )
            sizes = [n for n in sizes if n <= cxl_size]
            max_bytes = sizes[-1]

    # The rotation span is the region clamped to what every source can hold.
    rotate_span = region_bytes
    if cxl_ptr is not None:
        rotate_span = min(rotate_span, cxl_size)

    # Populate page-table entries over the span so rotating accesses do not
    # fold a first-touch fault into the median (DAX mmaps fault lazily). The
    # sequential read leaves only the LLC-sized tail warm, so accesses stay
    # cold for both directions.
    if rotate:
        _prefault(dram_ptr, rotate_span)
        if cxl_ptr is not None:
            _prefault(cxl_ptr, rotate_span)

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
            try:
                _cuda_host_unregister(cxl_ptr)
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
