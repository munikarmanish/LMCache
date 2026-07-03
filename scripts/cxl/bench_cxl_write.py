#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark: DRAM -> CXL pool write bandwidth, by copy method.

The CXL cross-node pull (`PushKVToCXL`) is dominated by the donor's CPU
``ctypes.memmove`` into the CXL pool, measured at ~1.35 GB/s — vs ~22 GB/s
for CXL->GPU reads. This script isolates that one operation and compares
copy methods so we can tell whether 1.35 GB/s is a ``memmove`` inefficiency
(fixable in software) or the device's real CPU-write ceiling.

It mmaps the DAX device directly (like cxl/bootstrap.py), allocates a DRAM
source buffer, and times each method writing the source into the pool. No
LMCache stack needed.

Methods:
  - ctypes.memmove        : the current donor path (baseline)
  - torch copy_           : torch's optimized CPU memcpy
  - numpy copyto          : numpy's optimized memcpy
  - nontemporal (movnt)   : streaming stores that bypass the cache (a tiny
                            C helper compiled on the fly); often much faster
                            for write-only-to-device traffic
  - cuda memcpy (H2H)     : register the pool with cudaHostRegister and let
                            the GPU's DMA engine do the copy (DRAM->CXL via
                            cudaMemcpy); tests whether a DMA engine beats CPU
                            stores into CXL

Run on a node with the real CXL device (NOT in a sandbox that SIGBUSes on
DAX mmap). Use the lmcache venv python:

  ~/.virtualenvs/lmcache/bin/python scripts/cxl/bench_cxl_write.py \
      --dev /dev/dax0.0 --chunk-mib 32 --chunks 105 --iters 5

Reports GB/s per method (median of ``--iters``). Higher is better.
"""

# Standard
from __future__ import annotations
import argparse
import ctypes
import mmap
import os
import statistics
import subprocess
import sys
import tempfile
import time

# Third Party
import numpy as np

# Non-temporal-store helper: streaming stores (_mm_stream) bypass the cache,
# which for write-only traffic into device memory can be far faster than
# regular stores that pull the destination line into cache first. Compiled
# on the fly with cc, mirroring lmcache/v1/storage_backend/cxl/_native.
_NT_SRC = r"""
#include <stddef.h>
#include <stdint.h>
#if defined(__x86_64__)
#include <stdint.h>
#include <immintrin.h>
/* Non-temporal copy: scalar head to 32-byte-align dst (VMOVNTDQ FAULTS on
   an unaligned dst — not just slower), then 32-byte streaming stores, then
   a scalar tail, then sfence. src may be unaligned (loadu). */
void nt_copy(void* dst, const void* src, size_t n) {
    char* d = (char*)dst;
    const char* s = (const char*)src;
    size_t i = 0;
    size_t head = ((size_t)(-(uintptr_t)d)) & (size_t)31;
    if (head > n) head = n;
    for (; i < head; ++i) d[i] = s[i];
    size_t aligned_end = i + ((n - i) & ~(size_t)31);
    for (; i < aligned_end; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(s + i));
        _mm256_stream_si256((__m256i*)(d + i), v);
    }
    for (; i < n; ++i) d[i] = s[i];
    _mm_sfence();
}
int nt_available(void) { return 1; }
#else
void nt_copy(void* dst, const void* src, size_t n) {
    __builtin_memcpy(dst, src, n);
}
int nt_available(void) { return 0; }
#endif
"""


def _load_nt_lib():
    """Compile + load the non-temporal copy helper. Returns (fn, available)
    or (None, False) if cc is unavailable."""
    cc = os.environ.get("CC", "cc")
    tmpdir = tempfile.mkdtemp(prefix="cxl-nt-")
    src_path = os.path.join(tmpdir, "nt_copy.c")
    so_path = os.path.join(tmpdir, "nt_copy.so")
    with open(src_path, "w") as f:
        f.write(_NT_SRC)
    try:
        subprocess.run(
            [cc, "-O3", "-march=native", "-shared", "-fPIC", src_path, "-o", so_path],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(
            f"[bench] could not build nt_copy.so ({e}); skipping nontemporal",
            file=sys.stderr,
        )
        return None, False
    lib = ctypes.CDLL(so_path)
    lib.nt_copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.nt_copy.restype = None
    lib.nt_available.restype = ctypes.c_int
    return lib, bool(lib.nt_available())


def _open_pool(dev_path: str, want_bytes: int) -> tuple[int, mmap.mmap, int]:
    """mmap the DAX device (or a regular file for a DRAM control run).

    Returns (base_addr, mmap_obj, size). Mirrors cxl/bootstrap's mmap.
    """
    fd = os.open(dev_path, os.O_RDWR)
    try:
        st = os.fstat(fd)
        size = st.st_size
        if size == 0:
            # Character device (DAX) reports size 0 via fstat; fall back to
            # the requested size (the caller sized it to the workload).
            size = want_bytes
        mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)
    base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
    return base, mm, size


def _time_method(name: str, fn, iters: int, total_bytes: int) -> None:
    # Warm up once (first touch faults pages / populates TLB).
    fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    med = statistics.median(samples)
    gbps = (total_bytes / med) / 1e9
    print(f"  {name:<22} {med * 1000:8.2f} ms   {gbps:7.2f} GB/s")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dev", default="/dev/dax0.0", help="CXL DAX device path")
    p.add_argument("--chunk-mib", type=int, default=32, help="per-chunk size (MiB)")
    p.add_argument("--chunks", type=int, default=105, help="number of chunks")
    p.add_argument("--iters", type=int, default=5, help="timed iterations per method")
    p.add_argument(
        "--dram-control",
        action="store_true",
        help="also mmap a tmpfs/regular file as a DRAM->DRAM control baseline",
    )
    p.add_argument(
        "--skip-cuda",
        action="store_true",
        help="skip the cudaHostRegister + cudaMemcpy method",
    )
    p.add_argument(
        "--thread-sweep",
        action="store_true",
        help="sweep NT-write aggregate bandwidth across worker-thread counts "
        "(mirrors the donor's thread pool; reveals whether NT scales)",
    )
    p.add_argument(
        "--thread-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="worker counts for --thread-sweep (default: 1 2 4 8)",
    )
    args = p.parse_args()

    chunk_bytes = args.chunk_mib * 1024 * 1024
    total_bytes = chunk_bytes * args.chunks
    print(
        f"[bench] dev={args.dev} chunk={args.chunk_mib}MiB chunks={args.chunks} "
        f"total={total_bytes / (1 << 30):.2f} GiB iters={args.iters}"
    )

    # DRAM source: one chunk's worth, reused for every chunk write (the
    # donor reads distinct L1 objects, but for write-bandwidth the source
    # content is irrelevant; reusing one keeps the source cache-hot, which
    # is the favorable case — so any slowness is the DESTINATION side).
    src = np.ones(chunk_bytes, dtype=np.uint8)
    src_ptr = src.ctypes.data

    base, mm, size = _open_pool(args.dev, total_bytes)
    if size < total_bytes:
        print(
            f"[bench] WARNING: pool size {size / (1 << 30):.2f} GiB < workload "
            f"{total_bytes / (1 << 30):.2f} GiB; reducing chunks to fit",
            file=sys.stderr,
        )
        args.chunks = max(1, size // chunk_bytes)
        total_bytes = chunk_bytes * args.chunks

    offsets = [i * chunk_bytes for i in range(args.chunks)]

    print("\n== DRAM -> CXL write bandwidth (median of iters) ==")

    # 1. ctypes.memmove — the current donor path.
    def m_memmove():
        for off in offsets:
            ctypes.memmove(base + off, src_ptr, chunk_bytes)

    _time_method("ctypes.memmove", m_memmove, args.iters, total_bytes)

    # 2. torch copy_ into a tensor viewing the pool.
    try:
        # Third Party
        import torch

        pool_t = torch.frombuffer(mm, dtype=torch.uint8)
        src_t = torch.from_numpy(src)

        def m_torch():
            for off in offsets:
                pool_t[off : off + chunk_bytes].copy_(src_t)

        _time_method("torch copy_", m_torch, args.iters, total_bytes)
    except Exception as e:
        print(f"  torch copy_            skipped ({e})", file=sys.stderr)

    # 3. numpy copyto into an ndarray viewing the pool.
    pool_np = np.frombuffer(mm, dtype=np.uint8)

    def m_numpy():
        for off in offsets:
            np.copyto(pool_np[off : off + chunk_bytes], src)

    _time_method("numpy copyto", m_numpy, args.iters, total_bytes)

    # 4. non-temporal streaming stores.
    nt_lib, nt_ok = _load_nt_lib()
    if nt_lib is not None and nt_ok:

        def m_nt():
            for off in offsets:
                nt_lib.nt_copy(base + off, src_ptr, chunk_bytes)

        _time_method("nontemporal (movnt)", m_nt, args.iters, total_bytes)
    elif nt_lib is not None:
        print("  nontemporal (movnt)    skipped (non-x86 build)", file=sys.stderr)

    # 5. cudaHostRegister the pool + cudaMemcpy (DMA engine does the write).
    if not args.skip_cuda:
        try:
            # Third Party
            import torch

            # Register the pool as pinned host memory so cudaMemcpy can DMA
            # into it. Source is a pinned host tensor.
            cudart = ctypes.CDLL("libcudart.so")
            # cudaHostRegister(ptr, size, flags=cudaHostRegisterDefault=0)
            rc = cudart.cudaHostRegister(
                ctypes.c_void_p(base), ctypes.c_size_t(size), ctypes.c_uint(0)
            )
            if rc != 0:
                raise RuntimeError(f"cudaHostRegister failed rc={rc}")
            try:
                src_pinned = torch.ones(chunk_bytes, dtype=torch.uint8, pin_memory=True)
                src_pin_ptr = src_pinned.data_ptr()
                # cudaMemcpy kind 0 = cudaMemcpyHostToHost (DMA-capable when
                # both ends are pinned/registered).
                memcpy = cudart.cudaMemcpy
                memcpy.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_int,
                ]

                def m_cuda_h2h():
                    for off in offsets:
                        memcpy(
                            ctypes.c_void_p(base + off),
                            ctypes.c_void_p(src_pin_ptr),
                            ctypes.c_size_t(chunk_bytes),
                            0,  # cudaMemcpyHostToHost
                        )

                _time_method("cuda memcpy H2H", m_cuda_h2h, args.iters, total_bytes)
            finally:
                cudart.cudaHostUnregister(ctypes.c_void_p(base))
        except Exception as e:
            print(f"  cuda memcpy H2H        skipped ({e})", file=sys.stderr)

    # Thread-scaling sweep: the donor copies chunks with a thread pool
    # (LMCACHE_CXL_DONOR_WORKERS, default 8). If NT stores don't scale
    # across threads on this device, aggregate bandwidth COLLAPSES with
    # more workers — this sweep finds the sweet spot. Each thread NT-copies
    # a disjoint subset of the chunks concurrently, exactly like the donor.
    if nt_lib is not None and nt_ok and args.thread_sweep:
        # Third Party
        from concurrent.futures import ThreadPoolExecutor

        print("\n== NT write, thread scaling (aggregate GB/s) ==")
        for nthreads in args.thread_counts:

            def m_nt_threads(nt=nthreads):
                # Partition offsets across nt workers, round-robin.
                buckets = [offsets[k::nt] for k in range(nt)]

                def _worker(my_offsets):
                    for off in my_offsets:
                        nt_lib.nt_copy(base + off, src_ptr, chunk_bytes)

                with ThreadPoolExecutor(max_workers=nt) as ex:
                    list(ex.map(_worker, buckets))

            _time_method(
                f"NT x{nthreads} threads", m_nt_threads, args.iters, total_bytes
            )

    # Optional: DRAM->DRAM control (a tmpfs file) to show the non-CXL ceiling.
    if args.dram_control:
        with tempfile.NamedTemporaryFile(prefix="cxl-dram-ctl-", delete=False) as f:
            f.truncate(total_bytes)
            ctl_path = f.name
        try:
            cbase, cmm, _ = _open_pool(ctl_path, total_bytes)

            def m_ctl():
                for off in offsets:
                    ctypes.memmove(cbase + off, src_ptr, chunk_bytes)

            print("\n== DRAM -> DRAM control (tmpfs/regular file) ==")
            _time_method("ctypes.memmove (DRAM)", m_ctl, args.iters, total_bytes)
            cmm.close()
        finally:
            os.unlink(ctl_path)

    # The numpy/torch views over the mmap still export buffers, so an
    # explicit mmap.close() would raise "cannot close exported pointers".
    # Best-effort close; the OS reclaims the mapping on process exit.
    try:
        mm.close()
    except BufferError:
        pass
    print(
        "\n[bench] Interpretation: if nontemporal or cuda H2H is much faster "
        "than ctypes.memmove, the donor's CXL write is a software issue "
        "(fixable). If all methods cluster near ~1.4 GB/s, that's the "
        "device's CPU-write ceiling and the fix must avoid CPU writes "
        "(e.g. RDMA the pull, or write CXL from the GPU)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
