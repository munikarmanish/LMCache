"""Probe cudaHostRegister limits on a devdax (CXL) mapping.

Usage:
    pin_probe_dax.py SIZE_GIB [OFFSET_GIB]         single registration
    pin_probe_dax.py SIZE_GIB --chunks N           N back-to-back registrations
                                                   of SIZE_GIB each, none released
                                                   until all are attempted

The chunked mode distinguishes a per-call limit from a global budget: if the
driver's ceiling is per-call (a page_count-sized array allocated per
cudaHostRegister), every chunk succeeds and the total pinned far exceeds the
single-call ceiling. If it is a global budget, a later chunk fails even though
the same size succeeded on its own.
"""

import ctypes
import mmap
import os
import sys

DEV_PATH = "/dev/dax0.0"


def dax_size(path: str) -> int:
    name = os.path.basename(path)
    with open(f"/sys/bus/dax/devices/{name}/size") as f:
        return int(f.read().strip())


cudart = ctypes.CDLL("libcudart.so")

size = int(sys.argv[1]) << 30
offset = 0
chunks = 1
if len(sys.argv) > 2:
    if sys.argv[2] == "--chunks":
        chunks = int(sys.argv[3])
    else:
        offset = int(sys.argv[2]) << 30

fd = os.open(DEV_PATH, os.O_RDWR)
st = os.fstat(fd)
dev_size = st.st_size if st.st_size > 0 else dax_size(DEV_PATH)
total = offset + size * chunks
if total > dev_size:
    print(f"requested offset+size*chunks {total} > device size {dev_size}")
    os.close(fd)
    sys.exit(1)

# Hold every mapping and registration alive until all chunks are attempted, so
# that a failure reflects accumulated driver state rather than a fresh context.
bufs = []
pinned = []
failed = False
for i in range(chunks):
    chunk_offset = offset + size * i
    print(f"[{i}] mmaping {size >> 30} GiB at offset {chunk_offset >> 30} GiB")
    buf = mmap.mmap(
        fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=chunk_offset
    )
    bufs.append(buf)
    ptr = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(buf)))

    print(f"[{i}] pinning with cudaHostRegister")
    rc = cudart.cudaHostRegister(ptr, ctypes.c_size_t(size), 0)
    cumulative = (size * (i + 1)) >> 30
    print(f"[{i}] result: {rc}   (cumulative pinned: {cumulative} GiB)")
    if rc != 0:
        failed = True
        break
    pinned.append(ptr)

print(f"total pinned: {(size * len(pinned)) >> 30} GiB in {len(pinned)} registration(s)")

for ptr in pinned:
    cudart.cudaHostUnregister(ptr)
for buf in bufs:
    buf.close()
os.close(fd)
sys.exit(1 if failed else 0)
