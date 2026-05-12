import ctypes
import mmap
import os
import stat
import sys

MADV_HUGEPAGE = 14
MADV_POPULATE_WRITE = 23

libc = ctypes.CDLL("libc.so.6", use_errno=True)
cudart = ctypes.CDLL("libcudart.so")

size = int(sys.argv[1]) << 30
print(f"mmaping anonymous buffer of size {size >> 30} GiB");
buf = mmap.mmap(-1, size, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, mmap.PROT_READ | mmap.PROT_WRITE)
ptr = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(buf)))

print("requesting THP with MADV_HUGEPAGE")
if libc.madvise(ptr, ctypes.c_size_t(size), MADV_HUGEPAGE) != 0:
    print(f"  madvise(MADV_HUGEPAGE) failed: errno={ctypes.get_errno()}")

print("prefaulting with MADV_POPULATE_WRITE")
if libc.madvise(ptr, ctypes.c_size_t(size), MADV_POPULATE_WRITE) != 0:
    print(f"  madvise(MADV_POPULATE_WRITE) failed: errno={ctypes.get_errno()}")

print("pinning buffer with cudaHostRegister");
rc = cudart.cudaHostRegister(ptr, ctypes.c_size_t(size), 0)
print(f"result: {rc}")
if rc == 0:
    cudart.cudaHostUnregister(ptr)
buf.close()
