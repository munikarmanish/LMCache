import ctypes
import mmap
import os
import stat
import sys

DEV_PATH = "/dev/dax0.0"


def dax_size(path: str) -> int:
    name = os.path.basename(path)
    with open(f"/sys/bus/dax/devices/{name}/size") as f:
        return int(f.read().strip())


cudart = ctypes.CDLL("libcudart.so")

size = int(sys.argv[1]) << 30
fd = os.open(DEV_PATH, os.O_RDWR)
st = os.fstat(fd)
dev_size = st.st_size if st.st_size > 0 else dax_size(DEV_PATH)
if size > dev_size:
    print(f"requested size {size} > device size {dev_size}")
    os.close(fd)
    sys.exit(1)

print(f"mmaping CXL buffer of size {size >> 30} GiB");
buf = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
ptr = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(buf)))

print("pinning buffer with cudaHostRegister");
rc = cudart.cudaHostRegister(ptr, ctypes.c_size_t(size), 0)
print(f"result: {rc}")
if rc == 0:
    cudart.cudaHostUnregister(ptr)
buf.close()
os.close(fd)
