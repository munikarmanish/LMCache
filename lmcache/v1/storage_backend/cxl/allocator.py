# SPDX-License-Identifier: Apache-2.0
"""CXLMemoryAllocator: implements MemoryAllocatorInterface over a NodeHeap.

Returned `MemoryObj`s are `TensorMemoryObj`s whose `raw_data` is a
`torch.from_numpy` view into the `cudaHostRegister`'d CXL pool. Pointer
lifetime is tied to the CXL pool mmap and is stable for the allocator's
entire lifetime.
"""

# Standard
import ctypes
from typing import List, Optional, Union

# Third Party
import numpy as np
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.cxl.heap import NodeHeap, OutOfChunks

logger = init_logger(__name__)


class CXLMemoryAllocator(MemoryAllocatorInterface):
    """Serves MemoryObjs whose payloads live in the CXL pool.

    The heap hands out pool-relative offsets; the allocator wraps
    those offsets into `TensorMemoryObj`s that the backend and
    subsequent `GPUConnector` can use transparently.
    """

    def __init__(
        self,
        heap: NodeHeap,
        pool_base: int,
    ):
        self._heap = heap
        self._pool_base = pool_base
        # offset -> numpy buffer backing the torch tensor. We keep the
        # numpy array alive because torch.from_numpy does not take
        # ownership; if the ndarray is collected the tensor's data
        # pointer becomes dangling. We deliberately do NOT hold a
        # strong reference to the MemoryObj itself — ref counting
        # drives its lifecycle, and a strong reference here would
        # defeat __del__ -> free.
        self._live: dict[int, np.ndarray] = {}

    # -------- MemoryAllocatorInterface ----------------------------------

    def allocate(
        self,
        shapes: Union[torch.Size, List[torch.Size]],
        dtypes: Union[torch.dtype, List[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        shapes, dtypes = self._adapt_shapes_and_dtypes(shapes, dtypes)
        size_bytes = _compute_size_bytes(shapes, dtypes)
        if size_bytes > self._heap.chunk_size:
            raise ValueError(
                f"requested allocation of {size_bytes} bytes exceeds heap "
                f"chunk size {self._heap.chunk_size}"
            )
        try:
            offset = self._heap.alloc()
        except Exception:
            return None
        return self._wrap(offset, shapes, dtypes, fmt, size_bytes)

    def batched_allocate(
        self,
        shapes: Union[torch.Size, List[torch.Size]],
        dtypes: Union[torch.dtype, List[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        if batch_size <= 0:
            return []
        shapes, dtypes = self._adapt_shapes_and_dtypes(shapes, dtypes)
        size_bytes = _compute_size_bytes(shapes, dtypes)
        if size_bytes > self._heap.chunk_size:
            raise ValueError(
                f"requested allocation of {size_bytes} bytes exceeds heap "
                f"chunk size {self._heap.chunk_size}"
            )
        try:
            offsets = self._heap.alloc_batch(batch_size)
        except Exception:
            return None
        try:
            return [
                self._wrap(off, shapes, dtypes, fmt, size_bytes)
                for off in offsets
            ]
        except Exception:
            # Back out any offsets we took but didn't successfully wrap.
            for off in offsets:
                if off not in self._live:
                    self._heap.free(off)
            raise

    def free(
        self,
        memory_obj: MemoryObj,
        allocator_type: Optional[str] = None,
    ):
        offset = memory_obj.meta.address - self._pool_base
        if offset not in self._live:
            # Already freed or foreign — silently ignore to match
            # LocalCPU's tolerance for late ref_count_down calls.
            return
        self._live.pop(offset, None)
        memory_obj.invalidate()
        self._heap.free(offset)

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        offsets = []
        for obj in memory_objs:
            offset = obj.meta.address - self._pool_base
            if offset not in self._live:
                continue
            self._live.pop(offset, None)
            obj.invalidate()
            offsets.append(offset)
        if offsets:
            self._heap.free_batch(offsets)

    def memcheck(self) -> bool:
        stats = self._heap.stats()
        # Live count should equal allocated slots.
        live = len(self._live)
        allocated = stats.total_slots - stats.free_slots
        if live != allocated:
            logger.warning(
                "CXLMemoryAllocator live=%d but heap-allocated=%d", live, allocated
            )
            return False
        return True

    def close(self):
        # Leaving live objects is the caller's problem; we don't force
        # invalidation because a MemoryObj's __del__ may still fire.
        pass

    # -------- helpers ---------------------------------------------------

    def _wrap(
        self,
        offset: int,
        shapes: List[torch.Size],
        dtypes: List[torch.dtype],
        fmt: MemoryFormat,
        size_bytes: int,
    ) -> MemoryObj:
        chunk_size = self._heap.chunk_size
        # Build a stable numpy array over the CXL bytes. We use uint8
        # because the underlying storage is opaque; TensorMemoryObj
        # views it at its logical dtype via the `tensor` property.
        buf_type = ctypes.c_uint8 * chunk_size
        ctypes_buf = buf_type.from_address(self._pool_base + offset)
        np_buf = np.frombuffer(ctypes_buf, dtype=np.uint8, count=chunk_size)
        raw_data = torch.from_numpy(np_buf)
        # For TensorMemoryObj.get_size() to return size_bytes, the
        # metadata carries `shapes`/`dtypes`. Single-shape allocations
        # also set `shape`/`dtype` for backwards-compat paths.
        primary_shape = shapes[0]
        primary_dtype = dtypes[0]
        meta = MemoryObjMetadata(
            shape=primary_shape,
            dtype=primary_dtype,
            address=self._pool_base + offset,
            phy_size=chunk_size,
            ref_count=0,
            pin_count=0,
            fmt=fmt,
            shapes=shapes if len(shapes) > 1 else None,
            dtypes=dtypes if len(dtypes) > 1 else None,
        )
        obj = TensorMemoryObj(raw_data=raw_data, metadata=meta, parent_allocator=self)
        # Keep the numpy buffer alive as long as the offset is live.
        self._live[offset] = np_buf
        return obj


def _compute_size_bytes(
    shapes: List[torch.Size], dtypes: List[torch.dtype]
) -> int:
    total = 0
    for shape, dtype in zip(shapes, dtypes, strict=True):
        total += int(shape.numel()) * torch.tensor([], dtype=dtype).element_size()
    return total
