# SPDX-License-Identifier: Apache-2.0
"""L1-manager-backed `local_copy_provider` for the CXL donor side.

When a peer issues a `PushKVToCXLMsg`, the donor needs to read the
chunk's bytes from this node's local DRAM tier and copy them into a
CXL chunk it then commits. In MP mode, "local DRAM tier" is the
`L1Manager`. This module provides the bridge:

    provider = L1LocalCopyProvider(l1_manager, metadata)
    cxl_donor = CXLDonor(..., local_copy_provider=provider)

The provider's contract (see `cross_node.LocalCopyProvider`) is:

    __call__(key_str: str) -> Optional[MemoryObj]

Returning a MemoryObj implies the caller will eventually call
`memory_obj.ref_count_down()`. We wrap the L1Manager's
reserve_read / unsafe_read / finish_read sequence behind a custom
MemoryObj subclass whose `ref_count_down` triggers `finish_read`,
matching the contract donor code expects.
"""

# Future
from __future__ import annotations

# Standard
from typing import Optional
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.memory_management import MemoryObj, MemoryObjMetadata
from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)


def _cek_chunk_hash_to_objkey_bytes(chunk_hash: int) -> bytes:
    """Recover the ObjectKey.chunk_hash bytes from a CacheEngineKey int.

    The L2 adapter's `_object_key_to_chunk_hash` reinterprets the first
    8 bytes of `ObjectKey.chunk_hash` as a little-endian unsigned u64
    and stores that int as `CacheEngineKey.chunk_hash`. To recover the
    original 8 bytes here, we reverse: write the int back as 8 bytes
    little-endian unsigned. The bytes themselves are byte-order-
    independent — only the integer interpretation differs.

    NOTE: this round-trip is byte-exact only when the original
    ObjectKey.chunk_hash is exactly 8 bytes (i.e. the hash algorithm
    yields 64 bits). The builtin Python hash satisfies this. blake3 /
    sha256 produce 32-byte digests that get truncated by the adapter,
    so the donor cannot reconstruct them; lookup will fail-soft (None)
    for those.
    """
    return chunk_hash.to_bytes(8, byteorder="little", signed=False)


class _L1FinishReadOnDrop:
    """Duck-typed proxy: forwards attribute access to an inner MemoryObj
    and overrides `ref_count_down` to call `L1Manager.finish_read` once.

    The CXL donor only uses a small slice of the MemoryObj surface
    (`raw_data`, `meta`, `get_size()`, `ref_count_down()`). Rather than
    subclass the abstract MemoryObj (which has many abstract methods),
    we wrap the inner object via __getattr__ and override only what we
    need to manage the L1 read lock lifecycle.
    """

    def __init__(
        self,
        inner: MemoryObj,
        l1: L1Manager,
        obj_key: ObjectKey,
    ):
        # Use object.__setattr__ to bypass our own __setattr__ logic
        # if any.
        self._inner = inner
        self._l1 = l1
        self._obj_key = obj_key
        self._released = False
        self._lock = threading.Lock()

    def __getattr__(self, name: str):
        # Falls through for everything we don't override above.
        # Note: __getattr__ is only called if normal lookup fails, so
        # the explicit attributes below take precedence.
        return getattr(self._inner, name)

    def ref_count_down(self) -> None:
        # Donor calls this exactly once per acquisition. On the first
        # call, finish_read on the L1 entry; further calls are no-op.
        with self._lock:
            if self._released:
                return
            self._released = True
        try:
            self._l1.finish_read([self._obj_key])
        except Exception:
            logger.exception(
                "L1Manager.finish_read failed for %s", self._obj_key
            )


class L1LocalCopyProvider:
    """Donor-side bridge: fetch a chunk from L1Manager by CacheEngineKey string.

    Passed to `CXLDonor` as its `local_copy_provider`. On each call:
    1. Parse the wire-form CacheEngineKey string.
    2. Recover the ObjectKey (chunk_hash bytes, model_name, kv_rank).
    3. reserve_read on L1Manager. If miss, return None.
    4. unsafe_read to get the underlying MemoryObj.
    5. Wrap so that the donor's ref_count_down releases the L1 read lock.

    The kv_rank is taken from `metadata` (the same LMCacheMetadata the
    L2 adapter uses to bridge ObjectKey→CacheEngineKey on the requester
    side). For TP=1 this is always 0.
    """

    def __init__(self, l1_manager: L1Manager, metadata: LMCacheMetadata):
        self._l1 = l1_manager
        self._metadata = metadata
        self._kv_rank = ObjectKey.ComputeKVRank(
            world_size=metadata.world_size,
            global_rank=metadata.worker_id,
            local_world_size=metadata.local_world_size,
            local_rank=metadata.local_worker_id,
        )

    def __call__(self, key_str: str) -> Optional[MemoryObj]:
        try:
            cek = CacheEngineKey.from_string(key_str)
        except Exception:
            logger.warning("could not parse CacheEngineKey: %s", key_str)
            return None

        if cek.model_name != self._metadata.model_name:
            return None

        try:
            ch_bytes = _cek_chunk_hash_to_objkey_bytes(cek.chunk_hash)
        except (OverflowError, ValueError):
            return None

        obj_key = ObjectKey(
            chunk_hash=ch_bytes,
            model_name=cek.model_name,
            kv_rank=self._kv_rank,
        )

        results = self._l1.reserve_read([obj_key])
        err, mem_obj = results.get(obj_key, (L1Error.KEY_NOT_EXIST, None))
        if err != L1Error.SUCCESS or mem_obj is None:
            return None

        # Wrap so finish_read fires when donor calls ref_count_down.
        return _L1FinishReadOnDrop(mem_obj, self._l1, obj_key)
