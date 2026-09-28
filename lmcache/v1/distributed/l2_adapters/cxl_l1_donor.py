# SPDX-License-Identifier: Apache-2.0
"""L1-manager-backed `local_copy_provider` for the CXL donor side.

When a peer issues a `PushKVToCXLMsg`, the donor needs to read the
chunk's bytes from this node's local DRAM tier and copy them into a
CXL chunk it then commits. In MP mode, "local DRAM tier" is the
`L1Manager`. This module provides the bridge:

    provider = L1LocalCopyProvider(l1_manager)
    cxl_donor = CXLDonor(..., local_copy_provider=provider)

The provider's contract (see `cross_node.LocalCopyProvider`) is:

    __call__(chunk_hash: bytes, model_name: str, kv_rank: int,
             cache_salt: str) -> Optional[MemoryObj]

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
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)


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
            logger.exception("L1Manager.finish_read failed for %s", self._obj_key)


class L1LocalCopyProvider:
    """Donor-side bridge: fetch a chunk from L1Manager by ObjectKey identity.

    Passed to `CXLDonor` as its `local_copy_provider`. On each call:
    1. Rebuild the ObjectKey from the identity the requester sent.
    2. reserve_read on L1Manager. If miss, return None.
    3. unsafe_read to get the underlying MemoryObj.
    4. Wrap so that the donor's ref_count_down releases the L1 read lock.

    The ObjectKey identity (chunk_hash bytes, model_name, kv_rank,
    cache_salt) arrives over the wire from the requester rather than
    being reconstructed locally: L1Manager is keyed on the full
    ObjectKey, and the pool's u64 index hash is one-way. Taking
    kv_rank from local `metadata` would also be wrong under TP>1,
    where the requester may ask for any rank's shard, not this
    donor's.
    """

    def __init__(self, l1_manager: L1Manager):
        self._l1 = l1_manager

    def __call__(
        self,
        chunk_hash: bytes,
        model_name: str,
        kv_rank: int,
        cache_salt: str,
    ) -> Optional[MemoryObj]:
        try:
            obj_key = ObjectKey(
                chunk_hash=chunk_hash,
                model_name=model_name,
                kv_rank=kv_rank,
                cache_salt=cache_salt,
            )
        except ValueError:
            # ObjectKey enforces its own field invariants; a peer that
            # sends something malformed is a miss, not a crash.
            logger.warning(
                "malformed ObjectKey identity from peer (model_name=%r)",
                model_name,
            )
            return None

        results = self._l1.reserve_read([obj_key])
        err, mem_obj = results.get(obj_key, (L1Error.KEY_NOT_EXIST, None))
        if err != L1Error.SUCCESS or mem_obj is None:
            return None

        # Wrap so finish_read fires when donor calls ref_count_down.
        return _L1FinishReadOnDrop(mem_obj, self._l1, obj_key)
