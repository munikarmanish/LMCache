# SPDX-License-Identifier: Apache-2.0
"""Donor side of the RDMA/NIXL peer L2 adapter.

When a peer issues a ``RemoteLookupReq``, this node is the *donor*: it
must (a) confirm it has each requested chunk in its local L1 tier,
(b) read-lock the matching L1 objects so they survive the peer's
one-sided RDMA READ window, and (c) return each chunk's page index (its
position in the NIXL descriptor list over the registered L1 buffer) so
the peer can address the READ. A later ``RemoteUnlockReq`` releases the
read-locks.

Read-lock lifecycle and leasing
--------------------------------
Each granted pin is recorded under ``(lease_id, ObjectKey)`` with a
monotonic-deadline lease. The matching ``RemoteUnlockReq`` releases the
pins for its lease. If the requester dies between lookup and unlock, a
background sweep (``sweep_expired``, driven by the transport server)
releases pins whose lease elapsed — the same best-effort, self-healing
stance the CXL GC takes toward orphaned state. This guarantees a peer
chunk is never stranded read-locked forever.

The page index returned for a chunk is ``MemoryObj.meta.address``: in the
paged L1 allocator one ``MemoryObj`` is exactly one page and
``meta.address`` *is* its descriptor index, which is precisely the
``remote_index`` NIXL's ``make_prepped_xfer`` expects. This matches how
``pd_backend`` returns ``mem_obj.meta.address`` as ``remote_indexes``.
"""

# Future
from __future__ import annotations

# Standard
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.nixl_peer_messages import (
    RemoteLookupReq,
    RemoteLookupResp,
    RemoteUnlockReq,
    RemoteUnlockResp,
    WireKey,
)

logger = init_logger(__name__)


def wire_key_to_object_key(wire_key: WireKey) -> ObjectKey:
    """Reconstruct an ``ObjectKey`` from its wire form.

    Args:
        wire_key: The serialized key received over the control plane.

    Returns:
        The reconstructed ``ObjectKey``. The chunk hash is decoded from
        hex byte-exactly, so it round-trips any hash width.

    Raises:
        ValueError: If ``chunk_hash_hex`` is not valid hex.
    """
    return ObjectKey(
        chunk_hash=bytes.fromhex(wire_key.chunk_hash_hex),
        model_name=wire_key.model_name,
        kv_rank=wire_key.kv_rank,
        cache_salt=wire_key.cache_salt,
    )


def object_key_to_wire_key(key: ObjectKey) -> WireKey:
    """Serialize an ``ObjectKey`` to its wire form.

    Args:
        key: The object key to serialize.

    Returns:
        A ``WireKey`` carrying a byte-exact hex of the chunk hash plus
        the model name, kv_rank, and cache salt.
    """
    return WireKey(
        chunk_hash_hex=key.chunk_hash.hex(),
        model_name=key.model_name,
        kv_rank=key.kv_rank,
        cache_salt=key.cache_salt,
    )


class NixlPeerDonor:
    """Serves remote lookups/unlocks against this node's L1 tier.

    Thread-safety: ``handle_lookup`` / ``handle_unlock`` / ``sweep_expired``
    may be invoked from the transport server thread. All access to the
    pin table is guarded by ``self._lock``. The underlying ``L1Manager``
    is independently thread-safe.
    """

    def __init__(
        self,
        l1_manager: L1Manager,
        peer_agent_id: str,
        lease_seconds: float,
        page_size: int,
    ):
        """Initialize the donor.

        Args:
            l1_manager: This node's L1 manager, queried for local chunks.
            peer_agent_id: This node's NIXL peer id, echoed to requesters
                so they select the correct remote transfer handler.
            lease_seconds: How long a granted read-lock survives without
                an explicit unlock before ``sweep_expired`` reclaims it.
                Must be positive.
            page_size: The L1 buffer's NIXL page size (``align_bytes`` =
                one chunk). A ``MemoryObj.meta.address`` is a byte offset
                into the registered L1 buffer; the NIXL one-sided READ is
                addressed by *descriptor index* (one descriptor per page),
                so the page index reported to the requester is
                ``address // page_size``. Must be positive.

        Raises:
            ValueError: If ``lease_seconds`` or ``page_size`` is not
                positive.
        """
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self._l1 = l1_manager
        self._peer_agent_id = peer_agent_id
        self._lease_seconds = lease_seconds
        self._page_size = page_size
        self._lock = threading.Lock()
        # (lease_id, ObjectKey) -> monotonic deadline. Presence means we
        # hold exactly one read-lock for that key under that lease.
        self._pins: dict[tuple[str, ObjectKey], float] = {}

    def handle_lookup(self, req: RemoteLookupReq) -> RemoteLookupResp:
        """Read-lock the requested chunks that exist locally and report
        their page indices.

        For each requested key the donor calls ``L1Manager.reserve_read``
        (acquiring one read-lock) and, on success, ``unsafe_read`` to
        recover the ``MemoryObj`` whose ``meta.address`` is the page
        index. A read-lock is held (recorded in the pin table under the
        request's lease) until the matching unlock or lease expiry.

        Args:
            req: The lookup request.

        Returns:
            A ``RemoteLookupResp`` with positional ``found`` /
            ``page_indices`` / ``sizes`` and this donor's peer id.
        """
        n = len(req.keys)
        found = [False] * n
        page_indices = [-1] * n
        sizes = [0] * n

        deadline = time.monotonic() + self._lease_seconds
        for i, wire_key in enumerate(req.keys):
            try:
                obj_key = wire_key_to_object_key(wire_key)
            except ValueError:
                logger.warning("RemoteLookup: bad wire key at index %d", i)
                continue

            reserve = self._l1.reserve_read([obj_key])
            err, _ = reserve.get(obj_key, (L1Error.KEY_NOT_EXIST, None))
            if err != L1Error.SUCCESS:
                continue

            # reserve_read returns the MemoryObj, but go through
            # unsafe_read for the address so we never act on a racing
            # release: we hold the read-lock now, so this is safe.
            read = self._l1.unsafe_read([obj_key])
            r_err, mem_obj = read.get(obj_key, (L1Error.KEY_NOT_EXIST, None))
            if r_err != L1Error.SUCCESS or mem_obj is None:
                # Could not recover the object after locking; drop the
                # lock we just took so we don't strand it.
                self._l1.finish_read([obj_key])
                continue

            addr = mem_obj.meta.address
            if addr % self._page_size != 0:
                # A chunk must sit on a page boundary for the one-sided
                # READ to address it by descriptor index. If it doesn't
                # (unexpected for the chunk-aligned L1 allocator), skip it
                # rather than hand back a wrong index.
                logger.warning(
                    "RemoteLookup: L1 address %d for %s is not page-aligned "
                    "(page_size=%d); skipping",
                    addr,
                    obj_key,
                    self._page_size,
                )
                self._l1.finish_read([obj_key])
                continue

            found[i] = True
            page_indices[i] = addr // self._page_size
            sizes[i] = mem_obj.get_size()
            with self._lock:
                self._pins[(req.lease_id, obj_key)] = deadline

        return RemoteLookupResp(
            found=found,
            page_indices=page_indices,
            sizes=sizes,
            peer_agent_id=self._peer_agent_id,
        )

    def handle_unlock(self, req: RemoteUnlockReq) -> RemoteUnlockResp:
        """Release the read-locks held for these keys under the lease.

        Args:
            req: The unlock request.

        Returns:
            A ``RemoteUnlockResp`` reporting how many pins were actually
            released (a pin already reclaimed by lease expiry counts as
            0, so the count is idempotency-safe).
        """
        released = 0
        for wire_key in req.keys:
            try:
                obj_key = wire_key_to_object_key(wire_key)
            except ValueError:
                continue
            with self._lock:
                present = self._pins.pop((req.lease_id, obj_key), None)
            if present is None:
                continue
            try:
                self._l1.finish_read([obj_key])
                released += 1
            except Exception:
                logger.exception("RemoteUnlock: finish_read failed for %s", obj_key)
        return RemoteUnlockResp(num_released=released)

    def sweep_expired(self) -> int:
        """Release pins whose lease deadline has passed.

        Driven periodically by the transport server so a requester that
        died between lookup and unlock cannot strand a peer chunk
        read-locked forever.

        Returns:
            The number of expired pins released this sweep.
        """
        now = time.monotonic()
        with self._lock:
            expired = [
                (lease_id, obj_key)
                for (lease_id, obj_key), deadline in self._pins.items()
                if deadline <= now
            ]
            for entry in expired:
                del self._pins[entry]
        for lease_id, obj_key in expired:
            try:
                self._l1.finish_read([obj_key])
            except Exception:
                logger.exception("lease sweep: finish_read failed for %s", obj_key)
        if expired:
            logger.warning(
                "NIXL peer donor swept %d expired read-lock(s)", len(expired)
            )
        return len(expired)

    def close(self) -> None:
        """Release every read-lock this donor still holds.

        Called on shutdown so no L1 chunk is left pinned by a peer that
        never unlocked.
        """
        with self._lock:
            remaining = list(self._pins.keys())
            self._pins.clear()
        for _lease_id, obj_key in remaining:
            try:
                self._l1.finish_read([obj_key])
            except Exception:
                logger.exception("donor close: finish_read failed for %s", obj_key)
        if remaining:
            logger.info(
                "NIXL peer donor released %d held read-lock(s) on close",
                len(remaining),
            )
