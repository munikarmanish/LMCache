# SPDX-License-Identifier: Apache-2.0
"""Cross-node `PushKVToCXL` fallback handler.

Plan reference: F1, F5. Used when a CXL_LOOKUP miss is followed by a
controller `batched_p2p_lookup` that points to a peer with the chunks
in its **local** L0/L1 tier (not in CXL).

Two roles:

- **Donor handler.** Server-side. Receives a PushKVToCXLMsg, looks up
  each key in the donor's local tier, copies the bytes into the
  pre-reserved CXL slots, commits them. Returns a count of the
  contiguous prefix that was committed.

- **Requester driver.** Client-side. Calls
  `CXLIndexWriter.reserve_slot(key, ...)` for each key in the
  contiguous prefix, builds the message, sends it to the donor,
  releases slots beyond `num_committed`. Returns the count of keys
  the requester can now CXL_LOOKUP successfully.

Transport-agnostic: the requester takes a `donor` object that exposes
`handle_push(msg) -> PushKVToCXLRetMsg`. In tests this is a direct
function call between two in-process backends. In production it would
be a ZMQ-wrapped RPC against the donor's worker process.
"""

# Standard
import ctypes
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.cxl.bootstrap import PoolHandle
from lmcache.v1.storage_backend.cxl.heap import NodeHeap
from lmcache.v1.storage_backend.cxl.index_writer import (
    CXLIndexWriter,
    ReserveOutcome,
)
from lmcache.v1.storage_backend.cxl.p2p_messages import (
    PushKVToCXLMsg,
    PushKVToCXLRetMsg,
    PushStatus,
)

logger = init_logger(__name__)


class LocalCopyProvider(Protocol):
    """Donor-side callback to fetch the local copy of a key.

    The donor's CXLBackend is a peer of the L0/L1 tiers, not their
    owner. We inject this callable so the cross-node module doesn't
    have a hard dependency on LocalCPUBackend or any specific
    local-tier implementation.

    Returns None if the donor no longer has a local copy (it was
    evicted between the directory's KVAdmitMsg and the push request —
    the directory is best-effort).
    """

    def __call__(self, key_str: str) -> Optional[MemoryObj]: ...


@dataclass
class _ReservedKey:
    """Bookkeeping for one reserved slot on the requester side."""

    key: CacheEngineKey
    slot_idx: int
    outcome: ReserveOutcome


class CXLDonor:
    """Donor-side handler. Owns a backend's index_writer + heap.

    Instantiated alongside CXLBackend on the donor node. Tests
    construct one directly per backend.
    """

    def __init__(
        self,
        handle: PoolHandle,
        index_writer: CXLIndexWriter,
        heap: NodeHeap,
        node_id: int,
        local_copy_provider: LocalCopyProvider,
    ):
        self._handle = handle
        self._iw = index_writer
        self._heap = heap
        self._node_id = node_id
        self._local = local_copy_provider

        # Thread pool used to parallelize per-chunk alloc+memcpy and
        # commit_slot in handle_push. Each chunk is independent (distinct
        # heap offsets, distinct slot_idx → distinct lock_id), so threads
        # don't contend on the same arbiter slot.
        workers = int(os.environ.get("LMCACHE_CXL_DONOR_WORKERS", "8"))
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, workers),
            thread_name_prefix="cxl-donor",
        )

    def close(self) -> None:
        """Shut down the worker pool. Idempotent."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            # Mark closed so a second close is a no-op.
            self._executor = None  # type: ignore[assignment]

    def handle_push(self, msg: PushKVToCXLMsg) -> PushKVToCXLRetMsg:
        """Donor-side processing of a PushKVToCXLMsg."""
        import time as _time
        t_start = _time.perf_counter_ns()
        t_local_ns = 0
        t_state_ns = 0
        t_alloc_ns = 0
        t_memcpy_ns = 0
        t_commit_ns = 0
        bytes_copied = 0

        # Epoch check — single fail for the whole batch (plan).
        if msg.epoch != int(self._handle.header.gen):
            return PushKVToCXLRetMsg(
                num_committed=0, status=PushStatus.EPOCH_STALE
            )

        if len(msg.keys) != len(msg.slot_idxs):
            raise ValueError(
                f"PushKVToCXLMsg shape mismatch: {len(msg.keys)} keys vs "
                f"{len(msg.slot_idxs)} slot_idxs"
            )

        if not msg.keys:
            return PushKVToCXLRetMsg(num_committed=0, status=PushStatus.OK)

        # Pre-pin local copies in order; the contiguous prefix of
        # successful pins is what we commit. First miss caps the
        # effective prefix.
        t0 = _time.perf_counter_ns()
        locals_: List[Optional[MemoryObj]] = []
        for key_str in msg.keys:
            obj = self._local(key_str)
            locals_.append(obj)
            if obj is None:
                break
        t_local_ns = _time.perf_counter_ns() - t0

        effective = sum(1 for o in locals_ if o is not None)
        if effective == 0:
            return PushKVToCXLRetMsg(num_committed=0, status=PushStatus.ALL_NACK)

        # Validate that each slot in the effective prefix is still
        # ALLOCATING(self) at the requester's epoch.
        t0 = _time.perf_counter_ns()
        for i in range(effective):
            slot = self._handle.slots()[msg.slot_idxs[i]]
            from lmcache.v1.storage_backend.cxl.layout import SLOT_STATE_ALLOCATING

            if (
                slot.line0.state != SLOT_STATE_ALLOCATING
                or slot.line0.owner_node_id != self._node_id
                or slot.line0.generation != msg.epoch
            ):
                effective = i
                break
        t_state_ns = _time.perf_counter_ns() - t0

        if effective == 0:
            for o in locals_:
                if o is not None:
                    o.ref_count_down()
            return PushKVToCXLRetMsg(num_committed=0, status=PushStatus.ALL_NACK)

        # Allocate chunks + copy bytes in parallel. Each worker is
        # independent: heap.alloc returns a unique offset under its own
        # lock; _copy_into_chunk writes to a unique chunk; locals_ entries
        # are read-only after pinning. Aggregate phase timings via a lock.
        chunk_offsets: List[Optional[int]] = [None] * effective
        sizes: List[int] = [0] * effective
        memcpy_ns_per: List[int] = [0] * effective
        alloc_ns_per: List[int] = [0] * effective

        def _alloc_and_copy(i: int) -> None:
            obj = locals_[i]
            assert obj is not None
            size_bytes = obj.get_size()
            sizes[i] = size_bytes
            t_a0 = _time.perf_counter_ns()
            chunk_off = self._heap.alloc()
            t_a1 = _time.perf_counter_ns()
            chunk_offsets[i] = chunk_off
            _copy_into_chunk(self._handle.base, chunk_off, obj, size_bytes)
            t_a2 = _time.perf_counter_ns()
            alloc_ns_per[i] = t_a1 - t_a0
            memcpy_ns_per[i] = t_a2 - t_a1

        try:
            futures = [
                self._executor.submit(_alloc_and_copy, i)
                for i in range(effective)
            ]
            for fut in futures:
                fut.result()  # surface exceptions
        except Exception:
            # Roll back any chunks we already grabbed.
            for off in chunk_offsets:
                if off is None:
                    continue
                try:
                    self._heap.free(off)
                except Exception:
                    pass
            for o in locals_:
                if o is not None:
                    o.ref_count_down()
            raise

        t_alloc_ns = sum(alloc_ns_per)
        t_memcpy_ns = sum(memcpy_ns_per)
        bytes_copied = sum(sizes)

        # Commit slots in parallel. Each commit takes the distributed
        # lock for its own slot_idx (distinct lock_id), so parallel
        # commits trigger parallel arbiter grants. Semantic change vs.
        # the previous sequential path: we no longer "stop at first
        # failure" since commits can complete out of order. Instead we
        # compute the longest contiguous prefix of successful commits,
        # which is what callers use as num_committed.
        commit_ok: List[bool] = [False] * effective
        t0 = _time.perf_counter_ns()

        def _commit_one(i: int) -> None:
            obj = locals_[i]
            assert obj is not None
            try:
                self._iw.commit_slot(
                    slot_idx=msg.slot_idxs[i],
                    chunk_offset=chunk_offsets[i],  # type: ignore[arg-type]
                    chunk_len=obj.get_size(),
                    fmt=obj.meta.fmt,
                )
                commit_ok[i] = True
            except Exception:
                logger.exception("commit_slot failed at i=%d", i)
                commit_ok[i] = False

        futures = [self._executor.submit(_commit_one, i) for i in range(effective)]
        for fut in futures:
            fut.result()  # surface any unexpected exceptions
        t_commit_ns = _time.perf_counter_ns() - t0

        # Longest contiguous prefix of successful commits.
        num_committed = 0
        for ok in commit_ok:
            if ok:
                num_committed += 1
            else:
                break

        # Free chunks beyond the committed prefix. Two cases:
        #   - commit succeeded but lies past a gap (slot is VALID):
        #     evict to TOMB first, then free the chunk. We can't expose
        #     these — the caller expects a contiguous prefix and would
        #     otherwise see VALID slots it didn't ask for.
        #   - commit failed (slot is still ALLOCATING): the requester
        #     side releases these via release_slot_for_donor based on
        #     the returned num_committed, so we only free the chunk.
        for j in range(num_committed, effective):
            off = chunk_offsets[j]
            if off is None:
                continue
            if commit_ok[j]:
                # Slot is VALID; flip to TOMB before freeing the chunk
                # to avoid exposing a stale offset.
                try:
                    self._iw.evict(msg.slot_idxs[j])
                except Exception:
                    logger.exception(
                        "evict failed for past-prefix slot %d",
                        msg.slot_idxs[j],
                    )
            try:
                self._heap.free(off)
            except Exception:
                pass

        # Drop our refs on the local copies.
        for o in locals_:
            if o is not None:
                o.ref_count_down()

        if num_committed == 0:
            status = PushStatus.ALL_NACK
        elif num_committed == len(msg.keys):
            status = PushStatus.OK
        else:
            status = PushStatus.PARTIAL

        total_ms = (_time.perf_counter_ns() - t_start) / 1_000_000.0
        memcpy_gbps = (
            (bytes_copied / (t_memcpy_ns / 1e9)) / 1e9
            if t_memcpy_ns > 0
            else 0.0
        )
        logger.info(
            "CXL donor handle_push: n_keys=%d effective=%d committed=%d "
            "bytes=%d total_ms=%.3f local_ms=%.3f state_ms=%.3f "
            "alloc_ms=%.3f memcpy_ms=%.3f memcpy_gbps=%.3f commit_ms=%.3f",
            len(msg.keys),
            effective,
            num_committed,
            bytes_copied,
            total_ms,
            t_local_ns / 1_000_000.0,
            t_state_ns / 1_000_000.0,
            t_alloc_ns / 1_000_000.0,
            t_memcpy_ns / 1_000_000.0,
            memcpy_gbps,
            t_commit_ns / 1_000_000.0,
        )

        return PushKVToCXLRetMsg(num_committed=num_committed, status=status)


class DonorEndpoint(Protocol):
    """Transport-agnostic donor handle the requester calls.

    In tests this is a direct CXLDonor. In production it's a ZMQ
    wrapper that serializes the message, sends it to the donor's
    worker, and deserializes the response.
    """

    def handle_push(self, msg: PushKVToCXLMsg) -> PushKVToCXLRetMsg: ...


@dataclass
class RemoteFetchResult:
    """Outcome of a cross-node REMOTE_FETCH attempt."""

    num_satisfied: int
    """Length of the contiguous prefix the caller can now CXL_LOOKUP."""

    status: PushStatus


def remote_fetch(
    requester_node_id: int,
    keys: List[CacheEngineKey],
    index_writer: CXLIndexWriter,
    donor_node_id: int,
    donor: DonorEndpoint,
    sender_id: str,
    epoch: int,
) -> RemoteFetchResult:
    """Reserve slots for `keys` on the donor's behalf and trigger a push.

    Returns the count of keys the caller can now CXL_LOOKUP. Releases
    any slots beyond `num_committed`.
    """
    if not keys:
        return RemoteFetchResult(num_satisfied=0, status=PushStatus.OK)

    # Reserve slots, owning each on behalf of the donor.
    reserved: List[_ReservedKey] = []
    for key in keys:
        result = index_writer.reserve_slot_for_donor(key, donor_node_id)
        if result.outcome == ReserveOutcome.INDEX_FULL:
            break
        if result.slot_idx is None:
            # WAIT_FOR_OTHER with no slot_idx means probe collision —
            # treat as terminal.
            break
        reserved.append(
            _ReservedKey(
                key=key, slot_idx=result.slot_idx, outcome=result.outcome
            )
        )

    if not reserved:
        return RemoteFetchResult(num_satisfied=0, status=PushStatus.ALL_NACK)

    # Subset of (key, slot_idx) pairs that the donor needs to DMA.
    # ALREADY_PRESENT slots are already VALID — no DMA needed.
    # WAIT_FOR_OTHER means another writer is mid-flight on this key.
    # Both are "satisfied" from the requester's perspective once the
    # state stabilizes.
    push_indices = [
        i for i, r in enumerate(reserved) if r.outcome == ReserveOutcome.RESERVED
    ]
    push_keys = [reserved[i].key for i in push_indices]
    push_slots = [reserved[i].slot_idx for i in push_indices]

    if not push_keys:
        # Everything was already present or in-flight — nothing for
        # the donor to do. Caller's next CXL_LOOKUP will see them.
        return RemoteFetchResult(
            num_satisfied=_count_settled(reserved, index_writer),
            status=PushStatus.OK,
        )

    msg = PushKVToCXLMsg(
        sender_id=sender_id,
        keys=[k.to_string() for k in push_keys],
        slot_idxs=push_slots,
        epoch=epoch,
    )

    try:
        ack = donor.handle_push(msg)
    except Exception:
        # Donor unreachable / errored: release everything we reserved.
        for r in reserved:
            if r.outcome == ReserveOutcome.RESERVED:
                try:
                    index_writer.release_slot_for_donor(r.slot_idx, donor_node_id)
                except Exception:
                    pass
        raise

    # Release slots beyond ack.num_committed (within the push subset).
    if ack.status == PushStatus.EPOCH_STALE:
        # Donor saw stale gen — release everything we reserved.
        for r in reserved:
            if r.outcome == ReserveOutcome.RESERVED:
                try:
                    index_writer.release_slot_for_donor(r.slot_idx, donor_node_id)
                except Exception:
                    pass
        return RemoteFetchResult(num_satisfied=0, status=PushStatus.EPOCH_STALE)

    committed = ack.num_committed
    for j, r_idx in enumerate(push_indices):
        if j >= committed:
            try:
                index_writer.release_slot_for_donor(
                    reserved[r_idx].slot_idx, donor_node_id
                )
            except Exception:
                pass

    return RemoteFetchResult(
        num_satisfied=_count_settled(reserved, index_writer, committed_pushes=committed),
        status=ack.status,
    )


def _count_settled(
    reserved: List[_ReservedKey],
    index_writer: CXLIndexWriter,
    committed_pushes: Optional[int] = None,
) -> int:
    """Length of the contiguous prefix that is now usable by the requester.

    The contract for `num_satisfied`: the smallest i such that
    reserved[0..i] are all "VALID and matching" from the requester's
    POV. ALREADY_PRESENT counts as satisfied trivially. RESERVED
    entries count if they fall within the donor's `committed_pushes`
    window. WAIT_FOR_OTHER currently does not count — caller can poll
    and re-check on a follow-up lookup.
    """
    push_seen = 0
    n = 0
    for r in reserved:
        if r.outcome == ReserveOutcome.ALREADY_PRESENT:
            n += 1
            continue
        if r.outcome == ReserveOutcome.RESERVED:
            if committed_pushes is None or push_seen >= committed_pushes:
                break
            push_seen += 1
            n += 1
            continue
        # WAIT_FOR_OTHER or anything else: stop.
        break
    return n


def _copy_into_chunk(
    pool_base: int, chunk_offset: int, src: MemoryObj, size_bytes: int
) -> None:
    """Memcpy `size_bytes` from `src.raw_data` into the CXL chunk."""
    # Third Party
    import torch

    src_tensor = src.raw_data
    if src_tensor.dtype != torch.uint8:
        src_tensor = src_tensor.view(torch.uint8)
    src_tensor = src_tensor.reshape(-1)[:size_bytes].contiguous()
    ctypes.memmove(pool_base + chunk_offset, src_tensor.data_ptr(), size_bytes)
