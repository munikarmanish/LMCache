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
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Tuple

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.cxl.bootstrap import PoolHandle
from lmcache.v1.storage_backend.cxl.fast_copy import fast_copy_to_cxl
from lmcache.v1.storage_backend.cxl.heap import NodeHeap
from lmcache.v1.storage_backend.cxl.index_writer import (
    CommitPin,
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

        # Thread pool used to parallelize the per-chunk DRAM→CXL memcpy in
        # handle_push (alloc and commit are now batched under a single lock
        # hold, so they no longer use the pool). Default 4: the CXL pool's
        # write bandwidth saturates at ~23 GB/s with ~4 concurrent NT-store
        # copies, so more workers only split the same bandwidth thinner while
        # adding GIL/scheduler contention. Override via the env var.
        workers = int(os.environ.get("LMCACHE_CXL_DONOR_WORKERS", "4"))
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

    def _alloc_chunks(self, n: int) -> List[Optional[int]]:
        """Allocate up to ``n`` chunks, preferring one batched lock hold.

        Fast path: ``heap.alloc_batch(n)`` grabs all ``n`` chunks under a
        single lock acquisition. If the heap is out of regions mid-batch
        (``alloc_batch`` raises and may leak the chunks it already took), fall
        back to per-chunk ``alloc()`` and stop at the first failure, returning
        the contiguous prefix actually obtained. Callers cap the batch to the
        returned length.

        Args:
            n: Desired chunk count.

        Returns:
            A list of pool-relative offsets, length ``<= n`` (shorter only
            when the heap ran out of regions).
        """
        if n <= 0:
            return []
        try:
            return list(self._heap.alloc_batch(n))
        except Exception:
            logger.warning(
                "CXL donor: alloc_batch(%d) failed (heap full?); "
                "falling back to per-chunk alloc for the available prefix",
                n,
            )
        offsets: List[Optional[int]] = []
        for _ in range(n):
            try:
                offsets.append(self._heap.alloc())
            except Exception:
                break
        return offsets

    def handle_push(self, msg: PushKVToCXLMsg) -> PushKVToCXLRetMsg:
        """Donor-side processing of a PushKVToCXLMsg."""
        import time as _time

        t_start = _time.perf_counter_ns()
        t_local_ns = 0
        t_state_ns = 0
        t_alloc_ns = 0
        t_memcpy_ns = 0
        t_memcpy_wall_ns = 0
        t_commit_ns = 0
        bytes_copied = 0

        # Epoch check — single fail for the whole batch (plan).
        if msg.epoch != int(self._handle.header.gen):
            return PushKVToCXLRetMsg(num_committed=0, status=PushStatus.EPOCH_STALE)

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

        # Allocate ALL chunks up front in a single heap lock acquisition.
        # The previous per-chunk ``heap.alloc()`` inside each parallel worker
        # serialized the 8 GIL-bound workers on the heap's global lock and
        # crossed the rack-wide region lock once per region boundary mid-batch
        # — tens of ms of pure overhead at long prompts. ``alloc_batch`` does
        # the whole batch under one lock hold, confining region claims to one
        # place. The DRAM→CXL copies still run in parallel (the device, not
        # the allocator, is the bandwidth limiter).
        sizes: List[int] = [obj.get_size() for obj in locals_[:effective]]  # type: ignore[union-attr]
        t_a0 = _time.perf_counter_ns()
        chunk_offsets: List[Optional[int]] = self._alloc_chunks(effective)
        t_alloc_ns = _time.perf_counter_ns() - t_a0
        # The heap could not satisfy the whole batch (out of regions). Cap the
        # batch to what we got so the rest of the pipeline (copy/commit) only
        # touches allocated chunks; the requester treats the short prefix as a
        # PARTIAL and may retry the tail with another peer. The capped-off
        # locals are still ref-dropped by the single ``locals_`` loop at the
        # end (do NOT drop them here, or they double-decrement).
        if len(chunk_offsets) < effective:
            effective = len(chunk_offsets)
            sizes = sizes[:effective]
        if effective == 0:
            for o in locals_:
                if o is not None:
                    o.ref_count_down()
            return PushKVToCXLRetMsg(num_committed=0, status=PushStatus.ALL_NACK)

        # Copy bytes in parallel. Each worker writes to its own pre-allocated
        # chunk; ``locals_`` entries are read-only after pinning, so there is
        # no shared mutable state between workers. ``memcpy_ns_per`` records
        # per-thread copy time (summed for the per-thread bandwidth);
        # ``t_memcpy_wall_ns`` is the wall-clock of the whole parallel phase
        # (for the achieved aggregate bandwidth).
        memcpy_ns_per: List[int] = [0] * effective

        def _copy_one(i: int) -> None:
            obj = locals_[i]
            assert obj is not None
            t_c0 = _time.perf_counter_ns()
            _copy_into_chunk(
                self._handle.base,
                chunk_offsets[i],  # type: ignore[arg-type]
                obj,
                sizes[i],
            )
            memcpy_ns_per[i] = _time.perf_counter_ns() - t_c0

        t_m0 = _time.perf_counter_ns()
        try:
            futures = [self._executor.submit(_copy_one, i) for i in range(effective)]
            for fut in futures:
                fut.result()  # surface exceptions
        except Exception:
            # Roll back the chunks we allocated for this batch.
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
        t_memcpy_wall_ns = _time.perf_counter_ns() - t_m0

        t_memcpy_ns = sum(memcpy_ns_per)
        bytes_copied = sum(sizes)

        # Commit all slots in one batched lock acquisition. Each slot's lock
        # is distinct (slot_idx → lock_id), so a single ``acquire_batch``
        # grants the whole batch in ~one arbiter sweep instead of one sweep
        # per slot (the previous per-chunk parallel commits each paid a sweep
        # round-trip). Born-pinned: the requester's resident retrieve releases
        # the pin after the H2D drains; folding the pin into commit removes a
        # separate per-chunk pin round-trip on the requester side. A slot that
        # fails validation is reported False (not raised); we take the longest
        # contiguous prefix of successes as num_committed, and the past-prefix
        # cleanup below unpins+evicts any committed slot beyond it.
        t0 = _time.perf_counter_ns()
        objs = [locals_[i] for i in range(effective)]
        commit_ok: List[bool] = self._iw.commit_slot_batch(
            slot_idxs=[msg.slot_idxs[i] for i in range(effective)],
            chunk_offsets=[chunk_offsets[i] for i in range(effective)],  # type: ignore[misc]
            chunk_lens=[o.get_size() for o in objs],  # type: ignore[union-attr]
            fmts=[o.meta.fmt for o in objs],  # type: ignore[union-attr]
            pin=CommitPin.BORN_PINNED,
        )
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
                # Slot is VALID and born-pinned; drop the commit-born pin
                # first (evict refuses pinned slots), then flip to TOMB
                # before freeing the chunk to avoid exposing a stale offset.
                # These past-prefix slots are never returned to the requester,
                # so no one else holds or will release this pin.
                try:
                    self._iw.unpin(msg.slot_idxs[j])
                except Exception:
                    logger.exception(
                        "unpin failed for past-prefix slot %d",
                        msg.slot_idxs[j],
                    )
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
        # Per-thread bandwidth: bytes ÷ summed per-worker copy time. This is
        # the rate a SINGLE NT-store copy achieves (it does not reflect the
        # parallelism). The achieved aggregate is the wall-clock rate below.
        memcpy_gbps = (
            (bytes_copied / (t_memcpy_ns / 1e9)) / 1e9 if t_memcpy_ns > 0 else 0.0
        )
        # Aggregate bandwidth: bytes ÷ wall-clock of the parallel copy phase —
        # the real device write throughput across all workers. When this
        # plateaus as workers are added, the CXL pool's write bandwidth (not
        # the worker count) is the limiter.
        agg_gbps = (
            (bytes_copied / (t_memcpy_wall_ns / 1e9)) / 1e9
            if t_memcpy_wall_ns > 0
            else 0.0
        )
        logger.info(
            "CXL donor handle_push: n_keys=%d effective=%d committed=%d "
            "bytes=%d total_ms=%.3f local_ms=%.3f state_ms=%.3f "
            "alloc_ms=%.3f memcpy_ms=%.3f memcpy_wall_ms=%.3f "
            "memcpy_gbps=%.3f agg_gbps=%.3f commit_ms=%.3f",
            len(msg.keys),
            effective,
            num_committed,
            bytes_copied,
            total_ms,
            t_local_ns / 1_000_000.0,
            t_state_ns / 1_000_000.0,
            t_alloc_ns / 1_000_000.0,
            t_memcpy_ns / 1_000_000.0,
            t_memcpy_wall_ns / 1_000_000.0,
            memcpy_gbps,
            agg_gbps,
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

    born_pinned: List[bool] = field(default_factory=list)
    """Per-index flag over the satisfied prefix (length == ``num_satisfied``):
    True where the donor freshly committed the slot born-pinned (the
    requester must NOT pin again — the pin is already held on its behalf),
    False where the slot was ALREADY_PRESENT (the requester still needs to
    pin it). Lets the lookup path skip the redundant per-chunk pin
    round-trip for the freshly-fetched majority."""


def remote_fetch(
    requester_node_id: int,
    keys: List[CacheEngineKey],
    index_writer: CXLIndexWriter,
    donor_node_id: int,
    donor: DonorEndpoint,
    sender_id: str,
    epoch: int,
    timings: Optional[dict] = None,
) -> RemoteFetchResult:
    """Reserve slots for `keys` on the donor's behalf and trigger a push.

    Returns the count of keys the caller can now CXL_LOOKUP. Releases
    any slots beyond `num_committed`.

    Args:
        timings: Optional out-dict; when provided, the seconds spent in the
            per-key slot ``reserve`` phase and the donor ``rpc`` round-trip
            are accumulated under those keys (for profiling). Left untouched
            when None.
    """
    if not keys:
        return RemoteFetchResult(num_satisfied=0, status=PushStatus.OK)

    import time as _time

    t_reserve0 = _time.perf_counter()
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
            _ReservedKey(key=key, slot_idx=result.slot_idx, outcome=result.outcome)
        )

    if timings is not None:
        timings["reserve"] = timings.get("reserve", 0.0) + (
            _time.perf_counter() - t_reserve0
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
        settled, born_pinned = _count_settled(reserved, index_writer)
        return RemoteFetchResult(
            num_satisfied=settled,
            status=PushStatus.OK,
            born_pinned=born_pinned,
        )

    msg = PushKVToCXLMsg(
        sender_id=sender_id,
        keys=[k.to_string() for k in push_keys],
        slot_idxs=push_slots,
        epoch=epoch,
    )

    t_rpc0 = _time.perf_counter()
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

    if timings is not None:
        timings["rpc"] = timings.get("rpc", 0.0) + (_time.perf_counter() - t_rpc0)

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

    settled, born_pinned = _count_settled(
        reserved, index_writer, committed_pushes=committed
    )
    return RemoteFetchResult(
        num_satisfied=settled,
        status=ack.status,
        born_pinned=born_pinned,
    )


def _count_settled(
    reserved: List[_ReservedKey],
    index_writer: CXLIndexWriter,
    committed_pushes: Optional[int] = None,
) -> Tuple[int, List[bool]]:
    """Contiguous prefix usable by the requester, plus per-index pin state.

    The contract for `num_satisfied`: the smallest i such that
    reserved[0..i] are all "VALID and matching" from the requester's
    POV. ALREADY_PRESENT counts as satisfied trivially. RESERVED
    entries count if they fall within the donor's `committed_pushes`
    window. WAIT_FOR_OTHER currently does not count — caller can poll
    and re-check on a follow-up lookup.

    Returns:
        ``(num_satisfied, born_pinned)`` where ``born_pinned[i]`` is True
        when ``reserved[i]`` was freshly committed by the donor (and is
        therefore already pinned on the requester's behalf), False when it
        was ALREADY_PRESENT (still needs a pin). ``len(born_pinned) ==
        num_satisfied``.
    """
    push_seen = 0
    n = 0
    born_pinned: List[bool] = []
    for r in reserved:
        if r.outcome == ReserveOutcome.ALREADY_PRESENT:
            n += 1
            born_pinned.append(False)
            continue
        if r.outcome == ReserveOutcome.RESERVED:
            if committed_pushes is None or push_seen >= committed_pushes:
                break
            push_seen += 1
            n += 1
            born_pinned.append(True)
            continue
        # WAIT_FOR_OTHER or anything else: stop.
        break
    return n, born_pinned


def _copy_into_chunk(
    pool_base: int, chunk_offset: int, src: MemoryObj, size_bytes: int
) -> None:
    """Copy `size_bytes` from `src.raw_data` (DRAM) into the CXL chunk.

    Uses non-temporal streaming stores (~5x faster than ``memmove`` for
    write-only traffic into the CXL pool) when the native helper is
    available, falling back to ``memmove`` otherwise. See
    ``cxl/fast_copy.py``.
    """
    # Third Party
    import torch

    src_tensor = src.raw_data
    if src_tensor.dtype != torch.uint8:
        src_tensor = src_tensor.view(torch.uint8)
    src_tensor = src_tensor.reshape(-1)[:size_bytes].contiguous()
    fast_copy_to_cxl(pool_base + chunk_offset, src_tensor.data_ptr(), size_bytes)
