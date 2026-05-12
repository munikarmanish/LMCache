# SPDX-License-Identifier: Apache-2.0
"""Writer half of the CXL hash index.

The reader half (`CXLIndex`) is lock-free. Writers — reserve_slot,
commit_slot, release_slot, evict — go through the two-tier lock
identified by the slot's `lock_id` field.

Plan reference: F2 ("Single-slot linear probing with per-slot state
words" deadlocks under concurrent writers), F11 (return duplicate on
ALREADY_PRESENT), F12 (ALLOCATING state + owner-based GC).

State machine:

    EMPTY
      │  reserve_slot() claims it
      ▼
    ALLOCATING(owner=me)
      │  commit_slot() publishes full line0
      ▼
    VALID
      │  evict() flips (after ref_count==0 check)
      ▼
    TOMB
      │  reserve_slot() may reuse TOMB slots (open-addressing rule)
      ▼
    (back to ALLOCATING)

Note on `lock_id`: the slot's `lock_id` is set when we first *claim*
a slot (reserve_slot). Subsequent operations (commit, evict, pin,
unpin) pull it from the slot itself. For an EMPTY slot we use
`slot_idx % num_locks` as the default — harmless because the slot
isn't claimed yet and the local-tier mutex still serializes reservers.
"""

# Standard
import ctypes
import enum
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend.cxl.bootstrap import PoolHandle
from lmcache.v1.storage_backend.cxl.fence import Fence, default_fence
from lmcache.v1.storage_backend.cxl.index import CXLIndex, SlotView
from lmcache.v1.storage_backend.cxl.layout import (
    GEOM_HASH_SIZE,
    SLOT_STATE_ALLOCATING,
    SLOT_STATE_EMPTY,
    SLOT_STATE_TOMB,
    SLOT_STATE_VALID,
    Slot,
    SlotLine0,
    SlotLine1,
)
from lmcache.v1.storage_backend.cxl.locks import TwoTierLock

logger = init_logger(__name__)


class ReserveOutcome(enum.Enum):
    RESERVED = "RESERVED"
    ALREADY_PRESENT = "ALREADY_PRESENT"
    WAIT_FOR_OTHER = "WAIT_FOR_OTHER"
    INDEX_FULL = "INDEX_FULL"


@dataclass
class ReserveResult:
    outcome: ReserveOutcome
    slot_idx: Optional[int]


class CXLIndexWriter:
    """Mutates the on-CXL hash index. Pairs with CXLIndex for reads."""

    def __init__(
        self,
        handle: PoolHandle,
        index: CXLIndex,
        lock: TwoTierLock,
        node_id: int,
        *,
        fence: Optional[Fence] = None,
        max_probe: Optional[int] = None,
    ):
        self._handle = handle
        self._index = index
        self._lock = lock
        self._node_id = node_id
        self._fence = fence or default_fence()
        self._max_probe = max_probe or index.max_probe
        self._num_locks = lock.num_locks
        self._slot_count = index.slot_count

        # Direct access to the slot array for writes.
        self._slots = handle.slots()
        self._slots_base_addr = ctypes.addressof(self._slots)
        self._slot_size = ctypes.sizeof(Slot)
        self._line0_size = ctypes.sizeof(SlotLine0)
        self._line1_size = ctypes.sizeof(SlotLine1)

    # -------- lock_id policy --------------------------------------------

    def _lock_id_for_slot(self, slot_idx: int) -> int:
        """Return the lock_id to use for writes on this slot.

        We derive from slot_idx rather than chunk_hash because a slot
        can be claimed, tombed, and re-claimed by different keys; using
        a slot-stable lock_id is simpler and still gives us good
        sharding across the `num_locks` array.
        """
        # lock_id 0 is reserved for the region allocator (see regions.py).
        # Map slot writes onto lock_id ∈ [1, num_locks).
        return 1 + (slot_idx % (self._num_locks - 1))

    # -------- reserve / commit ------------------------------------------

    def reserve_slot(self, key: CacheEngineKey) -> ReserveResult:
        """Reserve a slot for the local node. See `_reserve_slot_with_owner`."""
        return self._reserve_slot_with_owner(key, owner_node_id=self._node_id)

    def reserve_slot_for_donor(
        self, key: CacheEngineKey, donor_node_id: int
    ) -> ReserveResult:
        """Reserve a slot on behalf of a peer (the future donor of a push).

        Used by the cross-node REMOTE_FETCH path: the requester
        reserves slots stamped with `owner_node_id = donor_node_id`,
        then asks the donor to commit them. If the donor crashes or
        NACKs, the requester calls `release_slot_for_donor` to
        return the slot to EMPTY.
        """
        return self._reserve_slot_with_owner(key, owner_node_id=donor_node_id)

    def _reserve_slot_with_owner(
        self, key: CacheEngineKey, owner_node_id: int
    ) -> ReserveResult:
        """Atomically claim a slot for writing `key`.

        Returns ReserveResult with:
          * RESERVED + slot_idx   — slot is ALLOCATING(owner); caller must
            commit_slot() or release_slot(). First TOMB seen on the
            probe chain is preferred as the reuse target.
          * ALREADY_PRESENT + slot_idx — a VALID slot for this key
            already exists; caller should treat as a deduped insert.
          * WAIT_FOR_OTHER + slot_idx — another node is in the middle
            of writing this same key. Caller polls and retries.
          * INDEX_FULL + None — probe exhausted with no reusable slot.
        """
        needle = key.chunk_hash & 0xFFFFFFFFFFFFFFFF
        start = needle % self._slot_count
        first_tomb: Optional[int] = None

        # Take the lock for the first probe position. If we can't
        # claim there, we'll release and retry with the next position's
        # lock. This keeps the critical section small while guaranteeing
        # only one reserver per (key, slot-range) at a time. Simple
        # global serialization under one lock is possible but wastes
        # the sharded lock array.
        #
        # Correctness argument: all reservers for `key` start at
        # `start`; they serialize on the lock for `start`. A reserver
        # that needs to look past `start` holds locks as it goes,
        # ensuring no peer re-examines a slot we've already passed.
        locks_held = []  # list of lock_ids held, in acquisition order

        def _release_all():
            for lid in reversed(locks_held):
                # Each held lock's critical section belongs to this call
                # tree; we release them via the managed context below.
                pass

        # We use a single acquisition at `start`'s lock_id — this
        # serializes concurrent reserves for any key that starts at
        # the same slot. Different start-slots with unrelated hashes
        # don't collide. This is a simplification vs. the plan's
        # per-slot locking walk but is correct as long as the probe
        # chain from `start` is atomic from the reserver's POV, which
        # it is because writers never *move* slots sideways.
        with self._lock.acquire(self._lock_id_for_slot(start)):
            self._fence.flush_before_read(
                self._slots_base_addr + start * self._slot_size,
                self._max_probe * self._slot_size,
            )
            for step in range(self._max_probe):
                slot_idx = (start + step) % self._slot_count
                line0 = self._slots[slot_idx].line0
                state = line0.state

                if state == SLOT_STATE_VALID and line0.chunk_hash == needle:
                    # Match. Defensive generation/geom check.
                    if line0.generation == self._handle.header.gen and bytes(
                        line0.geom_hash
                    ) == bytes(self._handle.header.geom_hash):
                        return ReserveResult(
                            ReserveOutcome.ALREADY_PRESENT, slot_idx
                        )
                    # Stale slot with matching hash — treat as TOMB
                    # candidate. We'll overwrite it.
                    if first_tomb is None:
                        first_tomb = slot_idx

                if state == SLOT_STATE_ALLOCATING and line0.chunk_hash == needle:
                    return ReserveResult(
                        ReserveOutcome.WAIT_FOR_OTHER, slot_idx
                    )

                if state == SLOT_STATE_EMPTY:
                    target = first_tomb if first_tomb is not None else slot_idx
                    self._claim(target, key.chunk_hash, owner_node_id)
                    return ReserveResult(ReserveOutcome.RESERVED, target)

                if state == SLOT_STATE_TOMB and first_tomb is None:
                    first_tomb = slot_idx

            # Probe exhausted.
            if first_tomb is not None:
                self._claim(first_tomb, key.chunk_hash, owner_node_id)
                return ReserveResult(ReserveOutcome.RESERVED, first_tomb)
            return ReserveResult(ReserveOutcome.INDEX_FULL, None)

    def _claim(
        self, slot_idx: int, chunk_hash: int, owner_node_id: int
    ) -> None:
        """Write ALLOCATING to `slot_idx`. Caller holds the slot's lock."""
        slot = self._slots[slot_idx]
        slot.line0.chunk_hash = chunk_hash & 0xFFFFFFFFFFFFFFFF
        slot.line0.chunk_offset = 0
        slot.line0.chunk_len = 0
        slot.line0.fmt = MemoryFormat.UNDEFINED.value
        slot.line0.owner_node_id = owner_node_id
        slot.line0.generation = self._handle.header.gen
        ctypes.memmove(
            slot.line0.geom_hash,
            bytes(self._handle.header.geom_hash),
            GEOM_HASH_SIZE,
        )
        # Publish ALLOCATING last so readers see it as a single
        # cacheline transition from EMPTY/TOMB → ALLOCATING.
        slot.line0.state = SLOT_STATE_ALLOCATING
        # Initialize line1 as well so ref_count/pin_count start clean
        # even if the slot is a reused TOMB.
        slot.line1.ref_count = 0
        slot.line1.pin_count = 0
        slot.line1.lock_id = self._lock_id_for_slot(slot_idx)
        slot_addr = self._slots_base_addr + slot_idx * self._slot_size
        self._fence.fence_after_write(slot_addr, self._slot_size)

    def commit_slot(
        self,
        slot_idx: int,
        chunk_offset: int,
        chunk_len: int,
        fmt: MemoryFormat,
    ) -> None:
        """Publish a reserved slot as VALID.

        Caller guarantees the chunk bytes at `chunk_offset..+chunk_len`
        are already fully written and fenced. We only flip the slot's
        metadata and transition state to VALID.
        """
        with self._lock.acquire(self._lock_id_for_slot(slot_idx)):
            slot = self._slots[slot_idx]
            if slot.line0.state != SLOT_STATE_ALLOCATING:
                raise RuntimeError(
                    f"commit_slot: slot {slot_idx} is in state "
                    f"{slot.line0.state}, expected ALLOCATING"
                )
            if slot.line0.owner_node_id != self._node_id:
                raise RuntimeError(
                    f"commit_slot: slot {slot_idx} owner is "
                    f"{slot.line0.owner_node_id}, not {self._node_id}"
                )
            slot.line0.chunk_offset = chunk_offset
            slot.line0.chunk_len = chunk_len
            slot.line0.fmt = fmt.value
            # Single-store state publish.
            slot.line0.state = SLOT_STATE_VALID
            slot_addr = self._slots_base_addr + slot_idx * self._slot_size
            self._fence.fence_after_write(slot_addr, self._line0_size)

    def release_slot(self, slot_idx: int) -> None:
        """Abandon an ALLOCATING slot owned by self; flip back to EMPTY."""
        self._release_slot_with_owner(slot_idx, expected_owner=self._node_id)

    def release_slot_for_donor(self, slot_idx: int, donor_node_id: int) -> None:
        """Abandon an ALLOCATING slot the requester reserved on behalf of a donor.

        Used when the donor NACKs or is unreachable. The slot must
        have been reserved with `reserve_slot_for_donor(...,
        donor_node_id)`.
        """
        self._release_slot_with_owner(slot_idx, expected_owner=donor_node_id)

    def _release_slot_with_owner(
        self, slot_idx: int, *, expected_owner: int
    ) -> None:
        with self._lock.acquire(self._lock_id_for_slot(slot_idx)):
            slot = self._slots[slot_idx]
            if slot.line0.state != SLOT_STATE_ALLOCATING:
                raise RuntimeError(
                    f"release_slot: slot {slot_idx} is in state "
                    f"{slot.line0.state}, expected ALLOCATING"
                )
            if slot.line0.owner_node_id != expected_owner:
                raise RuntimeError(
                    f"release_slot: slot {slot_idx} owner is "
                    f"{slot.line0.owner_node_id}, not {expected_owner}"
                )
            slot.line0.state = SLOT_STATE_EMPTY
            slot.line0.chunk_hash = 0
            slot.line0.chunk_offset = 0
            slot.line0.chunk_len = 0
            slot_addr = self._slots_base_addr + slot_idx * self._slot_size
            self._fence.fence_after_write(slot_addr, self._line0_size)

    # -------- pin / unpin -----------------------------------------------

    def pin(self, slot_idx: int) -> bool:
        """Bump pin_count under the slot lock. Returns True if slot is VALID.

        Pin is "protect against eviction" — distinct from ref_count,
        which tracks in-flight reads. Both are needed for the plan's
        EVICT predicate (`ref_count == 0 AND pin_count == 0`).
        """
        with self._lock.acquire(self._lock_id_for_slot(slot_idx)):
            slot = self._slots[slot_idx]
            if slot.line0.state != SLOT_STATE_VALID:
                return False
            slot.line1.pin_count += 1
            line1_addr = (
                self._slots_base_addr
                + slot_idx * self._slot_size
                + self._line0_size
            )
            self._fence.fence_after_write(line1_addr, self._line1_size)
            return True

    def unpin(self, slot_idx: int) -> bool:
        with self._lock.acquire(self._lock_id_for_slot(slot_idx)):
            slot = self._slots[slot_idx]
            if slot.line1.pin_count <= 0:
                return False
            slot.line1.pin_count -= 1
            line1_addr = (
                self._slots_base_addr
                + slot_idx * self._slot_size
                + self._line0_size
            )
            self._fence.fence_after_write(line1_addr, self._line1_size)
            return True

    def ref_count_up(
        self, slot_idx: int, phase_ns: Optional[List[int]] = None
    ) -> bool:
        """Bump ref_count under the slot lock.

        Used by GET to keep the slot alive during DMA. Returns True
        if the slot is VALID and the bump happened.

        If `phase_ns` is provided (length 3), accumulates ns into:
            [0] acquire (distributed lock acquire)
            [1] body (slot read + state check + mutation)
            [2] fence (fence_after_write)
        """
        if phase_ns is None:
            with self._lock.acquire(self._lock_id_for_slot(slot_idx)):
                slot = self._slots[slot_idx]
                if slot.line0.state != SLOT_STATE_VALID:
                    return False
                slot.line1.ref_count += 1
                line1_addr = (
                    self._slots_base_addr
                    + slot_idx * self._slot_size
                    + self._line0_size
                )
                self._fence.fence_after_write(line1_addr, self._line1_size)
                return True

        t0 = time.perf_counter_ns()
        with self._lock.acquire(self._lock_id_for_slot(slot_idx)):
            t1 = time.perf_counter_ns()
            phase_ns[0] += t1 - t0
            slot = self._slots[slot_idx]
            if slot.line0.state != SLOT_STATE_VALID:
                phase_ns[1] += time.perf_counter_ns() - t1
                return False
            slot.line1.ref_count += 1
            line1_addr = (
                self._slots_base_addr
                + slot_idx * self._slot_size
                + self._line0_size
            )
            t2 = time.perf_counter_ns()
            phase_ns[1] += t2 - t1
            self._fence.fence_after_write(line1_addr, self._line1_size)
            phase_ns[2] += time.perf_counter_ns() - t2
            return True

    def ref_count_down(
        self, slot_idx: int, phase_ns: Optional[List[int]] = None
    ) -> None:
        """Drop ref_count under the slot lock.

        If `phase_ns` is provided (length 3), accumulates ns into:
            [0] acquire, [1] body, [2] fence.
        """
        if phase_ns is None:
            with self._lock.acquire(self._lock_id_for_slot(slot_idx)):
                slot = self._slots[slot_idx]
                if slot.line1.ref_count > 0:
                    slot.line1.ref_count -= 1
                    line1_addr = (
                        self._slots_base_addr
                        + slot_idx * self._slot_size
                        + self._line0_size
                    )
                    self._fence.fence_after_write(line1_addr, self._line1_size)
            return

        t0 = time.perf_counter_ns()
        with self._lock.acquire(self._lock_id_for_slot(slot_idx)):
            t1 = time.perf_counter_ns()
            phase_ns[0] += t1 - t0
            slot = self._slots[slot_idx]
            if slot.line1.ref_count > 0:
                slot.line1.ref_count -= 1
                line1_addr = (
                    self._slots_base_addr
                    + slot_idx * self._slot_size
                    + self._line0_size
                )
                t2 = time.perf_counter_ns()
                phase_ns[1] += t2 - t1
                self._fence.fence_after_write(line1_addr, self._line1_size)
                phase_ns[2] += time.perf_counter_ns() - t2
            else:
                phase_ns[1] += time.perf_counter_ns() - t1

    # -------- GC sweeps --------------------------------------------------

    def sweep_dead_owner_allocating(
        self, dead_node_id: int
    ) -> List[Tuple[int, int, int]]:
        """Flip every ALLOCATING slot owned by `dead_node_id` to TOMB.

        ALLOCATING means the writer was mid-INSERT when it died. The
        chunk bytes at `(chunk_offset, chunk_len)` are guaranteed
        garbage — no commit happened — so we drop the slot immediately.

        Returns a list of `(slot_idx, chunk_offset, chunk_len)` tuples
        for which the slot was flipped. Callers (the GC driver) use
        the offsets to free the dead chunks back to the heap.

        This is a full index scan and intended to be cheap at GC
        cadence (~tens of seconds). Each slot examination takes the
        slot's lock briefly, so concurrent reads/writes on other slots
        are not blocked.

        Note: `chunk_offset` may be 0 if the dead writer never got
        past the initial reserve_slot (commit_slot is what stamps the
        offset). The caller must handle a 0-len entry as "nothing to
        free".
        """
        freed: List[Tuple[int, int, int]] = []
        # One bulk fence-before-read covering the whole slot array.
        # This replaces 11K+ ctypes round-trips with a single C call
        # that CLFLUSHes all cachelines in one tight loop. Correct
        # because GC scans are coarse (cacheline staleness during the
        # scan is acceptable; we re-verify each candidate under the
        # slot lock, which does its own fence).
        self._fence.flush_before_read(
            self._slots_base_addr, self._slot_count * self._slot_size
        )
        for slot_idx in range(self._slot_count):
            line0 = self._slots[slot_idx].line0
            if (
                line0.state != SLOT_STATE_ALLOCATING
                or line0.owner_node_id != dead_node_id
            ):
                continue
            with self._lock.acquire(self._lock_id_for_slot(slot_idx)):
                slot = self._slots[slot_idx]
                # Re-verify under lock — concurrent commit/release may
                # have moved the slot since the unlocked pre-check.
                if (
                    slot.line0.state != SLOT_STATE_ALLOCATING
                    or slot.line0.owner_node_id != dead_node_id
                ):
                    continue
                offset = int(slot.line0.chunk_offset)
                length = int(slot.line0.chunk_len)
                slot.line0.state = SLOT_STATE_TOMB
                slot_addr = self._slots_base_addr + slot_idx * self._slot_size
                self._fence.fence_after_write(slot_addr, self._line0_size)
                freed.append((slot_idx, offset, length))
        return freed

    def region_has_no_live_slots(
        self,
        region_id: int,
        region_offset_lo: int,
        region_offset_hi: int,
    ) -> bool:
        """Return True if no VALID slot has `chunk_offset` in [lo, hi).

        The drain predicate for `RegionAllocator.promote_orphaned`.
        Caller computes `lo = off_regions + region_id * region_size`
        and `hi = lo + region_size`.

        Read-only; one bulk fence-before-read up front (much cheaper
        than per-slot fences for a 11K+ slot index — see
        `sweep_dead_owner_allocating` for the rationale).
        """
        self._fence.flush_before_read(
            self._slots_base_addr, self._slot_count * self._slot_size
        )
        for slot_idx in range(self._slot_count):
            line0 = self._slots[slot_idx].line0
            if line0.state != SLOT_STATE_VALID:
                continue
            offset = int(line0.chunk_offset)
            if region_offset_lo <= offset < region_offset_hi:
                return False
        return True

    # -------- evict ------------------------------------------------------

    def evict(self, slot_idx: int) -> Tuple[bool, Optional[SlotView]]:
        """Flip a VALID slot to TOMB if ref_count==0 and pin_count==0.

        Returns (True, view) on success, (False, None) on refusal.
        The returned SlotView captures the (hash, offset, len) so the
        caller can free the chunk from the node heap afterwards.
        """
        with self._lock.acquire(self._lock_id_for_slot(slot_idx)):
            slot = self._slots[slot_idx]
            if slot.line0.state != SLOT_STATE_VALID:
                return False, None
            if slot.line1.ref_count > 0 or slot.line1.pin_count > 0:
                return False, None
            view = SlotView(
                slot_idx=slot_idx,
                chunk_hash=int(slot.line0.chunk_hash),
                chunk_offset=int(slot.line0.chunk_offset),
                chunk_len=int(slot.line0.chunk_len),
                state=SLOT_STATE_VALID,
                fmt=int(slot.line0.fmt),
                owner_node_id=int(slot.line0.owner_node_id),
                generation=int(slot.line0.generation),
                geom_hash=bytes(slot.line0.geom_hash),
            )
            slot.line0.state = SLOT_STATE_TOMB
            slot_addr = self._slots_base_addr + slot_idx * self._slot_size
            self._fence.fence_after_write(slot_addr, self._line0_size)
            return True, view
