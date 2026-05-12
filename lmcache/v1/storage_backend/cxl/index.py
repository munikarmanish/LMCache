# SPDX-License-Identifier: Apache-2.0
"""Lockless hash-index lookup over the CXL-resident slot array.

The hash index is N open-addressed slots of 128 B each, stored in the
CXL pool. A slot's `line0` (read-mostly, 64 B) holds the identity:
chunk_hash, chunk_offset, chunk_len, state, fmt, owner, generation,
geom_hash. A writer's final `CLFLUSH(&line0); MFENCE` publishes the
whole line atomically from any reader's point of view.

This module implements `CXL_LOOKUP`: the lock-free read path.

Correctness invariants (plan "Locking Summary"):

- A reader never mutates the index.
- `state == EMPTY` terminates the probe (no key was ever placed here).
- `state == TOMB` and `state == ALLOCATING` are skipped — keep probing.
- A slot that matches by hash AND is `VALID` AND whose generation /
  geom_hash match the header is a hit.
- Torn reads are impossible at the line0 granularity on x86: a line0
  fits one cacheline, so either the full pre- or post-publish value
  is visible, never a mix.
"""

# Standard
import ctypes
from dataclasses import dataclass
from typing import Optional

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend.cxl.bootstrap import PoolHandle
from lmcache.v1.storage_backend.cxl.fence import Fence, default_fence
from lmcache.v1.storage_backend.cxl.layout import (
    GEOM_HASH_SIZE,
    SLOT_STATE_ALLOCATING,
    SLOT_STATE_EMPTY,
    SLOT_STATE_TOMB,
    SLOT_STATE_VALID,
    Slot,
    SlotLine0,
)

logger = init_logger(__name__)

# Bound on linear-probe length. If the index load factor is reasonable
# (~0.5), the 99th-percentile probe count is O(1). We cap at 64 so a
# pathologically-loaded index never turns a lookup into an O(N) scan.
DEFAULT_MAX_PROBE = 64


# Sentinels used internally by `_read_slot_consistent` to signal probe
# control flow without overloading SlotView with sentinel cases.
_PROBE_TERMINATE = object()
_PROBE_CONTINUE = object()


@dataclass(frozen=True)
class SlotView:
    """Immutable snapshot of a slot's line0 at a specific point in time.

    `CXL_LOOKUP` returns one of these on a hit. Callers that proceed
    to do a GET re-acquire the slot lock and re-check `state`/`hash`
    against this snapshot to close the post-lookup race (plan F9).
    """

    slot_idx: int
    chunk_hash: int
    chunk_offset: int
    chunk_len: int
    state: int
    fmt: int
    owner_node_id: int
    generation: int
    geom_hash: bytes


class CXLIndex:
    """Lock-free reader over the hash index in a bootstrapped PoolHandle.

    The writer side (INSERT / reserve_slot / commit / EVICT) lives in
    a later slice; this class deliberately exposes only the read path.
    """

    def __init__(
        self,
        handle: PoolHandle,
        fence: Optional[Fence] = None,
        max_probe: int = DEFAULT_MAX_PROBE,
    ):
        self._handle = handle
        self._fence = fence or default_fence()
        self._slot_count = handle.layout.index_slot_count
        if max_probe <= 0:
            raise ValueError("max_probe must be positive")
        # Cap probe length at index size so we never revisit slots.
        self._max_probe = min(max_probe, self._slot_count)

        # Cache addresses we'll reach for on every lookup.
        import ctypes
        self._slots = handle.slots()
        self._slots_base_addr = ctypes.addressof(self._slots)
        self._slot_struct_size = ctypes.sizeof(Slot)
        self._line0_size = ctypes.sizeof(SlotLine0)

        # Freeze a local copy of the header's geom_hash and generation
        # at construction. Callers rebuild the index object if the
        # header's generation bumps (plan F8).
        self._header_geom_hash = bytes(handle.header.geom_hash)
        self._header_generation = int(handle.header.gen)

    # -------- public API -------------------------------------------------

    @property
    def slot_count(self) -> int:
        return self._slot_count

    @property
    def max_probe(self) -> int:
        return self._max_probe

    def lookup(self, key: CacheEngineKey) -> Optional[SlotView]:
        """Return a SlotView for `key` if it is VALID in the index, else None.

        Lock-free. Safe against any concurrent writer. May return None
        for a key that is concurrently being written (ALLOCATING); the
        caller retries or treats it as a miss, which is the behavior
        we want — an incomplete write should not be observable.
        """
        return self._lookup_by_hash(key.chunk_hash)

    def contains(self, key: CacheEngineKey) -> bool:
        return self.lookup(key) is not None

    # -------- internals --------------------------------------------------

    def _lookup_by_hash(self, chunk_hash: int) -> Optional[SlotView]:
        # Mask into u64 because the stored field is unsigned.
        needle = chunk_hash & 0xFFFFFFFFFFFFFFFF
        start = needle % self._slot_count
        for i in range(self._max_probe):
            slot_idx = (start + i) % self._slot_count
            view = self._read_slot_consistent(slot_idx, needle)
            if view is _PROBE_TERMINATE:
                # Open-addressing invariant: first EMPTY ends the probe.
                return None
            if view is _PROBE_CONTINUE:
                continue
            return view  # type: ignore[return-value]
        # MAX_PROBE exhausted: table is effectively full on this chain.
        return None

    def _read_slot_consistent(
        self, slot_idx: int, needle: int
    ) -> "Optional[SlotView] | object":
        """Seqlock-style consistent read of a slot's line0.

        Returns:
        - SlotView on a verified-consistent VALID match.
        - _PROBE_CONTINUE to keep probing (TOMB, ALLOCATING, mismatch,
          or the slot raced — torn snapshot detected, retry).
        - _PROBE_TERMINATE if the slot is EMPTY (open-addressing end).

        The protocol: read `state`; if VALID and matches, snapshot all
        line0 fields, then re-read `state` and `chunk_hash`. If they
        haven't changed, the snapshot is consistent. If they have, a
        concurrent writer flipped the slot mid-read; we treat this as
        "keep probing" and the next round will see the stable post-
        publish value (or eventually find the key elsewhere).
        """
        # Bounded retry to avoid pathological livelock if a single
        # writer is hammering this slot. After RETRIES, fall through
        # to "keep probing" — correctness is preserved either way.
        RETRIES = 4
        for _ in range(RETRIES):
            line0 = self._read_line0(slot_idx)
            state_before = line0.state
            if state_before == SLOT_STATE_EMPTY:
                return _PROBE_TERMINATE
            if state_before != SLOT_STATE_VALID:
                # TOMB or ALLOCATING — no need for a consistent snapshot.
                return _PROBE_CONTINUE
            hash_before = line0.chunk_hash
            if hash_before != needle:
                return _PROBE_CONTINUE

            # Snapshot the full line0 contents.
            snap_offset = int(line0.chunk_offset)
            snap_len = int(line0.chunk_len)
            snap_fmt = int(line0.fmt)
            snap_owner = int(line0.owner_node_id)
            snap_gen = int(line0.generation)
            snap_geom = bytes(line0.geom_hash)

            # Re-read state and hash with a fresh fence; if either
            # changed, our snapshot may be torn.
            line0_after = self._read_line0(slot_idx)
            if (
                line0_after.state != SLOT_STATE_VALID
                or line0_after.chunk_hash != needle
            ):
                # Slot was rewritten mid-snapshot. Retry from the top
                # of this slot — we may find the new VALID state, or
                # a non-match that lets us keep probing.
                continue

            # Defensive checks against stale-generation / geom drift.
            if snap_gen != self._header_generation:
                return _PROBE_CONTINUE
            if snap_geom != self._header_geom_hash:
                return _PROBE_CONTINUE

            return SlotView(
                slot_idx=slot_idx,
                chunk_hash=int(hash_before),
                chunk_offset=snap_offset,
                chunk_len=snap_len,
                state=SLOT_STATE_VALID,
                fmt=snap_fmt,
                owner_node_id=snap_owner,
                generation=snap_gen,
                geom_hash=snap_geom,
            )
        return _PROBE_CONTINUE

    def _read_line0(self, slot_idx: int) -> SlotLine0:
        """Snapshot line0 for `slot_idx` into a stable local copy.

        Important: returns a **copy** (not a live view), so subsequent
        field reads off the returned object are guaranteed consistent
        with each other. The seqlock validation in
        `_read_slot_consistent` re-snapshots the line and compares
        state+hash to detect concurrent rewrites.

        On real CXL, fence_before_read invalidates the cacheline; the
        following memmove issues a fresh load from the device. On
        StubFence (single-host tests), it's a plain memcpy, which
        still gives us atomicity at the field-read level.
        """
        slot_addr = self._slots_base_addr + slot_idx * self._slot_struct_size
        self._fence.flush_before_read(slot_addr, self._line0_size)
        snap = SlotLine0()
        ctypes.memmove(
            ctypes.addressof(snap), slot_addr, self._line0_size
        )
        return snap


def _slot_probe_order(chunk_hash: int, slot_count: int, max_probe: int):
    """Yield the probe sequence for a given hash. Exposed for tests."""
    needle = chunk_hash & 0xFFFFFFFFFFFFFFFF
    start = needle % slot_count
    for i in range(max_probe):
        yield (start + i) % slot_count
