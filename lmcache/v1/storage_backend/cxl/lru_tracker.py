# SPDX-License-Identifier: Apache-2.0
"""Node-local LRU recency tracker for CXL slots.

The CXL pool is shared across nodes, but maintaining a *global* LRU order
would require a CXL write (and cross-node fence) on every read — the exact
cost the eviction design avoids. Instead each node keeps a purely DRAM view
of the recency of the slots *it* touches. A slot hot on another node reads as
cold here; that is an accepted approximation of the goal, which is only to
give a node a sensible victim order when it must evict its own chunks to make
room after the global pool is exhausted (see ``CXLStore`` eviction ladder).

The tracker holds only this node's owned slot indices. It is consulted when a
store fails to claim a new region: :meth:`coldest` yields victims oldest-first,
and the backend evicts them (skipping pinned / in-flight ones) back into its
node heap's free-list.

Cost model:
  - :meth:`touch` (on every read/store): one ``OrderedDict.move_to_end`` under a
    DRAM lock. No CXL access, no fence, no cross-node coordination.
  - :meth:`coldest` (only on eviction): pops from the cold end, O(1) each.

Thread-safety: every method takes an internal mutex. The critical sections are
O(1) dict operations, so contention with the store/read path is negligible.
"""

# Standard
from collections import OrderedDict
from typing import List
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class NodeLRUTracker:
    """DRAM-only LRU order over the slots this node owns.

    Recency is keyed by ``slot_idx``. The most-recently-touched slot sits at
    the end of the backing ``OrderedDict``; the coldest sits at the front.
    """

    def __init__(self):
        # slot_idx -> None; ordering is the recency (front = coldest).
        self._order: "OrderedDict[int, None]" = OrderedDict()
        self._lock = threading.Lock()

    def touch(self, slot_idx: int) -> None:
        """Mark ``slot_idx`` as most-recently-used.

        Inserts the slot if it is not already tracked (a store commit), or
        moves it to the hot end if it is (a read or re-store). Idempotent.

        Args:
            slot_idx: Index of the slot this node just read or committed.
        """
        with self._lock:
            self._order[slot_idx] = None
            self._order.move_to_end(slot_idx)

    def forget(self, slot_idx: int) -> None:
        """Drop ``slot_idx`` from recency tracking.

        Called when a slot leaves this node's ownership: eviction, remove, or
        bulk clear. A slot not currently tracked is ignored.

        Args:
            slot_idx: Index of the slot to stop tracking.
        """
        with self._lock:
            self._order.pop(slot_idx, None)

    def clear(self) -> None:
        """Forget every tracked slot (used by ``CXLStore.clear``)."""
        with self._lock:
            self._order.clear()

    def coldest(self, n: int) -> List[int]:
        """Return up to ``n`` coldest slot indices, oldest first.

        This is a *snapshot* — it does not remove the slots from tracking. The
        caller evicts each in turn and calls :meth:`forget` for the ones it
        actually reclaims, so a slot that can't be evicted (pinned / in-flight)
        stays tracked and is retried on the next eviction pass.

        Args:
            n: Maximum number of victims to return.

        Returns:
            Up to ``n`` slot indices ordered coldest-first. Empty if ``n <= 0``
            or nothing is tracked.
        """
        if n <= 0:
            return []
        with self._lock:
            out: List[int] = []
            for slot_idx in self._order:  # iterates front (coldest) -> back
                out.append(slot_idx)
                if len(out) >= n:
                    break
            return out

    def tracked_count(self) -> int:
        """Return the number of slots currently tracked (debug / tests)."""
        with self._lock:
            return len(self._order)
