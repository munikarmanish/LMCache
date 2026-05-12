# SPDX-License-Identifier: Apache-2.0
"""P2P message types for the CXL cross-node `PushKVToCXL` fallback.

Plan reference: F1, F5, "Cross-node fallback" section. The hot path
(warm CXL hit) does not touch these messages — they only exist for
the case where:

  1. Node B's CXL_LOOKUP misses.
  2. The built-in `batched_p2p_lookup` returns a donor (Node A) that
     has the chunks in its **local** L0/L1 tier, not in CXL yet.
  3. Node B reserves CXL slots on Node A's behalf (owner=A) and asks
     A to publish its local copies into those slots.

Modeled on `BatchedLookupAndPutMsg` in p2p_backend.py.
"""

# Standard
import enum
from typing import Union

# Third Party
import msgspec


class CXLP2PMsgBase(msgspec.Struct, tag=True):
    """Base class for CXL P2P messages."""

    pass


class PushKVToCXLMsg(CXLP2PMsgBase):
    """Requester → donor. "Publish your local copies of these keys into
    the CXL slots I've already reserved for you."

    The slot ids are pre-reserved by the requester via
    `CXLIndexWriter.reserve_slot(key, owner=donor_node_id)`. The donor
    only needs to (a) confirm it has each key locally, (b) DMA the
    bytes into the chunk inside its node-heap-allocated region for
    the slot, (c) commit each slot.

    Fields are ordered the same way `BatchedLookupAndPutMsg` orders
    its fields so the wire format is intuitive to anyone who has
    already seen the existing P2P path.
    """

    # Requester's identity, for logging / response routing.
    sender_id: str

    # Contiguous prefix of CacheEngineKeys, in `to_string()` form.
    keys: list[str]

    # CXL slot ids, one per key, all pre-reserved with owner=donor and
    # state=ALLOCATING. The donor commits them to VALID after DMA.
    slot_idxs: list[int]

    # Generation observed at the time of reservation. The donor
    # rejects the whole batch if its current header.gen differs
    # (controller restarted in between).
    epoch: int


class PushStatus(enum.IntEnum):
    OK = 0  # All keys committed.
    PARTIAL = 1  # First N committed, the rest skipped.
    EPOCH_STALE = 2  # Donor saw mismatched generation; nothing committed.
    ALL_NACK = 3  # Donor had no local copy for any key.


class PushKVToCXLRetMsg(CXLP2PMsgBase):
    """Donor → requester acknowledgment.

    Partial-success semantics: `num_committed` is the length of the
    **contiguous prefix** of keys that reached VALID. Any keys past
    that point were not published; the requester must release their
    pre-reserved slots.
    """

    num_committed: int

    status: PushStatus


CXLP2PMsg = Union[
    PushKVToCXLMsg,
    PushKVToCXLRetMsg,
]
