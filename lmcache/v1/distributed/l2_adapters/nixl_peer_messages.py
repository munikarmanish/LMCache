# SPDX-License-Identifier: Apache-2.0
"""Control-plane message types for the RDMA/NIXL peer L2 adapter.

The data plane is a one-sided NIXL RDMA READ (the requester reads bytes
straight out of a peer's registered L1 buffer). A one-sided READ is
addressed by *page indices into the peer's registered L1 buffer*, so the
requester must first ask the peer two things over this control plane:

  1. Which page indices hold the chunks for these hashes?
     (``RemoteLookupReq`` -> ``RemoteLookupResp.page_indices``)
  2. Read-lock those chunks so they survive the READ window.
     (the peer ``reserve_read``s them on lookup; the requester releases
     them with ``RemoteUnlockReq`` once the READ has landed.)

This mirrors how ``pd_backend`` exchanges ``remote_indexes`` via an
alloc/query RPC before its RDMA transfer, and how the CXL adapter uses
``PushKVToCXLMsg`` / ``PushKVToCXLRetMsg`` for its cross-node fallback.
The hot path (an L1 hit) never touches these messages — they exist only
for the L1-miss-then-peer-fetch path.

Wire keys: an ``ObjectKey`` is serialized to a ``WireKey`` (chunk-hash
hex + kv_rank + cache_salt). ``model_name`` is implied by the rack
geometry but carried for a defensive cross-check on the donor side.
"""

# Standard
import enum
from typing import Union

# Third Party
import msgspec


class NixlPeerMsgBase(msgspec.Struct, tag=True):
    """Base class for all NIXL-peer control-plane messages."""

    pass


class WireKey(msgspec.Struct):
    """Wire-safe serialization of an ``ObjectKey``.

    Two peers in one rack share model geometry and dtype, so the tuple
    ``(chunk_hash, kv_rank, cache_salt)`` uniquely identifies a chunk
    across hosts. ``model_name`` is included so the donor can reject a
    request that somehow crossed model boundaries.

    Fields:
        chunk_hash_hex: ``ObjectKey.chunk_hash`` as a lowercase hex
            string (byte-exact, length-preserving — unlike the CXL
            adapter's lossy u64 truncation, this round-trips any hash
            width).
        model_name: ``ObjectKey.model_name``.
        kv_rank: ``ObjectKey.kv_rank``.
        cache_salt: ``ObjectKey.cache_salt``.
    """

    chunk_hash_hex: str
    model_name: str
    kv_rank: int
    cache_salt: str = ""


class RemoteLookupReq(NixlPeerMsgBase):
    """Requester -> donor: "Do you have these chunks? If so, read-lock
    them and tell me their page indices so I can RDMA-READ them."

    The donor ``reserve_read``s each matching L1 chunk (holding a read
    lock that protects the READ window) and returns its page index. The
    requester later releases each held lock with ``RemoteUnlockReq``.

    Fields:
        sender_id: Requester identity, for logging / lease bookkeeping.
        lease_id: Opaque per-request id the donor stamps on every pin it
            grants for this request, so a later ``RemoteUnlockReq`` (or a
            lease-expiry sweep) can release exactly these pins.
        keys: The chunks to look up, in request order.
    """

    sender_id: str
    lease_id: str
    keys: list[WireKey]


class RemoteLookupResp(NixlPeerMsgBase):
    """Donor -> requester acknowledgment for a ``RemoteLookupReq``.

    Positional, one entry per requested key:

    Fields:
        found: ``found[i]`` is ``True`` iff the donor has key ``i`` and
            successfully read-locked it (so its page index is valid and
            the chunk is protected until unlocked).
        page_indices: ``page_indices[i]`` is the donor's L1 page index
            for key ``i`` when ``found[i]`` is ``True``, else ``-1``.
            This is the ``remote_index`` for the NIXL one-sided READ.
        sizes: ``sizes[i]`` is the chunk's byte size when found, else 0.
        peer_agent_id: The donor's NIXL peer id, used by the requester to
            select the right remote transfer handler. Echoes the peer id
            the requester already knows; carried for explicitness.
    """

    found: list[bool]
    page_indices: list[int]
    sizes: list[int]
    peer_agent_id: str


class RemoteUnlockReq(NixlPeerMsgBase):
    """Requester -> donor: "Release the read-locks I took under this
    lease for these keys; their READ has landed (or been abandoned)."

    Fire-and-forget at the adapter level, but ZMQ REP still requires a
    reply, so the donor answers with ``RemoteUnlockResp``.

    Fields:
        sender_id: Requester identity.
        lease_id: The lease the pins were granted under (from the
            matching ``RemoteLookupReq``).
        keys: The chunks to unlock.
    """

    sender_id: str
    lease_id: str
    keys: list[WireKey]


class RemoteUnlockResp(NixlPeerMsgBase):
    """Donor -> requester acknowledgment for a ``RemoteUnlockReq``.

    Fields:
        num_released: How many of the requested pins were actually
            released (a pin already dropped by lease expiry counts as 0).
    """

    num_released: int


class NixlPeerStatus(enum.IntEnum):
    """Coarse status codes shared across responses for logging."""

    OK = 0
    PARTIAL = 1
    ERROR = 2


NixlPeerMsg = Union[
    RemoteLookupReq,
    RemoteLookupResp,
    RemoteUnlockReq,
    RemoteUnlockResp,
]
