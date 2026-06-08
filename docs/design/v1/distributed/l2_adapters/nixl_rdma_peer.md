# RDMA/NIXL Peer L2 Adapter

A read-only (pull-only) L2 tier that, on an L1 miss, looks up KV chunks
on a **static list of remote peers** and pulls the hits straight into
this node's L1 cache over **one-sided RDMA** using NIXL.

The adapter lives at
[`nixl_peer_l2_adapter.py`](lmcache/v1/distributed/l2_adapters/nixl_peer_l2_adapter.py)
and is wired into the MP-mode `StorageManager` as a `--l2-adapter` type
(`nixl_peer`) alongside `cxl`, `nixl_store`, and `mock`.

It is the network analogue of the
[CXL adapter](lmcache/v1/distributed/l2_adapters/cxl_l2_adapter.py): same
"L1-first, L2 fetches the misses from peers and prefetches them into L1"
shape, but the peers are on other hosts reachable by RDMA rather than a
shared CXL pool.

---

## 1. Goals & Non-Goals

**Goals**

- On an L1 miss, find missing chunks on remote peers and pull them into
  L1 over RDMA so the subsequent `retrieve` is a plain L1 read.
- Use **one-sided RDMA READ** so the peer's CPU is not in the data path
  (the requester drives the transfer).
- Hold a **remote read-lock** on a peer's chunk for exactly the READ
  window, and always release it — so peers can still evict.
- Reuse the existing, hardware-validated agent↔agent NIXL machinery
  ([`NixlChannel`](lmcache/v1/transfer_channel/nixl_channel.py)) rather
  than re-implement RDMA.
- Static peer list from config at startup. No cluster controller, no
  dynamic membership.

**Non-Goals (deliberately out of scope for v1)**

- Storing/pushing chunks to peers. STORE is a no-op; this tier is
  pull-only and relies on `store_policy="lazy"` (see §6).
- Dynamic peer discovery / membership changes / a directory service.
- Cross-host eviction coordination. Each peer evicts its own L1 freely;
  the remote read-lock only protects the in-flight READ window.
- L2-resident GPU-direct retrieve (`supports_l2_resident_retrieve()`
  stays `False` — chunks land in L1 and retrieve is unchanged).
- Authentication / untrusted peers / multi-tenancy.

---

## 2. Where it sits

```
        Node A (this node)                         Node B (a peer)
 +-----------------------------+          +-----------------------------+
 | StorageManager              |          | StorageManager              |
 |   L1Manager (DRAM)          |          |   L1Manager (DRAM)          |
 |   NixlPeerL2Adapter         |          |   NixlPeerL2Adapter         |
 |     - ZMQ control client ---|--RPC---->|--- ZMQ control server       |
 |       (per peer)            | lookup/  |    (RemoteLookup/Unlock)    |
 |     - NixlChannel (RDMA) <==|==READ====|==> peer L1 buffer (reg'd)   |
 |     - donor: L1 read-lock   |<---RPC---|--- donor: L1 read-lock      |
 +-----------------------------+          +-----------------------------+

 Control plane: ZMQ REQ/REP — "do you have hashes H[]? readlock + give
                me your page indices" ; "release these readlocks".
 Data plane:    one-sided NIXL RDMA READ from peer's registered L1
                buffer into this node's registered L1 buffer.
```

Every node runs **both** sides: a control-plane REP server + donor (so
peers can look up and read its L1) and, per configured peer, a control
client + a NIXL remote-agent handle (so it can fetch from that peer).
The roles are symmetric, matching the CXL adapter's donor-server +
peer-clients design.

---

## 2a. Peer connection lifecycle (off the request path)

The NIXL handshake (agent-metadata + transfer-descriptor exchange) is a
blocking ZMQ round-trip to a peer's init side-channel. It is **never**
done synchronously at startup (that would hang the whole MP server until
every peer is up) and is kept **out of the lookup/load critical path** so
a request never pays for a first-time handshake. Three mechanisms, in
order of how a peer typically becomes connected:

1. **Eager (background, at startup).** The adapter kicks off a bounded
   outbound handshake to each peer on its asyncio loop. If the peer isn't
   up yet, the attempt times out quietly and the peer stays unconnected —
   the server runs fine alone and all lookups MISS immediately.
2. **Inbound handshake (the steady state).** When a peer later starts and
   handshakes *us*, its `NixlMemRegRequest` registers its transfer handler
   on our channel. A small `on_peer_registered` hook on `NixlChannel`
   fires the adapter's callback, which marks that peer **connected** — so
   we can READ from it without ever doing our own outbound handshake. This
   is how a node that booted first picks up a node that booted later.
3. **Lazy on first hit (bounded fallback).** If a lookup finds a hit on a
   peer that neither of the above has connected yet, the adapter attempts
   the handshake once, bounded by `control_timeout_ms`. On failure those
   keys fall back to MISS for that request (and the remote read-lock the
   peer took is released, not stranded).

A peer's connection is memoized (a per-peer flag + lock), so the handshake
runs at most once regardless of which mechanism wins the race.

---

## 3. The two planes

| Plane | Carries | Transport | Who initiates |
|---|---|---|---|
| Control | "have these hashes? readlock + page indices", "unlock these" | ZMQ REQ/REP (msgspec) | Requester (this node) |
| Data | the KV bytes | one-sided NIXL RDMA READ | Requester (this node) |

**Why a control plane is required even with one-sided RDMA.** A NIXL
one-sided READ is addressed by *page indices into the peer's registered
L1 buffer* (`make_prepped_xfer(..., remote_xfer_handler,
remote_indexes)`). The requester cannot know which indices hold a given
chunk hash, nor that those bytes will stay put during the READ. So a
control RPC must: (1) resolve hash → page indices on the peer, and
(2) read-lock those chunks so they survive the READ window. This mirrors
how `pd_backend` exchanges `remote_indexes` via an alloc/query RPC before
its RDMA transfer.

**Addressing: byte offset → descriptor index.** A NIXL prepped transfer
is addressed by *descriptor index* — the channel registers one descriptor
per `align_bytes` (one chunk) of the L1 buffer, so descriptor `i` covers
bytes `[i·align_bytes, (i+1)·align_bytes)`. But the v1 L1 allocator
(`TensorMemoryAllocator`) stores `MemoryObj.meta.address` as a **byte
offset**, not a page index. So a chunk's descriptor index is
`meta.address // align_bytes`:

- the **donor** converts its chunk's offset to an index in
  `RemoteLookupResp.page_indices` ([`nixl_peer_donor.py`](lmcache/v1/distributed/l2_adapters/nixl_peer_donor.py));
- the **requester** converts each local destination buffer's offset the
  same way before the READ (`_NixlReadChannel.read_chunks` in
  [`nixl_peer_l2_adapter.py`](lmcache/v1/distributed/l2_adapters/nixl_peer_l2_adapter.py)).

This is why the adapter drives `make_prepped_xfer` through its own
`_NixlReadChannel` wrapper rather than `NixlChannel.batched_read`: the
latter passes `meta.address` (the byte offset) to NIXL verbatim, which is
correct only for a *paged* allocator (where `meta.address` already is the
index, as in `pd_backend`) and produces a "local index out of range" error
for the byte-offset L1 allocator used here. Both nodes must register L1
with the same `align_bytes`, which holds when the rack shares model
geometry + dtype.

---

## 4. The three operations

### 4.1 LOOKUP (`submit_lookup_and_lock_task`)

```
for each requested key:                       (this node, bg loop)
  - L1 is checked by the StorageManager BEFORE L2 is consulted, so every
    key handed to this adapter is already an L1 miss.
  - group the miss keys, and for each peer (in config order):
        RemoteLookupReq{hashes[], generation}  --ZMQ-->  peer
        peer: for each hash, reserve_read its L1 chunk (read-lock held);
              return RemoteLookupResp{found_mask, page_indices[],
                                      sizes[], peer_agent_id}
  - for every hash a peer claims + locked, record
        (key -> peer_id, remote_index, size)
    in an in-flight "remote pin table" and set the bitmap bit.
  - a key found on no peer stays 0 in the bitmap (true miss).
```

The returned bitmap = "this adapter can satisfy these keys." The pins on
the peers are now **held** and tracked in the remote pin table keyed by
`(task lineage, key)`. We do not READ yet — load is a separate phase.

Concurrency across peers: the per-peer RemoteLookup RPCs for one task are
issued concurrently (one ZMQ REQ socket per peer; the bg loop fans them
out). First peer to claim a key wins; later peers are told to unlock any
duplicate they also locked (a small "unlock the losers" RPC) so we never
leak a remote read-lock.

### 4.2 LOAD (`submit_load_task(keys, l1_buffers)`)

```
- For each key the controller asks us to load, look up its entry in the
  remote pin table -> (peer_id, remote_index).
- Group by peer. For each peer, one NixlChannel.batched_read:
      local_indices  = [dst.meta.address for dst in that peer's l1_buffers]
      remote_indexes = [entry.remote_index for those keys]
      channel.batched_read(l1_buffers,
                           {"sender_id": peer_id,
                            "remote_indexes": remote_indexes})
  This is a one-sided RDMA READ: peer bytes -> our L1 buffer. No peer CPU.
- set bitmap bit per key whose READ returned DONE.
```

The destination `l1_buffers` are the write-locked L1 objects the
`PrefetchController` reserved. After the READ lands, the controller flips
them write→read-locked; retrieve then reads them from L1 unchanged.

### 4.3 UNLOCK (`submit_unlock(keys)`) — releasing the remote read-lock

The `PrefetchController` calls `submit_unlock` for **every key in the
load plan** once load completes (`_unlock_all_plan_keys`). That is our
hook to release the peer's read-lock:

```
- Group the keys by the peer that holds their pin (from the pin table).
- Per peer: RemoteUnlockReq{hashes[]} --ZMQ--> peer
      peer: for each hash, finish_read its L1 chunk (drop the read-lock).
- Drop the entries from the local remote pin table.
```

`submit_unlock` is fire-and-forget per the interface contract: it must
*eventually* succeed and never be retried by the caller, so the adapter
retries transient RPC failures internally and logs unrecoverable ones.

**Donor side read-lock lifecycle.** On the peer, a RemoteLookup
read-locks via `L1Manager.reserve_read`; the matching RemoteUnlock calls
`L1Manager.finish_read`. To guarantee a lock is never stranded if the
requester dies between lookup and unlock, the donor stamps each pin with
a lease (generation + monotonic deadline) and a background sweep drops
pins whose lease expired — the same "best-effort, self-healing" stance
the CXL GC takes toward orphaned state. (Reuses the `_L1FinishReadOnDrop`
idea from [`cxl_l1_donor.py`](lmcache/v1/distributed/l2_adapters/cxl_l1_donor.py).)

---

## 5. RETRIEVE is unchanged

By the time `retrieve` runs, every hit chunk is in L1 (pulled during the
prefetch LOAD phase) and read-locked by the `PrefetchController`. The
retrieve path reads from L1 and calls `finish_read` exactly as it does
for an L1-native hit. This adapter adds nothing to the retrieve hot path,
and the chunk stays in L1 (subject to normal L1 eviction) for future
requests — satisfying the "keep them in L1 until evicted" requirement.

---

## 6. STORE is a no-op (LazyStorePolicy)

With `store_policy="lazy"`
([`store_policy.py`](lmcache/v1/distributed/storage_controllers/store_policy.py)),
`select_store_targets` returns `{}`, so the `StoreController` never hands
this adapter a store task. For defense in depth, `submit_store_task` is
also implemented as an inert success (records 0 bytes, touches no peer),
so even a non-lazy misconfiguration cannot push KV to peers. Peers serve
only what already lives in their own L1 from their own traffic.

---

## 7. Configuration surface

```jsonc
// one entry in --l2-adapter
{
  "type": "nixl_peer",
  "node_id": 0,                         // this node's id; distinct per rack
  "peers": [                            // static peer list
    { "node_id": 1,
      "control_url": "tcp://hostB:8500",  // peer's ZMQ control REP server
      "init_url":    "tcp://hostB:8501" } // peer's NIXL handshake side-channel
  ],
  "control_bind_url": "tcp://0.0.0.0:8500", // our control REP server
  "init_bind_url":    "tcp://0.0.0.0:8501", // our NIXL handshake server
  "nixl_backends": ["UCX"],             // NIXL data-plane backend(s)
  "control_timeout_ms": 30000,
  "lease_ms": 60000,                    // remote read-lock lease / sweep window

  // geometry — must match across the rack (page-size / dtype agreement)
  "model_name": "...", "world_size": 8,
  "kv_dtype_str": "torch.bfloat16",
  "kv_shape": [32, 2, 256, 8, 128],
  "use_mla": false, "cluster_chunk_size": 256
}
```

The factory needs both `l1_memory_desc` (to register the L1 buffer with
NIXL for RDMA) and `l1_manager` (for the donor side to read-lock/serve
local chunks) — it opts into both kwargs, which the registry forwards.

---

## 8. Failure modes

| What fails | Detection | Recovery |
|---|---|---|
| Peer has no copy of a hash | `found_mask` bit 0 in RemoteLookupResp | Key stays a miss; bitmap bit 0; retrieve recomputes. |
| Peer evicted the chunk between lookup and READ | Read-lock is held across the window, so this cannot happen for a locked hit. If lookup itself races eviction, the peer simply reports not-found. | n/a / miss. |
| Control RPC to a peer times out | ZMQ recv timeout | That peer contributes no hits this round; other peers still consulted. Logged. |
| RDMA READ errors (`status == ERR`) | `NixlChannel.batched_read` raises | Affected keys' bitmap bits stay 0 (load miss); pins still released via `submit_unlock`. |
| Requester dies between lookup and unlock | Donor lease expires | Donor sweep calls `finish_read` on expired pins; chunk becomes evictable again. |
| Geometry / page-size mismatch between peers | Rejected at handshake (page size disagreement) or guarded by geometry fields | Peer connection refused; logged. |

---

## 9. Reused components

- [`NixlChannel`](lmcache/v1/transfer_channel/nixl_channel.py) —
  agent↔agent handshake (`lazy_init_peer_connection`,
  `get_agent_metadata`/`add_remote_agent`) and `batched_read`
  (one-sided RDMA READ by page index). Already used by `pd_backend`.
- The donor read-lock pattern from
  [`cxl_l1_donor.py`](lmcache/v1/distributed/l2_adapters/cxl_l1_donor.py)
  (`reserve_read` → serve → `finish_read`).
- The adapter skeleton (asyncio loop on a daemon thread, task-id-keyed
  result dicts, three distinct eventfds) from
  [`cxl_l2_adapter.py`](lmcache/v1/distributed/l2_adapters/cxl_l2_adapter.py)
  / [`mock_l2_adapter.py`](lmcache/v1/distributed/l2_adapters/mock_l2_adapter.py).
- The registry/factory wiring (`register_l2_adapter_type` /
  `register_l2_adapter_factory`).
