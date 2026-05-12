# CXL-Backed KV Cache for LMCache

A shared-memory L2 tier that lets multiple LMCache nodes (and the
vLLM engines they front) reuse each other's KV chunks at near-DRAM
latency, without the network hops of RDMA/NIXL.

This document is the design we shipped — the code lives under
[`lmcache/v1/storage_backend/cxl/`](lmcache/v1/storage_backend/cxl)
and is wired into the MP-mode storage manager via
[`CXLL2Adapter`](lmcache/v1/distributed/l2_adapters/cxl_l2_adapter.py).

---

## 1. Goals & Non-Goals

**Goals**

- Add CXL 2.0 Type-3 shared memory as an L2 tier between local DRAM
  (L1) and remote storage (L3).
- Let multiple LMCache instances on different nodes share one CXL
  pool and reuse each other's KV.
- Make a CXL hit a pure load/store probe — no controller in the hot
  path.
- Provide a cross-node fallback (`PushKVToCXL`) when one peer has the
  KV only in local DRAM.
- Stay correct under concurrent access despite no hardware coherence
  and no hardware atomics across hosts.

**Non-Goals (deliberately out of scope)**

- Multi-tenancy, security, untrusted peers.
- Persistence across power cycles.
- Variable-sized chunks (each pool serves one fixed chunk size).
- Cross-rack CXL fabric (single rack assumed).
- Online schema evolution.

---

## 2. Platform Assumptions

```
+-------------------------------------------------------------+
|                    Single rack                              |
|                                                             |
|  +-----------+   +-----------+   +-----------+              |
|  |  Node A   |   |  Node B   |   |  Node C   |              |
|  |  GPU(s)   |   |  GPU(s)   |   |  GPU(s)   |              |
|  |  vLLM     |   |  vLLM     |   |  vLLM     |              |
|  |  + LMCache|   |  + LMCache|   |  + LMCache|              |
|  +-----+-----+   +-----+-----+   +-----+-----+              |
|        |               |               |                    |
|  +-----+-----+   +-----+-----+   +-----+-----+              |
|  | mmap of   |   | mmap of   |   | mmap of   |              |
|  | /dev/dax  |   | /dev/dax  |   | /dev/dax  |              |
|  +-----+-----+   +-----+-----+   +-----+-----+              |
|        |               |               |                    |
|        v               v               v                    |
|     +======================================================+|
|     |          CXL 2.0 Type-3 shared memory pool          ||
|     |               (one /dev/dax0.0)                     ||
|     +======================================================+|
|                                                             |
+-------------------------------------------------------------+
```

What CXL 2.0 gives us:

- Every node `mmap`s the same `/dev/dax0.0` and gets a virtual address
  range that maps to the same physical bytes.
- Loads and stores are issued from the host CPU at near-DRAM latency
  (TraCT measured ~640 ns, ~10 GB/s on a Niagara 2.0 device).
- DMA engines (GPU and NIC) can access CXL memory directly when it's
  pinned via `cudaHostRegister`.

What it does **not** give us:

- **No HW cache coherence across hosts.** A store on Node A is not
  automatically visible in Node B's L1/L2 cache. Visibility requires
  explicit `CLFLUSH` (writer) and a fence-then-load sequence (reader).
- **No HW atomics across hosts.** `LOCK CMPXCHG` is atomic only within
  one host. Two hosts can both succeed at "atomic" CAS on the same
  cacheline. Cross-host mutual exclusion must come from a software
  protocol — never from CAS on shared memory.

Two consequences drive the design:

- All cross-host synchronization is software: ownership + a single-
  writer arbitrator, not shared-memory CAS.
- All cross-host visibility is software: explicit `CLFLUSH` + `MFENCE`
  discipline. (`CLFLUSHOPT` is too weakly ordered; we use `CLFLUSH`.)

The pool is **volatile**: bytes do not survive a power cycle, and
`CLFLUSH` is used purely for cross-host visibility, not durability.

---

## 3. High-Level Architecture

```
                                                           +-----------------------+
                                                           |   Built-in LMCache    |
                                                           |  Cluster Controller   |
                                                           |  (RegistryTree, ZMQ)  |
                                                           |                       |
                                                           |  Tracks LOCAL tier    |
                                                           |  presence only.       |
                                                           |  CXL state is NOT     |
                                                           |  in the directory.    |
                                                           +-----+-----------+-----+
                                                                 ^           ^
                                              ZMQ                |           |   ZMQ
                                       (KVAdmit, KVEvict,        |           |
                                        BatchedP2PLookup)        |           |
                                                                 |           |
                              +----------------------------------+           +------------------+
                              |                                                                  |
                +-------------v-------------+                                  +-----------------v-----------+
                | LMCache MP server (Node A)|                                  | LMCache MP server (Node B)  |
                |                           |                                  |                             |
                | StorageManager:           |                                  | StorageManager:             |
                |   L1Manager (DRAM)        |                                  |   L1Manager (DRAM)          |
                |   CXLL2Adapter <===+      |                                  |   CXLL2Adapter <===+        |
                |     CXLBackend     |      |                                  |     CXLBackend     |        |
                |       index        |      |                                  |       index        |        |
                |       index_writer |      |                                  |       index_writer |        |
                |       lock+manager |      |                                  |       locks       |        |
                |       regions+heap |      |                                  |       regions+heap |        |
                |                    |      |                                  |                    |        |
                |    cudaHostRegister|      |                                  |    cudaHostRegister|        |
                +------+-------------+------+                                  +------+-------------+--------+
                       ^             |                                                ^             |
                       | CUDA IPC    | mmap                                           | CUDA IPC    | mmap
                       | (KV blocks) | (POOL)                                         | (KV blocks) | (POOL)
                       |             v                                                |             v
                +------+--------+   +--------------------------------------------------+--------+
                | vLLM(s) on A  |   |              Shared CXL pool /dev/dax0.0                  |
                |   paged KV    |   |                                                           |
                +---------------+   +-----------------------------------------------------------+
                                                                                  ^
                                                                                  | mmap
                                                                                  |
                                                                            +-----+--------+
                                                                            | vLLM(s) on B |
                                                                            |   paged KV   |
                                                                            +--------------+
```

**Three distinct planes:**

| Plane | Role | Crosses the network? |
|---|---|---|
| Data plane | KV bytes flow GPU ↔ CXL ↔ GPU via memcpy / DMA. | No. |
| Control plane (CXL) | Hash index, region bitmap, locks. Lives in CXL itself, accessed via load/store + clflush. | No. |
| Control plane (cluster) | Built-in LMCache controller's `RegistryTree` of **local-tier** presence, plus `BatchedP2PLookup` RPCs for cold-miss fallback. | Yes (ZMQ over TCP). |

**Why split the controllers this way?** A warm CXL hit is the entire
point of this tier. If the hot-path lookup had to RPC a controller,
we'd reintroduce the network hop CXL is supposed to eliminate. So:

- "Is this in CXL?" is a load/store on the CXL hash index — no
  controller.
- "Does any peer have this in its **local DRAM** tier?" is consulted
  only on a CXL miss, via the existing `RegistryTree` + `BatchedP2PLookup`
  ([cache_controller/](lmcache/v1/cache_controller)). We reuse the
  controller verbatim — `CXLBackend` does NOT emit `KVAdmitMsg`/`KVEvictMsg`,
  and `RegistryTree` never carries a `location="CXLBackend"` entry.

---

## 4. Pool Layout

The pool is one `mmap` of `/dev/dax0.0`. All metadata sections live at
fixed offsets recorded in the header so any attaching node can
discover them. Code in
[`layout.py`](lmcache/v1/storage_backend/cxl/layout.py).

```
offset 0
+----------------------------------------------------------+
|                       Header (4 KiB)                     |
|  magic, layout_version, gen, geom_hash[16],              |
|  region_size, region_count, index_slot_count,            |
|  num_locks, max_nodes, search_hint,                      |
|  off_global_locks, off_region_bitmap, off_region_descs,  |
|  off_index, off_regions, pool_size                       |
+----------------------------------------------------------+
|              Global locks  (NUM_LOCKS x MAX_NODES x 64B) |
|  global_lock[lock_id][node_id] = LockSlot{state, seq}    |
|  state: IDLE | WAITING | LOCKED                          |
+----------------------------------------------------------+
|              Region bitmap (R/8 bytes, cacheline-aligned)|
|  bit i: 1 = region i is allocated                        |
+----------------------------------------------------------+
|              Region descriptors (R x 64 B)               |
|  RegionDesc[i] = { owner_node_id, claim_epoch,           |
|                    last_heartbeat_seen }                 |
|  owner_node_id sentinels:                                |
|     0xFFFF = FREE, 0xFFFE = ORPHANED                     |
+----------------------------------------------------------+
|              Hash index  (N x 128 B = 2 cachelines/slot) |
|  Open-addressed, linear probe.                           |
|  Each slot is two cachelines (line0 + line1) — see §5.   |
+----------------------------------------------------------+
|              Region payload area (R x region_size)       |
|                                                          |
|  +-- region 0 (256 MiB default) -------------------+     |
|  |  carved into chunk_size-byte slots by the       |     |
|  |  owner's per-node DRAM heap                     |     |
|  +-------------------------------------------------+     |
|                          ...                             |
|  +-- region R-1 -----------------------------------+     |
|  +-------------------------------------------------+     |
+----------------------------------------------------------+
end of /dev/dax0.0
```

**Three nested granularities** (terminology used throughout this doc):

| Granularity | Owner | Where ownership state lives |
|---|---|---|
| **Pool** = the whole CXL device (`/dev/dax0.0`). | The rack. | Header. |
| **Region** = 256 MiB (default) coarse alloc unit. | One node. | Region bitmap + descriptor. |
| **Chunk** = one KV payload (e.g. 64 KiB–MiB). | One node. | Per-node DRAM free-list (not in CXL). |

A reader does not need to know who owns the chunk's region — only that
the index slot says VALID and the bytes are at `pool_base + offset`.
Ownership matters only for writing and for GC.

**Sizing heuristics** (see §11): default to 256 MiB regions for
64–128 GiB pools and 8–16 nodes. Generally
`region_count ≈ max(8 × node_count, 32)` with `region_size ≥ 16 ×
max_chunk_size` so a node always fits plenty of chunks per region.

### Slot layout (128 B = 2 cachelines)

```
line 0 (read-mostly, 64 B) — fits in one cacheline so a writer's
                             final store-of-state publishes everything
                             atomically from the reader's POV
+--------------------------------------------------------------------+
| u64 chunk_hash         |  identity                                 |
| u64 chunk_offset       |  bytes from pool_base                     |
| u32 chunk_len          |  payload length                           |
| u32 state              |  EMPTY | ALLOCATING | VALID | TOMB        |
| u16 fmt                |  MemoryFormat enum                        |
| u16 owner_node_id      |  who claimed it                           |
| u32 generation         |  must equal header.gen                    |
| u8  geom_hash[16]      |  cluster-wide config fingerprint          |
| u8  pad[24]                                                        |
+--------------------------------------------------------------------+

line 1 (hot mutable, 64 B) — isolated from line0 so ref_count ping-pong
                             doesn't hurt readers
+--------------------------------------------------------------------+
| u32 ref_count          |  active reads in flight (DMA window)      |
| u32 pin_count          |  caller-imposed eviction protection       |
| u64 lru_prev           |  index of previous LRU slot               |
| u64 lru_next           |  index of next LRU slot                   |
| u16 lock_id            |  which global_lock row protects this slot |
| u8  pad[38]                                                        |
+--------------------------------------------------------------------+
```

Why two cachelines: line0 is read by every lookup; line1 changes on
every pin/unpin/get. Putting them in one cacheline would make every
ref_count bump invalidate readers' lookups.

### `geom_hash`: the cluster-config fingerprint

Stored in the header at bootstrap and on every slot at insert. It's a
16-byte `blake2b` of:

- `model_name`, `world_size`
- `kv_dtype`, `kv_shape = (num_layers, kv_size, chunk_size, num_heads, head_size)`
- `use_mla`, `chunk_size`

A peer with mismatched config sees `geom_hash` mismatch and refuses
to attach. Slots whose `geom_hash` doesn't match the current header
are skipped on lookup. This is what makes raw bytes on CXL safely
reusable across processes.

### `generation` (epoch)

Bumped by the controller on every restart (plan flaw F8). Slots that
survived a restart have a stale `generation` and are treated as
non-hits. This prevents a peer that had a partial probe in flight
during a controller restart from observing slots that point into
now-free memory.

---

## 5. Component Architecture (per-node MP server)

```
   +-------------------------------------------------------------+
   |                     LMCache MP Server                       |
   |                                                             |
   |  +------------+    +-----------------+    +--------------+  |
   |  | L1 Manager |    | StorageManager  |    | Cluster      |  |
   |  | (DRAM tier)|<-->| (orchestration) |<-->| controller   |  |
   |  +------------+    +--------+--------+    | (LMCacheCtlr)|  |
   |                             |              +--------------+  |
   |                             v                                |
   |               +---------------------------+                  |
   |               |     CXLL2Adapter          |                  |
   |               |  (eventfd async surface)  |                  |
   |               |  store/lookup/load/unlock |                  |
   |               +-------------+-------------+                  |
   |                             |                                |
   |                             v                                |
   |        +-------------------------------------------+         |
   |        |                CXLBackend                 |         |
   |        |     (synchronous AllocatorBackend)        |         |
   |        +-+-----+-----+-----+-------+-----+---------+         |
   |          |     |     |     |       |     |                   |
   |          v     v     v     v       v     v                   |
   |     +-------+ +---+ +---+ +-----+ +---+ +---+                |
   |     |Pool   | |Lck| |Rgn| |Heap | |Idx| |Wtr|  CXL package   |
   |     |Handle | |Mgr| |Allc |     | |   | |   |                |
   |     +---+---+ +---+ +---+ +-----+ +---+ +---+                |
   |         |                                                    |
   |         | mmap + cudaHostRegister                            |
   |         v                                                    |
   +-------------------------------------------------------------+
                               |
                               v
                  +------------------------------+
                  |  Shared CXL pool /dev/dax0.0 |
                  +------------------------------+
```

The layered stack lets each layer have one job:

| Layer | Module | Responsibility |
|---|---|---|
| Pool handle | [`bootstrap.py`](lmcache/v1/storage_backend/cxl/bootstrap.py) | `open` → `mmap` → header init/verify → `cudaHostRegister`. |
| Visibility | [`fence.py`](lmcache/v1/storage_backend/cxl/fence.py) | `flush_before_read` / `fence_after_write` abstraction. `StubFence` for in-process tests; CLFLUSH-based fence for real CXL. |
| Cross-host lock | [`locks.py`](lmcache/v1/storage_backend/cxl/locks.py) + [`lock_manager.py`](lmcache/v1/storage_backend/cxl/lock_manager.py) | TraCT two-tier lock + single-writer arbitrator thread. |
| Region allocator | [`regions.py`](lmcache/v1/storage_backend/cxl/regions.py) | Bitmap + descriptors. claim / release / `gc_dead_node` / `promote_orphaned`. |
| Per-node heap | [`heap.py`](lmcache/v1/storage_backend/cxl/heap.py) | DRAM free-list of fixed-size chunks over claimed regions. |
| Hash index reader | [`index.py`](lmcache/v1/storage_backend/cxl/index.py) | Lock-free seqlock-style lookup. Returns `SlotView` snapshots. |
| Hash index writer | [`index_writer.py`](lmcache/v1/storage_backend/cxl/index_writer.py) | reserve / commit / release / evict / pin / refcount under slot lock. |
| MemoryObj allocator | [`allocator.py`](lmcache/v1/storage_backend/cxl/allocator.py) | Wraps offsets into `TensorMemoryObj`s viewing the cudaHostRegistered pool. |
| Synchronous backend | [`cxl_backend.py`](lmcache/v1/storage_backend/cxl_backend.py) | `AllocatorBackendInterface`: contains/get/put/pin/unpin/remove. |
| Async adapter | [`cxl_l2_adapter.py`](lmcache/v1/distributed/l2_adapters/cxl_l2_adapter.py) | `L2AdapterInterface`: eventfd-based store/lookup/load tasks. Wraps `CXLBackend`. |
| Cross-node fallback | [`cross_node.py`](lmcache/v1/storage_backend/cxl/cross_node.py) + [`p2p_messages.py`](lmcache/v1/storage_backend/cxl/p2p_messages.py) | `PushKVToCXL` requester/donor handlers. |

**Backend-vs-adapter split:** `CXLBackend` is a synchronous
`AllocatorBackendInterface` implementation that's complete and
testable on its own. `CXLL2Adapter` is a thin shim that wraps it in
eventfd-based async semantics for the MP server's `L2AdapterInterface`.
This split keeps the algorithm logic decoupled from the IPC contract
and makes both layers independently fuzzable.

---

## 6. Locking & Concurrency

Plan reference: F2, F3, "Locking Summary".

### What needs synchronization, what doesn't

```
                           +-----------+   +-----------+
                           | Reader 1  |   | Reader 2  |
                           +-----+-----+   +-----+-----+
                                 |               |
                                 |               |
                  fence_before_read    fence_before_read
                     ; load slot          ; load slot         (no lock)
                                 |               |
                                 v               v
   +========================== CXL slot (line0 + line1) ===========================+
                                 ^               ^
                                 |               |
            takes(slot.lock_id)  |               |  takes(slot.lock_id)
                                 |               |
                           +-----+-----+   +-----+-----+
                           | Writer 1  |   | Writer 2  |    serialized via the
                           |  INSERT   |   |   EVICT   |    two-tier lock
                           +-----------+   +-----------+
```

Reads are lock-free. They use a seqlock-style snapshot:

1. Snapshot line0 into a local `SlotLine0` (memmove + fence).
2. Check state and chunk_hash.
3. Re-snapshot line0; if state or chunk_hash changed, retry.

Why this is needed: the plan's "line0 fits in one cacheline so publish
is atomic" only protects single field reads. If a reader peels off
multiple fields one at a time and a writer rewrites the slot in
between, the reader observes a torn snapshot. The seqlock solves
this with bounded retry.

### The two-tier lock

`global_lock[NUM_LOCKS][MAX_NODES]` lives in CXL. Workers never CAS
it (no cross-host atomics). Instead:

```
                             +------------------------+
                             |  Lock-manager thread   |
                             |  (one per rack)        |
                             |                        |
                             |  scans the array,      |
                             |  picks one WAITING per |
                             |  lock_id with the      |
                             |  smallest seq, flips   |
                             |  to LOCKED             |
                             +------------+-----------+
                                          |
                              writes      |   continuously
                              CLFLUSH     |   sweeps
                                          v
       global_lock[lock_id]:    +-------+-------+-------+-------+
       (per-node columns)       |Node 0 |Node 1 |Node 2 |Node 3 |
                                +-------+-------+-------+-------+
                                |IDLE   |WAITING|LOCKED |IDLE   |
                                +---^---+--^----+---^---+-------+
                                    |      |        |
                       fence_after_write   |        |
                                    |      |        |
                                +---+--+ +-+----+ +-+-----+
                                |Node 0| |Node 1| |Node 2 |
                                |  no  | |waiter| |holder |
                                | one  | |       | |       |
                                +------+ +-------+ +-------+
```

Acquisition (per node, plan F2):

```
acquire(local_lock[lock_id])               # DRAM pthread_mutex
write global_lock[lock_id][me] = WAITING   # CLFLUSH, store seq monotonic
poll  global_lock[lock_id][me] == LOCKED   # fence-before-read each loop
... critical section ...
write global_lock[lock_id][me] = IDLE      # CLFLUSH
release(local_lock[lock_id])
```

This is correct without HW atomics because **only the lock manager
writes the LOCKED field**. Workers only write WAITING and IDLE on
their own row. The lock manager is the single writer to LOCKED, so
its decisions cannot race.

### What takes the lock

| Operation | Local lock | Global lock | Notes |
|---|:-:|:-:|---|
| `CXL_LOOKUP` (read line0) | no | no | Pure load + fence + seqlock validation. |
| `GET` pin / unpin | yes | yes briefly | Two short critical sections around a lock-free DMA. |
| `INSERT` reserve | yes | yes | Writes ALLOCATING + owner; CLFLUSH; release. Donor DMA is lock-free. |
| `INSERT` commit | yes | yes | Writes offset/len/fmt; flips state to VALID. |
| `EVICT` | yes | yes | Refuses if `ref_count > 0` or `pin_count > 0`; flips to TOMB. |
| Region bitmap claim | yes | yes | Rare: ~once per 256 MiB of churn. |
| Chunk alloc/free in heap | yes (local) | **no** | Per-node DRAM free-list. |
| Controller RPCs | n/a | n/a | ZMQ; not CXL locks. |

Sharded locks: `lock_id = slot_idx % (num_locks - 1) + 1` (lock 0 is
reserved for the region allocator). With NUM_LOCKS = 4096 and 16
nodes × 8 concurrent writers, the chance of two unrelated keys
colliding on the same lock is ~3%.

---

## 7. The Slot State Machine

```
                                  reserve_slot()
                  +------+      claims via probe         +-----------------+
                  | EMPTY|---------------------------+--->|   ALLOCATING    |
                  +------+                           |    |   (owner=me)    |
                     ^                               |    +--------+--------+
                     |                               |             |
                     |   release_slot()              |             | commit_slot()
                     |   (NACK / OOM / restart)      |             | publishes VALID
                     +-------------------------------+             v
                                                          +-----------------+
                                                          |      VALID      |<---+
                                                          +--------+--------+    |
                                                                   |             |
                                              evict() refuses if   |             |
                                                ref/pin > 0        |             | reuse: a TOMB
                                                                   v             | seen on probe
                                                          +-----------------+    | becomes the
                                                          |      TOMB       |----+ first claim
                                                          +-----------------+      target of a
                                                                                   future reserve
```

State transitions:

| From | To | Operation | Conditions |
|---|---|---|---|
| EMPTY | ALLOCATING | reserve_slot | always (probe found a free position) |
| TOMB | ALLOCATING | reserve_slot | first TOMB seen on probe chain (open-addressing reuse) |
| ALLOCATING | VALID | commit_slot | caller wrote the chunk bytes |
| ALLOCATING | EMPTY | release_slot | caller bailed out (NACK / OOM) |
| ALLOCATING | TOMB | GC | owner died mid-write; bytes are junk |
| VALID | TOMB | evict | `ref_count == 0 && pin_count == 0` |

Readers skip TOMB and ALLOCATING during probes; only VALID with a
matching `chunk_hash`, current `generation`, and matching `geom_hash`
counts as a hit.

---

## 8. Workflows

### 8.1 Local PUT (warm-write to CXL)

```
 vLLM                CXLBackend            Lock        Heap         Index slot      CXL chunk
   |                     |                  |            |              |               |
   |--store(key, obj)--->|                  |            |              |               |
   |                     | reserve_slot:    |            |              |               |
   |                     | acquire(lock_id) |----------->|              |               |
   |                     | probe & claim    |            |              |               |
   |                     |     CAS ALLOCATING (owner=me, gen=header.gen)|               |
   |                     |     CLFLUSH ; MFENCE                         |               |
   |                     | release(lock_id) |<-----------|              |               |
   |                     |                  |            |              |               |
   |                     |  heap.alloc(size)|----------->|              |               |
   |                     |                  |    deque pop, perhaps     |               |
   |                     |                  |    region.claim()         |               |
   |                     |  return offset   |<-----------|              |               |
   |                     |                                                              |
   |                     |  memmove(pool_base+off, src, size)  ----------- bytes ------>|
   |                     |  fence_after_write(payload)                                  |
   |                     |                                                              |
   |                     | commit_slot:                                                 |
   |                     | acquire(lock_id) |----------->|                              |
   |                     |    write offset, len, fmt; state = VALID                     |
   |                     |    CLFLUSH ; MFENCE                                          |
   |                     | release          |<-----------|                              |
   |                     |                                                              |
   |<--ack(success)------|                                                              |
```

Failure rollback at any point: `heap.free(offset)` and
`release_slot(slot)`.

### 8.2 Local GET (CXL hit)

```
 vLLM                  CXLBackend                Index reader        Index writer       CXL chunk
   |                       |                          |                   |                |
   |--retrieve(key)------->|                          |                   |                |
   |                       | CXL_LOOKUP(key)--------->|                   |                |
   |                       |              snapshot line0 (lock-free)      |                |
   |                       |              seqlock validate                |                |
   |                       |<------------SlotView(off, len)               |                |
   |                       |                                                               |
   |                       | ref_count_up(slot)------------------->| acquire(lock_id)      |
   |                       |                                              |  ref_count++   |
   |                       |                                              |  CLFLUSH       |
   |                       |<-------------------------------------|                        |
   |                       |                                                               |
   |                       | re-verify slot still VALID, chunk_hash matches                |
   |                       |                                                               |
   |                       | wrap MemoryObj viewing pool[off..off+len]  <----- bytes ------|
   |<---MemoryObj----------|                                                               |
   |                                                                                       |
   |  ... (caller reads via MemoryObj; eventually drops the ref) ...                       |
   |                                                                                       |
   |                       |--MemoryObj.ref_count_down()-----------> ref_count_down(slot)  |
```

The pin (ref_count++) blocks any concurrent EVICT until the caller
releases the MemoryObj.

### 8.3 Cross-node fallback: `PushKVToCXL`

The hot path is `8.2` — Node B looks up a CXL slot and reads bytes
directly. The cross-node fallback handles the case where the chunk
exists only in another node's local DRAM tier:

```
 vLLM           Node B's          Cluster              Node A's          Node A's
 (B)           CXLBackend        controller           CXLBackend        local tier
   |                |                  |                    |                  |
   |--retrieve----->|                  |                    |                  |
   |                | CXL_LOOKUP -> MISS                    |                  |
   |                |                  |                    |                  |
   |                |--BatchedP2PLookup(hashes)-->|         |                  |
   |                |                  | RegistryTree probe |                  |
   |                |<--{donor=A, location="LocalCPUBackend", num_hit, peer_url}|
   |                |                  |                    |                  |
   |                | reserve_slot_for_donor(key, owner=A) for each key        |
   |                |   (slots are now ALLOCATING(A))       |                  |
   |                |                                                          |
   |                |--PushKVToCXLMsg{keys[], slot_idxs[], epoch}-->|          |
   |                |                                       |                  |
   |                |                                       | for each key:    |
   |                |                                       |   pin local copy |
   |                |                                       |<-----------------|
   |                |                                       | heap.alloc(size) |
   |                |                                       | memmove src->CXL |
   |                |                                       | fence            |
   |                |                                       | commit_slot      |
   |                |                                       | unpin local      |
   |                |                                                          |
   |                |<--PushKVToCXLRetMsg{num_committed=N, status=OK|PARTIAL}--|
   |                |                                                          |
   |                | for j > num_committed: release_slot_for_donor(slot, A)   |
   |                |                                                          |
   |                | retry CXL_LOOKUP --> HIT (now)                           |
   |<--MemoryObj----|                                                          |
```

Properties:

- **Bytes never cross the controller.** The controller is consulted
  exactly once, returns a peer URL, and the data flows directly
  Node A → CXL → Node B.
- **The requester reserves the slots, not the donor.** That puts
  retry/timeout logic in the requester. If the donor is slow or
  dies, the requester can call `release_slot_for_donor` and move on.
- **Whole-batch epoch check.** If the controller restarted between
  reservation and push, the donor sees a stale `epoch` and rejects
  the entire batch (`EPOCH_STALE`). The requester releases all its
  reservations.
- **Partial-success is normal.** The donor commits a contiguous
  prefix of `num_committed` keys. The requester releases the tail.
  This handles the case where the donor evicted some keys after
  admitting them — common, since the directory is best-effort.
- **`ALREADY_PRESENT` short-circuits.** If a key is already VALID in
  CXL by the time the requester reserves, it's not in the push
  subset; the donor never hears about it.

### 8.4 Disaggregated prefill (P/D)

```
 router           Prefill (P)                     CXL pool        Decode (D)
   |                 |                                |               |
   |--Request------->|                                |               |
   |                 | probe CXL index for prefix     |
   |                 |======= warm prefix HIT =======>|<-------- DMA -------+
   |                 | compute (N - hit) chunks         |                    |
   |                 | put new chunks to CXL (8.1)===> |                    |
   |                 |                                                       |
   |  first_token <--|                                                      |
   |                                                                        |
   |--StreamRequest(req)--------------------> D                             |
   |                                          | probe CXL ===========>|<----+
   |                                          |  HIT all N chunks
   |                                          |<====== DMA CXL->GPU ====
   |                                          | decode loop
   |  stream tokens <---------------------- D
```

What this gains over the existing NIXL-based PD path:

- Prefillers share each other's KV via CXL (no recompute on a hot
  prefix even when the request lands on a different prefiller).
- Decoders read prefill output directly from CXL with no network
  hop.
- Partial-prefix hits work too: a long prompt that overlaps an
  existing prefix gets the warm bytes from CXL and only computes
  the tail.

---

## 9. Region Allocator & GC

### 9.1 Two-tier allocation

```
   global region pool                          per-node DRAM heap
   (CXL bitmap + descs)                        (no CXL traffic)
   +-------------------+
   | bit | owner       |       claim()         +-----------------+
   |  0  | OWNER_FREE  |---------------------->| free chunks deque|   alloc() / free()
   |  1  | node 3      |                       | per claimed     |   touch only DRAM
   |  2  | node 7      |                       | region          |
   |  3  | OWNER_ORPHANED ---+                 +-----------------+
   |  4  | node 3      |    | promote when            |
   |  5  | OWNER_FREE  |<---+ all VALID slots         | hand out
   |  ...|             |     drained                  | offsets
   +-------------------+                              v
                                                +-----------+
                                                | new chunk |
                                                +-----------+
```

Why two tiers (TraCT §3.5):

- The global bitmap is on CXL, so every alloc/free crossing that
  layer is a cross-host metadata update — expensive (CLFLUSH, lock
  acquisition).
- Per-node DRAM free-lists keep chunk-level alloc/free entirely
  local. Cross-host traffic happens only at region boundaries
  (every ~256 MiB of churn).

Bitmap not free-list (plan F4):

- At realistic scales (256–512 regions total) bitmap scan is ~8
  cacheline loads — negligible vs. the lock + CLFLUSH overhead.
- Bitmaps are self-healing. A node crashing mid-free leaves at most
  one stuck bit; GC sweeps it. A free-list can end up with a
  dangling pointer that poisons subsequent allocs.
- Bitmap + descriptors are redundant by design: the bitmap is the
  alloc fast path, the descriptors are what GC scans to find a
  dead node's regions.

### 9.2 Dead-node GC

```
   step 0   Node A heartbeats stop arriving at the cluster controller.

   step 1   Controller marks A dead, calls `gc_dead_node(A)` on the
            CXL backend running on the elected GC node.

   step 2   Region descriptors:                     Region bitmap:
            for i where owner == A:                 unchanged
              owner = OWNER_ORPHANED                (bits stay set)
              CLFLUSH

   step 3   Slot sweep: for slot where
              owner == A and state == ALLOCATING:
              state = TOMB
              CLFLUSH
            (these chunks were mid-write; bytes are guaranteed junk)

   step 4   `VALID` slots owned by A are LEFT ALONE.
            They're still readable by any peer. Eventually, normal
            LRU eviction TOMBs them and frees their chunks back to
            A's (orphaned) heap — which we don't have access to in
            DRAM, so the chunks just sit there.

   step 5   `promote_orphaned(region)` runs periodically. When an
            ORPHANED region has zero live `VALID` slots pointing
            into it, it's promoted to FREE — bitmap bit cleared,
            descriptor reset.
```

Why the two-step (`ORPHANED` then `FREE`) dance:

- A dead node's DRAM-resident free-list is lost. We don't know which
  bytes inside its regions are holes (already-freed chunks) vs. live
  payload still backing `VALID` slots that peers are reading.
- Reusing the region's bytes immediately would corrupt a live
  reader's DMA.
- Worst-case capacity loss: `node_count × region_size` of effective
  space held in orphan state until LRU drains it. Acceptable.

### 9.3 Why regions are 256 MiB by default

| Pool | Nodes | Region size | Regions |
|---|---|---|---|
| 64 GiB | 8 | 256 MiB | 256 |
| 64 GiB | 16 | 128 MiB | 512 |
| 256 GiB | 16 | 1 GiB | 256 |
| 1 TiB | 32 | 2 GiB | 512 |

Heuristic: `region_count ≈ max(8 × node_count, 32)`,
`region_size = pool_size / region_count` rounded to a power of two,
`region_size ≥ 16 × max_chunk_size`.

Larger regions → fewer global-lock crossings (good — that's the
dominant cost). Smaller regions → finer-grained reclamation on node
death (marginal, since the dead node's `VALID` chunks drain via LRU
anyway). 256 MiB is a defensible default for the 64–128 GiB pools we
expect.

---

## 10. MP-Mode Data Path & GPU Connector

### 10.1 The two-copy path (MP mode)

```
 vLLM process                          MP server process
 +----------------+                   +-------------------------+
 |                |   1. CUDA IPC     |                         |
 |  paged KV      |<------------------+ server holds GPU ptrs   |
 |  blocks (HBM)  |                   |   to vLLM's KV blocks   |
 +-------+--------+                   |   via cudaIpc handle    |
         ^                            |                         |
         |                            |  +--------------------+ |
         | 3. GPU<->GPU DMA           |  | tmp_gpu_buffer     | |
         | (NVLink/PCIe)              |  | (server's HBM)     | |
         |                            |  +---------+----------+ |
         |                            |            ^            |
         |  +-------------------------+            |            |
         |  | multi_layer_kv_transfer                            |
         |  | (CUDA kernel)                                      |
         |  +----------------------------------------------------+
         |                                         |
         |                                         | 2. cudaMemcpyAsync
         |                                         |    H2D (pinned)
         |                                         |
         |                            +------------+-----------+
         |                            |  CXL chunk             |
         |                            |  (cudaHostRegister'd)  |
         |                            +------------------------+
```

Why two copies, not one (plan F7):

- vLLM's paged KV blocks are GPU memory; the MP server reaches them
  via **CUDA IPC handles** ([vllm_multi_process_adapter.py:340](lmcache/integration/vllm/vllm_multi_process_adapter.py#L340)).
- CUDA IPC shares **GPU** memory across processes — not host memory.
- The CXL pool is host memory in the MP server's address space. vLLM
  cannot map it (no cross-process host pinning in CUDA).
- So the path is: CXL (host, pinned) → server tmp HBM → vLLM HBM.

What CXL avoids vs. naïve mmap (TraCT §4.4):

- The **DRAM bounce buffer** that a `cudaMemcpy` from an
  un-pinned mmap would create. We `cudaHostRegister` the whole pool
  at backend init, so the first hop is a single pinned-host DMA.
- Cross-node network bytes for shared prefixes. Sibling MP servers
  read each other's writes from CXL; no NIXL/RDMA needed.

What CXL doesn't avoid:

- The final GPU→GPU IPC copy. That's a hard CUDA-IPC limitation, not
  a CXL property.

### 10.2 No new GPUConnector needed

`GPUConnectorInterface` is the layer **above** backends — one shared
instance per CacheEngine, created in
[manager.py:387](lmcache/v1/manager.py#L387). It scatters/gathers
between flat `MemoryObj` buffers and vLLM's paged KV blocks.

In MP mode, the MP server doesn't even use `GPUConnector` directly —
it calls `lmc_ops.multi_layer_kv_transfer` on the IPC-opened vLLM
pointers ([server.py:399](lmcache/v1/multiprocess/server.py#L399)).
For CXL-backed `MemoryObj`s, this works unchanged: `data_ptr()`
returns a pointer into the `cudaHostRegister`'d pool, which CUDA
treats as any other pinned-host buffer.

In single-process deployments, the same `CXLBackend` runs alongside
vLLM in one process. The shared `GPUConnector` issues one batched
H2D directly from CXL into vLLM's paged blocks — a one-copy path.

---

## 11. Deployment Modes: In-Process vs. MP-Server

LMCache has two ways to host the storage manager: in-process (the
v1 default, where `CXLBackend` lives in the vLLM worker process) or
MP-server mode (one `LMCache MP server` process per node, vLLM
processes connect over ZMQ + CUDA IPC). The CXL backend supports
both. They have meaningfully different cost profiles.

### 11.1 The two modes side by side

**In-process** (the v1 path, no separate server):

```
   vLLM worker process
   +----------------------------------------+
   |  vLLM engine                           |
   |  +-----------------+                   |
   |  | LMCacheEngine   |                   |
   |  |  StorageManager |   GPUConnector    |
   |  |   CXLBackend    |--- 1 GPU copy --->| vLLM's paged KV (HBM)
   |  +--------+--------+   (CXL host -> HBM)
   |           |
   |           | mmap + cudaHostRegister
   |           v
   +-----------+----------------------------+
               |
               v
   +-----------+----------------------------+
   |          CXL pool /dev/dax0.0          |
   +----------------------------------------+
```

**MP-server** (`CXLL2Adapter` in the server, vLLM connects):

```
   vLLM worker process              MP server process
   +-----------------+              +-----------------+
   | vLLM engine     |              |  StorageManager |
   |  paged KV blocks|<-----+       |   CXLL2Adapter  |
   |  (HBM)          |      |       |     CXLBackend  |
   +--------+--------+      |       +--------+--------+
            |               |                | mmap + cudaHostRegister
            | CUDA IPC      | tmp_gpu_buffer v
            | handles       | (server HBM)   |
            v               | + IPC ptrs     v
       (server scatters     | into vLLM HBM  CXL pool
        into vLLM HBM via   |                /dev/dax0.0
        IPC-opened pointers)|

  Per request: 1 ZMQ RPC, 2 GPU copies (CXL -> server HBM -> vLLM HBM)
```

### 11.2 Per-request cost comparison

| Cost | In-process | MP-server | Difference |
|---|---|---|---|
| LMCache lookup | Direct call | ZMQ over TCP | A few µs of IPC. |
| GPU copies for one chunk | 1: CXL → vLLM HBM | 2: CXL → server HBM → vLLM HBM | The second copy is GPU↔GPU over NVLink (~600 GB/s) or PCIe (~64 GB/s). |
| Per-vLLM-process overhead | Each process maps `/dev/dax0.0`, runs / attaches a lock manager, holds its own DRAM heap | One server holds all CXL state; vLLM holds only IPC handles | MP saves ~64 GB of mmapped pages per extra vLLM process. |
| GPU memory overhead | None | `tmp_gpu_buffer` per (instance, GPU) — one chunk's worth | A few MiB per registered worker. |
| CXL pool initialization cost | Each vLLM process pays it at startup | Paid once per node | `cudaHostRegister`'ing 64 GiB takes real time and pins host memory. |

### 11.3 What each mode buys you

**In-process advantages**

- One GPU copy instead of two. For long-context prefill streaming
  many MB of KV per request, the server-HBM bounce is real
  bandwidth.
- No ZMQ chatter on retrieve/store.
- Simpler crash story: vLLM and LMCache live and die together; no
  "MP server died, vLLM is now talking to nothing" failure mode.
- Simpler deployment: one process, one config, no extra service.
  Tests, benchmarks, and demos are dramatically easier to set up.

**MP-server advantages**

- **One CXL initializer per node.** The lock-manager thread *must*
  be a per-rack singleton — multiple managers would race on the
  LOCKED writes. In-process mode forces an election between vLLM
  workers, or the workers all attach without running a manager
  (and an external process must run one). MP mode makes this
  trivial: the MP server is the one place a manager lives.
- **Avoiding redundant CXL state per worker.** With TP=8, in-
  process mode means 8 processes each mmap'ing the pool, each
  holding their own DRAM heap, each potentially trying to claim
  regions. The cross-host lock has to arbitrate 8× the contention.
  MP mode collapses this to one client of the pool per node.
- **Sharing the L1 (DRAM) cache across vLLM workers.** The MP
  server's `L1Manager` is one DRAM cache shared by every vLLM
  process on the node. In-process mode duplicates it per worker.
  For large `max_local_cpu_size` this is the dominant win — much
  bigger than the CXL second-copy cost.
- **vLLM lifecycle decoupling.** A vLLM crash doesn't kill cached
  KV. A new vLLM worker registers and starts hitting the existing
  cache.
- **Disaggregated prefill (P/D).** Prefill workers and decode
  workers are usually distinct vLLM processes. Sharing KV between
  them through one MP server (which sees their writes via CUDA
  IPC) is the natural shape; in-process mode requires
  re-implementing P/D handoff.
- **A single point to extend.** Cluster-controller integration,
  cross-node `PushKVToCXL` fallback, observability, eviction
  policy — adding any of these in MP mode is one place to touch
  vs. N concurrent processes.

### 11.4 Decision rule

Choose **in-process** when:

- Single vLLM process on the node (no TP, no model multiplexing).
- No P/D disaggregation.
- Latency budget is tight and chunks are large enough that the
  second GPU copy shows up in profiles.
- You're prototyping, debugging, or benchmarking the CXL backend
  itself.

Choose **MP-server** when:

- Multiple vLLM workers on the node share KV (TP, multi-instance,
  P/D).
- You want one place to manage cluster-controller integration,
  P2P fallback, observability.
- Crash isolation between cache and serving matters.
- DRAM-cache size or CPU-side coordination is the bottleneck, not
  GPU bandwidth.

### 11.5 Is the second GPU copy a CXL problem?

No — it's an MP-mode property, independent of which L2 backend is
underneath. Comparable numbers:

- **CXL → server HBM** (pinned-host DMA): ~10 GB/s. For a 1 MiB
  chunk, ~100 µs. Bandwidth-limited, identical in both modes.
- **Server HBM → vLLM HBM** (CUDA IPC scatter via NVLink): sub-µs
  per chunk on NVLink, ~30 µs on PCIe Gen5. Only paid in MP mode.
- **RDMA/NIXL round-trip** (the alternative we're replacing):
  hundreds of µs on a healthy fabric, more under load.

So **CXL is a ~10× win over RDMA in either deployment mode**;
MP mode adds a small constant on top. The decision between in-
process and MP-server is about operational properties (worker
count, P/D, lifecycle), not about CXL itself.

If profiles ever show the server-HBM bounce dominating real
workload latency (long-context prefill on slow PCIe links is the
worst case), the right fix is the cross-process `cudaHostRegister`
route called out as out-of-scope for v1: have vLLM also `mmap`
`/dev/dax0.0` and `cudaHostRegister` it itself, with the MP server
handing out CXL offsets via lightweight RPC. That gets back to one
GPU copy without giving up MP mode's other benefits — but it's a
real chunk of work and only worth doing once measurements demand
it.

### 11.6 Recommendation

- **Production: MP-server.** The operational benefits (cache
  survives vLLM restarts, shared L1 across workers, P/D handoff,
  single integration surface) are substantial; the per-request
  GPU bandwidth cost is small relative to the CXL win over RDMA.
- **Development / benchmarks: in-process.** Simpler to set up, and
  it isolates CXL's contribution without the IPC variable. Useful
  for pinning down whether a regression is in the CXL backend or
  somewhere in MP mode's surrounding plumbing.

---

## 12. Configuration Surface

```jsonc
// --l2-adapter argument to the MP server
{
  "type": "cxl",
  "dev_path": "/dev/dax0.0",       // required
  "node_id": 3,                    // required, distinct per rack
  "chunk_size_bytes": 65536,       // required, must divide region_size
  "region_size": 268435456,        // default 256 MiB
  "initialize": false,             // exactly one node per rack: true
  "generation": 1,                 // bump on controller restart
  "run_lock_manager": false,       // exactly one node per rack: true

  // geom_hash inputs — must match across the rack
  "model_name": "deepseek-r1-distill-llama-8b",
  "world_size": 8,
  "kv_dtype_str": "torch.float16",
  "kv_shape": [32, 2, 256, 8, 128],
  "use_mla": false,
  "cluster_chunk_size": 256,

  // worker identity (does not feed geom_hash)
  "worker_id": 5,
  "local_world_size": 8,
  "local_worker_id": 5
}
```

Tunables we kept in the config rather than auto-deriving:

- `region_size`: workload-dependent. Default 256 MiB; raise on larger
  pools.
- `initialize` / `run_lock_manager`: per-rack singletons. The
  bootstrap node sets both true; everyone else sets both false.
- `generation`: bumped explicitly on controller restart so a stale
  peer's lookups reject stale slots.

---

## 13. Failure Modes & Recovery

| What fails | Detection | Recovery |
|---|---|---|
| Writer crashes mid-INSERT (slot ALLOCATING) | Heartbeat timeout. | GC slot sweep flips ALLOCATING→TOMB; chunk bytes ignored. Region containing the chunk goes ORPHANED. |
| Writer crashes after VALID, with live readers | Heartbeat timeout. | `VALID` slots stay readable. Region orphaned. LRU drains over time; `promote_orphaned` reclaims when empty. |
| Donor of a `PushKVToCXL` evicts the local copy after the directory advertised it | Donor returns `num_committed < N`. | Requester releases unused slot reservations. Caller falls back to GPU recompute for the tail. |
| Controller restarts | Generation counter bumps. | All in-flight slots become epoch-stale; peers' lookups skip them. Donors reject stale-epoch push messages. |
| Geom mismatch between peers | `geom_hash` field check at attach. | Peer refuses to attach. Logs the digest mismatch. |
| Partial CXL torn read | Seqlock retry in `CXL_LOOKUP`. | Up to 4 retries; after that, treated as MISS (correctness preserved either way). |
| Lock-manager stalls | Per-acquire timeout in `TwoTierLock`. | Caller raises `LockAcquisitionTimeout`; surfaces as a put failure that the L1 manager retries. |

---

## 14. Implementation Status

**Shipped and validated on hardware**

- [`lmcache/v1/storage_backend/cxl/`](lmcache/v1/storage_backend/cxl)
  — pool layout, fence abstraction, two-tier lock + manager,
  region/heap allocators, lock-free index reader (with seqlock),
  index writer, MemoryObj allocator, cross-node `PushKVToCXL` driver,
  periodic GC thread, ZMQ-based P2P transport.
- [`lmcache/v1/storage_backend/cxl_backend.py`](lmcache/v1/storage_backend/cxl_backend.py)
  — `CXLBackend(AllocatorBackendInterface)`, single-process complete.
- [`lmcache/v1/distributed/l2_adapters/cxl_l2_adapter.py`](lmcache/v1/distributed/l2_adapters/cxl_l2_adapter.py)
  — `CXLL2Adapter(L2AdapterInterface)`, plus
  [`CXLL2AdapterConfig`](lmcache/v1/distributed/l2_adapters/config.py)
  registered in `create_l2_adapter`. CXL is now a `--l2-adapter`
  type alongside `mock` and `nixl_store`.
- **Real-CXL `Fence` implementation.** `CLFlushFence` issues
  `CLFLUSH`+`MFENCE` via a small C `.so`
  ([`_native/cxl_fence.c`](lmcache/v1/storage_backend/cxl/_native/cxl_fence.c))
  loaded with `ctypes`. Auto-built on first import on x86; falls back
  to `StubFence` on non-x86 or when no compiler is available.
  `default_fence()` auto-selects `CLFlushFence` whenever the `.so` is
  loadable. `StubFence` remains correct for single-host deployments
  (MP-mode + multiple vLLM procs, in-process tests).
- **Periodic GC thread**
  ([`gc.py`](lmcache/v1/storage_backend/cxl/gc.py)). Rack-wide
  singleton; on each tick, polls a caller-injected `LivenessProvider`
  and runs three idempotent steps: `gc_dead_node` (mark dead-owned
  regions ORPHANED), `sweep_dead_owner_allocating` (flip stranded
  ALLOCATING slots to TOMB), `promote_orphaned` (drain check then
  ORPHANED → FREE). No coupling to the cluster controller — by
  design, GC pulls liveness rather than waiting on a push event.
- **ZMQ wire-up of `PushKVToCXLMsg`/`PushKVToCXLRetMsg`**
  ([`p2p_transport.py`](lmcache/v1/storage_backend/cxl/p2p_transport.py))
  via `CXLP2PServer` (REP) and `CXLP2PClient` (REQ). Decoupled from
  `P2PBackend` so it doesn't drag in the `LocalCPUBackend` dependency
  there. Each donor node runs one `CXLP2PServer` alongside its
  backend and lock manager; clients hold one `CXLP2PClient` per
  donor URL.
- 202 tests under `tests/v1/storage_backend/cxl/` (CXL package) and
  `tests/v1/distributed/l2_adapters/` (MP-mode integration). All
  pre-existing LMCache backend tests continue to pass.
- **Cross-host validation completed.** Two-node test rig (Node A +
  Node B, both `mmap`'ing the same `/dev/dax0.0`) successfully
  exercised: warm CXL hits across hosts (Node A writes, Node B reads
  via load/store + `CLFlushFence`), and the full ZMQ-driven
  `PushKVToCXL` round-trip (Node B requests, Node A reads from local
  tier, allocates CXL chunk, copies bytes, commits slot, Node B
  reads back via `CXL_LOOKUP`). Driven by
  [`scripts/cxl_cross_host_test.py`](scripts/cxl_cross_host_test.py),
  which is the recommended manual smoke test on new hardware.

**Cluster-controller integration (shipped)**

[`controller_integration.py`](lmcache/v1/storage_backend/cxl/controller_integration.py)
provides two adapters that bridge the CXL backend to the existing
LMCache cluster controller's worker registry. Production deployments
import these instead of writing the same plumbing each time.

- `ControllerLivenessProvider` — wraps an `LMCacheWorker` and turns
  the controller's `QueryWorkerInfoMsg` (with `instance_id="all"`)
  into a `LivenessProvider` that the GC accepts unchanged. Takes an
  explicit `InstanceMapping` list (lmcache `instance_id` → CXL
  `node_id`), configured at boot. Stale-heartbeat threshold is
  configurable; failsafe behavior on RPC error is to treat all known
  node ids as alive (a transient outage must NOT trigger reclaim).
- `ControllerDonorRouter` + `ControllerBackedFetch` — turn
  `BatchedP2PLookupMsg` into a `DonorRoute` (donor's CXL node id +
  CXL P2P URL) and produce a cached `CXLP2PClient` keyed by URL.
  `ControllerBackedFetch.fetch(keys)` is what the CXL adapter calls
  on a CXL miss: one RPC to the controller, one ZMQ round-trip to
  the donor, return a `RemoteFetchResult`.

The mapping between LMCache `(instance_id, worker_id)` and CXL
`node_id` is intentionally explicit and configured per deployment:
one CXL `node_id` per LMCache *instance* (regardless of TP rank).
That choice keeps the CXL `MAX_NODES` budget from being consumed by
TP fan-out and matches the deployment shape (one MP server per
node = one CXL identity per node).

---

## 15. References

- TraCT (https://arxiv.org/html/2512.18194v1) — two-tier locking,
  per-node heaps, `cudaHostRegister`, GPU↔CXL DMA, prefix-cache
  index design.
- Maru (https://github.com/xcena-dev/maru) — control-plane / data-
  plane split, `mmap`+handle API.
- LMCache built-in cluster controller —
  [`lmcache/v1/cache_controller/`](lmcache/v1/cache_controller).
  We reuse `RegistryTree`, `RegisterMsg`, `BatchedKVOperationMsg`,
  and `BatchedP2PLookupMsg` verbatim; CXL adds no new RPCs.
