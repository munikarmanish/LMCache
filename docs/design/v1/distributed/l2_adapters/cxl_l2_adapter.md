# CXL L2 Adapter

An **L2 tier backed by a CXL 2.0 shared-memory pool** exposed as a DAX device
(`/dev/dax0.0`, or a plain file for tests). Every participating node mmaps the
*same* physical pool and reads/writes a shared hash index + a shared heap of KV
chunks through a rack-wide distributed lock. On an L1 miss, a node finds the
chunk in the shared pool (a local pointer read + DMA), or — if only a *peer's*
local DRAM has it — pulls it into the pool via a one-shot `PushKVToCXL` RPC and
then reads it back.

The adapter lives at
[`cxl_l2_adapter.py`](lmcache/v1/distributed/l2_adapters/cxl_l2_adapter.py) and
is wired into the MP-mode `StorageManager` as a `--l2-adapter` type (`cxl`)
alongside `nixl_peer`, `nixl_store`, and `mock`. It is a thin async/eventfd
shim over the synchronous
[`CXLBackend`](lmcache/v1/storage_backend/cxl_backend.py); the real machinery
(pool layout, index, heap, locks, cross-node protocol) lives under
[`lmcache/v1/storage_backend/cxl/`](lmcache/v1/storage_backend/cxl/).

It is the **shared-memory analogue** of the
[NIXL peer adapter](docs/design/v1/distributed/l2_adapters/nixl_rdma_peer.md):
same "L1-first, L2 fetches the misses" shape, but the second tier is a coherent
shared CXL pool rather than a set of RDMA-reachable peers. Unlike the NIXL peer
adapter (pull-only, chunks land in L1), the CXL adapter can also serve a hit
**straight to GPU** without a DRAM bounce (§7).

> **Doc-vs-code note.** Some module docstrings under `storage_backend/cxl/`
> still describe the backend as a "single-process skeleton / step 5" with
> features "not yet wired." That prose lags the code: the cross-node P2P push,
> born-pinned commit, batched locking, and GPU-direct retrieve described here
> are all implemented and wired. Trust this doc + the source, not the older
> docstrings.

---

## 1. Goals & Non-Goals

**Goals**

- Provide a rack-wide L2 tier out of a shared CXL pool: any node can read any
  chunk another node stored, as a local pointer + DMA (no network on the read
  path).
- On an L1 miss where only a *peer's* DRAM holds the chunk, pull it into the
  pool with one `PushKVToCXL` RPC, then serve it locally.
- Serve hits **GPU-direct** (CXL → GPU HBM `cudaMemcpyAsync`) when the caller
  wants an L2-resident retrieve, skipping the L1 DRAM bounce.
- Keep the read/pin/commit hot paths cheap at long prompts by **batching**
  distributed-lock acquisitions (one arbiter sweep for a whole batch, not one
  per chunk).
- Self-heal orphaned state (dead-node regions, stranded ALLOCATING slots) with
  a background GC, and invalidate a whole generation of slots on controller
  restart via an epoch check.

**Non-Goals (v1)**

- No cluster controller / dynamic membership: peers are a static list from
  config (Alternative A). Liveness for GC is injected, not discovered here.
- No cross-rack coherence: one pool = one rack sharing one `/dev/dax`.
- No authentication / untrusted peers / multi-tenancy on the pool itself.
- No automatic pool sizing across heterogeneous devices — `dev_path`,
  `region_size`, `chunk_size_bytes` are operator-set and must agree per rack.

---

## 2. Where it sits

```
        Node 0 (initializer)                        Node 1
 +------------------------------+          +------------------------------+
 | StorageManager               |          | StorageManager               |
 |   L1Manager (DRAM)           |          |   L1Manager (DRAM)           |
 |   CXLL2Adapter               |          |   CXLL2Adapter               |
 |     CXLBackend               |          |     CXLBackend               |
 |     CXLDonor  <--PushKVToCXL-|--RPC-----|--- CXLP2PClient (requester)  |
 |     CXLP2PClient ----RPC-----|--------->|--- CXLDonor                  |
 +------------------------------+          +------------------------------+
              \                                        /
               \        one shared /dev/dax0.0        /
                \  (both nodes mmap the SAME pool)    /
                 v                                    v
 +-------------------------------------------------------------------+
 | Header | Global locks | Region bitmap | Region descrs | Index | Regions
 +-------------------------------------------------------------------+
                 ^
                 | WAITING->LOCKED flips only
        cxl_lock_manager (C sidecar, run on exactly ONE node)
```

Every node runs a `CXLBackend` over the same pool. Two roles for the cross-node
push (symmetric, like the NIXL peer adapter): a **donor** server
([`CXLDonor`](lmcache/v1/storage_backend/cxl/cross_node.py)) so peers can ask it
to push its local DRAM copies into the pool, and per configured peer a
**requester** client ([`CXLP2PClient`](lmcache/v1/storage_backend/cxl/p2p_transport.py)).
Exactly one node **initializes** the pool (writes a fresh header) and exactly
one node runs the **lock-manager arbiter**.

---

## 3. The shared pool layout

Defined in [`layout.py`](lmcache/v1/storage_backend/cxl/layout.py). The pool is
one `/dev/dax` mmap carved into fixed sections whose offsets are recorded in the
header and computed by `PoolLayout.compute`:

```
Header (4 KiB) | Global locks | Region bitmap | Region descriptors | Hash index | Regions
```

**Header** ([`Header`](lmcache/v1/storage_backend/cxl/layout.py), 4 KiB at
offset 0): `magic`, `layout_version`, `gen` (the **generation / epoch**, bumped
each bootstrap), `geom_hash` (16-byte blake2b of the model geometry), and the
sizing/offset fields (`region_size`, `region_count`, `index_slot_count`,
`num_locks`, `max_nodes`).

**Hash index** — an open-addressed table of fixed slots. Each
[`Slot`](lmcache/v1/storage_backend/cxl/layout.py) is 128 B = two cachelines,
deliberately split to avoid false sharing:

| Line | Role | Key fields |
|---|---|---|
| `line0` (read-mostly, 64 B) | published identity of the chunk | `chunk_hash`, `chunk_offset` (bytes from pool base), `chunk_len`, `state`, `fmt`, `owner_node_id`, `generation`, `geom_hash` |
| `line1` (hot-mutable, 64 B) | liveness counters + LRU | `ref_count`, `pin_count`, `lru_prev/next`, `lock_id` |

Everything a lookup reads lives in `line0`, so a writer's final fenced store of
`line0` **publishes the whole chunk atomically**.

**Slot state machine** (`layout.py`; states `EMPTY=0`, `ALLOCATING=1`,
`VALID=2`, `TOMB=3`):

```
EMPTY --reserve_slot()--> ALLOCATING(owner=me) --commit_slot()--> VALID
                                                                    |
  (TOMB slots are reused by      TOMB <--evict() (needs ref_count==0
   reserve under open-addressing)  ^            AND pin_count==0)
```

- **`ref_count`** keeps a slot alive during a DMA read (`read_into`).
- **`pin_count`** keeps a slot alive across a lookup→retrieve window ("locked").
  `evict` refuses a slot unless **both** counters are 0
  ([`index_writer.py`](lmcache/v1/storage_backend/cxl/index_writer.py)).

**Heap / regions** ([`heap.py`](lmcache/v1/storage_backend/cxl/heap.py)): a
`NodeHeap` is a per-node free-list over CXL regions, one fixed `chunk_size`. A
claimed region is carved into `region_size // chunk_size` fixed slots. `alloc()`
hands out one pool-relative offset; **`alloc_batch(n)`** grabs `n` under a single
lock hold (used by the donor push, §5). Chunk addressing is uniform everywhere:
the host address of a chunk is `pool.base + line0.chunk_offset`.

---

## 4. The three L2 operations

The adapter is a single asyncio loop on a daemon thread (mirrors
`MockL2Adapter`), with **three distinct eventfds** (store / lookup / load) that
the `PrefetchController` and `StoreController` poll. Each `submit_*` allocates a
task id, schedules work on the loop, and on completion writes a result dict +
`eventfd_write(...)`. See
[`cxl_l2_adapter.py`](lmcache/v1/distributed/l2_adapters/cxl_l2_adapter.py).

### 4.1 LOOKUP-AND-LOCK (`submit_lookup_and_lock_task` → `_do_lookup`)

Every key handed here is already an L1 miss (the `StorageManager` checks L1
first).

1. **First pass — batched pin of local hits.**
   `self._backend.pin_batch(ce_keys)` pins *all* keys in **one batched lock
   acquisition** (≈ one arbiter sweep) instead of one sweep per key — the
   dominant warm-lookup cost at long prompts. A pin succeeds only for a
   currently-`VALID` slot, so the `True` positions are exactly the local CXL
   hits; the rest are misses. "Pin" = bump `pin_count`, i.e. the slot cannot be
   evicted while we hold it — this is the "lock" in lookup-and-lock.
2. **Second pass — misses → cross-node fetch.** If there are misses and static
   peers are configured, `_try_remote_fetch_misses` hands each peer the
   contiguous prefix of misses starting at the first miss and calls
   `remote_fetch` (§5). Freshly-committed slots come back **born-pinned** (the
   donor already holds the pin on our behalf), so we only verify+cache them and
   skip a redundant per-chunk pin; `ALREADY_PRESENT` slots still need a pin.

The returned bitmap = "this adapter can satisfy these keys," with their pins now
held.

### 4.2 LOAD (`submit_load_task` → `_do_load`)

For each key the controller asks us to load into a caller-provided L1 buffer,
`CXLBackend.read_into` runs the fast hit path: index lookup → `ref_count_up`
(keep alive during DMA) → re-verify the slot → `ctypes.memmove` from
`pool.base + chunk_offset` into the destination → `ref_count_down` in `finally`.
Per-chunk copies fan out over a `ThreadPoolExecutor`
(`LMCACHE_CXL_LOAD_WORKERS`, default 8). The caller owns the destination
`MemoryObj`; the adapter never frees it. (Load is skipped entirely for keys
served GPU-direct — see §7.)

### 4.3 UNLOCK (`submit_unlock` → `_do_unlock`)

Fire-and-forget per the interface contract (must *eventually* succeed, never
retried by the caller). Drops the pins taken in lookup via
`self._backend.unpin_batch(ce_keys)` — again **one batched lock acquisition**
for the whole set instead of one sweep per key.

---

## 5. Cross-node `PushKVToCXL` (when only a peer's DRAM has the chunk)

A CXL miss means the chunk isn't in the *shared pool* — but a peer may hold it
in its own local DRAM (from its own traffic). The requester asks that peer to
copy it into the pool. Transport-agnostic:
[`remote_fetch`](lmcache/v1/storage_backend/cxl/cross_node.py) (requester) drives
a `donor.handle_push(msg)` against
[`CXLDonor`](lmcache/v1/storage_backend/cxl/cross_node.py) (server), over ZMQ
in production or a direct call in tests. Messages live in
[`p2p_messages.py`](lmcache/v1/storage_backend/cxl/p2p_messages.py).

**Requester (`remote_fetch`): reserve → push → settle**

1. **Reserve** a slot per key on the *donor's behalf*
   (`reserve_slot_for_donor` stamps `owner_node_id = donor`, `ALLOCATING`).
   Outcomes: `RESERVED` (needs a DMA), `ALREADY_PRESENT` (already `VALID`),
   `WAIT_FOR_OTHER` (another writer in flight), `INDEX_FULL` (stop).
2. **Push** the `RESERVED` slot_idxs to the donor with the current `epoch`.
3. **Settle**: release any slots beyond `ack.num_committed`; report the
   contiguous satisfied prefix and a per-key `born_pinned[]` flag.

**Donor (`handle_push`)**

1. **Epoch check** — reject the whole batch if `msg.epoch != header.gen`
   (`EPOCH_STALE`); this is how a controller restart invalidates in-flight
   pushes.
2. **Pre-pin** local copies in order; the first miss caps the effective prefix.
3. **Validate** each slot is still `ALLOCATING`, owned by this node, at `epoch`.
4. **`alloc_batch`** all chunks in one heap-lock hold (falls back to a
   per-chunk prefix if the heap runs out mid-batch → requester sees `PARTIAL`).
5. **Copy** DRAM → CXL in parallel (default 4 workers,
   `LMCACHE_CXL_DONOR_WORKERS`) via the NT-store `fast_copy` (§8).
6. **`commit_slot_batch(..., pin=BORN_PINNED)`** — commit every slot in one
   batched lock acquisition, publishing each `VALID` with `pin_count == 1` in
   the *same* critical section. This folds the requester's pin into the donor's
   commit, removing a separate per-chunk pin round-trip.

**Contiguous-prefix contract.** `num_committed` is the length of the longest
contiguous prefix of successful commits from the first key. Any gap/miss/failure
truncates it; committed-but-past-the-gap slots are unpinned, evicted to `TOMB`,
and their chunks freed so the requester only ever sees a clean prefix. `status`
is `OK` (all keys), `PARTIAL` (short prefix — the requester may retry the tail on
the next peer), `ALL_NACK`, or `EPOCH_STALE`.

---

## 6. The distributed lock and why batching matters

[`TwoTierLock`](lmcache/v1/storage_backend/cxl/locks.py) is a cross-host,
sharded lock backed by a `global_lock[NUM_LOCKS][MAX_NODES]` array in the pool.

- Workers never CAS shared memory. They only **store** transitions: `WAITING`
  when joining a lock's line, `IDLE` when releasing. A single **lock-manager
  arbiter** does every `WAITING → LOCKED` flip (smallest `seq` wins → FIFO-ish
  fairness). A per-node DRAM `threading.Lock` per `lock_id` serializes
  intra-node waiters so only one contends for a given CXL row.
- The arbiter runs as a **C sidecar process**
  ([`lock_manager_proc.py`](lmcache/v1/storage_backend/cxl/lock_manager_proc.py),
  built on demand) in production, because the Python arbiter's sweep balloons
  ~20× (≈10 ms → ≈215 ms) under GIL contention from donor-commit load. An
  in-process Python `LockManager` is the test/no-compiler fallback.

**`acquire_batch`** ([`locks.py`](lmcache/v1/storage_backend/cxl/locks.py))
collapses N lock acquisitions into ≈ one arbiter sweep: (1) take every distinct
lock's local mutex in ascending `lock_id` order — a global order, so two batch
callers can't deadlock; (2) publish `WAITING` to *all* rows (cheap stores, no
blocking); (3) spin until every row is `LOCKED`, rechecking only the still-pending
set. Because distinct `lock_id`s are arbitrated independently within one sweep,
the whole batch is granted together.

The batched index ops built on it —
[`commit_slot_batch`, `pin_batch`, `unpin_batch`](lmcache/v1/storage_backend/cxl/index_writer.py)
— are what make lookup/unlock/commit cheap at long prompts (§4, §5). Each maps
slot_idxs → distinct lock_ids and reports per-slot `False` on validation failure
instead of raising, matching the contiguous-prefix contract.

---

## 7. L2-resident GPU-direct retrieve

`supports_l2_resident_retrieve()` returns `True`. The whole pool is
`cudaHostRegister`'d at bootstrap
([`bootstrap.py`](lmcache/v1/storage_backend/cxl/bootstrap.py)), so a copy from
the pool to GPU HBM is a real async DMA with **no DRAM bounce buffer**. This is
the CXL adapter's key advantage over the pull-into-L1 NIXL peer path.

- **`gpu_src_view(key)`** ([`cxl_backend.py`](lmcache/v1/storage_backend/cxl_backend.py))
  resolves `(n_bytes, pool.base + chunk_offset)`. It does **not** bump
  `ref_count` — the resident path relies on the `pin_count` already held since
  `lookup_and_lock` to keep the slot alive for the DMA.
- **`submit_h2d` / `submit_h2d_batch`** issue `cudaMemcpyAsync` (via
  `lmc_ops.lmcache_memcpy_async`) for each chunk on the caller's current CUDA
  stream and return opaque tokens. The batch variant resolves all source views
  lock-free and assigns all tokens under a single critical section; the DMAs
  still pipeline on the stream.
- **`release_after_h2d` / `release_after_h2d_batch`** drop the pins after the
  stream drains (a stream-ordered host callback). The batch variant resolves all
  tokens under one lock, then `unpin_batch` — one arbiter sweep for the whole
  teardown.

---

## 8. Fast DRAM → CXL copy (`fast_copy`)

The donor push writes chunk bytes from local DRAM into the pool — **write-only**
traffic into device memory. A plain `memmove` uses regular stores that pull each
destination cacheline into cache first (read-for-ownership); on a real CXL device
this measured **~2 GB/s**. Non-temporal 32-byte streaming stores
(`_mm256_stream_si256` / `VMOVNTDQ`) bypass the cache and write straight through
— **~10.6 GB/s**, ~5×.

[`fast_copy_to_cxl`](lmcache/v1/storage_backend/cxl/fast_copy.py) uses the native
helper ([`_native/cxl_copy.c`](lmcache/v1/storage_backend/cxl/_native/cxl_copy.c),
compiled on demand, x86-only) when available and falls back to `ctypes.memmove`
otherwise — **correctness never depends on the optimization**. The copy ends
with `_mm_sfence()` so the weakly-ordered streaming stores are globally ordered
before any later fence the caller issues to publish the slot metadata.

---

## 9. Configuration surface

`CXLL2AdapterConfig`
([`cxl_l2_adapter.py`](lmcache/v1/distributed/l2_adapters/cxl_l2_adapter.py)). A
`--l2-adapter` value is one JSON object; `"type": "cxl"` selects this config
class from the registry.

```jsonc
{
  "type": "cxl",
  "dev_path": "/dev/dax0.0",          // the shared CXL DAX device (or a file for tests)
  "node_id": 0,                        // distinct per rack
  "chunk_size_bytes": 33554432,        // 32 MiB; must divide region_size
  "region_size": 268435456,            // 256 MiB; power-of-two, >= chunk_size_bytes
  "pool_size_override": 137438953472,  // 128 GiB cap on the mapping (optional)
  "max_nodes": 2,                      // shrinks the arbiter's lock-table sweep vs the 64 default

  // Bootstrap — exactly ONE node per rack sets each of these:
  "initialize": true,                  // node 0 only: writes a fresh header, zeroes the pool
  "run_lock_manager": true,            // one node only: runs the arbiter sidecar
  "generation": 2,                     // bump on controller restart to invalidate stale slots

  // Static-peer cross-node fetch (Alternative A — no controller):
  "peers": [ { "node_id": 1, "url": "tcp://NODE1_HOST:8447" } ],  // peer's CXLP2PServer
  "cxl_p2p_bind_url": "tcp://0.0.0.0:8447",  // our donor server bind
  "cxl_p2p_timeout_ms": 30000,               // requester ZMQ timeout (donor push can take seconds)

  // geom_hash inputs — MUST match across the rack:
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "world_size": 1, "kv_dtype_str": "torch.float16",
  "kv_shape": [32, 2, 256, 8, 128], "use_mla": false,
  "cluster_chunk_size": 256           // LMCache token-chunk size (distinct from chunk_size_bytes)
}
```

`build_cxl_adapter_from_config` builds the `CXLBackend`, and — only if `peers`
*and* an `l1_manager` are present — an `L1LocalCopyProvider`, a `CXLDonor`, a
`CXLP2PServer` at `cxl_p2p_bind_url`, and one `CXLP2PClient` per peer. With peers
but no `l1_manager` it warns and disables cross-node fetch.

---

## 10. Failure modes & GC

| What fails | Detection | Recovery |
|---|---|---|
| Chunk not in pool nor any peer | `pin_batch` miss + all peers `found=0` | Key stays a miss; retrieve recomputes. |
| Controller restart (stale generation) | donor `msg.epoch != header.gen` → `EPOCH_STALE`; slot re-check on `reserve` | Requester releases all reserved slots; stale-but-matching slots become `TOMB` candidates. |
| Heap out of regions mid-push | `alloc_batch` short prefix | `effective` capped → `PARTIAL`; requester retries tail on next peer. |
| Born-pinned slot not `VALID` at verify | requester's verify step | Requester unpins the orphaned pin so it doesn't leak. |
| A node dies holding regions / ALLOCATING slots | GC liveness oracle | [`gc.py`](lmcache/v1/storage_backend/cxl/gc.py) sweep (~30 s, rack-wide singleton): orphan dead-node regions, flip their ALLOCATING slots to `TOMB`, promote drained ORPHANED regions to `FREE`. |
| Lock-manager stalls / dies | waiter timeout | `LockAcquisitionTimeout` after 30 s. Run the arbiter as the C sidecar to avoid GIL starvation in the first place. |
| Eviction races a live reader/holder | `evict` checks `ref_count == 0 AND pin_count == 0` | Refuses to evict a busy/pinned slot. |

---

## 11. Reused / related components

- [`CXLBackend`](docs/design/v1/storage_backend/cxl_backend.md)
  ([source](lmcache/v1/storage_backend/cxl_backend.py)) — the synchronous pool
  backend (bootstrap, index reader/writer, heap, lock) this adapter shims.
- [`cross_node.py`](lmcache/v1/storage_backend/cxl/cross_node.py) /
  [`p2p_transport.py`](lmcache/v1/storage_backend/cxl/p2p_transport.py) — the
  donor/requester push protocol and ZMQ transport.
- The adapter skeleton (asyncio loop, task-id result dicts, three eventfds) is
  shared with
  [`mock_l2_adapter.py`](lmcache/v1/distributed/l2_adapters/mock_l2_adapter.py)
  and the [NIXL peer adapter](docs/design/v1/distributed/l2_adapters/nixl_rdma_peer.md).
- Controller contracts (store/lookup/load/unlock, eventfd discipline,
  `LazyStorePolicy`) are documented in
  [`overall.md`](docs/design/v1/distributed/l2_adapters/overall.md).
