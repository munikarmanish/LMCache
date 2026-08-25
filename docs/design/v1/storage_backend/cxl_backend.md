# CXLBackend

`CXLBackend` is the **synchronous storage backend over a CXL shared-memory
pool**. It implements `AllocatorBackendInterface` so it plugs into LMCache's
`StorageManager` like any other backend, and it is the engine that the
[CXL L2 adapter](docs/design/v1/distributed/l2_adapters/cxl_l2_adapter.md) wraps
with an async/eventfd shim.

Where the L2 adapter is about *scheduling* (task ids, eventfds, the
controller-facing store/lookup/load/unlock contract), `CXLBackend` is about
*mechanism*: it owns the mmap'd pool and turns a `(key, bytes)` into a committed,
addressable chunk in shared memory, and a `key` back into a copy of those bytes —
all under the rack-wide distributed lock.

Source:
[`cxl_backend.py`](lmcache/v1/storage_backend/cxl_backend.py). The primitives it
composes live under
[`lmcache/v1/storage_backend/cxl/`](lmcache/v1/storage_backend/cxl/).

> **Doc-vs-code note.** The module docstring still calls this a "single-process
> skeleton for step 5" with cross-node push "not yet wired." That prose is stale:
> the cross-node donor
> ([`cross_node.py`](lmcache/v1/storage_backend/cxl/cross_node.py)) drives this
> backend's index writer and heap, and the GPU-direct and batched-lock paths are
> live. This doc describes the backend as it actually is.

---

## 1. Responsibilities

`CXLBackend` is the single owner, per MP-server process, of one CXL pool. It:

- **Bootstraps** the pool: mmap the `/dev/dax` device, read or write the header,
  and `cudaHostRegister` the whole mapping so chunks can DMA to GPU.
- **Wires the subsystems** that operate on the pool (§3) and hands them a single
  shared distributed lock.
- **Runs the store path** (`batched_submit_put_task` → `_put_one`): reserve a
  slot, allocate a chunk, copy bytes, commit VALID.
- **Runs the read paths**: `read_into` (fast memcpy into a caller buffer),
  `get_blocking` (materialize a `MemoryObj`), and `gpu_src_view` (a pointer for a
  GPU-direct DMA).
- **Manages slot liveness**: `pin`/`unpin` (eviction protection across a
  lookup→use window) and `ref_count` (protection across an in-flight DMA), plus
  their batched variants.
- **Exposes an allocator** (`get_memory_allocator`) so the pool can also serve as
  a `MemoryAllocatorInterface` target.

It deliberately does **not** emit `KVAdmitMsg`/`KVEvictMsg` to the cache
controller: CXL residency is discovered by *looking in the shared index*, not by
broadcasting local-tier admit/evict events.

---

## 2. Where it sits

```
        StorageManager
             │  put / get / pin / remove  (AllocatorBackendInterface)
             ▼
     ┌──────────────────────────────────────────────────────┐
     │ CXLBackend                                            │
     │   bootstrap → PoolHandle (mmap + header + hostReg)    │
     │   TwoTierLock  ── shared by all subsystems ──┐        │
     │   RegionAllocator → NodeHeap → CXLMemoryAllocator     │
     │   CXLIndex (lock-free reads) / CXLIndexWriter (writes)│
     │   key→slot cache, in-flight-put set                   │
     └───────────────┬──────────────────────────────────────┘
                     │ (same object, shared with)
        CXLL2Adapter ┘   ── async/eventfd shim, controller contract
        CXLDonor         ── cross-node PushKVToCXL uses the index_writer + heap
```

The same `CXLBackend` instance is shared by the L2 adapter (which schedules
operations onto it) and, on the donor side, by the cross-node push handler (which
reserves/commits slots and allocates chunks through it). All three go through the
one `TwoTierLock`, so concurrent local puts, cross-node commits, and pins are
mutually consistent.

---

## 3. Composed subsystems

`CXLBackend.__init__` runs the full bootstrap and wires these, in order:

| Subsystem | Source | Role |
|---|---|---|
| Bootstrap → `PoolHandle` | [`bootstrap.py`](lmcache/v1/storage_backend/cxl/bootstrap.py) | mmap the DAX device, validate/write the header (magic, `gen`, `geom_hash`), `cudaHostRegister` the mapping. |
| `TwoTierLock` | [`locks.py`](lmcache/v1/storage_backend/cxl/locks.py) | rack-wide sharded lock; one shared instance for every writer. |
| Lock-manager arbiter | [`lock_manager_proc.py`](lmcache/v1/storage_backend/cxl/lock_manager_proc.py) (C sidecar) / [`lock_manager.py`](lmcache/v1/storage_backend/cxl/lock_manager.py) (Python fallback) | the single writer that flips `WAITING → LOCKED`. Run on **one** node. |
| `RegionAllocator` → `NodeHeap` | [`regions.py`](lmcache/v1/storage_backend/cxl/regions.py), [`heap.py`](lmcache/v1/storage_backend/cxl/heap.py) | claim coarse regions, carve them into fixed `chunk_size` chunks, hand out pool-relative offsets. |
| `CXLMemoryAllocator` | [`allocator.py`](lmcache/v1/storage_backend/cxl/allocator.py) | expose the heap as a `MemoryAllocatorInterface`. |
| `CXLIndex` / `CXLIndexWriter` | [`index.py`](lmcache/v1/storage_backend/cxl/index.py), [`index_writer.py`](lmcache/v1/storage_backend/cxl/index_writer.py) | lock-free hash-index reads; lock-protected reserve/commit/evict/pin. |

The pool layout (header / index slots / heap / lock rows) and the slot state
machine (`EMPTY → ALLOCATING → VALID → TOMB`, `ref_count`/`pin_count`) are
documented in the
[CXL adapter design doc §3](docs/design/v1/distributed/l2_adapters/cxl_l2_adapter.md)
and in [`layout.py`](lmcache/v1/storage_backend/cxl/layout.py); this doc does not
repeat them.

**Lock-manager placement.** The arbiter is run out-of-process by default
(`use_process_lock_manager=True`) because the in-process Python arbiter is starved
of the GIL under donor-commit load (its sweep time inflated ~20×). Exactly one
node per rack should `run_lock_manager`; a compiler-less environment falls back to
the Python thread.

---

## 4. The store path (`_put_one`)

`batched_submit_put_task` completes **synchronously** and inline (there is no
async DMA yet on the store side); it loops `_put_one` per key and swallows
per-key failures so one bad key can't fail the batch. Each `_put_one` is a full
INSERT lifecycle:

1. **Mark in-flight** (so `exists_in_put_tasks` answers correctly mid-insert).
2. **Reserve a slot** (`reserve_slot`, with bounded retries on
   `WAIT_FOR_OTHER`). `ALREADY_PRESENT` → cache the slot and return (dedup);
   `INDEX_FULL` → drop the put and log.
3. **Allocate a chunk** from the heap (reject payloads larger than
   `chunk_size_bytes`), evicting local cold chunks on region exhaustion — see
   §4.1.
4. **Copy** `src.raw_data` into the chunk.
5. **Commit** the slot VALID (`commit_slot`) and cache the key→slot mapping. The
   commit also refreshes the slot's node-local LRU recency (§4.1).

On any failure after reservation, it **rolls back**: free the chunk and release
the slot back to EMPTY, so a partial insert never leaves a stranded ALLOCATING
slot or a leaked chunk. The `finally` clears the in-flight mark.

### 4.1 Node-local LRU eviction (`_alloc_chunk_with_eviction`)

The CXL pool is shared across nodes, but the backend deliberately keeps **no
global LRU** — maintaining cross-node recency would require a CXL write and a
cross-node fence on every read, the exact cost the read path is built to avoid.
Instead each node keeps a **DRAM-only** recency map (`NodeLRUTracker`, keyed by
`slot_idx`) over the slots *it* owns, and evicts its own cold chunks when it can
no longer grow.

**The allocation ladder** (chunk allocation, step 3 above):

1. **`heap.alloc()`** — serve from the free-list, else **claim a new region**
   from the global pool. Claiming is always preferred; while the pool has FREE
   regions, no eviction ever happens (eviction costs nothing until the pool is
   full).
2. **On `NoRegionAvailable`** (the global pool is exhausted — this node's region
   count is now fixed) — evict this node's coldest chunks back into the
   free-list via `_evict_cold_slots`, then retry with **`heap.alloc_no_claim()`**
   (free-list only, never claims). The freed slot is reused **in place**; no
   region is trimmed or reclaimed.
3. **If eviction frees nothing** (every cold slot is pinned or has an in-flight
   read), the retry raises `OutOfChunks` and the store is dropped (logged) —
   today's exhaustion behavior, now only reached when the node genuinely cannot
   make room.

**Watermark and victim order.** `_evict_cold_slots` drains toward a node-local
occupancy floor: `occupied / total` owned slots, where `total = owned_regions ×
slots_per_region` — both DRAM counts from `NodeHeap.occupancy()`, no CXL read.
The floor is `evict_low_watermark` (default 0.8); eviction frees
`max(store_shortfall, occupied − floor)` chunks, so it drops a batch down to the
floor rather than one-at-a-time (avoiding a re-trigger on the very next store),
but a store needing more than that evicts past the floor to the last
non-pinned chunk. Victims come oldest-first from the LRU; a pinned / in-flight
one is refused by `index_writer.evict` (the §6 guard) and skipped.

**Why the trigger is exhaustion, not a usage threshold.** In-region reuse frees
a slot without releasing a region, so a *claimed-region* watermark could never
fall — it would trigger forever. Gating on `NoRegionAvailable` instead means the
region count is fixed at the moment eviction runs, so the occupancy ratio is a
quantity eviction can actually lower, and the feedback loop converges.

**Recency updates (true LRU).** `_cache_slot` — called on every local
resolution (read hit, pin, store commit) — refreshes the slot's recency, so a
chunk kept warm by reads survives even if it was stored first. `forget` runs on
`remove`/`evict`; `clear` drops the whole map. A chunk hot on *another* node
reads as cold here; that approximation is accepted (the goal is only a sensible
local victim order, not global optimality).

**Batch admission.** `batched_submit_put_task` runs one eviction pass for the
whole batch up front (`_ensure_batch_space`): if the pool is exhausted and
eviction cannot free room for every chunk in the batch, the **entire batch is
dropped** rather than storing a partial prefix. The per-chunk ladder above still
guards the single-put path.

**Scope.** A node evicts only chunks in regions it owns (matching `clear`'s
ownership discipline). It never touches donor slots or other nodes' regions, so
a node hoarding regions is not reclaimed by this path — that is the dead-node
GC's job (`cxl/gc.py`), which is separate.

The cross-node donor path is the *batched* analogue of this lifecycle —
reserve-all → copy-all (NT streaming stores) → commit-all under one lock hold,
optionally born-pinned — described in the
[CXL adapter doc §5](docs/design/v1/distributed/l2_adapters/cxl_l2_adapter.md).

---

## 5. The read paths

All three start with a **lock-free** `CXLIndex.lookup(key)` (readers never take
the distributed lock), then protect the chunk for the duration of the use:

- **`read_into(key, dst_ptr, dst_size)`** — the fast hit path used by L2 load.
  `ref_count_up` → re-verify the slot didn't get evicted between lookup and pin →
  `ctypes.memmove` from `pool.base + chunk_offset` into the caller's buffer →
  `ref_count_down` in `finally`. It skips building a `MemoryObj` wrapper (the
  numpy/torch/ctypes wrapping dominated the per-chunk cost) and can record an
  11-slot per-phase ns timing breakdown for profiling.
- **`get_blocking(key)`** — same protect-and-verify, but returns a materialized
  `MemoryObj` view over the chunk (for callers that want a Python-level tensor).
- **`gpu_src_view(key)`** — returns `(n_bytes, pool.base + chunk_offset)` for a
  GPU-direct `cudaMemcpyAsync`. It does **not** bump `ref_count`; the
  GPU-direct path instead relies on the `pin_count` held since the caller's
  lookup-and-lock. This is what enables the L2-resident retrieve (CXL → GPU with
  no DRAM bounce), see the
  [CXL adapter doc §7](docs/design/v1/distributed/l2_adapters/cxl_l2_adapter.md).

The `ref_count`-then-re-verify dance is the core race guard: eviction flips a
slot to TOMB **only if `ref_count == 0 AND pin_count == 0`**, so a reader that
successfully bumps `ref_count` and then re-confirms the slot identity is
guaranteed the bytes stay put for the copy.

---

## 6. Liveness: pins vs ref-counts

Two independent counters on each slot keep it alive for two different windows:

- **`ref_count`** — held for the duration of a single read/DMA (`read_into`,
  `get_blocking`). Short-lived, taken and dropped inside one call.
- **`pin_count`** — held across a *lookup → later use* window (the "lock" in the
  adapter's lookup-and-lock). `pin`/`unpin` and their batched forms
  (`pin_batch`/`unpin_batch`) bump/drop it; the batched forms resolve N keys under
  a single distributed-lock acquisition (≈ one arbiter sweep) instead of one per
  key — the dominant warm-path cost at long prompts.

`remove(key)` and eviction both refuse a slot while either counter is non-zero.

### 6.1 Bulk clear (`clear`)

`clear()` is the whole-node counterpart of `remove(key)`: it deletes **all of
this node's chunks** and returns their now-empty regions to the global pool. It
is the reset used between benchmark arms (e.g. `scripts/cxl/clear_cache.sh`) and
wherever a node must drop its entire CXL residency without a restart.

Three steps, layering the same primitives `remove` uses:

1. **`index_writer.clear_owned_slots()`** — one bulk fence-before-read primes the
   slot array, then every `VALID` slot whose `owner_node_id` is this node is
   flipped to `TOMB` under its slot lock, and its `chunk_offset` is collected.
   Returns `(freed_offsets, skipped_busy)`.
2. **`heap.free_batch(freed_offsets)`** — the offsets go back to the node heap's
   free-list, so the regions holding them become fully free.
3. **`heap.trim()`** — every fully-empty region is released to the pool.

Two invariants inherited from the single-key path:

- **Busy chunks are skipped, never force-freed.** A slot with `ref_count > 0`
  (in-flight read/GPU copy) or `pin_count > 0` stays `VALID`; it is counted in
  `slots_skipped_busy`. A region still holding one will not trim. This preserves
  the reader-vs-evictor guard (§6) — a concurrent read is never invalidated.
- **Scope is this node only.** Donor slots owned by other nodes live in the same
  shared index but their offsets belong to other heaps; `clear` leaves them
  `VALID` and frees nothing on their behalf.

Returns a `ClearResult(chunks_deleted, slots_skipped_busy, regions_released)`.

---

## 7. Configuration & lifecycle

Constructed from a `CXLBackendConfig` + `LMCacheMetadata`. The config mirrors the
adapter-facing fields (see the
[CXL adapter doc §9](docs/design/v1/distributed/l2_adapters/cxl_l2_adapter.md) for
the JSON surface): `dev_path`, `node_id`, `chunk_size_bytes`, `region_size`,
`initialize`, `generation`, `run_lock_manager`, `use_process_lock_manager`,
`pool_size_override`, `max_nodes`, `evict_low_watermark` (§4.1; the node-local
occupancy floor eviction drains toward, default 0.8, range (0.0, 1.0]).

- After `__init__` the pool is mapped, host-registered, the lock manager is
  running (if enabled), and the backend serves put/get immediately.
- **`initialize`** (one node only) writes a fresh header and bumps `generation`;
  every other node re-attaches to the existing pool. `close()` does **not**
  re-initialize — a node restart re-attaches to the header it finds, so a warm
  pool survives an MP-server bounce.
- **`close()`** stops the lock manager and unmaps the pool.

---

## 8. Failure & safety properties

| Concern | How CXLBackend handles it |
|---|---|
| Partial insert (crash/exception mid-`_put_one`) | Rollback frees the chunk and releases the slot; no stranded ALLOCATING slot or leaked chunk. |
| Reader races an evictor | `ref_count_up` + re-verify slot identity; eviction refuses non-zero `ref_count`/`pin_count`. |
| Duplicate put of the same key | `reserve_slot` returns `ALREADY_PRESENT`; the put is a no-op. |
| Index full | `_put_one` drops the put and logs (no crash). |
| My regions full, pool has FREE regions | `heap.alloc()` claims a new region; no eviction (§4.1). |
| Pool globally exhausted | Evict this node's coldest chunks and reuse a freed slot in place (§4.1). |
| Pool exhausted + every cold slot pinned | Store (or whole batch) is dropped and logged; pinned chunks stay resident. |
| Payload larger than a chunk | `ValueError`, slot released. |
| Lock-manager GIL starvation | Run the arbiter as the C sidecar (default). |
| Stale generation after controller restart | Bump `generation` on init; the shared header's `gen` is the epoch checked by the cross-node push and slot reservation. |

---

## 9. Related docs

- [CXL L2 adapter](docs/design/v1/distributed/l2_adapters/cxl_l2_adapter.md) —
  the async/eventfd shim, the controller contract, the cross-node push protocol,
  batched locking, GPU-direct retrieve, and the pool-layout / slot-state details.
- [L2 controller overview](docs/design/v1/distributed/l2_adapters/overall.md) —
  how `StorageManager` and the store/prefetch controllers drive a backend.
- [`layout.py`](lmcache/v1/storage_backend/cxl/layout.py) — the on-device pool
  layout this backend reads and writes.
