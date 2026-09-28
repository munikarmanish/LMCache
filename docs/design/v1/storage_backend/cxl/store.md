# CXLStore

`CXLStore` is the **synchronous, content-addressed KV object store over a CXL
shared-memory pool**. It is the engine that the
[CXL L2 adapter](docs/design/v1/distributed/l2_adapters/cxl_l2_adapter.md) wraps
with an async/eventfd shim.

It is deliberately **not** a `StorageBackendInterface`. It was originally
written as an in-process backend and only later wrapped as an L2 adapter; that
inherited base class was the sole reason `CacheEngineKey` — which cannot
represent `kv_rank` or `cache_salt` — ever appeared on the CXL path. The store
is now keyed on `ObjectKey` directly, and nothing registers it as an in-process
backend. See
[`cxl_multi_tenant.md`](docs/design/v1/distributed/l2_adapters/cxl_multi_tenant.md)
§5.1.

Where the L2 adapter is about *scheduling* (task ids, eventfds, the
controller-facing store/lookup/load/unlock contract), `CXLStore` is about
*mechanism*: it owns the mmap'd pool and turns a `(key, bytes)` into a committed,
addressable chunk in shared memory, and a `key` back into a copy of those bytes —
all under the rack-wide distributed lock.

Source:
[`cxl/store.py`](lmcache/v1/storage_backend/cxl/store.py). The primitives it
composes live under
[`lmcache/v1/storage_backend/cxl/`](lmcache/v1/storage_backend/cxl/).

> **Key type.** The store speaks `ObjectKey`. The shared CXL index addresses
> slots by a single u64, so the full identity — content hash, `model_name`,
> `kv_rank`, `cache_salt` — is folded into that u64 by
> `object_key_to_chunk_hash`. Anything folded out would alias in the shared
> pool; that is exactly the TP>1 bug the old `CacheEngineKey` bridge caused.

---

## 1. Responsibilities

`CXLStore` is the single owner, per MP-server process, of one CXL pool. It:

- **Bootstraps** the pool: mmap the `/dev/dax` device, read or write the header,
  and `cudaHostRegister` the whole mapping so chunks can DMA to GPU.
  `dev_path` may also be `/dev/interleaved_dax` (the interleaved_dax kernel
  module, which stripes pages across the CXL modules for ~1.7x bandwidth). That
  device is not on the DAX bus, so its size is derived from the module's
  `config` parameter (`interleaved_dax_capacity_bytes` in `bootstrap.py`, mirrored
  in `cxl_lock_manager.c`). The interleave changes the logical-to-physical page
  mapping, so **every node sharing a pool must use the same device type and the
  same module `config`**, and a pool written through one mapping must be
  re-initialized (bump `generation`) before use through the other.
  For `/dev/interleaved_dax` the bootstrap also runs
  `madvise(MADV_POPULATE_WRITE)` over the mapping **before** `cudaHostRegister`
  (`PagePopulatePolicy.AUTO`). The device can only map 4 KiB pages, and
  `cudaHostRegister` pins them without marking the PTEs accessed/dirty, so the
  first CPU write to each page otherwise costs ~0.9 µs with no page fault. A
  cross-node donor push always writes never-touched chunks, so it ran at
  ~9 GB/s instead of ~40 GB/s. Populating is startup-neutral: the pass costs
  about what registration then saves by finding the PTEs present (~27 s total
  for 128 GiB either way). It is best-effort; failure only logs a warning.
- **Wires the subsystems** that operate on the pool (§3) and hands them a single
  shared distributed lock.
- **Runs the store path** (`put_batch` → `_put_one`): reserve a
  slot, allocate a chunk, copy bytes, commit VALID.
- **Runs the read paths**: `read_into` (memcpy into a caller-owned buffer)
  and `gpu_src_view` (a pointer for a
  GPU-direct DMA).
- **Manages slot liveness**: `pin`/`unpin` (eviction protection across a
  lookup→use window) and `ref_count` (protection across an in-flight DMA), plus
  their batched variants.
- **Separates tenants**: every slot carries a 16-byte tenant digest derived from
  its `ObjectKey` (plus the model's geometry salt, set through `set_geometry`),
  so one pool serves many models, TP degrees, and cache salts at once.

It deliberately does **not** emit `KVAdmitMsg`/`KVEvictMsg` to the cache
controller: CXL residency is discovered by *looking in the shared index*, not by
broadcasting local-tier admit/evict events.

---

## 2. Where it sits

```
        StorageManager
             │  L2AdapterInterface (store / lookup-and-lock / load / unlock)
             ▼
        CXLL2Adapter     ── async/eventfd shim, controller contract
             │  put_batch / read_into / gpu_src_view / pin / remove
             ▼
     ┌──────────────────────────────────────────────────────┐
     │ CXLStore                                              │
     │   bootstrap → PoolHandle (mmap + header + hostReg)    │
     │   TwoTierLock  ── shared by all subsystems            │
     │   RegionAllocator → HeapSet (one NodeHeap per size)   │
     │   CXLIndex (lock-free reads) / CXLIndexWriter (writes)│
     │   key→slot cache, node-local LRU                      │
     └───────────────┬──────────────────────────────────────┘
                     │ (same pool, index_writer and heaps shared with)
        CXLDonor     ┘   ── cross-node PushKVToCXL
```

The L2 adapter is the store's only caller; nothing registers it as an in-process
backend. On the donor side, the cross-node push handler reserves/commits slots
and allocates chunks through the same `index_writer` and `heaps`. Everything goes
through the one `TwoTierLock`, so concurrent local puts, cross-node commits, and
pins are mutually consistent.

---

## 3. Composed subsystems

`CXLStore.__init__` runs the full bootstrap and wires these, in order:

| Subsystem | Source | Role |
|---|---|---|
| Bootstrap → `PoolHandle` | [`bootstrap.py`](lmcache/v1/storage_backend/cxl/bootstrap.py) | mmap the DAX device, validate/write the header (magic, layout version, `gen`, sizing), `cudaHostRegister` the mapping. The header carries no model identity. |
| `TwoTierLock` | [`locks.py`](lmcache/v1/storage_backend/cxl/locks.py) | rack-wide sharded lock; one shared instance for every writer. |
| Lock-manager arbiter | [`lock_manager_proc.py`](lmcache/v1/storage_backend/cxl/lock_manager_proc.py) (C sidecar) / [`lock_manager.py`](lmcache/v1/storage_backend/cxl/lock_manager.py) (Python fallback) | the single writer that flips `WAITING → LOCKED`. Run on **one** node. |
| `RegionAllocator` → `HeapSet` → `NodeHeap` | [`regions.py`](lmcache/v1/storage_backend/cxl/regions.py), [`heap_set.py`](lmcache/v1/storage_backend/cxl/heap_set.py), [`heap.py`](lmcache/v1/storage_backend/cxl/heap.py) | claim coarse regions and carve them into chunks. `HeapSet` keeps one `NodeHeap` per distinct chunk byte size, created on first store, and routes a `free(offset)` back to the class owning that region. Sizes are exact-fit; the only waste is a region tail smaller than one chunk. |
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

`put_batch` completes **synchronously** and inline (there is no async DMA yet on
the store side); it loops `_put_one` per key and swallows per-key failures so
one bad key can't fail the batch. Each `_put_one` is a full INSERT lifecycle:

1. **Reserve a slot** (`reserve_slot(chunk_hash, tenant_digest)`, with bounded
   retries on `WAIT_FOR_OTHER`). `ALREADY_PRESENT` → cache the slot and return
   (dedup); `INDEX_FULL` → drop the put and log.
2. **Allocate a chunk** of exactly the payload's size from that size's heap
   class, evicting local cold chunks on region exhaustion — see §4.1. There is
   no configured chunk size; the optional `max_chunk_size_bytes` is only a guard
   against a garbled geometry, and a size larger than `region_size` fails loudly.
3. **Copy** `src.raw_data` into the chunk.
4. **Commit** the slot VALID (`commit_slot`) and cache the key→slot mapping. The
   commit also refreshes the slot's node-local LRU recency (§4.1).

On any failure after reservation, it **rolls back**: free the chunk and release
the slot back to EMPTY, so a partial insert never leaves a stranded ALLOCATING
slot or a leaked chunk.

### 4.1 Node-local LRU eviction (`_alloc_chunk_with_eviction`)

The CXL pool is shared across nodes, but the backend deliberately keeps **no
global LRU** — maintaining cross-node recency would require a CXL write and a
cross-node fence on every read, the exact cost the read path is built to avoid.
Instead each node keeps a **DRAM-only** recency map (`NodeLRUTracker`, keyed by
`slot_idx`) over the slots *it* owns, and evicts its own cold chunks when it can
no longer grow.

**The allocation ladder** (chunk allocation, step 3 above):

1. **`heaps.alloc(size)`** — serve from that size class's free-list, else **claim a new region**
   from the global pool. Claiming is always preferred; while the pool has FREE
   regions, no eviction ever happens (eviction costs nothing until the pool is
   full).
2. **On `NoRegionAvailable`** (the global pool is exhausted — this node's region
   count is now fixed) — evict this node's coldest chunks back into the
   free-list via `_evict_cold_slots`, then retry with
   **`heaps.alloc_no_claim(size)`** (free-list only, never claims). The freed
   slot is reused **in place**; no region is trimmed or reclaimed. The retry is
   scoped to the requested size class: a freed chunk of another size does not
   satisfy it.
3. **If eviction frees nothing** (every cold slot is pinned or has an in-flight
   read), the retry raises `OutOfChunks` and the store is dropped (logged) —
   today's exhaustion behavior, now only reached when the node genuinely cannot
   make room.

**Watermark and victim order.** `_evict_cold_slots` drains toward a node-local
occupancy floor: `occupied / total` owned slots, where `total = owned_regions ×
slots_per_region`, summed over the size classes — DRAM counts from
`HeapSet.occupancy()`, no CXL read.
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

**Batch admission.** `put_batch` runs one eviction pass for the
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

Both start with a **lock-free** `CXLIndex.lookup_by_hash(chunk_hash,
tenant_digest)` (readers never take the distributed lock). A slot matches only
if both the u64 hash and the 16-byte tenant digest agree, so another model's,
rank's, or geometry's chunk is a miss rather than a misread. The chunk is then
protected for the duration of the use:

- **`read_into(key, dst_ptr, dst_size)`** — the fast hit path used by L2 load.
  `ref_count_up` → re-verify the slot didn't get evicted between lookup and pin →
  `ctypes.memmove` from `pool.base + chunk_offset` into the caller's buffer →
  `ref_count_down` in `finally`. It skips building a `MemoryObj` wrapper (the
  numpy/torch/ctypes wrapping dominated the per-chunk cost) and can record an
  11-slot per-phase ns timing breakdown for profiling.
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

- **`ref_count`** — held for the duration of a single read/DMA (`read_into`).
  Short-lived, taken and dropped inside one call.
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
2. **`heaps.free_batch(freed_offsets)`** — each offset goes back to the
   free-list of the size class owning its region, so the regions holding them
   become fully free.
3. **`heaps.trim()`** — every fully-empty region, in every size class, is
   released to the pool.

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

Constructed from a `CXLStoreConfig` alone — no model metadata. The config
mirrors the adapter-facing fields (see the
[CXL adapter doc §9](docs/design/v1/distributed/l2_adapters/cxl_l2_adapter.md) for
the JSON surface): `dev_path`, `node_id`, `region_size`, `max_chunk_size_bytes`
(optional guard), `initialize`, `generation`, `run_lock_manager`,
`use_process_lock_manager`, `pool_size_override`, `max_nodes`, `num_locks`,
`evict_low_watermark` (§4.1; the node-local occupancy floor eviction drains
toward, default 0.8, range (0.0, 1.0]).

Model geometry arrives later, from the live model: when the serving engine
registers its KV caches, the adapter's `register_layout` calls
`set_geometry(model_name, geometry_salt)`, and the salt is folded into the
tenant digest of that model's chunks. Two nodes running one model under
different geometry therefore miss each other's chunks instead of misreading
them. Re-declaring a *different* geometry for a model in the same process
raises `ValueError`.

- After `__init__` the pool is mapped, host-registered, the lock manager is
  running (if enabled), and the backend serves put/get immediately.
- **`initialize`** (one node only) writes a fresh header and bumps `generation`;
  every other node re-attaches to the existing pool. `close()` does **not**
  re-initialize — a node restart re-attaches to the header it finds, so a warm
  pool survives an MP-server bounce.
- **`close()`** stops the lock manager and unmaps the pool.

---

## 8. Failure & safety properties

| Concern | How CXLStore handles it |
|---|---|
| Partial insert (crash/exception mid-`_put_one`) | Rollback frees the chunk and releases the slot; no stranded ALLOCATING slot or leaked chunk. |
| Reader races an evictor | `ref_count_up` + re-verify slot identity; eviction refuses non-zero `ref_count`/`pin_count`. |
| Duplicate put of the same key | `reserve_slot` returns `ALREADY_PRESENT`; the put is a no-op. |
| Two tenants collide on the u64 index hash | The 16-byte tenant digest differs, so the probe skips the slot; each tenant gets its own. |
| Same model, different geometry across nodes | Geometry salt makes the tenant digests differ; the nodes miss rather than misread (§7). |
| Index full | `_put_one` drops the put and logs (no crash). |
| My regions full, pool has FREE regions | `heaps.alloc()` claims a new region; no eviction (§4.1). |
| Pool globally exhausted | Evict this node's coldest chunks and reuse a freed slot in place (§4.1). |
| Pool exhausted + every cold slot pinned | Store (or whole batch) is dropped and logged; pinned chunks stay resident. |
| Payload larger than `max_chunk_size_bytes` or `region_size` | `ValueError`, slot released. |
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
