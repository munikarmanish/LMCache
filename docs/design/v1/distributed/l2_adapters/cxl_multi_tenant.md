# CXL as a Multi-Tenant KV Object Store

**Status:** implemented (PRs 1–5 of §5). Sections written in the future tense
describe the change as designed; §5 records what shipped.

Today one CXL pool serves exactly one `(model, TP, dtype, chunk_size)` tuple,
fixed at process start by hand-written JSON. This doc specifies the change to a
**content-addressed object store**: one pool serving many vLLM instances, many
models, and any TP degree concurrently, with the key — not the process config —
carrying tenant identity.

It supersedes the configuration surface described in §9 of
[`cxl_l2_adapter.md`](cxl_l2_adapter.md); everything else in that doc (pool
layout, locking, cross-node push, GPU-direct retrieve) is unchanged.

---

## 1. Motivation

Three problems share one root cause.

**A latent TP>1 corruption.** `_object_key_to_cache_engine_key`
([`cxl_l2_adapter.py:325-343`](../../../../../lmcache/v1/distributed/l2_adapters/cxl_l2_adapter.py))
takes `model_name` / `world_size` / `worker_id` from the adapter's *static
config* and keeps only `chunk_hash` from the incoming key. `ObjectKey.kv_rank`
and `cache_salt` are discarded. But `ipc_key_to_object_keys`
([`api.py:231-260`](../../../../../lmcache/v1/distributed/api.py)) fans one
lookup out to one `ObjectKey` per TP rank that differ *only* in `kv_rank` — the
token hash carries no rank. Under TP=2 both ranks collide on one index slot:
rank 1's store dedupes onto rank 0's, and a rank-1 retrieve is served rank 0's
shard. Silent KV corruption. Every other L2 adapter embeds `kv_rank` in the
storage key (e.g. `fs_l2_adapter.py:118`,
`s3_l2_adapter.py:77`); CXL is the outlier. Dropping `cache_salt` likewise
defeats per-user isolation
([`l2_per_user_quota.md`](l2_per_user_quota.md)).

**One model per pool.** `geom_hash` digests
`(model_name, world_size, kv_dtype, kv_shape, use_mla, chunk_size)` into a
pool-wide identity: a mismatch is a hard `RuntimeError` at attach
([`bootstrap.py:410-416`](../../../../../lmcache/v1/storage_backend/cxl/bootstrap.py)).
Two models cannot share a pool even though the on-media format is a byte blob.

**A single chunk size.** `chunk_size_bytes` is one fixed slab. Measured per-chunk
KV bytes at `chunk_size: 256` tokens:

| Model | TP=1 | TP=2 |
|---|---|---|
| Llama-3.1-8B | 32.00 MiB | 16.00 MiB |
| Llama-3.1-70B | 80.00 MiB | 40.00 MiB |
| Qwen3-32B | 64.00 MiB | 32.00 MiB |
| Qwen3-Next (12 full-attn layers) | 6.00 MiB | 3.00 MiB |

The deployed 32 MiB slab is exact for Llama-8B TP=1 and wrong for every other
row: Qwen3-32B TP=1 is **rejected at put**
([`cxl/store.py:462-465`](../../../../../lmcache/v1/storage_backend/cxl/store.py)),
and Qwen3-Next wastes 81% of every slot.

**The enabling observation:** none of this is a *format* constraint. The on-media
layout is already tenant-agnostic. See §2.

---

## 2. What the format already supports

Three facts make this a Python-layer change, not a format migration.

**`chunk_size_bytes` is absent from the header and the layout.**
`PoolLayout.compute` takes only `pool_size`, `region_size`, `num_locks`,
`max_nodes` ([`layout.py:255-260`](../../../../../lmcache/v1/storage_backend/cxl/layout.py)),
and `Header` ([`layout.py:89-111`](../../../../../lmcache/v1/storage_backend/cxl/layout.py))
has no chunk-size field. It is a *heap allocation-granularity* choice, held only
in process memory. Every slot already carries its own `chunk_offset` and
`chunk_len`.

**The read path is already model-agnostic.** `_build_memory_obj`
([`cxl/store.py:685-715`](../../../../../lmcache/v1/storage_backend/cxl/store.py))
returns a 1-D `uint8` `MemoryObj` sized to `chunk_len`, with the source comment:

> We don't know the caller's intended (shape, dtype) at get time — the slot only
> carries the format and a byte length. Return a 1-D uint8 MemoryObj sized to
> chunk_len; callers reinterpret via a GPUConnector that knows the KV geometry.

That is object-store semantics already. The bytes are never interpreted by the
backend.

**`geom_hash` is already a per-slot field.** `SlotLine0.geom_hash`
([`layout.py:177`](../../../../../lmcache/v1/storage_backend/cxl/layout.py)) is
stamped per insert. It is merely *compared against the header's* value
([`index.py:211`](../../../../../lmcache/v1/storage_backend/cxl/index.py)) rather
than against the requesting key's. The discriminator exists; it is wired as a
global assertion.

`SlotLine0` also has 16 spare padding bytes
(`_pad: c_uint8 * (CACHELINE_SIZE - 48)`), so a widened key discriminator fits
without growing the slot or touching `line1`.

---

## 3. Design

### 3.1 Tenant identity moves into the key

Define the **tenant tuple** as the fields that must never alias:

```
(model_name, world_size, kv_rank, cache_salt)
```

`kv_rank` already packs `world_size | global_rank | local_world_size |
local_rank` ([`api.py:104-152`](../../../../../lmcache/v1/distributed/api.py)),
so it distinguishes TP ranks and TP degrees. `model_name` separates models;
`cache_salt` separates users.

Replace `_object_key_to_chunk_hash` with a derivation over the whole `ObjectKey`:

```python
def _object_key_to_index_hash(key: ObjectKey) -> int:
    """Derive the 64-bit index hash from the full tenant identity."""
    h = blake2b(digest_size=8)
    h.update(key.chunk_hash)
    h.update(b"\x00")
    h.update(key.model_name.encode("utf-8"))
    h.update(b"\x00")
    h.update(key.kv_rank.to_bytes(8, "little"))
    h.update(b"\x00")
    h.update(key.cache_salt.encode("utf-8"))
    return int.from_bytes(h.digest(), "little")
```

The `\x00` separators matter: `ObjectKey.__post_init__` forbids `@` but permits
other characters in `model_name`, and NUL is rejected in `cache_salt`, so
unambiguous framing needs an explicit delimiter rather than concatenation.

This alone fixes the TP>1 corruption, and it does so the way every other adapter
already does. **It is independently correct and ships first** (§5, PR 1).

#### The donor path must carry ObjectKey identity

Making the index hash one-way breaks an assumption the cross-node push relied
on: `cxl_l1_donor._cek_chunk_hash_to_objkey_bytes` *inverted* the old u64 back
into `ObjectKey.chunk_hash` bytes so the donor could look up its L1 copy, which
is keyed on the full `ObjectKey`. A digest cannot be inverted, so without a
matching change the donor would reconstruct garbage, every `reserve_read` would
miss, and cross-node push would degrade **silently** to a permanent `ALL_NACK`.

The root cause is that `PushKVToCXLMsg.keys` carries
`CacheEngineKey.to_string()`, which structurally cannot represent an
`ObjectKey` — no `kv_rank`, no `cache_salt`, and an integer `chunk_hash`. The
donor was reconstructing identity it was never sent.

PR 1 therefore also sends the real identity: `PushKVToCXLMsg` gains
`obj_chunk_hashes` / `obj_model_names` / `obj_kv_ranks` / `obj_cache_salts`
(one entry per key), `remote_fetch` takes a parallel `obj_keys` list, and
`LocalCopyProvider.__call__` takes those four fields instead of a `key_str`.
The serialized `CacheEngineKey` still identifies the *CXL slot*; the new fields
identify the *donor's L1 entry*.

This also repairs a **pre-existing** bug the old helper documented but did not
solve: the inversion was byte-exact only for 8-byte hashes, so blake3/sha256
deployments (32-byte digests, truncated on the way in) already had a donor that
could never reconstruct a key. Raw `chunk_hash` bytes are now carried intact at
any length.

It is a P2P wire change, so both nodes must upgrade together — acceptable
because PR 1 invalidates pool contents anyway (§6).

#### Collision handling

A 64-bit index hash over a larger key space raises birthday-collision exposure.
Two distinct tenants hashing to one `chunk_hash` would today produce a false
hit, because the probe compares only the u64
([`index.py:186-188`](../../../../../lmcache/v1/storage_backend/cxl/index.py)).

Mitigation: store a **128-bit tenant discriminator** in the slot and compare it
on hit. Reuse the existing `geom_hash` field (§3.2) — it is already 16 bytes,
already per-slot, and already compared in the probe loop. At 2^20 resident
chunks the residual false-hit probability across both fields is ~2^-88, which is
negligible against silent corruption.

### 3.2 `geom_hash` becomes a per-key tenant digest

Redefine the 16-byte field from *pool-wide geometry* to *this chunk's tenant
digest*:

```python
def object_key_to_tenant_digest(key: ObjectKey) -> bytes:
    """16-byte digest of tenant identity."""
```

covering the same tenant tuple as the u64 index hash, at 16 bytes instead of 8.
Then change three comparison sites:

| Site | Today | Proposed |
|---|---|---|
| [`index.py:211`](../../../../../lmcache/v1/storage_backend/cxl/index.py) | `snap_geom != self._header_geom_hash` → miss | `snap_geom != requested_digest` → miss |
| [`index_writer.py:275-276`](../../../../../lmcache/v1/storage_backend/cxl/index_writer.py) | reuse if slot geom == header geom | reuse if slot geom == the digest being written |
| [`bootstrap.py:410-416`](../../../../../lmcache/v1/storage_backend/cxl/bootstrap.py) | attach fails on geom mismatch | attach checks `magic`, `layout_version`, sizing only |

The probe already treats a geom mismatch as `_PROBE_CONTINUE` — a fall-through to
the next slot, not an error — so open addressing handles two tenants colliding on
a `chunk_hash` correctly with **no change to the probe's control flow**. This is
why the change is small.

`header.geom_hash` becomes unused. Keep the field (it is fixed-size padding-free
in a versioned header) and zero it; do not renumber the struct.

#### Why the digest covers identity, not geometry

An earlier draft had the digest cover tenant identity **plus** KV geometry
(dtype, shapes), so a rack whose nodes disagreed about a model's geometry could
not exchange misinterpreted bytes. That is not implementable as specified,
because the two paths hold different information:

- **Store** has geometry — the `MemoryObj` carries `shape`, `dtype`, `fmt`.
- **Lookup** has only keys. `submit_lookup_and_lock_task(keys)` takes no
  `layout_desc`, and neither do `submit_unlock`, `submit_h2d`, or
  `release_after_h2d`.

A lookup therefore cannot recompute a geometry-bearing digest, so a store/lookup
pair would never match. Closing that gap requires plumbing geometry to the
adapter — a `register_layout(...)` hook on `L2AdapterInterface`, fed from
`gpu_transfer.register_kv_cache` — which is a larger, separable change.

The digest is therefore **tenant identity only**, and the two concerns are split:

| Concern | Where it lives |
|---|---|
| *Who* owns this chunk (model, TP rank, user) | Per slot, `line0.geom_hash` |
| *Whether this pool is readable at all* (format, sizing) | Header: `magic`, `layout_version`, `region_size`, `num_locks`, `max_nodes` |

**This gap is closed by PR 5 (§3.5).** As shipped in PR 3 it was a real
regression: a rack with the same `model_name` but a different dtype was no
longer rejected at attach, and two such nodes would exchange misinterpreted
bytes.

An earlier draft of this section claimed PR 4 would narrow it, on the theory
that registration-derived class sizes would make a geometry disagreement show up
as a size-class mismatch. **That turned out to be wrong.** PR 4 creates classes
from *stored bytes* rather than *registered geometry*, so a dtype mismatch
yields the same chunk size and no mismatch is observable. Only PR 5 closes it.

Removing the header-level check also removes bootstrap's last two uses of
`LMCacheMetadata`: `compute_geom_hash` (moved to a per-key digest, above) and
`_validate_sizing` (§3.4). **`bootstrap_pool` therefore stops taking
`LMCacheMetadata` at all** — the concrete mechanism by which pool creation is
decoupled from model identity.

**Removed from the config surface:** `model_name`, `world_size`, `kv_dtype_str`,
`kv_shape`, `use_mla`, `cluster_chunk_size`, `worker_id`, `local_world_size`,
`local_worker_id` — all nine. A chunk's tenant is derived from its own
`ObjectKey`, so nothing about the model is declared in JSON or kept in sync
across the rack. This removes the stale-`model_name` hazard, where launching a
different model against an unchanged config yielded an unchanged digest and
served stale KV.

It also removes the flat-`kv_shape` limitation outright, rather than by
replacing it: nothing in the pool describes KV geometry any more, so
hybrid-attention models (Qwen3-Next: 48 layers, `full_attention_interval: 4`, so
12 full-attention + 36 gated-delta-net layers) need no representation the
5-tuple could not express.

### 3.3 Per-geometry heap classes

Replace the single `NodeHeap` with a small set keyed by size class. Each claimed
region is dedicated to one class and carved into equal slots, preserving the
existing O(1) free-list with no coalescing — `heap.py`'s core invariant is
untouched. `free(offset)` maps an offset back to its region
([`heap.py:205-217`](../../../../../lmcache/v1/storage_backend/cxl/heap.py)), and
since a region belongs to exactly one class, the existing offset→region lookup
resolves the class with no extra slot state. The heap docstring already
anticipates this: *"Multiple sizes are served by multiple heaps."*

**Classes are exact per-geometry sizes, not power-of-two buckets.** The token
chunk count is uniform across every client of one MP server: `chunk_size` is a
single server-level value (`EngineContext._chunk_size`, from `mp_config`), and
the vLLM client *queries* the server for it rather than proposing its own —
`get_lmcache_chunk_size` sends `GET_CHUNK_SIZE` and takes what comes back,
asserting only that it is a multiple of its own block size
([`vllm_multi_process_adapter.py:215-231, 551-560`](../../../../../lmcache/integration/vllm/vllm_multi_process_adapter.py)).
Per-chunk bytes is therefore a pure function of the registered geometry, and the
set of live sizes is exactly the set of distinct registered geometries — known
at registration time.

Key heaps by **computed byte size**, not by model name: two models that happen to
share a geometry (Llama-3.1-8B TP=2 and Qwen3-32B TP=2 both land at 32 MiB, §1)
then share one class safely, since the tenant digest (§3.2) keeps their contents
distinct. This bounds the heap count by distinct sizes rather than distinct
models. Creation is driven by `LayoutDescRegistry`, which is already ref-counted
and keyed `(model_name, world_size)` and already retains a descriptor until the
last matching registration drops
([`engine_context.py:45-88`](../../../../../lmcache/v1/multiprocess/engine_context.py)) —
so class lifecycle comes for free.

Exact-fit classes eliminate internal fragmentation entirely, against
power-of-two's 2× worst case (Qwen3-Next TP=2 at 3 MiB would otherwise round to
a 4 MiB class). The cost moves to two other places, both cheaper:

- **Region tail waste.** `slots_per_region = region_size // class_bytes`, with
  the remainder unused, replacing today's `region_size % chunk_size_bytes == 0`
  validation ([`cxl_l2_adapter.py:165-168`](../../../../../lmcache/v1/distributed/l2_adapters/cxl_l2_adapter.py)).
  Bounded by one slot per region — 4 MiB of a 256 MiB region for a 6 MiB class,
  ~1.6%. Rounding classes up to divide `region_size` evenly would reintroduce
  exactly the internal fragmentation exact-fit exists to remove; take the tail.
- **Cross-class eviction does not help the triggering store.** Eviction drops
  the globally coldest chunks by LRU, which may belong to another class; the
  retry allocates from its own class's free-list, so it still fails. Evicting by
  recency across the node remains the right global policy — the freed regions
  become claimable once `trim()` returns them — but the store that triggered
  eviction may be dropped and only succeed on a later attempt. Making eviction
  class-targeted would mean evicting warm chunks over cold ones purely because
  of size, which is worse.
- **Region-level fragmentation.** A region belongs to one class, so a model that
  registers, fills regions, then unregisters strands them in its class.
  `heap.trim()` already sheds fully-empty regions; this makes it load-bearing.
  Trim when the last registration for a class drops.

### 3.4 `chunk_size_bytes` is removed entirely

Not made optional — removed. Its four uses each dissolve:

| Use | Disposition |
|---|---|
| Heap slab size ([`cxl/store.py:211`](../../../../../lmcache/v1/storage_backend/cxl/store.py)) | Derived per class from registered geometry (§3.3). |
| Max-size check at put ([`cxl/store.py:462-465`](../../../../../lmcache/v1/storage_backend/cxl/store.py)) | Vestigial under exact-fit — payload *is* the class size. Keep as an internal invariant against the selected class. |
| Read buffer size ([`cxl/store.py:693-695`](../../../../../lmcache/v1/storage_backend/cxl/store.py)) | Map `view.chunk_len`. Today it maps `chunk_size_bytes` while setting `shape = [chunk_len]` — an over-read that maps past the end of a smaller slot. |
| `lmcache_memcpy_async` trailing arg ([`cxl_l2_adapter.py:801, 873`](../../../../../lmcache/v1/distributed/l2_adapters/cxl_l2_adapter.py)) | Pass the slot's own `chunk_len` (already in hand from `gpu_src_view`); the copy length is separately `min(n_bytes, dst_size)`. |

`min_chunks_per_region` and `_validate_sizing`
([`bootstrap.py:81, 420-438`](../../../../../lmcache/v1/storage_backend/cxl/bootstrap.py))
go with it. That guard is unreachable today: it runs only when
`initialize=True`, returns early when `chunk_bytes <= 0` (which the default
`kv_shape=(0,0,0,0,0)` produces), defaults to `1` — a "can this hold one chunk"
test, not an amortization floor — and is not exposed in `CXLL2AdapterConfig`, so
no deployment can set it.

Its *intent* — keep region claims amortized against the DRAM free-list — is worth
preserving, and moves to class creation where the numbers are exact and every
node evaluates them:

- `slots_per_region == 0` → **hard error**. Reachable today: Llama-3.1-70B TP=1
  needs 80 MiB against a 256 MiB region is fine, but a smaller `region_size`
  would not be.
- `slots_per_region` small (< 4) → **warning**, not an error. A large-chunk model
  on a modest region is functional, just heavier on region-lock traffic.

**Nothing replaces it in config.** An optional `max_chunk_size_bytes` *cap*
remains — a safety bound so a garbled geometry fails at store time instead of
claiming an absurd slab — but it defaults to unset and is not a per-deployment
tuning value.

**As implemented, two details differ from the sketch above.**

*Classes are created at store time, not at registration.* The exact chunk size
is already on the `MemoryObj` being stored (`get_size()`), so `HeapSet` creates
a class on first use of a size and needs no `LayoutDescRegistry` plumbing at
all. This is strictly simpler than the registry-driven design, and it means the
class set reflects what is genuinely resident rather than what registered. The
consequence is that class *teardown* is not registration-driven either: a class
whose regions all empty is reclaimed by `trim()`, which is already the existing
mechanism.

*The `slots_per_region == 0` hard error lives in `NodeHeap`, not `HeapSet`* —
it is the same condition as "chunk larger than a region", so it is raised where
the slab is carved rather than duplicated in the multiplexer.

After §3.2 and §3.4, `cxl.base.json` contains only device and topology facts —
`dev_path`, `region_size`, `pool_size_override`, `num_locks`, `max_nodes`,
`generation`, plus node identity and peers. **Nothing in it changes when you
switch models or TP degree**, which was the goal the whole sequence was aimed
at.

Region-level accounting stays per-node, so `owner_node_id`, GC, and the
cross-node donor path are unaffected. `alloc_batch`'s single-lock-hold property
([`cxl_l2_adapter.md` §5](cxl_l2_adapter.md)) is preserved per class; a donor
push whose keys span classes takes one hold per class rather than one overall,
which is the only hot-path regression and is bounded by the class count.

---

### 3.5 Geometry agreement across the rack

`register_layout(model_name, world_size, layout_desc)` is added to
`L2AdapterInterface` — a default no-op, exactly mirroring the existing
`register_gpu_staging_buffer` — fanned out by `StorageManager` and called from
`gpu_transfer.register_kv_cache`. That is the first point the true geometry is
known, and it comes from the live model rather than from config.

The CXL adapter digests the geometry (per-group shapes and dtypes, plus
`world_size`) into a 16-byte salt and declares it via `CXLStore.set_geometry`.
The salt is folded into the tenant digest of every chunk for that model, so:

- Nodes that **agree** on geometry compute identical digests and share chunks as
  before.
- A node that **disagrees** computes different digests, so it never matches the
  other's slots. It **misses** rather than misreading — each node still serves
  its own traffic correctly; they simply stop sharing.

This is deliberately not a hard error. Refusing to start would take down an
engine over a condition only one side can see, and the cache is recomputable;
degrading to "no sharing" is the safe direction.

`remote_fetch` and `ControllerBackedFetch` take a `tenant_digest_fn` so slots
reserved on the requester's behalf carry *its* geometry — reserving under the
bare identity digest would reopen the hole from the cross-node path.

The salt defaults to empty, so a deployment whose writers never declare geometry
behaves exactly as it did before: identity separation never depended on it.

## 4. What stays pool-wide

Baked into the header at init and still requiring rack-wide agreement — all
*device/deployment* properties, none *model* properties:

- `region_size`, `num_locks`, `max_nodes`, `pool_size`, section offsets.
- `generation` — the restart epoch, semantics unchanged.
- `layout_version` — bumped by this change (§6).

Adding a model, changing TP, or attaching another vLLM instance requires
touching none of them.

### Pool scope vs. attacher scope

Worth stating precisely, because it is what makes §4's non-goal cheap.
`CXLBootstrapConfig` ([`bootstrap.py:57-82`](../../../../../lmcache/v1/storage_backend/cxl/bootstrap.py))
**does not take `node_id`** — the only occurrence in that module is
`descs[i].owner_node_id = OWNER_FREE`, initializing every region as unowned.
Bootstrap describes the *pool*, a shared node-independent artifact.

`node_id` is *attacher* identity and enters only afterward, in
`CXLBackend.__init__`: as the column index in `global_lock[lock_id][node_id]`
(bounds-checked against `header.max_nodes` read back from the pool), and as the
`owner_node_id` stamp on claimed regions and reserved slots. This is exactly why
`max_nodes` is a header field and `node_id` is not: the pool reserves lock-table
columns for N possible attachers without caring which ones appear, or when.

### Non-goals

- **No change to the node-identity model.** `node_id` remains 1:1 with a
  physical node. Two MP-server processes sharing a `node_id` corrupt the lock
  table — `TwoTierLock`'s local tier is a per-process `threading.Lock`
  ([`locks.py:109`](../../../../../lmcache/v1/storage_backend/cxl/locks.py)),
  so two processes on one row both enter the critical section and one's release
  frees the other's lock. Multiple vLLM instances on a node therefore attach to
  **one MP server**, which is already multi-instance capable: it keys GPU
  contexts by `instance_id` with per-entry `model_name`/`device`
  ([`gpu_transfer.py:142`](../../../../../lmcache/v1/multiprocess/modules/gpu_transfer.py)),
  and `cudaHostRegisterDefault` pins the pool portably across all devices in the
  process. Per-instance `node_id`s would need `max_nodes` raised — header-baked,
  so a re-init that wipes the pool — plus distinct P2P ports and a longer arbiter
  sweep.
- **No cross-tenant sharing.** Two models with byte-identical KV never dedupe;
  the tenant digest deliberately separates them.
- **No per-tenant quota.** Eviction stays global LRU. Per-tenant fairness is
  [`l2_per_user_quota.md`](l2_per_user_quota.md)'s concern.

---

## 5. Delivery

Three PRs, each independently correct and shippable.

**PR 1 — key derivation (§3.1).** Fold `kv_rank`, `model_name`, `cache_salt`
into the index hash. Fixes the TP>1 corruption and restores per-user isolation.
No format change, no config change; a pool written by the old code is invalidated
by bumping `generation`. *Prerequisite for any TP>1 run.*

**PR 2 — re-key the store on `ObjectKey` (§5.1).** A pure refactor with no
behavior or format change: `cxl_backend.py` becomes `cxl/store.py`, `CXLStore`
becomes `CXLStore`, the `AllocatorBackendInterface` base and ~250 lines of
in-process surface are deleted, and the `CacheEngineKey` bridge goes with them.
Makes PR 3 small, because `ObjectKey` carries the tenant identity right at the
comparison site.

**PR 3 — per-key tenant digest (§3.2).** Move the comparison from header to key,
relax the attach check to compatibility only, drop all nine model/TP config
fields and `LMCacheMetadata` from `bootstrap_pool` (and from `CXLStore` and
`L1LocalCopyProvider`, where it was already vestigial). Enables multi-model.
Bumps `layout_version` to 2. The digest covers tenant identity only — see
"Why the digest covers identity, not geometry" in §3.2, and its accepted risk.

`_validate_sizing` and `min_chunks_per_region` are removed here rather than in
PR 4: they were `bootstrap_pool`'s only other use of `LMCacheMetadata`, so the
parameter could not be dropped while they remained. They were already a no-op
(see §3.4).

**PR 4 — per-geometry heap classes (§3.3, §3.4).** Removes `chunk_size_bytes`
(replaced by an optional `max_chunk_size_bytes` cap) and adds `HeapSet`, which
multiplexes one `NodeHeap` per distinct byte size and routes `free(offset)` back
to the owning class. `NodeHeap` no longer requires `chunk_size` to divide
`region_size`; the tail is left unused. The donor allocates per size class, so a
push batch spanning geometries still takes one lock hold per distinct size.
Enables mixed chunk sizes in one pool.

The read-path over-read in §3.4's table needed no work: `_build_memory_obj` was
deleted with `get_blocking` in PR 2. The `lmcache_memcpy_async` trailing
argument now passes the chunk's own length.

**PR 5 — geometry agreement (§3.5).** Adds the `register_layout` hook and folds
a per-model geometry salt into the tenant digest, closing the gap PR 3 opened. It
is additive: nothing in PRs 1–4 forecloses or depends on it, and a deployment
that declares no geometry is unaffected.

PR 1 and PR 2 are independent of the rest and worth landing on their own.

### 5.1 Why the store is re-keyed, not adapted

`CXLStore` extended `AllocatorBackendInterface` because it was first written
as an in-process `StorageBackend` and only later wrapped as an L2 adapter. That
base class is the *sole* reason `CacheEngineKey` — which cannot represent
`kv_rank` or `cache_salt` — appeared in the CXL path at all. Nothing registers
the store in any storage-backend factory; its only production consumer is
`CXLL2Adapter`.

The coupling turned out to be seven lines, all using `.chunk_hash` as a dict
key. Every other method took the key and immediately handed it to
`index.lookup`, which itself called `_lookup_by_hash(key.chunk_hash)`. So the
re-key is mechanical, and the index/index-writer now take the u64 directly —
more honest than passing a key object they only destructure.

What the deletion removes, all with zero production callers: `get_blocking` and
its `_materialize` / `_SlotRefcountAdapter` chain (the L2 path uses `read_into`
and `gpu_src_view`), `exists_in_put_tasks` and its in-flight bookkeeping, the
four allocator methods plus `cxl/allocator.py` itself, the three `async`
wrappers, and `batched_contains`. `dst_device` goes too — the store never
touched it.

The adapter's four private reaches (`_pool`, `_node_id`, `_index_writer`,
`_heap`) become public accessors, satisfying the SLF rule in
`docs/coding_standards.md` §2. `scripts/cxl_cross_host_test.py` is ported to
`read_into`.

---

## 6. Compatibility

PR 1 and PR 2 both change what a given key resolves to, so **existing pool
contents are invalidated**. There is no migration path and none is warranted —
the pool is a cache; the KV is recomputable.

PR 3 bumps `layout_version` to 2, so an old reader attaching to a new pool fails
cleanly at the version check rather than misinterpreting slots.

**The C arbiter must be bumped in the same change.**
`_native/cxl_lock_manager.c` hardcodes `LAYOUT_VERSION` and refuses to attach on
a mismatch — but a refused arbiter *exits silently* while Python waiters spin
forever on a lock nobody grants. The observed symptom is a segfault deep in
`locks._wait_for_locked`, not a clean error, because the sidecar is gone and the
mapping is stale. `ProcessLockManager` rebuilds on source mtime, so touching the
`.c` is enough — but the constant must actually be changed. Deploy by
stopping all nodes, bumping `generation`, and restarting with `initialize: true`
on the bootstrap node.

---

## 7. Testing

Correctness obligations beyond the existing suite:

- **TP aliasing.** Two `ObjectKey`s differing only in `kv_rank` must map to
  distinct slots; a rank-1 retrieve must never return rank-0 bytes. This is the
  regression test for §1's corruption and belongs in PR 1.
- **Tenant isolation.** Same for `model_name` and for `cache_salt`.
- **Forced hash collision.** Inject two tenants with a colliding u64 and assert
  the digest comparison rejects the wrong one and open addressing finds the
  right one — the §3.1 collision path, which production traffic will not
  exercise.
- **Mixed-tenant pool.** Two models with different chunk sizes, stored and
  retrieved interleaved; assert no cross-tenant hit and no slot reuse across
  digests.
- **Size-class free/realloc.** Free a chunk in class A, allocate in class B,
  assert the class-A region is not corrupted and `free()` routes correctly.
- **Attach compatibility.** A node configured for a different model must now
  attach successfully (was `RuntimeError`) and simply not hit the other tenant's
  slots.
