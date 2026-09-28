# Per-Instance GPUDirect for the NIXL Peer Adapter

**Status:** implemented (PRs A–C of §5), with one deviation from the design:
`supports_l2_resident_retrieve()` stayed whole-adapter (all-or-nothing) rather
than per-instance — see §4.4. §1 describes the behavior *before* this change.

GPUDirect on the NIXL peer adapter assumes **one GPU staging buffer per MP
server**. That holds only for a single TP=1 vLLM instance. This doc specifies
the change to one GPU channel per registered instance.

Two deployments break it, and they are not the same shape:

- **One instance, TP>1.** Each vLLM worker registers its own buffer on its own
  device. Same model, same geometry, same buffer size — different pointers and
  devices.
- **Two independent instances on one node**, each TP=1, possibly serving
  *different models*. Different geometry, so different buffer sizes as well.

Both are the same fix, because the key (§3) is per-*worker-process*, not
per-rank. §5.1 covers what the second case adds.

Extends [`nixl_peer_gpudirect.md`](nixl_peer_gpudirect.md), which covers the
GPUDirect path itself (why `submit_h2d` blocks, the 2 MiB descriptor contract,
pin lifetime). Nothing there changes; this is about *how many* buffers exist.

The CXL adapter has no equivalent problem — see §6.

---

## 1. What actually happens today

Launching `TP=2 GPUDIRECT=1`:

```
Initialized cuda stream on device cuda:1
NIXL peer adapter: GPU staging buffer registered (ptr=0x7dfe82200000, ...)
Initialized cuda stream on device cuda:0
ERROR: L2 adapter 0 rejected the GPU staging buffer; ...
RuntimeError: GPU staging buffer is already registered
```

Two vLLM workers register two KV caches, so
[`register_kv_cache`](../../../../../lmcache/v1/multiprocess/modules/gpu_transfer.py)
runs twice and offers two staging buffers. The adapter holds a scalar
`self._gpu_channel` and raises on the second
([`nixl_peer_l2_adapter.py:1256-1257`](../../../../../lmcache/v1/distributed/l2_adapters/nixl_peer_l2_adapter.py)).

**The raise is swallowed.** `StorageManager.register_gpu_staging_buffer` catches
and logs it
([`storage_manager.py:775-783`](../../../../../lmcache/v1/distributed/storage_manager.py)),
because a GPUDirect failure must not take down KV-cache registration. So the
server comes up healthy and serves traffic — one rank on the GPU path, the rest
on DRAM. **The observable symptom is a silent halving of the optimization, not a
crash.**

### 1.1 The worse latent bug

`_NixlGpuReadChannel` stores a single `_buffer_base` and computes
`offset = gpu_ptr - self._buffer_base`
([`nixl_peer_l2_adapter.py:508`](../../../../../lmcache/v1/distributed/l2_adapters/nixl_peer_l2_adapter.py)).
Only `offset < 0` is rejected — there is no upper bound.

Today that is unreachable: the registration failure means rank 1 never reports
`supports_l2_resident_retrieve()`, so its pointers never arrive. But any fix that
accepts a second buffer without also making the base per-rank turns this into
**silent KV corruption**: a rank-1 pointer sitting above rank-0's base yields a
plausible in-range descriptor index into the *wrong device's* buffer, the READ
succeeds, and the scatter kernel writes another rank's bytes into the paged KV
cache. Adding an upper-bound check is worth doing regardless of the rest of this
design.

---

## 2. Why this is smaller than it looks

Three findings make the change requester-local.

**The donor has no GPU path at all.**
[`nixl_peer_donor.py`](../../../../../lmcache/v1/distributed/l2_adapters/nixl_peer_donor.py)
serves lookups out of its DRAM L1 and returns page indices into it. It cannot
tell a GPU read from a DRAM read, and needs no changes.

**GPUDirect is asymmetric.** Only the *local* side is VRAM: the requester READs
the peer's **DRAM L1** into its own GPU buffer. `_connect_peer_gpu_sync`
therefore handshakes the peer's `init_url` (DRAM agent), **not** its
`gpu_init_url` — documented at
[`nixl_peer_l2_adapter.py:1308-1322`](../../../../../lmcache/v1/distributed/l2_adapters/nixl_peer_l2_adapter.py).
`peers[].gpu_init_url` is never dialed; it is a capability flag, read only for
its non-emptiness at lines 1277 and 1339.

**No message carries a rank.** `RemoteLookupReq/Resp` and `RemoteUnlockReq/Resp`
have no rank field, and `_RemotePin` is built from the response and never sent.
So the pin can gain a rank field with no wire impact.

> **Trap:** `WireKey.kv_rank` is *not* a TP rank — it is `ObjectKey.kv_rank`, the
> KV-cache-role rank that is part of cache-key identity. Do not repurpose it.

---

## 3. The key: `instance_id`

`instance_id` is the registering vLLM worker's `os.getpid()`. It is the only
identifier available at **both** ends that is unique per worker process — which
is what makes it cover both deployment shapes with one mechanism:

| property | evidence |
|---|---|
| available at registration | `register_kv_cache(instance_id, ...)`, in scope at the `register_gpu_staging_buffer` call |
| available at retrieve | `retrieve(key, instance_id, ...)`, used to select the cache context |
| unique per TP rank | each TP worker is a separate process with its own PID |
| unique per instance | two independent vLLM servers are separate processes too, so the same property covers §4.4 without a second key |
| stable | it is the key of `_cache_contexts` for the registration's lifetime, matching the staging buffer's own "allocated once, never moved" guarantee |

**`worker_id` / `kv_rank` will not do.** Under MLA,
`extract_world_size_and_kv_rank` deliberately collapses TP ranks to the same
`kv_rank`, so two ranks would share a key. `instance_id` stays unique.

`cache_context.device` is arguably more semantically apt for a GPU registration —
an RDMA NIC cares about the device, not the PID — but it is derived rather than
declared, and `_cache_contexts` is not keyed by it. Use `instance_id` as the key
and carry `device` as payload (§4.2).

---

## 4. Design

### 4.1 Thread `instance_id` through four signatures

All four already have it in scope at the caller, or can:

| signature | change |
|---|---|
| `L2AdapterInterface.register_gpu_staging_buffer(gpu_ptr, size)` | `+ instance_id, device` |
| `StorageManager.register_gpu_staging_buffer(...)` | same, pass through |
| `StorageManager.submit_h2d_for_l2_resident(keys, adapter_indices, gpu_ptrs, sizes)` | `+ instance_id` |
| `L2AdapterInterface.submit_h2d_batch(keys, gpu_ptrs, dst_sizes)` | `+ instance_id` |
| `L2AdapterInterface.supports_l2_resident_retrieve()` | **unchanged** — stays parameterless; its caller has no worker identity (see §4.4) |

Also `unregister_kv_cache` needs an `unregister_gpu_staging_buffer(instance_id)`
counterpart, so a worker that goes away releases its NIXL agent rather than
leaking it for the process lifetime. The current code has no unregister at all —
acceptable for one buffer, not for N.

The base-class methods stay default no-ops, so the CXL adapter and every other
adapter are untouched.

### 4.2 One channel per instance

```python
self._gpu_channels: dict[int, GpuPeerDataChannel] = {}   # instance_id -> channel
```

- `register_gpu_staging_buffer` inserts; a duplicate `instance_id` with a
  *different* pointer is a real error (re-registration without unregister),
  while the same pointer is idempotent.
- `supports_l2_resident_retrieve()` stays whole-adapter: `True` only while
  every context that offered a buffer is registered. One rejected context
  disables GPUDirect for the whole server (see §4.4).
- `submit_h2d_batch` selects `self._gpu_channels[instance_id]` and fails that
  batch cleanly (`-1` tokens) if absent, rather than reaching for a global.
- `_NixlGpuReadChannel._buffer_base` is already per-channel, so it becomes
  correct automatically — **plus** an upper-bound check (§1.1).
- `close()` iterates the dict.

Grouping in `submit_h2d_batch` changes from `by peer` to `by peer` *within one
instance's channel* — the instance is fixed for the whole call, so this is a
lookup before the loop, not a second grouping dimension.

### 4.3 Per-instance NIXL identity

Two collisions to resolve, both in the requester's own process.

**The bind port.** Each `NixlChannel` binds a ZMQ REP socket on its
`peer_init_url`, so N ranks binding `gpu_init_bind_url` all hit `EADDRINUSE`
after the first.

The GPU channel's bound socket exists so a *peer* can dial in — but per §2 no
peer ever dials a `gpu_init_url`. **The GPU channel needs no inbound listener
at all.** If `NixlChannel` accepts `peer_init_url=None`, `_init_side_channels`
returns early and binds nothing, and the whole port question disappears. That is
the preferred fix; a rank-offset port range (`gpu_init_port + rank`) is the
fallback if some path still needs the listener.

**The agent id namespace.** `local_id` / `peer_id` are `f"node-{node_id}"`, and
the donor files remote xfer handlers under that key. Two ranks handshaking the
same peer would overwrite each other's handler. These become
`f"node-{node_id}-r{instance_id}"`.

This is the **only wire-visible change**, and it is compatible: both sides treat
the string as opaque, no msgspec schema changes, and it only widens a key
namespace. Distinct ids are in fact *required* — without them the donor's handler
dict silently aliases two ranks.

**`tp_rank` becomes the real device.** `NixlChannel` passes `tp_rank` as NIXL's
`dev_id` in the memory descriptor. It is currently `config.local_worker_id`,
which defaults to `0` for the whole MP server — so today a buffer on `cuda:1`
is registered claiming device 0. Pass the registering context's actual device
index instead. This is why §4.1 threads `device` alongside `instance_id`.

---

### 4.4 Two independent instances on one node

The second deployment shape — two separate vLLM servers on one node, each TP=1,
sharing one MP server — is covered by the same design, but it is worth being
explicit about what differs, because two of the differences are load-bearing.

**What is the same.** `instance_id` is `os.getpid()` of the registering worker,
so two independent instances get distinct ids exactly as two TP ranks do. The
channel dict, the per-instance `_buffer_base`, and the `instance_id`-threaded
signatures all work unchanged.

**What differs, and why it is fine:**

| | TP>1, one instance | Two independent instances |
|---|---|---|
| `model_name` | same for all ranks | **may differ** |
| geometry / buffer size | identical | **may differ** |
| `world_size` | same | may differ |
| device | one per rank | one per instance |

- **`ContextEntry` already carries per-instance `model_name` and `world_size`**,
  so nothing in the MP server assumes one model. This is the case the CXL work
  (`cxl_multi_tenant.md`) already had to solve at the storage layer.
- **Buffer sizes may differ.** `tmp_gpu_buffer_` is sized from *that instance's*
  geometry (`tmp_chunk_bytes_ * max_batch_size`). The 2 MiB alignment check in
  `gpu_channel_factory` is applied per buffer, so each instance registers or is
  rejected on its own merits — one model whose chunk bytes are not a 2 MiB
  multiple cannot disable GPUDirect for the other. Making that check per-buffer
  is automatic once the factory is called per instance; it is not extra work.
- **Pins cannot collide across models.** `_pins` is keyed by `ObjectKey`, which
  carries `model_name` and `cache_salt`, so two instances' pins are distinct even
  for identical content. No change needed.
- **`supports_l2_resident_retrieve()` becomes ambiguous.** With mixed instances,
  "can this adapter serve GPU-direct?" has no single answer — instance A may be
  registered while instance B was rejected for alignment. The
  `PrefetchController` reads it per request but not per instance.

That last point is the one genuinely new requirement, and it is why §4.1 threads
`instance_id` into the retrieve path rather than only into registration: a
per-instance answer needs a per-instance question. Two options:

1. **Keep the predicate coarse** (`bool(self._gpu_channels)`) and let
   `submit_h2d_batch` return `-1` for an unregistered instance. Simple, but a
   `-1` token is treated as an unrecoverable retrieve failure by the caller —
   which is wrong when the correct behavior is "use the DRAM path".
2. **Make the predicate per-instance** —
   `supports_l2_resident_retrieve(instance_id)` — so the controller routes
   instance B down the DRAM path before a resident plan is ever built.

**Neither option survived contact with the call site.** Option 2 assumed the
predicate's caller could supply an `instance_id`. It cannot: it is the
`PrefetchController`, reached from `lookup` on the vLLM **scheduler** adapter —
one per instance, no worker identity, and under TP it would need every rank's
answer rather than one. `register_kv_cache` and `retrieve` are on the **worker**
adapter and do have `instance_id`; `lookup` does not. The plan is built before
any worker is involved.

Option 1 (return `-1` and let the caller cope) is worse than it looks: a `-1`
token is treated as an *unrecoverable* retrieve failure, because the resident
path deliberately skipped the L1 load that would have been the fallback.

**What shipped is all-or-nothing.** `supports_l2_resident_retrieve()` stays
parameterless and answers `True` only while every registered context has a
channel. A context that fails to register latches `_gpu_direct_disabled` and
tears down the channels already built, so the server falls back uniformly and
says so loudly. This fixes both real deployment shapes — TP>1, and co-located
instances of the same model — and refuses cleanly rather than partially in the
mixed-geometry case.

Making it genuinely per-instance would mean carrying a worker identity from the
scheduler's lookup into the plan, or keeping an L1 plan alive as a shadow
fallback. Both are larger changes than the case currently justifies.

## 5. Delivery

**PR A — bound the offset (§1.1). DONE.** `_NixlGpuReadChannel` now takes
`buffer_size` and rejects a destination outside `[base, base + size)`, plus a
chunk that starts inside but runs past the end. The "already registered"
rejection logs a warning naming TP / multiple instances as the cause and stating
that the affected context falls back to DRAM — necessary because the caller
swallows the exception, so the message is the only signal an operator gets.

This converts the silent-corruption risk into a clean failure *before* the rest
lands, and is correct on its own even if PRs B and C never ship.

**PR B — thread `instance_id` (§4.1). DONE.** `register_gpu_staging_buffer`
gains `instance_id` and `device`; `submit_h2d` / `submit_h2d_batch` and
`submit_h2d_for_l2_resident` gain `instance_id`; a new
`unregister_gpu_staging_buffer` is called from `unregister_kv_cache`. CXL
ignores the new arguments (documented at its overrides) and is unaffected.

**PR C — per-instance channels (§4.2, §4.3, §4.4). DONE.** `_gpu_channels`
is a dict keyed by `instance_id`, with `_gpu_direct_disabled` latching the
all-or-nothing rule. `_Peer.gpu_connected` became a `set[int]` — each context
has its own NIXL agent and so handshakes each peer separately. The GPU channel
is now built with `peer_init_url=None` (no inbound listener, so no port
collision) and with the buffer's real device index as NIXL's `dev_id`.

**`supports_l2_resident_retrieve` stayed parameterless.** The design sketched
a per-instance predicate, but the prefetch controller that calls it runs on the
vLLM *scheduler* path and has no worker identity — see §4.4. The all-or-nothing
latch is what makes a parameterless answer safe.

---

## 6. Why CXL is unaffected

Worth stating, since the two adapters look similar here. CXL
`cudaHostRegister`s its whole pool once at bootstrap with
`cudaHostRegisterDefault`, which is portable across every device in the process.
`gpu_src_view` returns a *host* pointer, and the destination comes from whichever
caller's staging buffer — so there is no per-rank state to get wrong. This is
validated: TP=2 and two instances per node both work on CXL today.

The NIXL adapter differs because an RDMA NIC must have the destination
*registered*, and a registration is bound to one device and one address range.

---

## 7. Testing

- **Two registrations succeed.** Two `instance_id`s with distinct pointers both
  register and both report GPU-direct capable.
- **Routing.** A retrieve for instance A never reads into instance B's buffer —
  the regression test for §1.1. Inject two channels with known bases and assert
  the descriptor index derives from the *caller's* base.
- **Out-of-range destination is refused.** A pointer above a channel's buffer end
  raises rather than computing a plausible index (§1.1).
- **Unregister.** A worker that unregisters releases its channel; the remaining
  rank keeps serving.
- **Partial registration.** One rank registers, another fails: the first still
  serves GPU-direct, the second cleanly falls back to DRAM.
- **TP=1 unchanged.** The single-instance path behaves exactly as before.

For the two-independent-instance shape (§4.4):

- **Different models coexist.** Two instances registering different geometries
  both get GPU-direct, with different buffer sizes.
- **One rejected, one served.** An instance whose buffer fails the 2 MiB
  alignment check does not disable GPUDirect for the other, and its own
  retrieves take the DRAM path rather than failing.
- **All-or-nothing predicate.** `supports_l2_resident_retrieve()` answers `True`
  while every offered buffer is registered, and `False` for the whole adapter
  once any context is rejected — with the channels already built torn down.
