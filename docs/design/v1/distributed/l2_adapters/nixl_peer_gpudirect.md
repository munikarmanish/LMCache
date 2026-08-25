# GPUDirect for the RDMA/NIXL Peer L2 Adapter

**Status:** implemented, opt-in (`enable_gpu_direct`, default off).
**Validated on two-node RDMA hardware (2026-08-25): 1.43× on the cross-node
path** — 30k-token TTFT 0.407 s → 0.284 s, with the stage-level attribution in
§11. An earlier run that appeared to show 3.3× was an artifact of two bugs
(both fixed); see the postmortem in §10.
**Scope:** give the [NIXL peer adapter](docs/design/v1/distributed/l2_adapters/nixl_rdma_peer.md)
an L2-resident retrieve path — a one-sided RDMA READ that lands **directly in
GPU memory** — so a remote hit skips the L1/DRAM bounce, the way the
[CXL adapter](docs/design/v1/distributed/l2_adapters/cxl_l2_adapter.md) §7 skips it.

---

## 1. What we are removing

Today a remote hit costs two hops through DRAM:

```
peer L1 (DRAM)  --RDMA READ-->  our L1 (DRAM)  --cudaMemcpyAsync-->  GPU staging  --kernel-->  paged KV
                 ^ submit_load_task              ^ retrieve _retrieve_loop
```

With GPUDirect the READ targets the GPU staging buffer directly:

```
peer L1 (DRAM)  --RDMA READ (GPUDirect)-->  GPU staging  --kernel-->  paged KV
```

That removes the local DRAM landing **and** the H2D copy, and it removes the
L1 write-reserve for those keys (no L1 capacity consumed, no L1 eviction
pressure from remote pulls).

The NIC still reads the peer's DRAM on the far side — this is one-sided
GPUDirect RDMA on the *requester's* side only. Making the donor serve from
its own GPU is out of scope.

---

## 2. The one hard problem: RDMA completion is not stream-ordered

This is the crux of the design, and the reason this is **not** a drop-in reuse
of the CXL `submit_h2d` path.

The existing L2-resident contract
([`base.py:403`](lmcache/v1/distributed/l2_adapters/base.py#L403)) is:

> `submit_h2d` enqueues the copy **on the caller's current CUDA stream** and
> returns immediately without synchronizing.

The retrieve loop
([`gpu_transfer.py:719-784`](lmcache/v1/multiprocess/modules/gpu_transfer.py#L719-L784))
depends on exactly that. Per batch it:

1. calls `submit_h2d_for_l2_resident(...)` to fill the staging slots, then
2. immediately enqueues `multi_layer_block_kv_transfer` on the **same stream**,
   which reads those staging slots.

For CXL this is safe purely by **stream ordering**: the `cudaMemcpyAsync` and
the scatter kernel are on one stream, so the kernel cannot start before the
DMA retires. Nothing on the CPU has to wait.

**A NIXL RDMA READ has no such ordering.** It is issued to the NIC, not to the
CUDA stream, and its completion is observed by CPU-side polling
(`agent.check_xfer_state(handle)` — see
[`nixl_peer_l2_adapter.py:336-342`](lmcache/v1/distributed/l2_adapters/nixl_peer_l2_adapter.py#L336-L342)).
The CUDA stream knows nothing about it. If we merely *issue* the READ inside
`submit_h2d` and return, the scatter kernel is free to run against a staging
buffer the NIC has not finished writing — **silent KV corruption**, and
exactly the class of bug the design doc already records once (the
"3.3 GB READ completed in 1.6 ms with corrupt tails" incident, §3 of the
peer doc).

So the RDMA-to-GPU transfer **must be CPU-completed before the scatter kernel
for that batch is enqueued.** There is no way around this without a
NIC↔stream signalling mechanism — which, as §9 shows, the installed NIXL
cannot express at any layer reachable from Python.

### 2.1 The resolution: `submit_h2d` blocks until the READ completes

`NixlPeerL2Adapter.submit_h2d_batch` will **issue the READ and poll it to
DONE before returning**. It is synchronous with respect to the RDMA, while
still honoring the interface contract as written — the contract says the
method must not *synchronize the CUDA stream*, and it does not; it never
touches the stream at all.

The cost is that the retrieve thread blocks for the RDMA duration instead of
overlapping it with the scatter kernel. This is an acceptable and, in fact,
largely unavoidable trade:

- It is still **strictly faster than today**, which pays the same RDMA
  latency during `submit_load_task` *plus* an L1 write-reserve *plus* a full
  DRAM→GPU H2D copy.
- The scatter kernel for batch *N* overlaps the RDMA of batch *N+1* anyway,
  because the loop enqueues the scatter (async) and then immediately issues
  the next batch's READ. So the pipeline is not serialized end-to-end — only
  the first batch's RDMA is fully exposed.

This blocking behavior is a **real semantic difference from CXL** and must be
documented on the method, in the design doc, and called out in review. An
adapter whose `submit_h2d` blocks for milliseconds is a legitimate
implementation of the interface, but a surprising one.

### 2.2 Rejected alternative: keep it async, sync at end of loop

We could issue all READs non-blocking and poll them all after the loop. This
does **not** work: the scatter kernels are enqueued *inside* the loop, one per
batch, each reading its batch's staging slots. Deferring completion past the
loop means every scatter kernel races its data. Fixing it would require
restructuring `_retrieve_loop` into two passes (all fills, then all scatters),
which (a) needs `max_batch_size` distinct staging slots live simultaneously —
the buffer is sized exactly for that, so it is possible, but (b) it changes a
shared hot path used by CXL and the L1 bounce path, for the benefit of one
adapter. Out of scope for this change; revisit only if profiling shows the
exposed first-batch RDMA actually matters.

---

## 3. The second problem: NIXL registers exactly one buffer per agent

`NixlAgentWrapper.__init__`
([`nixl_channel.py:634-707`](lmcache/v1/transfer_channel/nixl_channel.py#L634-L707))
registers **one** `(buffer_ptr, buffer_size)` region with **one** `mem_type`,
derived from a single `device` string: `"cpu"` → `cpu`, `"cuda"` → `VRAM`.
The peer adapter builds its channel over the L1 DRAM buffer with
`device="cpu"`
([`nixl_peer_l2_adapter.py:1186-1197`](lmcache/v1/distributed/l2_adapters/nixl_peer_l2_adapter.py#L1186-L1197)).

There is no way to add a second (GPU) region to that agent through the current
wrapper. Two options:

**Option A — a second `NixlChannel` over the GPU staging buffer (recommended).**
Build a second channel with `device="cuda"`, `buffer_ptr =
tmp_gpu_buffer_.data_ptr()`, `align_bytes = 2 MiB`, its own `init_bind_url`,
and its own handshake to each peer. The adapter then holds two data channels:
the existing DRAM one (still used by `submit_load_task`, which we keep as a
fallback) and a GPU one used by `submit_h2d_batch`.

> **The GPU channel handshakes the peer's *L1* agent, not its GPU agent.**
> GPUDirect here is **asymmetric**: only the local side is GPU memory — we
> READ the peer's DRAM L1 into our GPU staging buffer. The peer's
> `gpu_init_url` is therefore *not* a handshake target; it exists only so a
> peer can advertise that it also has GPUDirect enabled.
>
> Getting this wrong is silent and expensive (it cost a full bogus benchmark
> — see §10). Connecting the two GPU agents makes the remote dlist describe
> the peer's *staging buffer* (64 descriptors for 128 MiB), while the donor
> reports page indices into its 32 GiB L1 (up to ~16k). NIXL then rejects
> every index past the staging buffer:
> `makeXferReq: remote index out of range at index 0 with value 1872` →
> `NIXL_ERR_INVALID_PARAM`.

Cost: a second NIXL agent, a second handshake per peer, a second init port.
Benefit: **zero changes to `NixlChannel`/`NixlAgentWrapper`**, which are shared
with `pd_backend` and the disagg-prefill path. Given the peer adapter already
had to wrap the channel rather than change it (`_NixlReadChannel`), this stays
consistent with how the module has handled channel limitations so far.

**Option B — extend `NixlAgentWrapper` to register N regions.** Cleaner
long-term, but it changes a shared, hardware-validated component used by
`pd_backend`, `nixl_storage_backend`, and the disagg tests. The descriptor
index space becomes segmented (region 0's descriptors then region 1's), which
silently changes index arithmetic for every existing caller.

**Recommendation: Option A.** Take Option B only as a follow-up PR with its own
review, if a third registered region is ever needed.

### 3.1 The GPU staging buffer must be reachable from the adapter

`tmp_gpu_buffer_` lives on `GPUCacheContext`
([`gpu_context.py:144-148`](lmcache/v1/multiprocess/gpu_context.py#L144-L148)) —
note **not** `PlainGPUCacheContext` (same file, line 472), which is a distinct
class with no `get_tmp_gpu_buffer_flat` and no `max_batch_size`; only
`GPUCacheContext` feeds the retrieve path this change targets —
a single persistent `torch.empty(tmp_chunk_bytes * max_batch_size, uint8)`
allocation, never reallocated. That is exactly what registration needs (a
stable pointer for the process lifetime).

But it is created by the **GPU context**, while the adapter is built by the
**L2 adapter factory** from `l1_memory_desc`. The adapter is constructed
before the GPU context registers its KV caches. So the GPU channel cannot be
built in the factory.

**Resolution: deferred registration.** Add to the adapter:

```python
def register_gpu_staging_buffer(self, ptr: int, size: int) -> None
```

called once from the GPU-transfer module when the GPU context is ready
(alongside where it registers the other stream callbacks,
[`gpu_transfer.py:149-162`](lmcache/v1/multiprocess/modules/gpu_transfer.py#L149-L162)).
It builds the second `NixlChannel`, kicks off background handshakes to each
peer, and only then flips the adapter's `supports_l2_resident_retrieve()` to
`True`.

**`supports_l2_resident_retrieve()` therefore becomes dynamic** — `False`
until the GPU buffer is registered and at least one peer's GPU channel is
connected, `True` after. Verify the `PrefetchController` reads it per-request
(it does, at
[`prefetch_controller.py:727`](lmcache/v1/distributed/storage_controllers/prefetch_controller.py#L727),
inside `_transition_to_load_phase`) — so flipping it at runtime is safe and a
request in flight during the flip is not affected. This must be stated as an
explicit contract note on the base method, which currently reads as a static
property.

### 3.2 Alignment and size constraints — where 2 MiB actually comes from

Worth being precise, because the obvious reading ("2 MiB is the rule
everywhere") is wrong and leads to over-constraining the design.

2 MiB is **not** a global constant. It is the `align_bytes` of *one
registration*: `build_nixl_peer_adapter_from_config` picks it, `NixlChannel`
forwards it to `NixlAgentWrapper`, which slices that region into one
descriptor per `page_size` bytes
([`nixl_channel.py:698-703`](lmcache/v1/transfer_channel/nixl_channel.py#L698-L703)).
It became a cross-node wire contract for the **DRAM L1** region only because
the donor publishes `meta.address // page_size` and the requester expands by
`chunk_bytes // page_size` — both sides must agree there or the index math
desyncs.

The GPU staging buffer is a *separate registration on a separate agent*
(Option A above), so that contract does not bind it directly. In principle we
could register it at any granularity — e.g. `align_bytes = tmp_chunk_bytes_`,
one descriptor per staging slot, making slot *k* simply index *k*.

**But NIXL requires paired descriptors to be byte-identical in size.**
Verified on this machine against NIXL 1.3.0 / UCX, two agents, DRAM source →
VRAM destination:

- 2 MiB local ← 2 MiB remote: `post=PROC`, `final=DONE`, data correct
  (confirming GPUDirect RDMA into a CUDA tensor works here at all).
- 8 MiB local ← 2 MiB remote: rejected at request-build time —
  `makeXferReq: length mismatch at index pair 0`, `NIXL_ERR_INVALID_PARAM`.

So the local GPU descriptors **must** be 2 MiB, because the remote ones are
2 MiB by the wire contract. The constraint is real, but it is *derived from
descriptor-size symmetry*, not inherited from the L1 rule — and the
"expand both sides to a common byte granularity at different page sizes"
escape hatch does not exist.

Consequences, both of which must be validated at registration and **fail
loudly** (matching how the factory already rejects a non-2 MiB-multiple L1
buffer rather than silently desyncing):

- `tmp_gpu_buffer_.data_ptr()` must be 2 MiB-aligned. Measured: `torch.empty`
  returns 2 MiB-aligned pointers for allocations at these sizes (8/64/256/1024
  MiB all aligned) — but the caching allocator does not guarantee it, so check.
- `tmp_chunk_bytes_` must be a whole number of 2 MiB pages, or slot *k*'s base
  descriptor index is not `k * pages_per_chunk`. **This one genuinely fails
  for real geometries:** it is `layers × 2 × chunk_tokens × kv_heads ×
  head_dim × itemsize`, a product of powers of two only when every factor is.
  Llama-3.1-8B/70B at chunk 256 give exactly 32/80 MiB (fine), but 30 layers ×
  6 KV heads → 22.5 MiB and chunk=200 → 25 MiB both violate it.

If the check fires, the fix is an over-sized allocation with an aligned,
padded sub-view — a contained follow-up, not something to write speculatively
before a real config needs it.

Note the destination geometry differs from the DRAM path: there, the
destination is an L1 `MemoryObj` whose `meta.address` gives the descriptor
index. Here, the destination is staging slot `chunk_idx`, so the local base
index is `(chunk_idx * tmp_chunk_bytes_) // page_size` — computed from the
`gpu_ptr` the retrieve loop passes, as `(gpu_ptr - gpu_buffer_base) //
page_size`. The remote index still comes from the donor unchanged, so **the
wire contract is untouched** and a GPUDirect-capable node interoperates with
an unmodified peer.

---

## 4. Pin lifetime

Today the remote read-lock is released by `submit_unlock`, which the
`PrefetchController` calls for every key in the load plan once load completes
(§4.3 of the peer doc). Under L2-resident retrieve there **is no load phase**
— the controller keeps the lookup pin held and hands the key to the retrieve
handler.

The lifecycle becomes:

```
lookup    -> peer reserve_read, pin recorded in _pins        (unchanged)
retrieve  -> submit_h2d_batch: RDMA READ peer DRAM -> GPU staging, polled to DONE
          -> token issued per key
          -> release_after_h2d_batch: RemoteUnlockReq to the peer
```

Two things follow:

1. **`release_after_h2d_batch` maps to the existing `_do_unlock` logic** —
   group by `(peer, lease_id)`, send `RemoteUnlockReq`, drop `_pins` entries.
   It should reuse `_do_unlock` rather than duplicate the grouping, since the
   lease-grouping subtlety (two lookups on one peer under different leases)
   is easy to get wrong twice.
2. **The stream callback is now unnecessary but harmless.** For CXL,
   `release_after_h2d` must wait for the stream to drain because the DMA is
   in flight. For us the RDMA already completed inside `submit_h2d_batch`, so
   the pin could be dropped immediately. Keep it on the stream callback
   anyway — the shared consumer path
   ([`gpu_transfer.py:895-901`](lmcache/v1/multiprocess/modules/gpu_transfer.py#L895-L901))
   already routes it that way, releasing slightly late is harmless (it only
   delays the peer's ability to evict), and diverging would mean special-casing
   the adapter in the consumer.

**The abort path already works, but is easy to break.** On a failed retrieve
the consumer calls `submit_unlock_l2_resident` synchronously
([`gpu_transfer.py:908-915`](lmcache/v1/multiprocess/modules/gpu_transfer.py#L908-L915)),
which groups by adapter and calls `submit_unlock`
([`storage_manager.py:808-812`](lmcache/v1/distributed/storage_manager.py#L808-L812))
— which is exactly our existing remote-unlock entry point. So the abort path
needs **no new code**; it needs a test, because if it ever regresses, every
failed retrieve strands a remote read-lock on a peer until its lease expires,
throttling that peer's eviction with no local symptom.

One ordering subtlety: on the *success* path a key's pin is dropped by
`release_after_h2d_batch`, on the *failure* path by `submit_unlock`. Both
funnel into `_do_unlock`, which pops from `_pins` — so a key unlocked twice
is a no-op on the second call (the pop returns `None`). That is the existing
behavior and it is what makes the two paths safe to overlap.

---

## 5. Failure handling

A `submit_h2d_batch` that fails must return `-1` for the affected keys, not
raise — the consumer treats `-1` as a miss. But unlike CXL, **a miss here is
not recoverable**: the key was excluded from the L1 load plan, so there is no
L1 copy to fall back on. The staging slot holds stale bytes and the scatter
kernel will happily write them into the paged KV cache.

**The consumer did not handle `-1`, which was a latent correctness bug** —
fixed as part of this work. `batch_tokens` was only ever `extend`ed into
`h2d_tokens`; nothing inspected the values before the scatter kernel for that
batch was enqueued, so a `-1` meant the scatter silently wrote whatever stale
bytes the staging slot held into the paged KV cache, undetected.

This is invisible today only because a CXL `gpu_src_view` miss under a held
pin is near-impossible. An RDMA READ fails for ordinary network reasons, so
this adapter turns a latent bug into a reachable one. **Fixing it is part of
this change, not a follow-up.** Options, in order of preference:

1. **Fail the whole retrieve** on any `-1` from this adapter (return
   `retrieve_succeeded = False`), so the engine recomputes. Safe, coarse, and
   matches the existing "Some keys not found during retrieve!" early-return.
2. Zero the staging slot and let the model consume zeros — **unacceptable**,
   silent wrong output.

Go with (1). Note it benefits the CXL path too, so it is worth landing as its
own small preparatory commit ahead of the adapter work, where it can be
reviewed as the correctness fix it is rather than as a detail of a larger
feature.

---

## 6. Implementation steps — as built

Two deviations from the plan, both discovered during implementation:

- **The GPU handshake runs on a plain daemon thread, not the adapter's asyncio
  loop.** The first draft scheduled `_ensure_peer_gpu_connected` onto the bg
  loop and had `submit_h2d_batch` block on the resulting future. That is wrong
  twice over: it queues the TTFT-critical retrieve behind whatever else the
  loop is running, and it strands a pending coroutine if the adapter closes
  meanwhile (visible as `Task was destroyed but it is pending!` in the test
  run). The handshake is a blocking ZMQ round-trip with no async component, so
  it is now `_connect_peer_gpu_sync`, called directly on whichever thread
  needs it.
- **`register_gpu_staging_buffer` is on `L2AdapterInterface`**, defaulting to
  a no-op, rather than being duck-typed with `getattr` from the storage
  manager — the codebase has a real adapter interface and this belongs on it.



| # | Change | File |
|---|---|---|
| 1 | `_NixlGpuReadChannel`: READ into GPU staging by slot index. Mirrors `_NixlReadChannel` but computes the local base index from `(gpu_ptr - base) // page_size` instead of `meta.address`. | `nixl_peer_l2_adapter.py` |
| 2 | `register_gpu_staging_buffer(ptr, size)`: validate 2 MiB alignment of `ptr` and of the per-slot stride (raise `ValueError`), build the second `NixlChannel` with `device="cuda"`, start background handshakes. | `nixl_peer_l2_adapter.py` |
| 3 | `supports_l2_resident_retrieve()` → `True` only once (2) has run and a GPU-channel peer is connected. | `nixl_peer_l2_adapter.py` |
| 4 | `submit_h2d_batch`: group pinned keys by peer, one READ per peer into the destination slots, **poll to DONE**, issue tokens. `-1` per key on failure. Override the batch form (the base's per-key loop would issue one RDMA per chunk). | `nixl_peer_l2_adapter.py` |
| 5 | `release_after_h2d` / `release_after_h2d_batch`: resolve tokens → keys, delegate to the existing `_do_unlock` grouping. | `nixl_peer_l2_adapter.py` |
| 6 | Config: `gpu_init_bind_url`, peer `gpu_init_url`, and an explicit `enable_gpu_direct` flag (default `False`) so the DRAM path stays the default until this is hardware-validated. | `nixl_peer_l2_adapter.py` |
| 7 | Call `register_gpu_staging_buffer` once the GPU context exists. | `modules/gpu_transfer.py` |
| 8 | Handle a `-1` token from a resident adapter as a hard retrieve failure (§5). | `modules/gpu_transfer.py` |
| 9 | Document the dynamic `supports_l2_resident_retrieve()` and the "may block on the network" `submit_h2d` semantics. | `l2_adapters/base.py` |
| 10 | Rewrite the peer design doc's §1 non-goal ("L2-resident GPU-direct retrieve stays `False`") and §5, plus the CXL doc's claim that GPU-direct is "the CXL adapter's key advantage over the pull-into-L1 NIXL peer path". | both design docs |

**`enable_gpu_direct` defaults to `False`.** Both adapters and the retrieve
path are load-bearing, this needs real RDMA hardware to validate, and the
failure mode (§5) is silent corruption rather than a crash. An opt-in flag lets
the DRAM path stay the default while the new path is benchmarked on the two-node
setup.

### Tests

- `_NixlGpuReadChannel` index math: slot *k* → base index `k *
  pages_per_chunk`; full descriptor expansion per chunk (the bug §3 of the
  peer doc records); `ValueError` on a misaligned pointer and on a
  non-page-multiple stride.
- `submit_h2d_batch` with a fake channel: correct per-peer grouping, tokens
  issued in input order, `-1` on a channel that raises, and — critically —
  that it does **not** return before the fake channel reports DONE.
- Pin lifecycle: `submit_h2d_batch` → `release_after_h2d_batch` sends exactly
  one `RemoteUnlockReq` per `(peer, lease)` and empties `_pins`
  (`debug_held_pin_count() == 0`).
- **Abort path:** a retrieve that fails after `submit_h2d_batch` releases every
  remote pin (§4).
- `supports_l2_resident_retrieve()` is `False` before registration, `True`
  after — and a prefetch that starts before registration completes correctly
  via the DRAM path.

The existing fake-channel harness in
[`test_nixl_peer_l2_adapter.py`](tests/v1/distributed/l2_adapters/test_nixl_peer_l2_adapter.py)
covers all of this without RDMA hardware. End-to-end GPUDirect needs the
two-node setup and belongs in the `scripts/cxl` bench sweep, not in CI.

---

## 7. Explicitly out of scope

- **Stream-ordered RDMA completion (incl. IBGDA).** See §9 — this is not a
  backend swap and does not remove the §2 block with the API we have.
- **Donor-side GPU serving.** The peer still serves from its DRAM L1.
- **Restructuring `_retrieve_loop` into fill-all-then-scatter-all** (§2.2).
- **Multi-region NIXL registration** (§3, Option B).

---

## 8. Expected benefit, and the honest caveat

> **Superseded by §11 — kept for the record.** The prediction below was
> directionally right about the *risk* (the retrieve phase did get 65 ms
> slower) but wrong about the *source of the win*: it is `l2load` alone, not
> the H2D bounce or the L1 reserve. Read §11 for what was measured.


Removed per remote-hit chunk: one DRAM landing, one L1 write-reserve, one
DRAM→GPU H2D copy (`ret_h2d` in the `LMC_PROFILE` line), and the L1 capacity
the chunk would have occupied.

Not removed: the RDMA READ itself, which is the dominant cost and is now
partly exposed on the retrieve thread rather than hidden in the prefetch phase
(§2.1). The net win is therefore **larger for `l2load` + `l1rsv` + `ret_h2d`
than the loss on `ret_h2d`'s replacement**, but it moves latency from the
prefetch phase (which overlaps vLLM's own work) into the retrieve phase (which
is TTFT-critical).

That is a real risk to the headline metric and the reason for the opt-in flag:
**it must be measured on the two-node setup before being made the default.**
Compare the `PROFILE` line's `l2lk / l1rsv / l2load / pf_wait / ret_h2d`
stages between `enable_gpu_direct` on and off. If `pf_wait` was previously
absorbing the RDMA latency for free, this change could be TTFT-neutral or
worse despite doing strictly less work — in which case the fix is the
stream-ordered completion in §9, not this design.

---

## 9. Why IBGDA / GPUDirect Async does not remove the §2 block

The natural objection to §2 is: NVIDIA has GPUDirect Async — use it. Recording
why that does not apply here, so it is not re-litigated.

**Three distinct technologies get conflated under this name:**

| | What it does | Relevance |
|---|---|---|
| GPUDirect **RDMA** | NIC DMAs to/from VRAM, bypassing host memory | This is what §3 uses. Works today. |
| GPUDirect **Async, kernel-initiated (IBGDA)** | A CUDA *kernel* rings the NIC doorbell and posts work requests; CPU out of the *control* path | Solves doorbell latency, **not** our ordering problem |
| GPUDirect **Async, stream-ordered** | The transfer is queued *into a CUDA stream*, ordering against other stream work | This is what §2 would actually need |

Our problem is ordering the READ against the scatter kernel on our stream.
IBGDA's benefit is removing CPU doorbell latency for GPU-resident
communication (it is what NVSHMEM uses); it does **not** give a stream-ordered
completion for a transfer that *host* code issued. Getting ordering out of
IBGDA means inverting control — the scatter kernel, or a kernel before it,
posts the READ and polls its completion queue *device-side*. That is writing
device-side communication into the KV-transfer kernels plus a symmetric-heap
addressing scheme for peer memory. A different project, not a backend flag.

**And the installed NIXL cannot express it regardless.** Verified against
this repo's `.venv` (NIXL 1.3.0):

- The GPUNETIO plugin is present
  (`nixl_cu12.libs/nixl/libplugin_GPUNETIO.so`) but is **DOCA**-based —
  BlueField DPU / ConnectX with the DOCA stack, not plain IB verbs.
- Its symbols show it creating and synchronizing its **own internal** CUDA
  streams (`cudaStreamCreateWithFlags`, `"CUDA streams used for pool mode"`,
  `cudaStreamSynchronize`). It uses streams internally; it does not accept
  the caller's.
- Decisively: `make_prepped_xfer`, `transfer`, and `check_xfer_state` in
  `nixl_cu12/_api.py` take **no stream argument**, and the compiled
  `_bindings...so` exports no stream-related symbol at all. Completion is
  `getXferStatus` polling — there is no API through which to say "complete
  this transfer on this stream."

So even on correct hardware, a stream-ordered path means C++ against the DOCA
engine plus custom device-side kernels. **Conclusion: §2.1's blocking poll is
the design, not a placeholder.** Revisit only if profiling (§8) shows the
exposed RDMA is the dominant TTFT cost, and treat it as its own project.

---

## 10. Postmortem: the first two-node run

Recorded because both failures were invisible in the headline numbers, and
the second nearly shipped as a 3.3× "win".

### 10.1 `UCX_TLS=rc` blocked VRAM registration

`launch_node.sh` pinned `UCX_TLS=rc` for nixl mode — correct when the only
registered buffer was DRAM. But `UCX_TLS` is an **allowlist**, and it excludes
`cuda_copy`, the transport UCX needs to *recognize* GPU memory. Registration
failed with `VRAM memory is detected as host by UCX` → `NIXL_ERR_BACKEND`.

Misleading detail: UCX also logs *"UCX CUDA support was not found"*, which
reads like a build problem. It was not — the bundled UCX ships
`libuct_cuda.so`, and VRAM registration succeeded in a bare probe. The
allowlist was masking a capability that was present.

Fixed by widening to `rc,cuda_copy` only when `GPUDIRECT=1`. (`gdr_copy` is
absent from this UCX build; naming it only produces a per-launch WARN.)

This failed **soft** — the adapter logged the rejection and fell back to the
DRAM path, so the node came up and served traffic. A silent misconfiguration
looks exactly like a successful run.

### 10.2 The GPU channel handshook the wrong agent

The real bug. `_connect_peer_gpu_sync` connected our GPU agent to the peer's
**GPU** agent (`gpu_init_url`), making the remote dlist describe the peer's
128 MiB staging buffer — 64 descriptors. The donor reports indices into its
32 GiB L1 (up to ~16k), so NIXL rejected everything past 63:

```
makeXferReq: remote index out of range at index 0 with value 1872
-> NIXL_ERR_INVALID_PARAM
```

GPUDirect is **asymmetric**: only our side is GPU memory. The GPU channel must
handshake the peer's **L1/DRAM** agent (`init_url`). See §3.

**The benchmark this produced was invalid, and looked excellent.** Reported
3.3× (`ttft_b_cold` 0.407 → 0.121). What actually happened: every
`submit_h2d_batch` failed → returned `-1` → the §5 guard failed the retrieve →
vLLM recomputed the prefix. The measurement was not of a GPUDirect cache hit.
36 error events across a 3-repeat run.

**Why it wasn't caught:**

- Node A's log looked perfectly healthy — A serves local hits and never
  exercises the GPU path. The failures were only on node B, on the *other
  host*. Checking one node's log is not evidence.
- The unit tests' fake channel discarded `peer_init_url`, so no test could
  distinguish the two endpoints. Now asserted by
  `test_gpu_direct_handshakes_l1_not_gpu_url`, verified to fail against the
  bug.

**What worked:** the §5 hard-fail guard turned would-be silent KV corruption
into a loud, traceable error. Without it this would have produced plausible
garbage output instead of a stack trace.

### 10.3 Lesson for the next run

A GPUDirect result is only trustworthy when **node B's** log shows both
`GPUDirect retrieve enabled` and `GPU channel connected`, **and** zero
occurrences of `cannot be served GPU-direct`, `GPUDirect READ from peer ...
failed`, and `scattering stale`. Check the fallback counters before reading
the TTFT table — a failed GPUDirect path still produces numbers.

---

## 11. Measured result (2026-08-25, two-node RDMA)

Llama-3.1-8B, 30k-token prompt (117 chunks, 3.66 GB), g5 ↔ g6 over
ConnectX/UCX RC. Node A serves local L1 hits as an unchanged control; node B
does the cross-node fetch. Run health verified per §10.3 (2 enable lines, zero
fallbacks).

### 11.1 End-to-end TTFT

| | DRAM path | GPUDirect | change |
|---|---:|---:|---:|
| `ttft_a_local` (control) | 0.2092 | 0.2111 | +0.9% |
| `ttft_b_cold` | 0.4067 | 0.2841 | **−30.2%** |
| `ttft_b_warm` | 0.4039 | 0.2844 | **−29.6%** |

The control moving <1% is what makes the delta attributable to this change.

### 11.2 Where the time went (`LMC_PROFILE=1`, steady-state medians, ms)

| stage | DRAM | GPUDirect | Δ |
|---|---:|---:|---:|
| `l2load` — RDMA into L1 | 190.7 | **0.0** | −190.7 |
| `l1rsv` — L1 write reserve | 0.8 | **0.0** | −0.8 |
| `pf_wait` — vLLM waiting on prefetch | 202.9 | **12.6** | −190.3 |
| `ret_h2d` — H2D fill | 93.2 | **158.7** | **+65.5** |
| `ret_scat` — scatter kernel | 6.8 | 9.6 | +2.8 |
| **total** | **305.8** | **183.9** | **−121.9** |

(`pf_wait` overlaps the prefetch stages, so `total` is not the column sum.)

**The win is `l2load` → 0: the transfer happens once instead of twice.** The
prefetch-side RDMA-into-L1 disappears entirely and `pf_wait` collapses with it.

**§2.1's predicted cost is real and measurable.** `ret_h2d` grew by 65.5 ms
because it now contains the blocking RDMA rather than a DRAM→GPU copy, and
`h2d_cpu` (CPU launch overhead) jumps 1.4 → 164.7 ms — the retrieve thread
stalled in `check_xfer_state`. The blocking poll costs exactly what §2.1 said
it would; it is simply outweighed 3:1 by not transferring twice.

### 11.3 Two corrections to earlier reasoning in this doc

- **§8 speculated the win would come from removing the H2D bounce and the L1
  reserve. It does not.** `l1rsv` is 0.8 ms (noise) and `ret_h2d` *grew*. The
  entire benefit is `l2load`.
- **`ttft_b_warm` does not skip the cross-node fetch.** Every request logs
  `0 L1, 117 L2` and a full `l2load` — node B never retains these chunks in L1
  between requests, so "warm" re-fetches like "cold". Cold and warm improving
  by the same amount is therefore expected, and must NOT be read as evidence
  that the savings are fetch-independent.

### 11.4 Implication for §7/§9 (stream-ordered completion)

**GPUDirect is running at 92% of line rate. §9 is not worth building here.**

| | |
|---|---|
| RDMA link | 200 Gbps = **25.0 GB/s** raw |
| measured `ret_h2d` | 158.7 ms for 3.66 GB = **23.0 GB/s** |
| efficiency | **92% of line rate** |
| floor at line rate | 146.2 ms — only **~12 ms** of headroom exists |

A stream-ordered (DOCA/IBGDA) path could at best reclaim a slice of that 12 ms.
The transfer is wire-bound, so the blocking poll of §2.1 — however unpleasant it
looks in `h2d_cpu` = 164.7 ms — is *not* the limiter. The retrieve thread is
stalled because the bytes take that long to arrive, not because polling is slow.

**Two comparisons that look like contradictions and are not:**

- *The NIXL DRAM path hits 39.2 GB/s `ret_h2d`, and `bench_h2d.py` measures
  ~56 GB/s DRAM→GPU.* Those are **local PCIe** copies from pinned host memory;
  they never cross the network. PCIe bandwidth is not a ceiling a 200 Gbps
  network transfer can be measured against.
- *CXL's GPU-direct read also lands at ~22.6 GB/s.* Coincidence, and a fully
  separate cause: `bench_h2d.py` shows CXL→GPU saturating at **24.9 GB/s vs
  DRAM's 56.1 GB/s** (ratio 0.44) at ≥4 MiB — the CXL device's own read
  ceiling. No CXL is involved in the NIXL path. Do not read the matching
  numbers as a shared bottleneck.

The lesson for future analysis: **compare a transport against its own medium's
ceiling.** Both near-misses above came from measuring a network path against a
local-bus number.

### 11.5 Still open before making it the default

- Only one prompt length (30k) and 3 repeats. The two effects scale
  differently — `l2load` savings and `ret_h2d` cost both grow with payload —
  so sweep `--prompt-tokens` to find where the ratio holds.
- Concurrency untested: the blocking poll occupies a retrieve thread for
  ~160 ms, which is invisible at depth 1 but may throttle throughput under
  concurrent requests. This is the most likely place the trade turns bad.

---

## 12. Versus the CXL adapter (same bench, 2026-08-25)

CXL always serves L2-resident (`l1rsv=0`, `l2load=0`), so it is GPU-direct on
every request regardless of the `GPUDIRECT` toggle — that flag only governs the
nixl adapter. CXL alternates **cold** (chunk not yet in the shared pool) and
**warm** (already resident); NIXL pays the same cost every request.

| condition | total | l2lk | l2load | pf_wait | ret_h2d | h2d_cpu |
|---|---:|---:|---:|---:|---:|---:|
| CXL cold | 471.8 | **291.2** | 0.0 | 300.0 | 161.8 | 5.6 |
| CXL warm | **182.9** | 4.3 | 0.0 | 10.7 | 162.4 | 4.4 |
| NIXL DRAM | 305.8 | 5.0 | **190.7** | 202.9 | 93.2 | 1.2 |
| NIXL GPUDirect | **183.9** | 4.9 | 0.0 | 12.6 | 158.7 | **164.7** |

**NIXL GPUDirect reaches parity with CXL-warm** — 183.9 vs 182.9 ms — and has
no cold penalty at all. An RDMA peer tier now matches a coherent shared-memory
pool on the retrieve path.

**The two tiers amortize differently, and that is the real distinction.**
CXL cold spends its 291 ms in `l2lk`, sub-broken-down by `PROFILE-L2LK`:

| sub-stage | cold | warm |
|---|---:|---:|
| `rpc` — `PushKVToCXL` (peer DRAM → shared pool) | 195.3 | 0.0 |
| `reserve` — slot alloc under the distributed lock | 91.7 | 0.0 |
| `local_pin` | 0.5 | 3.7 |

Cold pays once to *populate the pool*; every later request for that prefix is a
pointer read. **CXL amortizes: 472 ms once, then 183 ms forever. NIXL does not:
~184 ms every time**, because node B never retains these chunks in L1
(`0 L1, 117 L2` on every request — see §11.3).

So the choice is workload-shaped, not a ranking:

- **Repeated access to the same prefix** → CXL wins after the first request.
- **One-shot / low-reuse cross-node hits** → NIXL GPUDirect wins outright
  (184 ms vs 472 ms), with no pool, no distributed lock, and no DAX device.

Both are bounded by their medium at large payloads: CXL by its ~24.9 GB/s
device read ceiling, NIXL by the 200 Gbps link (§11.4). They land within ~2% of
each other here by coincidence, not by a shared mechanism.
