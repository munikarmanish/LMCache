#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cross-host validation script for the CXL backend.

Drives a 3-scenario test against a real shared CXL pool (or, for
sanity checks, a regular file accessible to both hosts via NFS or
copied path). Designed to be run with one process per node:

    # On Node A (initializer):
    python scripts/cxl_cross_host_test.py \\
        --dev-path /dev/dax0.0 --node-id 0 --role server \\
        --listen-port 8447 \\
        --pool-size $((4 * 1024 * 1024 * 1024))

    # On Node B (after Node A logs "ready"):
    python scripts/cxl_cross_host_test.py \\
        --dev-path /dev/dax0.0 --node-id 1 --role client \\
        --donor-host 192.168.128.31 --donor-port 8447

The server process initializes the pool, writes a known sentinel slot,
runs a CXLP2PServer with a fake local tier, and waits for the client.
The client process attaches, reads the sentinel slot back via
CXL_LOOKUP, then runs a remote_fetch round-trip.

Permissions: /dev/dax0.0 is typically root-only. Either run with
sudo or pre-chmod the device for the test session.
"""

# Future
from __future__ import annotations

# Standard
from typing import Optional
import argparse
import json
import logging
import sys
import time

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.cxl.cross_node import CXLDonor, remote_fetch
from lmcache.v1.storage_backend.cxl.fence import (
    CLFlushFence,
    StubFence,
    default_fence,
    reset_default_fence,
)
from lmcache.v1.storage_backend.cxl.gc import (
    CXLGarbageCollector,
    CXLGCConfig,
)
from lmcache.v1.storage_backend.cxl.p2p_messages import PushStatus
from lmcache.v1.storage_backend.cxl.p2p_transport import (
    CXLP2PClient,
    CXLP2PServer,
)
from lmcache.v1.storage_backend.cxl.store import CXLStore, CXLStoreConfig

# Test parameters that BOTH sides must agree on so geom_hash matches.
SHARED_MODEL_NAME = "cxl-cross-host-test"
SHARED_KV_SHAPE = (4, 2, 16, 4, 64)
SHARED_KV_DTYPE = torch.float16
SHARED_CHUNK_SIZE = 16
DEFAULT_REGION_SIZE = 256 * 1024 * 1024  # 256 MiB
DEFAULT_POOL_SIZE = 128 * 1024 * 1024 * 1024  # 128 GiB — fits cudaHostRegister
# per-call cap on our test rig.

# Sentinel keys used by the warm-CXL test.
SENTINEL_KEYS = [0xCC000000 + i for i in range(4)]
SENTINEL_FILL_BASE = 0xA0


def _metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name=SHARED_MODEL_NAME,
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=SHARED_KV_DTYPE,
        kv_shape=SHARED_KV_SHAPE,
        chunk_size=SHARED_CHUNK_SIZE,
    )


def _make_key(int_hash: int) -> ObjectKey:
    md = _metadata()
    return ObjectKey(
        chunk_hash=int_hash.to_bytes(8, "little"),
        model_name=md.model_name,
        kv_rank=0,
        cache_salt="",
    )


def _make_obj(size_bytes: int, fill: int) -> TensorMemoryObj:
    data = torch.full((size_bytes,), fill, dtype=torch.uint8)
    meta = MemoryObjMetadata(
        shape=torch.Size([size_bytes]),
        dtype=torch.uint8,
        address=data.data_ptr(),
        phy_size=size_bytes,
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.KV_2LTD,
    )
    return TensorMemoryObj(raw_data=data, metadata=meta, parent_allocator=None)


class _LocalTier:
    """Simple key->payload mapping the donor reads from."""

    def __init__(self):
        self._store: dict[ObjectKey, tuple[bytes, MemoryFormat]] = {}

    def put(
        self, key: ObjectKey, payload: bytes, fmt: MemoryFormat = MemoryFormat.KV_2LTD
    ):
        self._store[key] = (payload, fmt)

    def __call__(
        self, chunk_hash: bytes, model_name: str, kv_rank: int, cache_salt: str
    ) -> Optional[MemoryObj]:
        record = self._store.get(
            ObjectKey(
                chunk_hash=chunk_hash,
                model_name=model_name,
                kv_rank=kv_rank,
                cache_salt=cache_salt,
            )
        )
        if record is None:
            return None
        payload, fmt = record
        size = len(payload)
        data = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        meta = MemoryObjMetadata(
            shape=torch.Size([size]),
            dtype=torch.uint8,
            address=data.data_ptr(),
            phy_size=size,
            ref_count=1,
            pin_count=0,
            fmt=fmt,
        )
        return TensorMemoryObj(raw_data=data, metadata=meta, parent_allocator=None)


# ---------- server ----------


def run_server(args) -> int:
    log = logging.getLogger("server")

    # Force the right fence: real CLFLUSH if the user's hardware needs it
    # (the default on x86 already does this — but be explicit for the
    # cross-host case).
    _force_fence(args.fence_mode)
    log.info("using fence: %s", type(default_fence()).__name__)

    cfg = CXLStoreConfig(
        dev_path=args.dev_path,
        node_id=args.node_id,
        max_chunk_size_bytes=args.chunk_size_bytes,
        region_size=args.region_size,
        initialize=True,
        generation=args.generation,
        run_lock_manager=True,
        pool_size_override=args.pool_size if args.pool_size > 0 else None,
    )
    backend = CXLStore(cfg)
    log.info(
        "server initialized pool: regions=%d, chunk_size=%d",
        backend.pool.layout.region_count,
        args.chunk_size_bytes,
    )

    # Warm-CXL test: write SENTINEL_KEYS into CXL so the client can read.
    for i, h in enumerate(SENTINEL_KEYS):
        backend.put_batch(
            [_make_key(h)],
            [_make_obj(args.payload_size, SENTINEL_FILL_BASE + i)],
        )
    log.info("wrote %d sentinel keys to CXL", len(SENTINEL_KEYS))

    # PushKV-test: stage a few keys in the local tier (NOT in CXL).
    local = _LocalTier()
    for i in range(args.num_push_keys):
        h = 0xDD000000 + i
        local.put(_make_key(h), bytes([(0xB0 + i) & 0xFF] * args.payload_size))
    log.info("staged %d keys in local tier (not in CXL)", args.num_push_keys)

    donor = CXLDonor(
        handle=backend.pool,
        index_writer=backend.index_writer,
        heaps=backend.heaps,
        node_id=backend.node_id,
        local_copy_provider=local,
    )
    p2p_server = CXLP2PServer(donor=donor, bind_url=f"tcp://0.0.0.0:{args.listen_port}")
    p2p_server.start()

    # Optional GC, useful if the client wants to test dead-node handling
    # later.
    gc: Optional[CXLGarbageCollector] = None
    if args.run_gc:
        # Liveness oracle: trivial alive-set provider.
        alive_box = {"alive": frozenset({args.node_id})}

        def liveness():
            return alive_box["alive"]

        gc = CXLGarbageCollector(
            allocator=backend.region_allocator,
            index_writer=backend.index_writer,
            pool=backend.pool,
            liveness=liveness,
            config=CXLGCConfig(sweep_interval_s=2.0),
        )
        gc.start()

    print(json.dumps({"status": "ready", "node_id": args.node_id}), flush=True)
    log.info("server READY; waiting until --hold-seconds=%s", args.hold_seconds)

    try:
        time.sleep(args.hold_seconds)
    except KeyboardInterrupt:
        pass

    if gc is not None:
        gc.stop()
    p2p_server.stop()
    backend.close()
    log.info("server done")
    return 0


# ---------- client ----------


def run_client(args) -> int:
    log = logging.getLogger("client")

    _force_fence(args.fence_mode)
    log.info("using fence: %s", type(default_fence()).__name__)

    cfg = CXLStoreConfig(
        dev_path=args.dev_path,
        node_id=args.node_id,
        max_chunk_size_bytes=args.chunk_size_bytes,
        region_size=args.region_size,
        initialize=False,
        run_lock_manager=False,
        pool_size_override=args.pool_size if args.pool_size > 0 else None,
    )
    backend = CXLStore(cfg)
    log.info("client attached")

    failures = []

    # ---- Scenario 1: warm CXL hit ----
    log.info("scenario 1: warm CXL hits")
    warm_hits = 0
    warm_mismatches = 0
    for i, h in enumerate(SENTINEL_KEYS):
        key = _make_key(h)
        if not backend.contains(key):
            failures.append(f"warm miss for h=0x{h:08x}")
            continue
        buf = torch.empty(args.payload_size, dtype=torch.uint8)
        n = backend.read_into(key, buf.data_ptr(), buf.numel())
        if n == 0:
            failures.append(f"warm read returned 0 bytes for h=0x{h:08x}")
            continue
        first_byte = int(buf[0])
        if first_byte != (SENTINEL_FILL_BASE + i) & 0xFF:
            warm_mismatches += 1
            failures.append(
                f"warm fill mismatch for h=0x{h:08x}: "
                f"got 0x{first_byte:02x}, want 0x{(SENTINEL_FILL_BASE + i) & 0xFF:02x}"
            )
        else:
            warm_hits += 1
    log.info(
        "scenario 1: warm_hits=%d mismatches=%d failures=%d",
        warm_hits,
        warm_mismatches,
        len(failures),
    )

    # ---- Scenario 2: PushKVToCXL via ZMQ ----
    log.info("scenario 2: PushKVToCXL via ZMQ")
    push_keys = [_make_key(0xDD000000 + i) for i in range(args.num_push_keys)]
    # Confirm none are in CXL yet.
    pre_hits = sum(1 for k in push_keys if backend.contains(k))
    if pre_hits != 0:
        failures.append(f"push-test: expected 0 CXL hits before push, got {pre_hits}")

    donor_url = f"tcp://{args.donor_host}:{args.donor_port}"
    client = CXLP2PClient(donor_url=donor_url)
    try:
        result = remote_fetch(
            requester_node_id=backend.node_id,
            keys=push_keys,
            tenant_digest_fn=backend.tenant_digest_for,
            index_writer=backend.index_writer,
            donor_node_id=args.donor_node_id,
            donor=client,
            sender_id=f"node-{args.node_id}",
            epoch=backend.epoch,
        )
        log.info(
            "remote_fetch result: num_satisfied=%d status=%s",
            result.num_satisfied,
            result.status.name,
        )
        if result.status not in (PushStatus.OK, PushStatus.PARTIAL):
            failures.append(f"push-test: unexpected status {result.status.name}")

        # Verify each pushed key is now retrievable from CXL with the
        # right bytes.
        post_hits = 0
        post_mismatches = 0
        for i, k in enumerate(push_keys[: result.num_satisfied]):
            buf = torch.empty(args.payload_size, dtype=torch.uint8)
            n = backend.read_into(k, buf.data_ptr(), buf.numel())
            if n == 0:
                failures.append(f"push-test: post-fetch miss for key #{i}")
                continue
            expect = (0xB0 + i) & 0xFF
            actual = int(buf[0])
            if actual != expect:
                post_mismatches += 1
                failures.append(
                    f"push-test: byte mismatch for key #{i}: "
                    f"got 0x{actual:02x}, want 0x{expect:02x}"
                )
            else:
                post_hits += 1
        log.info(
            "scenario 2: post_hits=%d post_mismatches=%d", post_hits, post_mismatches
        )
    finally:
        client.close()

    backend.close()

    summary = {
        "warm_hits": warm_hits,
        "warm_mismatches": warm_mismatches,
        "remote_fetch_satisfied": result.num_satisfied,
        "failures": failures,
    }
    print(json.dumps(summary, indent=2), flush=True)

    return 0 if not failures else 1


# ---------- shared ----------


def _force_fence(mode: str) -> None:
    """Forcibly select a fence implementation regardless of platform."""
    reset_default_fence()
    if mode == "auto":
        return  # let default_fence() auto-select
    if mode == "stub":
        # First Party
        from lmcache.v1.storage_backend.cxl.fence import set_default_fence

        set_default_fence(StubFence())
        return
    if mode == "clflush":
        # First Party
        from lmcache.v1.storage_backend.cxl.fence import (
            _try_build_clflush_lib,
            _try_load_clflush_lib,
            set_default_fence,
        )

        lib = _try_load_clflush_lib() or _try_build_clflush_lib()
        if lib is None:
            raise RuntimeError("--fence-mode=clflush requested but .so not loadable")
        set_default_fence(CLFlushFence(lib))
        return
    raise ValueError(f"unknown fence mode {mode!r}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=("server", "client"), required=True)
    p.add_argument(
        "--dev-path", required=True, help="Path to /dev/dax0.0 or a shared regular file"
    )
    p.add_argument("--node-id", type=int, required=True)
    p.add_argument(
        "--pool-size",
        type=int,
        default=DEFAULT_POOL_SIZE,
        help="Cap the usable pool size in bytes. Must match on "
        "client and server. 0 means use the device's full "
        f"size. Default: {DEFAULT_POOL_SIZE} "
        f"({DEFAULT_POOL_SIZE >> 30} GiB).",
    )
    p.add_argument("--region-size", type=int, default=DEFAULT_REGION_SIZE)
    p.add_argument("--chunk-size-bytes", type=int, default=64 * 1024)
    p.add_argument(
        "--payload-size",
        type=int,
        default=4096,
        help="Payload bytes per sentinel/push key",
    )
    p.add_argument("--generation", type=int, default=1)
    p.add_argument("--num-push-keys", type=int, default=4)
    p.add_argument(
        "--fence-mode",
        choices=("auto", "stub", "clflush"),
        default="clflush",
        help="Force a fence implementation. 'clflush' is the "
        "correct choice for cross-host CXL on hardware "
        "without HW coherence (our test rack).",
    )
    # Server-only.
    p.add_argument(
        "--listen-port",
        type=int,
        default=8447,
        help="(server) bind port for the CXLP2PServer",
    )
    p.add_argument(
        "--hold-seconds",
        type=int,
        default=120,
        help="(server) keep server alive for this long",
    )
    p.add_argument(
        "--run-gc", action="store_true", help="(server) start a periodic GC sweeper"
    )
    # Client-only.
    p.add_argument(
        "--donor-host",
        default="127.0.0.1",
        help="(client) hostname/IP of the server's CXLP2PServer",
    )
    p.add_argument(
        "--donor-port",
        type=int,
        default=8447,
        help="(client) port of the server's CXLP2PServer",
    )
    p.add_argument(
        "--donor-node-id",
        type=int,
        default=0,
        help="(client) the server's node_id (for slot ownership)",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.role == "server":
        return run_server(args)
    else:
        return run_client(args)


if __name__ == "__main__":
    sys.exit(main())
