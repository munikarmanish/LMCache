#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Two-node correctness + bandwidth probe for the NIXL one-sided READ.

Resolves the anomaly where the nixl_peer load reported moving 3.3 GB in
~1.6 ms (~17,600 Gbps — 88x over a 200 Gbps port). Either ``check_xfer_state``
returns ``DONE`` before the bytes land (a correctness bug) or the transfer
isn't really moving the bytes we think. This probe settles it by reading a
KNOWN BYTE PATTERN over the real ``NixlChannel`` and verifying it landed,
while timing the transfer directly.

It uses ``NixlChannel`` directly (the same agent/handshake/transfer the
adapter's ``_NixlReadChannel`` wraps), filling a registered buffer with a
per-page pattern on the donor and reading it back on the reader. No vLLM,
no control plane, no index translation — just the data path in question.

Roles (run donor first, then reader; both on the lmcache venv python):

  # on c1 (donor):
  ~/.virtualenvs/lmcache/bin/python scripts/cxl/probe_nixl_rdma.py donor \
      --bind 0.0.0.0:9600 --chunk-mib 32 --chunks 105

  # on c2 (reader), pointing at c1:
  ~/.virtualenvs/lmcache/bin/python scripts/cxl/probe_nixl_rdma.py reader \
      --peer-init <c1-ip>:9600 --chunk-mib 32 --chunks 105 --iters 5

The reader prints, per iteration: transfer ms, GB/s, and whether every
page's bytes matched the donor's known pattern. If GB/s is plausible
(<= ~25 GB/s for 200 Gbps) AND bytes match -> the path is honest. If GB/s
is impossible AND bytes match -> ``DONE`` is premature but the data still
landed via some local/loopback path (the link isn't the NIC). If bytes
DON'T match -> premature ``DONE`` correctness bug in the READ completion.

Descriptor-size sweep
---------------------
``--page-kib`` sets the NIXL descriptor (transfer) size. Both roles MUST
use the same value, and it must divide the chunk size. Omit it for one
descriptor per chunk (the chunk size). Re-run the donor+reader pair per
value to find where bandwidth plateaus — this picks the descriptor size
to use in the nixl_peer adapter:

  # 4 KiB descriptors (the adapter's current point; expect op-bound):
  donor  ... --page-kib 4
  reader ... --page-kib 4
  # 2 MiB descriptors (expected to hit line rate):
  donor  ... --page-kib 2048
  reader ... --page-kib 2048
  # 32 MiB (one descriptor per chunk; omit --page-kib)
"""

# Standard
from __future__ import annotations
import argparse
import statistics
import sys
import time

# Third Party
import numpy as np


def _make_buffer(total_bytes: int) -> tuple[int, np.ndarray]:
    """Allocate a page-aligned host buffer; return (ptr, ndarray)."""
    # numpy arrays are 64-byte aligned in practice; that's enough for the
    # transfer (the NIXL page size is the chunk size, MiB-aligned).
    arr = np.zeros(total_bytes, dtype=np.uint8)
    return arr.ctypes.data, arr


def _fill_pattern(arr: np.ndarray, chunk_bytes: int, chunks: int) -> None:
    """Write a distinct per-page pattern so the reader can verify each page
    landed (and didn't get a different page's bytes)."""
    for c in range(chunks):
        off = c * chunk_bytes
        # First 8 bytes of each chunk = the chunk index; rest = index byte.
        arr[off : off + 8] = np.frombuffer(
            (c + 1).to_bytes(8, "little"), dtype=np.uint8
        )
        arr[off + 8 : off + chunk_bytes] = (c + 1) & 0xFF


def _verify_pattern(arr: np.ndarray, chunk_bytes: int, chunks: int) -> tuple[int, int]:
    """Return (pages_ok, pages_bad) by checking the per-page pattern."""
    ok = bad = 0
    for c in range(chunks):
        off = c * chunk_bytes
        idx = int.from_bytes(arr[off : off + 8].tobytes(), "little")
        tag_ok = idx == (c + 1)
        body_ok = bool((arr[off + 8] == ((c + 1) & 0xFF)))
        if tag_ok and body_ok:
            ok += 1
        else:
            bad += 1
    return ok, bad


def _build_channel(
    role: str, buf_ptr: int, buf_size: int, page_size: int, init_url: str
):
    """Construct a NixlChannel over the given registered buffer."""
    # First Party
    from lmcache.v1.transfer_channel.nixl_channel import NixlChannel

    return NixlChannel(
        async_mode=False,
        device="cpu",
        role="both",
        buffer_ptr=buf_ptr,
        buffer_size=buf_size,
        align_bytes=page_size,
        tp_rank=0,
        peer_init_url=init_url,  # bare host:port; NixlChannel adds tcp://
        backends=["UCX"],
    )


def _page_bytes(args, chunk_bytes: int) -> int:
    """NIXL descriptor size in bytes. Defaults to the chunk size (one
    descriptor per chunk) when --page-kib is omitted."""
    if args.page_kib is None:
        return chunk_bytes
    pb = args.page_kib * 1024
    if pb <= 0 or chunk_bytes % pb != 0:
        raise ValueError(
            f"chunk size {chunk_bytes} must be a positive multiple of page {pb}"
        )
    return pb


def run_donor(args) -> int:
    chunk_bytes = args.chunk_mib * 1024 * 1024
    total = chunk_bytes * args.chunks
    page_bytes = _page_bytes(args, chunk_bytes)
    ptr, arr = _make_buffer(total)
    _fill_pattern(arr, chunk_bytes, args.chunks)
    print(
        f"[donor] buffer={total / (1 << 30):.2f} GiB chunks={args.chunks} "
        f"chunk={args.chunk_mib}MiB nixl_page={page_bytes // 1024}KiB; "
        f"pattern filled. Binding init at {args.bind}. "
        f"IMPORTANT: the reader MUST use the same --page-kib.",
        flush=True,
    )
    ch = _build_channel("donor", ptr, total, page_bytes, args.bind)
    # The donor just needs to stay alive so the reader can handshake and
    # one-sided READ from this registered buffer. Hold a ref to arr so it
    # isn't GC'd. Park until interrupted.
    print("[donor] ready; the reader can now connect. Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("[donor] stopping.", flush=True)
    finally:
        del arr
        try:
            ch.close()
        except Exception:
            pass
    return 0


def run_reader(args) -> int:
    chunk_bytes = args.chunk_mib * 1024 * 1024
    total = chunk_bytes * args.chunks
    page_bytes = _page_bytes(args, chunk_bytes)
    ptr, arr = _make_buffer(total)  # destination, starts all-zero
    ch = _build_channel("reader", ptr, total, page_bytes, args.reader_bind)

    peer_id = "donor"
    print(
        f"[reader] nixl_page={page_bytes // 1024}KiB; "
        f"handshaking donor at {args.peer_init} ...",
        flush=True,
    )
    ch.lazy_init_peer_connection(
        local_id="reader", peer_id=peer_id, peer_init_url=args.peer_init
    )
    print("[reader] connected.", flush=True)

    # Build local + remote descriptor indices, mirroring the adapter:
    # each chunk spans pages_per_chunk consecutive descriptors. With the
    # default page=chunk this is one descriptor per chunk; smaller pages
    # expand into more (smaller) RDMA ops per chunk. This is exactly the
    # knob we're sweeping to find where bandwidth plateaus.
    n = args.chunks
    pages_per_chunk = chunk_bytes // page_bytes
    remote_indexes: list[int] = []
    local_indexes: list[int] = []
    for c in range(n):
        base = c * pages_per_chunk
        for p in range(pages_per_chunk):
            remote_indexes.append(base + p)
            local_indexes.append(base + p)
    print(
        f"[reader] {n} chunks x {pages_per_chunk} pages = "
        f"{len(local_indexes)} RDMA descriptors per transfer",
        flush=True,
    )

    def _read_once():
        # Mirror _NixlReadChannel.read_chunks: prepped READ, poll to DONE.
        agent = ch.nixl_agent
        handle = agent.make_prepped_xfer(
            "READ",
            ch.nixl_wrapper.xfer_handler,
            local_indexes,
            ch.remote_xfer_handlers_dict[peer_id],
            remote_indexes,
        )
        agent.transfer(handle)
        while True:
            status = agent.check_xfer_state(handle)
            if status == "ERR":
                raise RuntimeError("NIXL READ failed")
            if status == "DONE":
                break
            time.sleep(0.0005)

    samples = []
    for it in range(args.iters):
        arr[:] = 0  # clear so a stale-from-last-iter read can't pass
        t0 = time.perf_counter()
        _read_once()
        dt = time.perf_counter() - t0
        samples.append(dt)
        ok, bad = _verify_pattern(arr, chunk_bytes, n)
        gbps = (total / dt) / 1e9
        gbits = gbps * 8
        print(
            f"[reader] iter={it} ms={dt * 1000:.3f} GB/s={gbps:.1f} "
            f"Gbps={gbits:.0f} pages_ok={ok}/{n} pages_bad={bad}",
            flush=True,
        )

    med = statistics.median(samples)
    print(
        f"\n[reader] median ms={med * 1000:.3f} GB/s={(total / med) / 1e9:.1f} "
        f"Gbps={((total / med) / 1e9) * 8:.0f}",
        flush=True,
    )
    print(
        "[reader] Interpretation:\n"
        "  - pages_bad>0  -> READ reports DONE before bytes land (correctness "
        "bug in completion handling).\n"
        "  - pages_ok=all AND Gbps > port limit -> DONE is honest but the path "
        "isn't the 200Gbps NIC (local/loopback/shared-mem); the link assumption "
        "is wrong.\n"
        "  - pages_ok=all AND Gbps <= port limit -> the path is fully honest.",
        flush=True,
    )
    del arr
    try:
        ch.close()
    except Exception:
        pass
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="role", required=True)

    # --page-kib sets the NIXL descriptor (transfer) size; both roles MUST
    # use the same value. Omit it for one-descriptor-per-chunk (chunk size).
    # Sweep it (re-run the donor+reader pair per value) to find where
    # bandwidth plateaus: 4 (=4KiB, the current adapter point, op-bound) ->
    # 2048 (=2MiB) -> omit (=32MiB chunk, line rate). KiB granularity so the
    # sweep can reproduce the small end where the adapter currently sits.
    d = sub.add_parser("donor", help="hold a pattern-filled registered buffer")
    d.add_argument("--bind", default="0.0.0.0:9600", help="init bind host:port")
    d.add_argument("--chunk-mib", type=int, default=32)
    d.add_argument("--chunks", type=int, default=105)
    d.add_argument(
        "--page-kib",
        type=int,
        default=None,
        help="NIXL descriptor size in KiB (default: chunk size = 1 desc/chunk). "
        "Must match the reader and divide the chunk size.",
    )

    r = sub.add_parser("reader", help="RDMA-READ from the donor and verify")
    r.add_argument("--peer-init", required=True, help="donor init host:port")
    r.add_argument(
        "--reader-bind",
        default="0.0.0.0:9601",
        help="this reader's own init bind host:port",
    )
    r.add_argument("--chunk-mib", type=int, default=32)
    r.add_argument("--chunks", type=int, default=105)
    r.add_argument("--iters", type=int, default=5)
    r.add_argument(
        "--page-kib",
        type=int,
        default=None,
        help="NIXL descriptor size in KiB (default: chunk size = 1 desc/chunk). "
        "MUST match the donor's --page-kib and divide the chunk size.",
    )

    args = p.parse_args()
    if args.role == "donor":
        return run_donor(args)
    return run_reader(args)


if __name__ == "__main__":
    sys.exit(main())
