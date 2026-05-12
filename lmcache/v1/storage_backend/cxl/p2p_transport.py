# SPDX-License-Identifier: Apache-2.0
"""ZMQ transport for the CXL cross-node `PushKVToCXL` fallback.

Wraps `CXLDonor.handle_push` (server side) and `remote_fetch` (client
side) in a thin ZMQ REP/REQ pair. Decoupled from `P2PBackend`:

- `P2PBackend` is built around `LocalCPUBackend` for receiving KV
  bytes into CPU buffers. It would need a refactor to host the CXL
  push handler cleanly. That's a worthwhile refactor in its own right
  but doesn't have to gate cross-node CXL.
- `CXLP2PServer` / `CXLP2PClient` are self-contained — one server
  per CXL-attached node (running alongside the lock manager and GC),
  one client per peer that wants to ask for pushes.

Wire format: msgspec-encoded `PushKVToCXLMsg` / `PushKVToCXLRetMsg`
plus the existing tag union. ZMQ REQ/REP gives us synchronous
request/reply semantics, which matches `handle_push`'s contract.
"""

# Standard
import threading
from typing import Optional

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.cxl.cross_node import CXLDonor, DonorEndpoint
from lmcache.v1.storage_backend.cxl.p2p_messages import (
    CXLP2PMsg,
    PushKVToCXLMsg,
    PushKVToCXLRetMsg,
    PushStatus,
)

logger = init_logger(__name__)


# Default port for the donor REP socket. Callers can override.
DEFAULT_CXL_P2P_PORT = 8447


class CXLP2PServer:
    """ZMQ REP server that handles incoming PushKVToCXLMsg requests.

    One per node. Runs the dispatch in a background thread; binds to
    `bind_url` until `stop()` is called.
    """

    def __init__(
        self,
        donor: CXLDonor,
        bind_url: str = f"tcp://*:{DEFAULT_CXL_P2P_PORT}",
        *,
        context: Optional[zmq.Context] = None,
    ):
        self._donor = donor
        self._bind_url = bind_url
        self._context = context or zmq.Context.instance()
        self._socket: Optional[zmq.Socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # The encoder/decoder are per-server so ZMQ message recv +
        # decode runs without acquiring a global lock.
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(CXLP2PMsg)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("CXLP2PServer already started")
        self._stop_event.clear()
        self._socket = self._context.socket(zmq.REP)
        # Hard timeouts on send/recv prevent a stuck client from
        # wedging the server thread.
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)
        self._socket.setsockopt(zmq.SNDTIMEO, 5000)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(self._bind_url)
        self._thread = threading.Thread(
            target=self._run, name="cxl-p2p-server", daemon=True
        )
        self._thread.start()
        logger.info("CXL P2P server listening on %s", self._bind_url)

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop_event.set()
        t = self._thread
        if t is not None:
            t.join(timeout_s)
            if t.is_alive():
                logger.warning(
                    "CXL P2P server thread did not exit within %.1fs", timeout_s
                )
            self._thread = None
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None
        try:
            self._donor.close()
        except Exception:
            logger.exception("CXLDonor.close failed during shutdown")

    # -------- internals --------------------------------------------------

    def _run(self) -> None:
        assert self._socket is not None
        while not self._stop_event.is_set():
            try:
                raw = self._socket.recv()
            except zmq.Again:
                continue  # RCVTIMEO; check stop_event and retry
            except zmq.ZMQError as e:
                if self._stop_event.is_set():
                    break
                logger.warning("CXL P2P server recv error: %s", e)
                continue

            try:
                msg = self._decoder.decode(raw)
            except msgspec.DecodeError as e:
                logger.warning("CXL P2P server decode error: %s", e)
                # ZMQ REP requires a response per request, even on
                # error — send a synthetic ALL_NACK so the client
                # doesn't hang.
                self._send(PushKVToCXLRetMsg(num_committed=0, status=PushStatus.ALL_NACK))
                continue

            if isinstance(msg, PushKVToCXLMsg):
                try:
                    reply = self._donor.handle_push(msg)
                except Exception:
                    logger.exception("CXL donor handler raised")
                    reply = PushKVToCXLRetMsg(
                        num_committed=0, status=PushStatus.ALL_NACK
                    )
                self._send(reply)
            else:
                logger.warning(
                    "CXL P2P server got unexpected message type %s",
                    type(msg).__name__,
                )
                self._send(
                    PushKVToCXLRetMsg(num_committed=0, status=PushStatus.ALL_NACK)
                )

    def _send(self, reply: PushKVToCXLRetMsg) -> None:
        assert self._socket is not None
        try:
            self._socket.send(self._encoder.encode(reply))
        except zmq.ZMQError as e:
            logger.warning("CXL P2P server send error: %s", e)


class CXLP2PClient(DonorEndpoint):
    """Client-side handle that calls a remote `CXLP2PServer`.

    Implements `DonorEndpoint`, so `remote_fetch(...)` accepts it
    directly. Each client owns one REQ socket per donor URL; sockets
    are created lazily on first use.

    Concurrency: ZMQ REQ sockets are NOT thread-safe. We serialize
    access to each socket with a lock — the request rate is naturally
    low (one round-trip per CXL miss), so this is not a bottleneck.
    """

    def __init__(
        self,
        donor_url: str,
        *,
        context: Optional[zmq.Context] = None,
        recv_timeout_ms: int = 5000,
        send_timeout_ms: int = 5000,
    ):
        self._donor_url = donor_url
        self._context = context or zmq.Context.instance()
        self._recv_timeout_ms = recv_timeout_ms
        self._send_timeout_ms = send_timeout_ms
        self._lock = threading.Lock()
        self._socket: Optional[zmq.Socket] = None
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(CXLP2PMsg)

    def handle_push(self, msg: PushKVToCXLMsg) -> PushKVToCXLRetMsg:
        with self._lock:
            sock = self._ensure_socket()
            try:
                sock.send(self._encoder.encode(msg))
                raw = sock.recv()
            except zmq.ZMQError:
                # On any ZMQ error, drop the socket so the next call
                # rebuilds it. REQ/REP is brittle to mid-flight errors
                # (the socket gets stuck in send-vs-recv state).
                self._reset_socket()
                raise
            reply = self._decoder.decode(raw)
            if not isinstance(reply, PushKVToCXLRetMsg):
                raise RuntimeError(
                    f"unexpected reply type: {type(reply).__name__}"
                )
            return reply

    def close(self) -> None:
        with self._lock:
            self._reset_socket()

    # -------- internals --------------------------------------------------

    def _ensure_socket(self) -> zmq.Socket:
        if self._socket is None:
            sock = self._context.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, self._recv_timeout_ms)
            sock.setsockopt(zmq.SNDTIMEO, self._send_timeout_ms)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(self._donor_url)
            self._socket = sock
        return self._socket

    def _reset_socket(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None
