# SPDX-License-Identifier: Apache-2.0
"""ZMQ control-plane transport for the RDMA/NIXL peer L2 adapter.

A thin ZMQ REP/REQ pair around the donor's ``handle_lookup`` /
``handle_unlock``. Modeled on the CXL adapter's
``CXLP2PServer`` / ``CXLP2PClient`` (same recv-timeout sweep loop, same
lazy-socket + per-socket-lock REQ client, same "always reply on ZMQ REP"
discipline).

- ``NixlPeerControlServer``: one per node. Binds a REP socket, dispatches
  ``RemoteLookupReq`` / ``RemoteUnlockReq`` to the donor, and runs the
  donor's lease-expiry sweep on each idle recv timeout (no extra thread).
- ``NixlPeerControlClient``: one per peer. Owns a lazily-created REQ
  socket, serialized by a lock (ZMQ REQ sockets are not thread-safe).
"""

# Standard
from typing import Optional
import threading

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.l2_adapters.nixl_peer_donor import NixlPeerDonor
from lmcache.v1.distributed.l2_adapters.nixl_peer_messages import (
    NixlPeerMsg,
    RemoteLookupReq,
    RemoteLookupResp,
    RemoteUnlockReq,
    RemoteUnlockResp,
)

logger = init_logger(__name__)

# Default port for the control REP socket. Callers can override.
DEFAULT_NIXL_PEER_CONTROL_PORT = 8500

# How often the server's idle recv loop runs the donor lease sweep.
_RECV_TIMEOUT_MS = 1000


def _normalize_zmq_url(url: str) -> str:
    """Return ``url`` with a ``tcp://`` scheme, adding one if absent.

    The control sockets bind/connect this value directly, so it needs a
    full ZMQ endpoint. We accept either ``tcp://host:port`` or a bare
    ``host:port`` (so control and init URLs can be written the same way
    in config) and normalize to the former. A URL that already carries a
    scheme (any ``<scheme>://``) is returned unchanged so non-TCP
    transports (e.g. ``ipc://``) still work.

    Args:
        url: A ZMQ endpoint, with or without a scheme.

    Returns:
        The endpoint with a ``tcp://`` scheme when none was present.
    """
    return url if "://" in url else f"tcp://{url}"


class NixlPeerControlServer:
    """ZMQ REP server for ``RemoteLookupReq`` / ``RemoteUnlockReq``.

    Runs dispatch in a background daemon thread; binds to ``bind_url``
    until ``stop()`` is called. On each ``RCVTIMEO`` idle tick it calls
    ``donor.sweep_expired()`` so stranded read-locks are reclaimed even
    when no requests arrive.
    """

    def __init__(
        self,
        donor: NixlPeerDonor,
        bind_url: str = f"tcp://*:{DEFAULT_NIXL_PEER_CONTROL_PORT}",
        *,
        context: Optional[zmq.Context] = None,
    ):
        """Initialize the control server.

        Args:
            donor: The donor that answers lookups/unlocks against L1.
            bind_url: ZMQ bind URL for the REP socket. May be a full
                ``tcp://host:port`` or a bare ``host:port`` (a ``tcp://``
                scheme is added when absent).
            context: Optional shared ZMQ context; a process-wide instance
                is used when omitted.
        """
        self._donor = donor
        self._bind_url = _normalize_zmq_url(bind_url)
        self._context = context or zmq.Context.instance()
        self._socket: Optional[zmq.Socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(NixlPeerMsg)

    def start(self) -> None:
        """Bind the REP socket and start the dispatch thread.

        Raises:
            RuntimeError: If the server was already started.
        """
        if self._thread is not None:
            raise RuntimeError("NixlPeerControlServer already started")
        self._stop_event.clear()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.RCVTIMEO, _RECV_TIMEOUT_MS)
        self._socket.setsockopt(zmq.SNDTIMEO, 5000)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(self._bind_url)
        self._thread = threading.Thread(
            target=self._run, name="nixl-peer-control-server", daemon=True
        )
        self._thread.start()
        logger.info("NIXL peer control server listening on %s", self._bind_url)

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop the dispatch thread, close the socket, and close the donor.

        Args:
            timeout_s: How long to wait for the dispatch thread to exit.
        """
        self._stop_event.set()
        t = self._thread
        if t is not None:
            t.join(timeout_s)
            if t.is_alive():
                logger.warning(
                    "NIXL peer control server thread did not exit within %.1fs",
                    timeout_s,
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
            logger.exception("NixlPeerDonor.close failed during shutdown")

    def bound_endpoint(self) -> str:
        """Return the actual endpoint the REP socket is bound to.

        Useful when ``bind_url`` used port 0 (OS-assigned port): the
        concrete ``tcp://host:port`` is only known after ``start()``.

        Returns:
            The resolved bind endpoint.

        Raises:
            RuntimeError: If the server has not been started.
        """
        if self._socket is None:
            raise RuntimeError("NixlPeerControlServer not started")
        return self._socket.getsockopt_string(zmq.LAST_ENDPOINT)

    # -------- internals --------------------------------------------------

    def _run(self) -> None:
        assert self._socket is not None
        while not self._stop_event.is_set():
            try:
                raw = self._socket.recv()
            except zmq.Again:
                # Idle tick: reclaim any read-locks whose lease elapsed.
                self._donor.sweep_expired()
                continue
            except zmq.ZMQError as e:
                if self._stop_event.is_set():
                    break
                logger.warning("NIXL peer control server recv error: %s", e)
                continue

            try:
                msg = self._decoder.decode(raw)
            except msgspec.DecodeError as e:
                logger.warning("NIXL peer control decode error: %s", e)
                # ZMQ REP must answer every request; reply with an empty
                # unlock ack so the client doesn't hang.
                self._send(RemoteUnlockResp(num_released=0))
                continue

            self._send(self._dispatch(msg))

    def _dispatch(self, msg: NixlPeerMsg) -> NixlPeerMsg:
        if isinstance(msg, RemoteLookupReq):
            try:
                return self._donor.handle_lookup(msg)
            except Exception:
                logger.exception("NIXL peer donor lookup handler raised")
                n = len(msg.keys)
                return RemoteLookupResp(
                    found=[False] * n,
                    page_indices=[-1] * n,
                    sizes=[0] * n,
                    peer_agent_id="",
                )
        if isinstance(msg, RemoteUnlockReq):
            try:
                return self._donor.handle_unlock(msg)
            except Exception:
                logger.exception("NIXL peer donor unlock handler raised")
                return RemoteUnlockResp(num_released=0)
        logger.warning(
            "NIXL peer control server got unexpected message type %s",
            type(msg).__name__,
        )
        return RemoteUnlockResp(num_released=0)

    def _send(self, reply: NixlPeerMsg) -> None:
        assert self._socket is not None
        try:
            self._socket.send(self._encoder.encode(reply))
        except zmq.ZMQError as e:
            logger.warning("NIXL peer control server send error: %s", e)


class NixlPeerControlClient:
    """Client handle that calls a remote ``NixlPeerControlServer``.

    Owns one lazily-created REQ socket per peer URL. ZMQ REQ sockets are
    not thread-safe, so access is serialized by a lock; the request rate
    is low (one round-trip per prefetch miss), so this is not a
    bottleneck.
    """

    def __init__(
        self,
        control_url: str,
        *,
        context: Optional[zmq.Context] = None,
        recv_timeout_ms: int = 30000,
        send_timeout_ms: int = 30000,
    ):
        """Initialize the control client.

        Args:
            control_url: The peer's ``NixlPeerControlServer`` URL. May be
                a full ``tcp://host:port`` or a bare ``host:port`` (a
                ``tcp://`` scheme is added when absent).
            context: Optional shared ZMQ context.
            recv_timeout_ms: REQ socket receive timeout.
            send_timeout_ms: REQ socket send timeout.
        """
        self._control_url = _normalize_zmq_url(control_url)
        self._context = context or zmq.Context.instance()
        self._recv_timeout_ms = recv_timeout_ms
        self._send_timeout_ms = send_timeout_ms
        self._lock = threading.Lock()
        self._socket: Optional[zmq.Socket] = None
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(NixlPeerMsg)

    def lookup(self, req: RemoteLookupReq) -> RemoteLookupResp:
        """Send a ``RemoteLookupReq`` and return the donor's response.

        Args:
            req: The lookup request.

        Returns:
            The donor's ``RemoteLookupResp``.

        Raises:
            zmq.ZMQError: On a transport error (the socket is reset so the
                next call rebuilds it).
            RuntimeError: If the reply is the wrong message type.
        """
        reply = self._round_trip(req)
        if not isinstance(reply, RemoteLookupResp):
            raise RuntimeError(f"unexpected lookup reply type: {type(reply).__name__}")
        return reply

    def unlock(self, req: RemoteUnlockReq) -> RemoteUnlockResp:
        """Send a ``RemoteUnlockReq`` and return the donor's response.

        Args:
            req: The unlock request.

        Returns:
            The donor's ``RemoteUnlockResp``.

        Raises:
            zmq.ZMQError: On a transport error (the socket is reset).
            RuntimeError: If the reply is the wrong message type.
        """
        reply = self._round_trip(req)
        if not isinstance(reply, RemoteUnlockResp):
            raise RuntimeError(f"unexpected unlock reply type: {type(reply).__name__}")
        return reply

    def close(self) -> None:
        """Close the underlying REQ socket."""
        with self._lock:
            self._reset_socket()

    # -------- internals --------------------------------------------------

    def _round_trip(self, msg: NixlPeerMsg) -> NixlPeerMsg:
        with self._lock:
            sock = self._ensure_socket()
            try:
                sock.send(self._encoder.encode(msg))
                raw = sock.recv()
            except zmq.ZMQError:
                # REQ/REP is brittle to a mid-flight error (the socket
                # wedges in send-vs-recv state); drop it so the next call
                # rebuilds.
                self._reset_socket()
                raise
            return self._decoder.decode(raw)

    def _ensure_socket(self) -> zmq.Socket:
        if self._socket is None:
            sock = self._context.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, self._recv_timeout_ms)
            sock.setsockopt(zmq.SNDTIMEO, self._send_timeout_ms)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(self._control_url)
            self._socket = sock
        return self._socket

    def _reset_socket(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None
