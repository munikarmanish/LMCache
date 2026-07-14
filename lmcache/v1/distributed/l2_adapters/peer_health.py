# SPDX-License-Identifier: Apache-2.0
"""Peer liveness tracking + background reconnection for static-peer L2 adapters.

Both the CXL and the NIXL-peer L2 adapters take a *static* list of peers
from config. A peer may be down when this node starts (the operator
launched one node first), may die mid-run, or may come up later. Without
liveness tracking, every request that consults a dead peer pays that
peer's full RPC timeout (seconds) before falling through — so a single
node launched alone stalls on every lookup miss waiting for peers that
will never answer.

``PeerHealthMonitor`` keeps that cost off the request path:

- Each peer has an ``alive`` flag. The request path calls
  :meth:`is_alive` and simply **skips** peers marked dead (an instant
  MISS for that peer), so a dead peer never blocks a lookup.
- The request path reports outcomes via :meth:`record_success` /
  :meth:`record_failure`. A failure demotes a peer to dead immediately,
  so at most one request pays the timeout when a live peer dies.
- A background daemon thread periodically probes the **dead** peers with
  a short, cheap, side-effect-free health check (the injected
  ``probe_fn``, e.g. a ZMQ ping on a 1 s timeout). A peer that answers is
  promoted back to alive. This is what lets a node started alone pick up
  a peer that comes online later, with no request ever blocking.

The probe function and the notion of "peer" are injected, so the monitor
is transport-agnostic: the CXL adapter probes via ``CXLP2PClient.ping``
and the NIXL adapter via a zero-key control lookup. Peers are identified
by their index into the adapter's peer list.
"""

# Future
from __future__ import annotations

# Standard
from typing import Callable
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class PeerHealthMonitor:
    """Tracks per-peer liveness and reconnects dead peers in the background.

    Thread-safe. A single background daemon thread runs the probe loop;
    the request-path methods (:meth:`is_alive`, :meth:`record_success`,
    :meth:`record_failure`) only touch a lock-guarded state list and never
    block on I/O.

    Peers are identified by their integer index into the owning adapter's
    peer list. The monitor holds no reference to the peers themselves — it
    calls ``probe_fn(peer_index)`` to test one and ``describe_fn`` (if
    given) only for log messages.
    """

    def __init__(
        self,
        num_peers: int,
        probe_fn: Callable[[int], bool],
        *,
        probe_interval_s: float,
        name: str = "peer-health",
        describe_fn: Callable[[int], str] | None = None,
        start_alive: bool = False,
    ):
        """Initialize the monitor (does not start the thread; call :meth:`start`).

        Args:
            num_peers: Number of configured peers. Peer ids are
                ``0 .. num_peers - 1``.
            probe_fn: Callable that probes one peer by index and returns
                ``True`` iff it is reachable. Must be cheap and
                self-bounded (carry its own short timeout) — the monitor
                calls it from the probe thread and does not impose its own
                timeout.
            probe_interval_s: Seconds between probe sweeps of the dead
                peers. Must be positive.
            name: Thread name suffix, for debuggability.
            describe_fn: Optional callable mapping a peer index to a short
                human-readable label used only in log lines. Defaults to
                ``"peer[<index>]"``.
            start_alive: If ``True``, peers begin ``alive`` (optimistic).
                Default ``False`` — peers begin ``dead`` and the first
                probe sweep (run immediately on thread start) promotes the
                reachable ones, so no request ever blocks on an unproven
                peer.

        Raises:
            ValueError: If ``num_peers`` is negative or
                ``probe_interval_s`` is not positive.
        """
        if num_peers < 0:
            raise ValueError("num_peers must be non-negative")
        if probe_interval_s <= 0:
            raise ValueError("probe_interval_s must be positive")

        self._probe_fn = probe_fn
        self._probe_interval_s = probe_interval_s
        self._name = name
        self._describe_fn = describe_fn or (lambda i: f"peer[{i}]")

        self._lock = threading.Lock()
        self._alive: list[bool] = [start_alive] * num_peers

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background probe thread.

        Idempotent-unsafe: raises if already started. With no peers the
        thread is not started (there is nothing to probe).

        Raises:
            RuntimeError: If the monitor was already started.
        """
        if self._thread is not None:
            raise RuntimeError("PeerHealthMonitor already started")
        if not self._alive:
            # No peers configured — nothing to monitor.
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"{self._name}-monitor", daemon=True
        )
        self._thread.start()

    def is_alive(self, peer_index: int) -> bool:
        """Return whether ``peer_index`` is currently believed reachable.

        The request path calls this to decide whether to consult a peer.
        A ``False`` result means "skip this peer" — do not issue an RPC
        that would block on its timeout.

        Args:
            peer_index: The peer's index into the adapter's peer list.

        Returns:
            The peer's current liveness flag.
        """
        with self._lock:
            return self._alive[peer_index]

    def record_success(self, peer_index: int) -> None:
        """Mark ``peer_index`` alive after a successful request to it.

        Cheap to call on every successful RPC; only logs on a transition.

        Args:
            peer_index: The peer's index into the adapter's peer list.
        """
        with self._lock:
            was_alive = self._alive[peer_index]
            self._alive[peer_index] = True
        if not was_alive:
            logger.info(
                "%s: %s is now ALIVE (request succeeded)",
                self._name,
                self._describe_fn(peer_index),
            )

    def record_failure(self, peer_index: int) -> None:
        """Mark ``peer_index`` dead after a failed request to it.

        Demotes the peer immediately so subsequent requests skip it (and
        do not each pay its timeout) until the background prober finds it
        reachable again.

        Args:
            peer_index: The peer's index into the adapter's peer list.
        """
        with self._lock:
            was_alive = self._alive[peer_index]
            self._alive[peer_index] = False
        if was_alive:
            logger.warning(
                "%s: %s marked DEAD (request failed); background prober will "
                "retry every %.1fs",
                self._name,
                self._describe_fn(peer_index),
                self._probe_interval_s,
            )

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop the background probe thread.

        Args:
            timeout_s: How long to wait for the thread to exit.
        """
        self._stop_event.set()
        t = self._thread
        if t is not None:
            t.join(timeout_s)
            if t.is_alive():
                logger.warning(
                    "%s: probe thread did not exit within %.1fs",
                    self._name,
                    timeout_s,
                )
            self._thread = None

    # -------- internals --------------------------------------------------

    def _run(self) -> None:
        """Probe-loop body: sweep dead peers, then wait one interval.

        The first sweep runs immediately (no initial wait) so peers that
        are up at startup are promoted within one probe timeout, off the
        request path. ``Event.wait`` doubles as the interval sleep and the
        stop signal, so shutdown is prompt.
        """
        while not self._stop_event.is_set():
            self._probe_dead_peers()
            # wait() returns True as soon as stop() is called, else times
            # out after the interval — either way we loop and re-check.
            self._stop_event.wait(self._probe_interval_s)

    def _probe_dead_peers(self) -> None:
        """Probe every peer currently marked dead; promote those that answer.

        Only dead peers are probed — a live peer's health is tracked by
        the request path's success/failure reports, so probing it would be
        redundant traffic. A probe that raises is treated as "still dead".
        """
        with self._lock:
            dead = [i for i, alive in enumerate(self._alive) if not alive]

        for peer_index in dead:
            if self._stop_event.is_set():
                return
            try:
                reachable = self._probe_fn(peer_index)
            except Exception:
                logger.debug(
                    "%s: probe of %s raised; leaving it DEAD",
                    self._name,
                    self._describe_fn(peer_index),
                )
                reachable = False
            if reachable:
                self.record_success(peer_index)
