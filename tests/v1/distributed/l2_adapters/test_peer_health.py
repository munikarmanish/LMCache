# SPDX-License-Identifier: Apache-2.0
"""Tests for PeerHealthMonitor and the CXL/NIXL liveness ping transports.

Covers:
  - PeerHealthMonitor: request-path is_alive/record_success/record_failure,
    background probe promotion of dead peers, and demote-then-reprobe.
  - CXLP2PClient.ping / NixlPeerControlClient.ping: live server answers True
    fast; a dead endpoint returns False within the short probe timeout (NOT
    the long data/control timeout).
"""

# Standard
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.peer_health import PeerHealthMonitor


# ---------------------------------------------------------------------------
# PeerHealthMonitor
# ---------------------------------------------------------------------------


def _poll_until(fn, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if fn():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met within timeout")


def test_config_validation():
    with pytest.raises(ValueError, match="num_peers"):
        PeerHealthMonitor(-1, lambda i: True, probe_interval_s=1.0)
    with pytest.raises(ValueError, match="probe_interval_s"):
        PeerHealthMonitor(1, lambda i: True, probe_interval_s=0.0)


def test_start_alive_false_begins_dead():
    m = PeerHealthMonitor(2, lambda i: False, probe_interval_s=10.0, start_alive=False)
    # Not started: nothing probes, all peers dead.
    assert not m.is_alive(0)
    assert not m.is_alive(1)


def test_start_alive_true_begins_alive():
    m = PeerHealthMonitor(2, lambda i: False, probe_interval_s=10.0, start_alive=True)
    assert m.is_alive(0)
    assert m.is_alive(1)


def test_record_success_and_failure_flip_state():
    m = PeerHealthMonitor(1, lambda i: False, probe_interval_s=10.0)
    assert not m.is_alive(0)
    m.record_success(0)
    assert m.is_alive(0)
    m.record_failure(0)
    assert not m.is_alive(0)


def test_background_prober_promotes_reachable_peer():
    # Peer 0 becomes reachable only after `flip`; the prober must pick it
    # up without any request-path call.
    reachable = threading.Event()

    def probe(i: int) -> bool:
        return reachable.is_set()

    m = PeerHealthMonitor(1, probe, probe_interval_s=0.05)
    m.start()
    try:
        # Still dead until we flip the fake endpoint up.
        time.sleep(0.15)
        assert not m.is_alive(0)
        reachable.set()
        _poll_until(lambda: m.is_alive(0))
    finally:
        m.stop()


def test_prober_reprobes_after_demote():
    # A peer that is reachable but demoted by a request failure must be
    # re-promoted by the prober.
    m = PeerHealthMonitor(1, lambda i: True, probe_interval_s=0.05)
    m.start()
    try:
        _poll_until(lambda: m.is_alive(0))
        m.record_failure(0)
        assert not m.is_alive(0)
        _poll_until(lambda: m.is_alive(0))
    finally:
        m.stop()


def test_prober_does_not_probe_live_peers():
    # Only dead peers are probed. A peer kept alive by the request path is
    # never handed to probe_fn.
    probed: list[int] = []

    def probe(i: int) -> bool:
        probed.append(i)
        return True

    m = PeerHealthMonitor(2, probe, probe_interval_s=0.05, start_alive=True)
    m.start()
    try:
        time.sleep(0.2)
        assert probed == []
    finally:
        m.stop()


def test_probe_exception_leaves_peer_dead():
    def probe(i: int) -> bool:
        raise RuntimeError("probe blew up")

    m = PeerHealthMonitor(1, probe, probe_interval_s=0.05)
    m.start()
    try:
        time.sleep(0.2)
        assert not m.is_alive(0)
    finally:
        m.stop()


def test_stop_is_prompt_and_idempotent_after_start():
    m = PeerHealthMonitor(1, lambda i: False, probe_interval_s=100.0)
    m.start()
    t0 = time.monotonic()
    m.stop()
    # Must not wait the full 100 s interval (Event.wait interrupts on stop).
    assert time.monotonic() - t0 < 5.0
    # A second stop is harmless.
    m.stop()


def test_start_twice_raises():
    m = PeerHealthMonitor(1, lambda i: False, probe_interval_s=10.0)
    m.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            m.start()
    finally:
        m.stop()


def test_no_peers_start_is_noop():
    m = PeerHealthMonitor(0, lambda i: True, probe_interval_s=1.0)
    m.start()  # no thread, no error
    m.stop()


# ---------------------------------------------------------------------------
# CXL ping transport
# ---------------------------------------------------------------------------


class _FakeCXLDonor:
    def handle_push(self, msg):
        raise AssertionError("ping must not reach handle_push")

    def close(self):
        pass


def test_cxl_ping_live_and_dead():
    # Third Party
    import zmq

    # First Party
    from lmcache.v1.storage_backend.cxl.p2p_transport import (
        CXLP2PClient,
        CXLP2PServer,
    )

    server = CXLP2PServer(donor=_FakeCXLDonor(), bind_url="tcp://127.0.0.1:0")
    server.start()
    # bound port -> reconnect a client at the concrete endpoint.
    endpoint = server._socket.getsockopt_string(zmq.LAST_ENDPOINT)
    client = CXLP2PClient(donor_url=endpoint)
    try:
        assert client.ping("node-0", timeout_ms=1000) is True
    finally:
        client.close()
        server.stop()

    # Dead endpoint: fast False, bounded by the short probe timeout.
    dead = CXLP2PClient(donor_url="tcp://127.0.0.1:9")
    try:
        t0 = time.monotonic()
        assert dead.ping("node-0", timeout_ms=300) is False
        assert time.monotonic() - t0 < 2.0
    finally:
        dead.close()


# ---------------------------------------------------------------------------
# NIXL control ping transport
# ---------------------------------------------------------------------------


def test_nixl_ping_dead_endpoint_is_fast_false():
    # First Party
    from lmcache.v1.distributed.l2_adapters.nixl_peer_transport import (
        NixlPeerControlClient,
    )

    dead = NixlPeerControlClient(
        "tcp://127.0.0.1:9", recv_timeout_ms=30000, send_timeout_ms=30000
    )
    try:
        t0 = time.monotonic()
        # Uses the short probe timeout, NOT the 30 s control timeout.
        assert dead.ping("node-0", timeout_ms=300) is False
        assert time.monotonic() - t0 < 2.0
    finally:
        dead.close()
