# SPDX-License-Identifier: Apache-2.0
"""Bridges between the CXL backend and the LMCache cluster controller.

Two adapters live here:

- `ControllerLivenessProvider` — turns the cluster controller's
  worker registry (queryable via `QueryWorkerInfoMsg` with
  `instance_id="all"`) into a `LivenessProvider` callable that the
  CXL `CXLGarbageCollector` accepts unchanged. Accepts a
  configuration mapping from LMCache `instance_id` to CXL `node_id`
  so the GC can speak in CXL terms.

- `ControllerDonorRouter` — turns the cluster controller's
  `BatchedP2PLookupMsg` into a donor lookup that produces a live
  `DonorEndpoint` for `remote_fetch`. Caches `peer_init_url ->
  CXLP2PClient` so each peer's TCP connection is reused.

Neither piece is required to use the CXL backend. Tests and single-
node deployments construct GC with a closure over a fake alive set
and call `remote_fetch` with a directly-built `CXLP2PClient`. These
adapters exist so production code doesn't have to write the same
plumbing.

Threading model:
  - Both adapters are thread-safe.
  - The controller worker (`lmcache_worker`) is asyncio-driven; we
    drive its async APIs via `asyncio.run_coroutine_threadsafe` onto
    the worker's loop, then `.result()` for the synchronous side.
"""

# Standard
import threading
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple
import asyncio
import time

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_controller.message import (
    BatchedP2PLookupMsg,
    BatchedP2PLookupRetMsg,
    QueryWorkerInfoMsg,
    QueryWorkerInfoRetMsg,
    WorkerInfo,
)
from lmcache.v1.storage_backend.cxl.cross_node import DonorEndpoint
from lmcache.v1.storage_backend.cxl.gc import LivenessProvider
from lmcache.v1.storage_backend.cxl.p2p_messages import (
    PushKVToCXLMsg,
    PushKVToCXLRetMsg,
    PushStatus,
)
from lmcache.v1.storage_backend.cxl.p2p_transport import CXLP2PClient

logger = init_logger(__name__)


# A small protocol describing what the LMCacheWorker side exposes.
# We deliberately don't import LMCacheWorker directly so this module
# stays importable in tests that don't have the full controller stack.
class _ControllerClient:
    """Subset of LMCacheWorker we use. See lmcache_worker.py."""

    loop: asyncio.AbstractEventLoop

    async def async_put_and_wait_msg(self, msg):
        ...


# -------- liveness ---------------------------------------------------


@dataclass(frozen=True)
class InstanceMapping:
    """One row of the (lmcache instance) → (CXL node id) mapping.

    A CXL pool is a flat array of `MAX_NODES` slots. The cluster
    administrator picks which LMCache instance occupies which CXL
    node id at boot time; the mapping is stable for the life of the
    pool. If an instance disappears (cluster controller deregisters
    it), the GC reclaims that node id's regions.
    """

    instance_id: str
    cxl_node_id: int


class ControllerLivenessProvider:
    """Polls the controller for alive instances; returns a CXL-node-id set.

    Construction:
      provider = ControllerLivenessProvider(
          worker=lmcache_worker,                 # LMCacheWorker
          query_event_id_prefix="cxl-gc-",
          mapping=[
              InstanceMapping("inst-A", cxl_node_id=0),
              InstanceMapping("inst-B", cxl_node_id=1),
              ...
          ],
          stale_after_s=120.0,                   # consider workers older
                                                  # than this dead even if
                                                  # the registry still has them
      )

    Usage by the GC:
      gc = CXLGarbageCollector(
          ...,
          liveness=provider,    # provider is callable: () -> FrozenSet[int]
      )

    Caching: each call sends a query and waits for the reply. Don't
    drive this faster than the controller's heartbeat interval; the
    GC's default 30-second sweep is comfortably above that.

    Thread safety: callable from any thread. The async send goes
    through `asyncio.run_coroutine_threadsafe(self._worker.loop)` so
    we don't hold the GIL waiting on a network round-trip.
    """

    def __init__(
        self,
        worker: _ControllerClient,
        mapping: List[InstanceMapping],
        *,
        query_event_id_prefix: str = "cxl-gc-",
        stale_after_s: float = 120.0,
        rpc_timeout_s: float = 5.0,
    ):
        self._worker = worker
        # instance_id -> cxl_node_id (forward direction). Validate
        # the inverse is unique — we don't allow two instances to
        # share a CXL node id.
        self._instance_to_node: Dict[str, int] = {}
        seen_node_ids: set[int] = set()
        for row in mapping:
            if row.cxl_node_id in seen_node_ids:
                raise ValueError(
                    f"duplicate cxl_node_id {row.cxl_node_id} in mapping"
                )
            seen_node_ids.add(row.cxl_node_id)
            self._instance_to_node[row.instance_id] = row.cxl_node_id
        self._known_node_ids: FrozenSet[int] = frozenset(seen_node_ids)
        self._query_event_id_prefix = query_event_id_prefix
        self._stale_after_s = stale_after_s
        self._rpc_timeout_s = rpc_timeout_s
        self._call_counter = 0
        self._call_counter_lock = threading.Lock()

    @property
    def known_node_ids(self) -> FrozenSet[int]:
        """The CXL node ids configured by the mapping. Ground truth for the GC."""
        return self._known_node_ids

    def __call__(self) -> FrozenSet[int]:
        """Return the set of CXL node ids whose LMCache instance is alive."""
        with self._call_counter_lock:
            self._call_counter += 1
            event_id = f"{self._query_event_id_prefix}{self._call_counter}"

        msg = QueryWorkerInfoMsg(
            event_id=event_id,
            instance_id="all",
            worker_ids=None,
        )
        try:
            reply = self._send(msg)
        except Exception:
            # Controller unreachable: fail safe by returning *all*
            # known node ids as alive. A transient outage must NOT
            # cause the GC to reclaim everyone's regions.
            logger.warning(
                "ControllerLivenessProvider RPC failed; treating all known "
                "instances as alive for this sweep"
            )
            return self._known_node_ids

        return self._distill_alive_node_ids(reply.worker_infos)

    def _distill_alive_node_ids(
        self, worker_infos: List[WorkerInfo]
    ) -> FrozenSet[int]:
        """An instance is "alive" if it has at least one fresh worker.

        "Fresh" = `last_heartbeat_time >= now - stale_after_s`.
        """
        now = time.time()
        cutoff = now - self._stale_after_s
        alive_instances: set[str] = set()
        for info in worker_infos:
            if info.last_heartbeat_time >= cutoff:
                alive_instances.add(info.instance_id)

        alive_node_ids: set[int] = set()
        for instance_id in alive_instances:
            cxl_node_id = self._instance_to_node.get(instance_id)
            if cxl_node_id is not None:
                alive_node_ids.add(cxl_node_id)
        return frozenset(alive_node_ids)

    def _send(self, msg: QueryWorkerInfoMsg) -> QueryWorkerInfoRetMsg:
        """Run the worker's async send on its event loop, block for reply."""
        future = asyncio.run_coroutine_threadsafe(
            self._worker.async_put_and_wait_msg(msg), self._worker.loop
        )
        ret = future.result(timeout=self._rpc_timeout_s)
        if not isinstance(ret, QueryWorkerInfoRetMsg):
            raise RuntimeError(
                f"unexpected reply to QueryWorkerInfoMsg: {type(ret).__name__}"
            )
        return ret


# -------- donor routing --------------------------------------------


@dataclass(frozen=True)
class DonorRoute:
    """Resolved donor for a CXL miss: who has it, where to reach them."""

    donor_node_id: int
    """The CXL node id that owns the slots we'll reserve on the donor's behalf."""

    donor_url: str
    """The donor's CXLP2PServer URL — what CXLP2PClient connects to."""

    num_hit: int
    """Length of the contiguous prefix the donor advertised."""


class ControllerDonorRouter:
    """Discovers donors via the controller and reuses CXLP2PClients.

    `BatchedP2PLookupMsg` returns `(instance_id, location, num_hit_chunks,
    peer_init_url)`. We translate `instance_id` to a CXL node id via
    the same mapping the LivenessProvider uses, and convert
    `peer_init_url` (the LMCache P2P init host:port) to the donor's
    CXL P2P server URL via a caller-supplied transformation.

    The translation step is deliberately pluggable: the LMCache P2P
    URL is used by `P2PBackend` for KV transfer; the CXL P2P URL is
    used by `CXLP2PServer`. They typically share a host but use
    different ports. The default `derive_donor_url_from_peer` assumes
    the same host and the configured CXL port (default 8447); pass a
    custom callable if your deployment uses a different scheme.
    """

    DEFAULT_CXL_PORT = 8447

    def __init__(
        self,
        worker: _ControllerClient,
        mapping: List[InstanceMapping],
        *,
        derive_donor_url: Optional[Callable[[str], str]] = None,
        rpc_timeout_s: float = 5.0,
    ):
        self._worker = worker
        self._instance_to_node: Dict[str, int] = {
            row.instance_id: row.cxl_node_id for row in mapping
        }
        self._derive_donor_url = derive_donor_url or self._default_derive_url
        self._rpc_timeout_s = rpc_timeout_s

        # Cache: donor_url -> CXLP2PClient. Reused across calls.
        self._client_cache: Dict[str, CXLP2PClient] = {}
        self._client_cache_lock = threading.Lock()

    @staticmethod
    def _default_derive_url(peer_init_url: str) -> str:
        """Default mapping: extract host from peer_init_url, use CXL_PORT.

        peer_init_url is "host:port" per `LMCacheWorker.p2p_init_url`.
        We strip the port and append our CXL P2P port.
        """
        # peer_init_url format: "host:port" (NOT "tcp://host:port")
        if ":" in peer_init_url:
            host = peer_init_url.rsplit(":", 1)[0]
        else:
            host = peer_init_url
        return f"tcp://{host}:{ControllerDonorRouter.DEFAULT_CXL_PORT}"

    def lookup(
        self,
        keys: List[CacheEngineKey],
        *,
        requester_instance_id: str,
        requester_worker_id: int,
    ) -> Optional[DonorRoute]:
        """Ask the controller who has these keys in their local tier.

        Returns None if the controller has no donor (caller should
        recompute on GPU). Otherwise returns the donor route plus a
        `num_hit` length the caller uses to slice the request.
        """
        msg = BatchedP2PLookupMsg(
            hashes=[k.chunk_hash for k in keys],
            instance_id=requester_instance_id,
            worker_id=requester_worker_id,
        )
        try:
            reply = self._send(msg)
        except Exception:
            logger.warning("ControllerDonorRouter RPC failed; no donor returned")
            return None

        # The reply's layout_info is a list of one tuple
        # (or empty / sentinel on failure).
        if not reply.layout_info:
            return None
        donor_inst, _location, num_hit, peer_url = reply.layout_info[0]
        if num_hit <= 0 or not peer_url:
            return None
        donor_node_id = self._instance_to_node.get(donor_inst)
        if donor_node_id is None:
            logger.warning(
                "controller returned donor instance %r which has no CXL node "
                "mapping; ignoring",
                donor_inst,
            )
            return None
        donor_url = self._derive_donor_url(peer_url)
        return DonorRoute(
            donor_node_id=donor_node_id,
            donor_url=donor_url,
            num_hit=int(num_hit),
        )

    def get_endpoint(self, donor_url: str) -> DonorEndpoint:
        """Return a (cached) CXLP2PClient pointed at `donor_url`."""
        with self._client_cache_lock:
            client = self._client_cache.get(donor_url)
            if client is None:
                client = CXLP2PClient(donor_url=donor_url)
                self._client_cache[donor_url] = client
            return client

    def close(self) -> None:
        """Close all cached clients. Idempotent."""
        with self._client_cache_lock:
            clients = list(self._client_cache.values())
            self._client_cache.clear()
        for c in clients:
            try:
                c.close()
            except Exception:
                logger.exception("CXLP2PClient.close failed")

    def _send(self, msg: BatchedP2PLookupMsg) -> BatchedP2PLookupRetMsg:
        future = asyncio.run_coroutine_threadsafe(
            self._worker.async_put_and_wait_msg(msg), self._worker.loop
        )
        ret = future.result(timeout=self._rpc_timeout_s)
        if not isinstance(ret, BatchedP2PLookupRetMsg):
            raise RuntimeError(
                f"unexpected reply to BatchedP2PLookupMsg: {type(ret).__name__}"
            )
        return ret


# -------- one-shot fetch using the router -------------------------


@dataclass
class ControllerBackedFetch:
    """Convenience: bind a router + index_writer to drive remote_fetch.

    Reasonable production wiring: the CXL adapter on a node holds one
    of these; on a CXL miss it calls `fetch(keys)` and gets back a
    `RemoteFetchResult` (or None if the controller returned no donor).
    """

    router: ControllerDonorRouter
    index_writer: object  # CXLIndexWriter — kept loose to avoid a cycle
    requester_node_id: int
    requester_instance_id: str
    requester_worker_id: int
    epoch_provider: Callable[[], int]
    sender_id: str

    def fetch(self, keys: List[CacheEngineKey]):
        """Look up a donor and run remote_fetch. None on no-donor."""
        # First Party
        from lmcache.v1.storage_backend.cxl.cross_node import remote_fetch

        route = self.router.lookup(
            keys,
            requester_instance_id=self.requester_instance_id,
            requester_worker_id=self.requester_worker_id,
        )
        if route is None:
            return None
        endpoint = self.router.get_endpoint(route.donor_url)
        # Restrict to the prefix the donor advertised — extra keys
        # would just be NACKed and waste a slot reservation.
        push_keys = keys[: route.num_hit]
        if not push_keys:
            return None
        return remote_fetch(
            requester_node_id=self.requester_node_id,
            keys=push_keys,
            index_writer=self.index_writer,
            donor_node_id=route.donor_node_id,
            donor=endpoint,
            sender_id=self.sender_id,
            epoch=self.epoch_provider(),
        )
