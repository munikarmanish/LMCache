# SPDX-License-Identifier: Apache-2.0
"""
Prefetch Controller: asynchronously prefetches data from L2 adapters into L1.

The controller runs a background thread with an event-driven loop that:
1. Accepts prefetch requests from external threads via submit_prefetch_request.
2. Submits lookup_and_lock tasks to all L2 adapters.
3. Computes a load plan, keeping the keys retained by the TrimPolicy
   (PREFIX, SEGMENTED_PREFIX, or SPARSE).
4. Reserves L1 write buffers and submits load tasks to L2 adapters.
5. On load completion, transitions L1 entries from write-locked to read-locked.
6. Reports the retained-key bitmap.
"""

# Standard
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable
import enum
import select
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.profile import PROFILE_ENABLED
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey, TrimPolicy
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.storage_controller import StorageControllerInterface
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    PrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.platform import (
    consume_fd,
    create_event_notifier,
)

logger = init_logger(__name__)


# HELPER FUNCTIONS
def merge_bitmaps(bitmaps: Iterable[Bitmap], num_keys: int) -> Bitmap:
    """Merge bitmaps with a bitwise OR into a ``num_keys``-sized bitmap.

    Always returns a ``num_keys``-sized bitmap (empty input -> all zeros), so
    downstream ``&`` operations never hit a size mismatch.
    """
    merged = Bitmap(num_keys)
    for bm in bitmaps:
        merged = merged | bm
    return merged


def build_trim_mask(
    found: Bitmap,
    num_keys: int,
    policy: TrimPolicy = TrimPolicy.PREFIX,
) -> Bitmap:
    """Subset of ``found`` to keep (load + read-lock + report); the rest is
    released.

    PREFIX trims at the first gap (leading contiguous run). The non-PREFIX
    policies keep every set bit, gaps included, and differ only in intent:
    SEGMENTED_PREFIX keeps the keys that loaded when an L2 hit fails to load
    into L1 (e.g. OOM) mid-prefix; SPARSE keeps an intentionally scattered set.

    Args:
        found: Bitmap of found keys, over key indices ``0..num_keys-1``.
        num_keys: Total number of requested keys.
        policy: Trim policy to apply (see :class:`TrimPolicy`).

    Returns:
        Bitmap of the retained key indices.
    """
    if policy is TrimPolicy.PREFIX:
        return Bitmap(num_keys, found.count_leading_ones())
    return found


def trim_load_plan_with_mask(
    load_plan: dict[int, Bitmap],
    mask: Bitmap,
) -> dict[int, Bitmap]:
    """Trim the load plan to the key indices set in ``mask`` (gap-tolerant).

    Args:
        load_plan: Mapping from adapter index to Bitmap of key indices.
        mask: Bitmap of key indices to retain.

    Returns:
        Trimmed load plan; adapter indices retaining no keys are dropped.
    """
    trimmed_plan: dict[int, Bitmap] = {}
    for adapter_idx, bitmap in load_plan.items():
        new_bitmap = bitmap & mask
        if new_bitmap.popcount() == 0:
            continue
        trimmed_plan[adapter_idx] = new_bitmap
    return trimmed_plan


# Poll timeout in milliseconds for the prefetch loop
PREFETCH_LOOP_POLL_TIMEOUT_MS = 500

PrefetchRequestId = int


class PrefetchPhase(enum.Enum):
    LOOKUP = enum.auto()
    PLAN_AND_LOAD = enum.auto()


@dataclass(frozen=True)
class L2ResidentTierInfo:
    """Per-key tier info for hits left L2-resident (GPU-direct retrieve).

    Reported alongside the prefetch result bitmap so the retrieve handler
    can drive ``submit_h2d`` for these keys (copying straight from L2 to
    GPU) instead of expecting them in L1. ``keys[i]`` lives on the adapter
    at ``adapter_indices[i]``; both tuples are parallel and indexed into
    the original request's key list.
    """

    keys: tuple[ObjectKey, ...] = ()
    adapter_indices: tuple[int, ...] = ()


@dataclass
class InFlightPrefetchRequest:
    """Tracks a single prefetch request across its lifecycle phases."""

    request_id: PrefetchRequestId
    keys: list[ObjectKey]
    layout_desc: MemoryLayoutDesc
    phase: PrefetchPhase
    extra_count: int = 0
    """Extra read locks per key (on top of the default 1) to acquire when
    transitioning from write-locked to read-locked.  Must match the
    ``extra_count`` used in the corresponding ``submit_prefetch_task`` call."""

    policy: TrimPolicy = TrimPolicy.PREFIX
    """Which retained-subset policy to apply (see :class:`TrimPolicy`)."""

    # Lookup phase: adapter_idx -> task_id (removed as results arrive)
    pending_lookup_tasks: dict[int, L2TaskId] = field(default_factory=dict)
    # Lookup phase: adapter_idx -> bitmap (populated as results arrive)
    lookup_results: dict[int, Bitmap] = field(default_factory=dict)

    # Load phase: adapter_idx -> bitmap of key indices to load
    load_plan: dict[int, Bitmap] = field(default_factory=dict)
    # Load phase: adapter_idx -> bitmap of key indices left L2-resident
    # (served GPU-direct at retrieve; never reserve/load into L1). The
    # lookup-phase pins for these keys are held until the retrieve handler
    # releases them, so they are NOT unlocked at finalize.
    l2_resident_plan: dict[int, Bitmap] = field(default_factory=dict)
    # Load phase: adapter_idx -> task_id (removed as results arrive)
    pending_load_tasks: dict[int, L2TaskId] = field(default_factory=dict)
    # Load phase: adapter_idx -> L1 bytes reserved for that adapter's
    # in-flight load.  Read by the inflight_load_memory_usage_bytes gauge.
    load_bytes_by_adapter: dict[int, int] = field(default_factory=dict)
    # Load phase: adapter_idx -> bitmap (populated as results arrive)
    load_results: dict[int, Bitmap] = field(default_factory=dict)
    # Load phase: keys that were write-reserved in L1
    write_reserved_keys: list[ObjectKey] = field(default_factory=list)
    write_reserved_objs: dict[ObjectKey, MemoryObj] = field(default_factory=dict)

    # Profiling (only populated when ``LMC_PROFILE`` is set). Absolute
    # ``time.perf_counter`` instants captured at phase boundaries on the
    # prefetch thread; differences are reported via query_prefetch_breakdown.
    prof_lookup_submit_at: float = 0.0
    prof_lookup_done_at: float = 0.0
    prof_l1_reserve_seconds: float = 0.0
    prof_load_submit_at: float = 0.0
    prof_load_done_at: float = 0.0

    def all_lookups_done(self) -> bool:
        return len(self.pending_lookup_tasks) == 0

    def all_loads_done(self) -> bool:
        return len(self.pending_load_tasks) == 0


class PrefetchController(StorageControllerInterface):
    """
    Asynchronously prefetches data from L2 adapters into L1 memory.

    The controller:
    1. Accepts prefetch requests via submit_prefetch_request (thread-safe).
    2. Runs a background thread that submits lookup_and_lock to all adapters.
    3. Uses PrefetchPolicy to compute a load plan from lookup results.
    4. Reserves L1 write buffers and submits load tasks to adapters.
    5. On completion, transitions loaded keys to read-locked state.
    6. Reports the number of prefix hits via query_prefetch_result.

    Args:
        l1_manager: The L1 manager instance.
        l2_adapters: List of L2 adapter instances.
        adapter_descriptors: Descriptors for each L2 adapter (same order).
        policy: The prefetch policy for load plan decisions.
        max_in_flight: Maximum number of concurrent prefetch requests.
    """

    # Singleton dispatch for the in-flight load gauges: tests may construct
    # multiple controllers but the OTel SDK only honors the first gauge
    # registration, so the callbacks read from the most recently built
    # instance via ``_gauge_target``.
    _gauges_registered: bool = False
    _gauge_target: "PrefetchController | None" = None

    def __init__(
        self,
        l1_manager: L1Manager,
        l2_adapters: list[L2AdapterInterface],
        adapter_descriptors: list[AdapterDescriptor],
        policy: PrefetchPolicy,
        max_in_flight: int = 8,
    ) -> None:
        self._l1_manager = l1_manager
        self._l2_adapters = l2_adapters
        self._adapter_descriptors = adapter_descriptors
        self._policy = policy
        self._max_in_flight = max_in_flight

        # In-flight request tracking (background thread only)
        self._in_flight_requests: dict[PrefetchRequestId, InFlightPrefetchRequest] = {}
        self._pending_queue: list[
            tuple[
                PrefetchRequestId,
                list[ObjectKey],
                MemoryLayoutDesc,
                int,
                TrimPolicy,
            ]
        ] = []

        # Shadow counters for status reporting (updated in background loop)
        self._status_in_flight_count: int = 0
        self._status_pending_count: int = 0
        self._status_lookup_phase_count: int = 0
        self._status_load_phase_count: int = 0

        # Thread-safe submission queue (external -> background)
        self._submission_lock = threading.Lock()
        self._submission_queue: list[
            tuple[
                PrefetchRequestId,
                list[ObjectKey],
                MemoryLayoutDesc,
                int,
                TrimPolicy,
            ]
        ] = []
        self._next_request_id: PrefetchRequestId = 0
        self._submission_efd = create_event_notifier()

        # Thread-safe lookup results (background -> external)
        self._lookup_results_lock = threading.Lock()
        self._completed_lookups: dict[PrefetchRequestId, int] = {}

        # Thread-safe prefetch results (background -> external)
        self._prefetch_results_lock = threading.Lock()
        self._completed_results: dict[PrefetchRequestId, Bitmap] = {}
        # L2-resident tier info, reported as a sidecar to the result bitmap
        # (the bitmap contract is unchanged). Keyed by request id; populated
        # in _complete_request, popped by query_prefetch_tier_info.
        self._completed_tier_info: dict[PrefetchRequestId, L2ResidentTierInfo] = {}
        # Per-request profiling breakdown (``LMC_PROFILE`` only): maps
        # ``{stage: seconds}`` for the L2 lookup/pin, L1 reserve, and L2 load
        # spans measured on the prefetch thread. Populated in _complete_request,
        # popped by query_prefetch_breakdown (folded into the external profile).
        self._completed_breakdowns: dict[PrefetchRequestId, dict[str, float]] = {}

        # Map eventfds to adapter indices for quick lookup in poll.
        # Relies on the L2AdapterInterface contract that every adapter
        # returns distinct fds for store/lookup/load, and no two adapters
        # share an fd.  See the docstrings in L2AdapterInterface.
        self._lookup_efd_to_adapter: dict[int, int] = {}
        self._load_efd_to_adapter: dict[int, int] = {}
        for i, adapter in enumerate(self._l2_adapters):
            self._lookup_efd_to_adapter[adapter.get_lookup_and_lock_event_fd()] = i
            self._load_efd_to_adapter[adapter.get_load_event_fd()] = i

        self._event_bus = get_event_bus()

        PrefetchController._gauge_target = self
        if not PrefetchController._gauges_registered:
            PrefetchController._gauges_registered = True
            register_gauge(
                "lmcache.l2_prefetch",
                "lmcache_mp.num_inflight_l2_loads",
                "L2 -> L1 prefetch load tasks currently executing, per adapter",
                lambda: (
                    PrefetchController._gauge_target.get_inflight_loads_observations()
                    if PrefetchController._gauge_target is not None
                    else []
                ),
            )
            register_gauge(
                "lmcache.l2_prefetch",
                "lmcache_mp.inflight_load_memory_usage_bytes",
                "L1 bytes reserved by in-flight L2 -> L1 prefetch loads, per adapter",
                lambda: (
                    PrefetchController._gauge_target.get_inflight_load_bytes_observations()
                    if PrefetchController._gauge_target is not None
                    else []
                ),
            )

        self._stop_flag = threading.Event()
        self._thread = threading.Thread(
            target=self._prefetch_loop,
            daemon=True,
        )

    # =========================================================================
    # External API (thread-safe)
    # =========================================================================

    def submit_prefetch_request(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
        extra_count: int = 0,
        policy: TrimPolicy = TrimPolicy.PREFIX,
    ) -> PrefetchRequestId:
        """
        Submit a prefetch request for the given keys.

        Thread-safe. Can be called from any thread.

        The retained subset of found keys is chosen by ``policy`` (see
        :class:`TrimPolicy`).  With the default ``PREFIX`` policy, only the
        **contiguous prefix** of found keys is loaded from L2: if L2 has keys
        {0, 1, 3, 4} but not key 2, only keys {0, 1} are loaded because the gap
        at index 2 breaks the prefix.  Keys outside the retained set are never
        transferred, saving I/O bandwidth and L1 memory.  Use
        :meth:`query_prefetch_result` to retrieve the retained set once the
        request completes.

        Args:
            keys: List of object keys to prefetch from L2 into L1.
                The ordering defines the prefix: index 0 is the first key.
            layout_desc: Memory layout for L1 write buffer allocation.
            extra_count: Extra read locks per key (on top of the default 1)
                to acquire when transitioning loaded keys from write-locked
                to read-locked.  Must match the ``extra_count`` used in the
                corresponding ``submit_prefetch_task`` call so that all TP
                workers can each consume one read lock.
            policy: Which retained-subset policy to apply (see
                :class:`TrimPolicy`).  Defaults to ``PREFIX``.

        Returns:
            A request ID for tracking via query_prefetch_result.
        """
        with self._submission_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            self._submission_queue.append(
                (request_id, keys, layout_desc, extra_count, policy)
            )
        self._submission_efd.notify()
        return request_id

    def query_lookup_result(self, request_id: PrefetchRequestId) -> int | None:
        """
        Query the keys that are found during the lookup for a specific request.

        Thread-safe. Returns the prefix-hit count if the lookup phase
        has completed, None if still in progress, or the prefetch request
        has already been consumed by query_prefetch_result.

        Args:
            request_id: The request ID from submit_prefetch_request.

        Returns:
            Number of prefix hits from the lookup phase, or None if not yet complete
            or if the request has already been consumed by a previous call to this
            method.

        Note:
            This function does not pop the result. The caller need to make sure to call
            the query_prefetch_result after calling this function, otherwise nobody
            will clean up the completed lookups dictionary, causing memory leak.
        """
        with self._lookup_results_lock:
            return self._completed_lookups.get(request_id, None)

    def query_prefetch_result(self, request_id: PrefetchRequestId) -> Bitmap | None:
        """
        Query the result of a prefetch request.

        Thread-safe. Returns the retained-key bitmap if the request
        has completed, None if still in progress. Each result can only
        be retrieved once (subsequent calls return None).

        Args:
            request_id: The request ID from submit_prefetch_request.

        Returns:
            Number of prefix hits, or None if not yet complete.

        Note:
            This function will pop the completed lookup results as well.
            Therefore, the caller need to make sure that never call
            query_lookup_result after calling this function, otherwise it will
            get None forever.
        """
        with self._prefetch_results_lock:
            result = self._completed_results.pop(request_id, None)
        if result is not None:
            with self._lookup_results_lock:
                self._completed_lookups.pop(request_id, None)
        return result

    def query_prefetch_tier_info(
        self, request_id: PrefetchRequestId
    ) -> L2ResidentTierInfo:
        """Pop the L2-resident tier info for a completed prefetch request.

        Sidecar to :meth:`query_prefetch_result` (which owns the result
        bitmap and its lifecycle). Returns an empty :class:`L2ResidentTierInfo`
        when the request had no L2-resident hits or is unknown, so callers
        can treat "no resident keys" and "not found" uniformly.

        Thread-safe. Each request's tier info can only be retrieved once.

        Args:
            request_id: The request ID from submit_prefetch_request.

        Returns:
            The request's :class:`L2ResidentTierInfo`, or an empty one.
        """
        with self._prefetch_results_lock:
            return self._completed_tier_info.pop(request_id, L2ResidentTierInfo())

    def query_prefetch_breakdown(
        self, request_id: PrefetchRequestId
    ) -> dict[str, float]:
        """Pop the per-stage profiling breakdown for a completed request.

        Sidecar to :meth:`query_prefetch_result`, populated only when
        ``LMC_PROFILE`` is set. Returns ``{stage: seconds}`` for the L2
        lookup/pin (``l2lk``), L1 reserve (``l1rsv``), and L2 load
        (``l2load``) spans measured on the prefetch thread, or an empty dict
        when profiling is disabled or the request is unknown.

        Thread-safe. Each request's breakdown can only be retrieved once.

        Args:
            request_id: The request ID from submit_prefetch_request.

        Returns:
            ``{stage: seconds}``, or an empty dict.
        """
        with self._prefetch_results_lock:
            return self._completed_breakdowns.pop(request_id, {})

    def report_status(self) -> dict:
        """Return a status dict for the prefetch controller."""
        is_healthy = self._thread.is_alive()
        with self._submission_lock:
            submission_queue_size = len(self._submission_queue)
        with self._prefetch_results_lock:
            completed_results_count = len(self._completed_results)
        return {
            "is_healthy": is_healthy,
            "thread_alive": is_healthy,
            "max_in_flight": self._max_in_flight,
            "submission_queue_size": submission_queue_size,
            "pending_queue_size": self._status_pending_count,
            "in_flight_request_count": self._status_in_flight_count,
            "lookup_phase_count": self._status_lookup_phase_count,
            "load_phase_count": self._status_load_phase_count,
            "completed_results_count": completed_results_count,
            "num_l2_adapters": len(self._l2_adapters),
        }

    def _snapshot_inflight_loads(self) -> dict[int, tuple[int, int]]:
        """``{adapter_idx: (count, reserved_bytes)}`` for in-flight L2 -> L1
        loads, computed via GIL-atomic ``dict.copy()`` snapshots so the
        OTel reader thread can call this concurrently with the prefetch
        loop without locking.
        """
        counts: dict[int, int] = defaultdict(int)
        bytes_by_adapter: dict[int, int] = defaultdict(int)
        for request in self._in_flight_requests.copy().values():
            for idx, reserved in request.load_bytes_by_adapter.copy().items():
                counts[idx] += 1
                bytes_by_adapter[idx] += reserved
        return {idx: (counts[idx], bytes_by_adapter[idx]) for idx in counts}

    def get_inflight_loads_observations(
        self,
    ) -> list[tuple[int | float, dict[str, object]]]:
        """Per-adapter ``(count, attributes)`` for the
        ``lmcache_mp.num_inflight_l2_loads`` gauge."""
        return [
            (
                count,
                {
                    "l2_name": self._adapter_descriptors[idx].type_name,
                    "adapter_index": idx,
                },
            )
            for idx, (count, _) in self._snapshot_inflight_loads().items()
        ]

    def get_inflight_load_bytes_observations(
        self,
    ) -> list[tuple[int | float, dict[str, object]]]:
        """Per-adapter ``(reserved_bytes, attributes)`` for the
        ``lmcache_mp.inflight_load_memory_usage_bytes`` gauge."""
        return [
            (
                reserved_bytes,
                {
                    "l2_name": self._adapter_descriptors[idx].type_name,
                    "adapter_index": idx,
                },
            )
            for idx, (_, reserved_bytes) in self._snapshot_inflight_loads().items()
        ]

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start the background prefetch loop thread."""
        logger.info("Starting PrefetchController...")
        self._thread.start()

    def stop(self) -> None:
        """
        Signal the loop to stop and wait for the thread to join.

        Cleans up any in-flight requests (releases L1 write locks,
        L2 locks) before returning.
        """
        self._stop_flag.set()
        self._submission_efd.notify()
        self._thread.join()
        self._cleanup_in_flight_requests()
        self._submission_efd.close()

    # =========================================================================
    # Background loop
    # =========================================================================

    def _prefetch_loop(self) -> None:
        """
        Main event-driven loop running in a background thread.

        Uses select.poll() to wait on:
        - The submission eventfd (new prefetch requests).
        - Each L2 adapter's lookup eventfd (completed lookups).
        - Each L2 adapter's load eventfd (completed loads).
        """
        poller = select.poll()
        submission_fd = self._submission_efd.fileno()
        poller.register(submission_fd, select.POLLIN)
        for efd in self._lookup_efd_to_adapter:
            poller.register(efd, select.POLLIN)
        for efd in self._load_efd_to_adapter:
            poller.register(efd, select.POLLIN)

        while not self._stop_flag.is_set():
            ready = poller.poll(PREFETCH_LOOP_POLL_TIMEOUT_MS)

            signaled_adapters: dict[PrefetchPhase, set[int]] = {
                phase: set() for phase in PrefetchPhase
            }
            for fd, events in ready:
                if not (events & select.POLLIN):
                    continue

                try:
                    consume_fd(fd)
                except (OSError, BlockingIOError):
                    pass

                try:
                    if fd == submission_fd:
                        self._drain_submission_queue()
                    elif fd in self._lookup_efd_to_adapter:
                        signaled_adapters[PrefetchPhase.LOOKUP].add(
                            self._lookup_efd_to_adapter[fd]
                        )
                    elif fd in self._load_efd_to_adapter:
                        signaled_adapters[PrefetchPhase.PLAN_AND_LOAD].add(
                            self._load_efd_to_adapter[fd]
                        )
                except Exception:
                    logger.exception(
                        "Unexpected error in prefetch loop while processing fd %d",
                        fd,
                    )

            if any(signaled_adapters.values()):
                for request in list(self._in_flight_requests.values()):
                    try:
                        self._advance_request(request, signaled_adapters)
                    except Exception:
                        logger.exception(
                            "Unexpected error advancing in-flight prefetch request %d",
                            request.request_id,
                        )

            try:
                self._start_pending_requests()
            except Exception:
                logger.exception(
                    "Unexpected error in prefetch loop while starting pending requests"
                )

    def _drain_submission_queue(self) -> None:
        """Move items from the thread-safe submission queue to the
        pending queue."""
        with self._submission_lock:
            items = self._submission_queue
            self._submission_queue = []
        self._pending_queue.extend(items)
        self._status_pending_count += len(items)

    def _start_pending_requests(self) -> None:
        """Start pending requests up to the max in-flight limit."""
        while (
            self._pending_queue and len(self._in_flight_requests) < self._max_in_flight
        ):
            request_id, keys, layout_desc, extra_count, policy = (
                self._pending_queue.pop(0)
            )
            self._status_pending_count -= 1
            self._start_lookup_phase(request_id, keys, layout_desc, extra_count, policy)

    # =========================================================================
    # Lookup phase
    # =========================================================================

    def _start_lookup_phase(
        self,
        request_id: PrefetchRequestId,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
        extra_count: int = 0,
        policy: TrimPolicy = TrimPolicy.PREFIX,
    ) -> None:
        """Submit lookup_and_lock to all adapters for a new request."""
        if not self._l2_adapters:
            self._complete_request(request_id, Bitmap(len(keys)))
            return

        lookup_submit_at = time.perf_counter() if PROFILE_ENABLED else 0.0
        pending_lookup_tasks: dict[int, L2TaskId] = {}
        for i, adapter in enumerate(self._l2_adapters):
            task_id = adapter.submit_lookup_and_lock_task(keys)
            pending_lookup_tasks[i] = task_id

        request = InFlightPrefetchRequest(
            request_id=request_id,
            keys=keys,
            layout_desc=layout_desc,
            phase=PrefetchPhase.LOOKUP,
            extra_count=extra_count,
            policy=policy,
            pending_lookup_tasks=pending_lookup_tasks,
            prof_lookup_submit_at=lookup_submit_at,
        )
        self._in_flight_requests[request_id] = request
        self._status_in_flight_count += 1
        self._status_lookup_phase_count += 1

        self._event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOOKUP_SUBMITTED,
                metadata={
                    "request_id": request_id,
                    "key_count": len(keys),
                    "adapter_count": len(pending_lookup_tasks),
                    "key_count_per_salt": Counter(k.cache_salt for k in keys),
                },
            )
        )

    # =========================================================================
    # Load phase
    # =========================================================================
    def _transition_to_load_phase(self, request: InFlightPrefetchRequest) -> None:
        """Compute load plan, reserve L1 buffers, and submit load tasks."""
        if PROFILE_ENABLED:
            request.prof_lookup_done_at = time.perf_counter()
        request.phase = PrefetchPhase.PLAN_AND_LOAD
        self._status_lookup_phase_count -= 1
        self._status_load_phase_count += 1

        # Step 1: get load plan from policy
        load_plan = self._policy.select_load_plan(
            request.keys,
            request.lookup_results,
            self._adapter_descriptors,
        )

        # Step 2: trim the load plan to the policy's retained subset
        num_keys = len(request.keys)
        merged_lookup = merge_bitmaps(load_plan.values(), num_keys)
        retained = build_trim_mask(merged_lookup, num_keys, request.policy)
        trimmed_plan = trim_load_plan_with_mask(load_plan, retained)

        # Step 2b: split off adapters that serve hits GPU-direct from L2
        # (L2-resident retrieve). Those keys skip the L1 reserve/load path
        # entirely; their lookup-phase pins are kept until the retrieve
        # handler drains them to GPU. Gated to PREFIX for v1 — under
        # SPARSE/SEGMENTED these adapters stay on the L1-bounce path.
        l2_resident_plan: dict[int, Bitmap] = {}
        if request.policy is TrimPolicy.PREFIX:
            for adapter_idx in list(trimmed_plan.keys()):
                if self._l2_adapters[adapter_idx].supports_l2_resident_retrieve():
                    l2_resident_plan[adapter_idx] = trimmed_plan.pop(adapter_idx)
        request.l2_resident_plan = l2_resident_plan
        # Bitmap of keys satisfied purely by a held L2-resident pin (no L1).
        resident_bitmap = merge_bitmaps(l2_resident_plan.values(), num_keys)

        if not trimmed_plan and not l2_resident_plan:
            # Nothing to load after trimming. Unlock all lookup locks and
            # complete with an empty retained set.
            self._unlock_all_lookups(request)
            self._update_lookup_results(request.request_id, 0)
            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_LOOKUP_COMPLETED,
                    metadata={
                        "request_id": request.request_id,
                        "prefix_hit_count": 0,
                    },
                )
            )
            self._complete_request(request.request_id, Bitmap(num_keys))
            return

        # Step 3: reserve L1 write buffers
        merged_bitmap = merge_bitmaps(trimmed_plan.values(), len(request.keys))
        keys_to_reserve = merged_bitmap.gather(request.keys)
        l1_mgr = self._l1_manager

        retentions = self._policy.select_l1_retentions(
            keys_to_reserve,
        )
        reserve_start = time.perf_counter() if PROFILE_ENABLED else 0.0
        write_results = l1_mgr.reserve_write(
            keys=keys_to_reserve,
            is_temporary=[not r for r in retentions],
            layout_desc=request.layout_desc,
            mode="new",
        )
        if PROFILE_ENABLED:
            request.prof_l1_reserve_seconds += time.perf_counter() - reserve_start

        # Step 4: filter to successfully reserved keys
        reserved_key_set: set[ObjectKey] = set()
        oom_keys: list[ObjectKey] = []
        for key, (err, mem_obj) in write_results.items():
            if err == L1Error.SUCCESS and mem_obj is not None:
                request.write_reserved_keys.append(key)
                request.write_reserved_objs[key] = mem_obj
                reserved_key_set.add(key)
            else:
                if err == L1Error.OUT_OF_MEMORY:
                    oom_keys.append(key)
                logger.debug(
                    "Prefetch request %d: reserve write failed for %s: %s",
                    request.request_id,
                    key,
                    err,
                )

        if oom_keys:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_ALLOCATION_FAILED,
                    metadata={"during": "l2_prefetch", "keys": oom_keys},
                )
            )
            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_FAILED,
                    metadata={"reason": "l1_oom", "keys": oom_keys},
                )
            )

        # Step 5: recompute load plan excluding failed reservations.
        # L2-resident keys are satisfied by their held pin (no L1 reserve),
        # so they count toward the retained prefix unconditionally.
        reserved_bitmap = Bitmap(num_keys)
        for i, key in enumerate(request.keys):
            if key in reserved_key_set:
                reserved_bitmap.set(i)
        reserved_bitmap = reserved_bitmap | resident_bitmap

        retained = build_trim_mask(reserved_bitmap, num_keys, request.policy)
        # Re-trim BOTH partitions with the same mask, then re-split so the
        # L1-load plan (request.load_plan) and the resident plan stay
        # disjoint. ``load_plan`` still holds the resident adapters, so we
        # remove them again after trimming.
        retrimmed = trim_load_plan_with_mask(load_plan, retained)
        request.l2_resident_plan = {
            idx: bm for idx, bm in retrimmed.items() if idx in l2_resident_plan
        }
        request.load_plan = {
            idx: bm for idx, bm in retrimmed.items() if idx not in l2_resident_plan
        }
        trimmed_plan = request.load_plan

        ## Step 6: phase 1 unlock — keys locked in lookup but neither in the
        ## L1-load plan nor kept L2-resident.
        self._unlock_unneeded_keys(request)

        if not trimmed_plan and not request.l2_resident_plan:
            # Nothing loadable after filtering
            if request.write_reserved_keys:
                l1_mgr.finish_write(request.write_reserved_keys)
                l1_mgr.delete(request.write_reserved_keys)
            self._update_lookup_results(request.request_id, 0)
            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_LOOKUP_COMPLETED,
                    metadata={
                        "request_id": request.request_id,
                        "prefix_hit_count": 0,
                    },
                )
            )
            self._complete_request(request.request_id, Bitmap(num_keys))
            return

        # If only L2-resident keys remain (no L1 load tasks to wait on),
        # finalize synchronously — there will be no load-completion event.
        if not trimmed_plan:
            self._update_lookup_results(
                request.request_id, retained.count_leading_ones()
            )
            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_LOOKUP_COMPLETED,
                    metadata={
                        "request_id": request.request_id,
                        "prefix_hit_count": retained.count_leading_ones(),
                    },
                )
            )
            self._finalize_load(request)
            return

        ## Step 7: submit load tasks per adapter
        if PROFILE_ENABLED:
            request.prof_load_submit_at = time.perf_counter()
        for adapter_idx, bitmap in trimmed_plan.items():
            per_adapter_keys = bitmap.gather(request.keys)
            per_adapter_objs = [
                request.write_reserved_objs[key] for key in per_adapter_keys
            ]
            task_id = self._l2_adapters[adapter_idx].submit_load_task(
                per_adapter_keys, per_adapter_objs
            )
            request.pending_load_tasks[adapter_idx] = task_id
            # Per-adapter byte accounting for L2_LOAD_TASK_* throughput
            # events.  Uniform layout per chunk -> size * count.
            total_bytes = (
                per_adapter_objs[0].get_size() * len(per_adapter_objs)
                if per_adapter_objs
                else 0
            )
            request.load_bytes_by_adapter[adapter_idx] = total_bytes

            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_LOAD_TASK_SUBMITTED,
                    metadata={
                        "request_id": request.request_id,
                        "adapter_index": adapter_idx,
                        "task_id": task_id,
                        "l2_name": self._adapter_descriptors[adapter_idx].type_name,
                        "key_count": len(per_adapter_keys),
                        "total_bytes": total_bytes,
                    },
                )
            )

        ## Step 8: update the lookup result based on the final load plan
        self._update_lookup_results(request.request_id, retained.count_leading_ones())

        self._event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOOKUP_COMPLETED,
                metadata={
                    "request_id": request.request_id,
                    "prefix_hit_count": retained.count_leading_ones(),
                },
            )
        )
        self._event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOAD_SUBMITTED,
                metadata={
                    "request_id": request.request_id,
                    "key_count": len(reserved_key_set),
                    "adapter_count": len(trimmed_plan),
                    "key_count_per_salt": Counter(
                        k.cache_salt for k in reserved_key_set
                    ),
                },
            )
        )

        logger.debug(
            "Prefetch request %d: submitted load tasks to %d adapters for %d keys",
            request.request_id,
            len(trimmed_plan),
            len(reserved_key_set),
        )

    def _update_lookup_results(
        self, request_id: PrefetchRequestId, prefix_hit_count: int
    ) -> None:
        """Store the prefix-hit count from the lookup phase."""
        with self._lookup_results_lock:
            self._completed_lookups[request_id] = prefix_hit_count

    def _advance_request(
        self,
        request: InFlightPrefetchRequest,
        signaled_adapters: dict[PrefetchPhase, set[int]],
    ) -> None:
        """State-transition dispatcher by phase: poll signaled adapters for
        the request's current phase via the per-phase helper, then trigger
        the phase transition when done."""
        phase_adapters = signaled_adapters[request.phase]
        if not phase_adapters:
            return
        if request.phase == PrefetchPhase.LOOKUP:
            self._poll_lookup_results(request, phase_adapters)
            if request.all_lookups_done():
                self._transition_to_load_phase(request)
        elif request.phase == PrefetchPhase.PLAN_AND_LOAD:
            self._poll_load_results(request, phase_adapters)
            if request.all_loads_done():
                self._finalize_load(request)

    def _poll_lookup_results(
        self,
        request: InFlightPrefetchRequest,
        signaled_adapters: set[int],
    ) -> None:
        """Query pending lookup-and-lock results from signaled adapters."""
        for adapter_idx in list(request.pending_lookup_tasks):
            if adapter_idx not in signaled_adapters:
                continue
            task_id = request.pending_lookup_tasks[adapter_idx]
            result = self._l2_adapters[adapter_idx].query_lookup_and_lock_result(
                task_id
            )
            if result is None:
                continue
            request.lookup_results[adapter_idx] = result
            del request.pending_lookup_tasks[adapter_idx]

    def _poll_load_results(
        self,
        request: InFlightPrefetchRequest,
        signaled_adapters: set[int],
    ) -> None:
        """Query pending load results from signaled adapters."""
        for adapter_idx in list(request.pending_load_tasks):
            if adapter_idx not in signaled_adapters:
                continue
            task_id = request.pending_load_tasks[adapter_idx]
            result = self._l2_adapters[adapter_idx].query_load_result(task_id)
            if result is None:
                continue
            request.load_results[adapter_idx] = result
            del request.pending_load_tasks[adapter_idx]
            request.load_bytes_by_adapter.pop(adapter_idx, None)

            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_LOAD_TASK_COMPLETED,
                    metadata={
                        "request_id": request.request_id,
                        "adapter_index": adapter_idx,
                        "task_id": task_id,
                        "l2_name": self._adapter_descriptors[adapter_idx].type_name,
                    },
                )
            )

    def _finalize_load(self, request: InFlightPrefetchRequest) -> None:
        """
        Finalize a completed load: build result bitmap, transition L1
        state, release read locks outside the retained set, and report the
        retained-key bitmap.

        Partial load failures can create gaps, so a loaded key may fall
        outside the policy's retained set; its read lock must be released.
        """
        if PROFILE_ENABLED:
            request.prof_load_done_at = time.perf_counter()
        num_keys = len(request.keys)

        # Scatter per-adapter local load results into global positions.
        # Each adapter's load bitmap is locally indexed (size == adapter's
        # key count).  The plan bitmap maps local → global indices via
        # get_indices_list().
        result_bitmap = Bitmap(num_keys)
        for adapter_idx, plan_bitmap in request.load_plan.items():
            load_bitmap = request.load_results.get(adapter_idx)
            if load_bitmap is None:
                continue
            plan_indices = plan_bitmap.get_indices_list()
            for global_i in load_bitmap.gather(plan_indices):
                result_bitmap.set(global_i)

        # ``result_bitmap`` so far covers only L1-loaded keys. The L1
        # lifecycle ops below (finish_write_and_reserve_read, finish_read,
        # failed-key cleanup) must operate on THIS L1-only set, since
        # resident keys have no L1 entry.
        l1_loaded_keys: list[ObjectKey] = result_bitmap.gather(request.keys)
        loaded_set = set(l1_loaded_keys)
        failed_keys = [k for k in request.write_reserved_keys if k not in loaded_set]

        # L2-resident keys are "loaded" by virtue of their held pin — they
        # are served GPU-direct at retrieve, never copied to L1. OR them
        # into the result so they count toward the reported retained prefix.
        resident_bitmap = merge_bitmaps(request.l2_resident_plan.values(), num_keys)
        result_bitmap = result_bitmap | resident_bitmap
        loaded_keys = result_bitmap.gather(request.keys)

        # Phase 2 unlock: release L2 locks for all keys in the load plan
        self._unlock_all_plan_keys(request)

        l1_mgr = self._l1_manager

        # Transition L1-loaded keys: write-locked -> read-locked
        # Use extra_count so that all TP workers each get their own read lock.
        # Resident keys are excluded — they were never write-reserved in L1.
        if l1_loaded_keys:
            l1_mgr.finish_write_and_reserve_read(
                l1_loaded_keys, extra_count=request.extra_count
            )

        # Clean up failed keys
        if failed_keys:
            l1_mgr.finish_write(failed_keys)
            l1_mgr.delete(failed_keys)

        self._event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOAD_COMPLETED,
                metadata={
                    "request_id": request.request_id,
                    "loaded_count": len(loaded_keys),
                    "failed_count": len(failed_keys),
                    "key_count_per_salt": Counter(k.cache_salt for k in loaded_keys),
                },
            )
        )

        # L2 prefetch-failure anomaly reporting: keys were reserved in L1
        # (expected to load from L2) but did not appear in the load bitmap.
        # Classified as ``not_found`` — the serde_failure reason will be
        # added once the serde PR lands and adapters can distinguish
        # deserialization errors from missing objects.
        if failed_keys:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_FAILED,
                    metadata={"reason": "not_found", "keys": failed_keys},
                )
            )

        # Release read locks for any L1-loaded key outside the retained set
        # (partial load failures can create gaps). Resident keys are
        # excluded — they hold a CXL pin, not an L1 read lock; that pin is
        # released by the retrieve handler (or the abort path), not here.
        retained = build_trim_mask(result_bitmap, num_keys, request.policy)
        released_bitmap = result_bitmap & (~retained) & (~resident_bitmap)
        released = released_bitmap.gather(request.keys)
        if released:
            l1_mgr.finish_read(released, extra_count=request.extra_count)

        self._complete_request(request.request_id, retained)

    # =========================================================================
    # Unlock helpers
    # =========================================================================

    def _unlock_unneeded_keys(self, request: InFlightPrefetchRequest) -> None:
        """Phase 1 unlock: keys locked in lookup but neither in the L1-load
        plan nor kept L2-resident.

        L2-resident keys keep their lookup pin (the retrieve handler drains
        them to GPU and releases the pin afterwards), so they are excluded
        from the unlock set here.
        """
        num_keys = len(request.keys)
        for adapter_idx, lookup_bitmap in request.lookup_results.items():
            plan_bitmap = request.load_plan.get(adapter_idx, Bitmap(num_keys))
            resident_bitmap = request.l2_resident_plan.get(
                adapter_idx, Bitmap(num_keys)
            )
            kept_bitmap = plan_bitmap | resident_bitmap
            to_unlock_bitmap = lookup_bitmap & (~kept_bitmap)
            unlock_keys = to_unlock_bitmap.gather(request.keys)
            if unlock_keys:
                self._l2_adapters[adapter_idx].submit_unlock(unlock_keys)

    def _unlock_all_plan_keys(self, request: InFlightPrefetchRequest) -> None:
        """Phase 2 unlock: release L2 locks for all keys in the load plan."""
        for adapter_idx, load_bitmap in request.load_plan.items():
            unlock_keys = load_bitmap.gather(request.keys)
            self._l2_adapters[adapter_idx].submit_unlock(unlock_keys)

    def _unlock_all_lookups(self, request: InFlightPrefetchRequest) -> None:
        """Unlock all keys locked during lookup (nothing to load case)."""
        for adapter_idx, lookup_bitmap in request.lookup_results.items():
            unlock_keys = lookup_bitmap.gather(request.keys)
            if unlock_keys:
                self._l2_adapters[adapter_idx].submit_unlock(unlock_keys)

    def _unlock_l2_resident_keys(self, request: InFlightPrefetchRequest) -> None:
        """Release the held pins for L2-resident keys.

        Used only on the shutdown/cleanup path: normally the retrieve
        handler releases these pins after the GPU-direct H2D lands, but at
        shutdown no retrieve will run, so we must unlock them here to avoid
        leaking pins (which would block eviction forever).
        """
        for adapter_idx, resident_bitmap in request.l2_resident_plan.items():
            unlock_keys = resident_bitmap.gather(request.keys)
            if unlock_keys:
                self._l2_adapters[adapter_idx].submit_unlock(unlock_keys)

    # =========================================================================
    # Completion and cleanup
    # =========================================================================

    def _complete_request(self, request_id: PrefetchRequestId, result: Bitmap) -> None:
        """Store the retained-key bitmap (and any L2-resident tier info) and
        remove the request from in-flight tracking.

        The tier info is built from the popped request's ``l2_resident_plan``
        masked to the retained set, so the retrieve handler learns exactly
        which keys to serve GPU-direct (and on which adapter).
        """
        removed = self._in_flight_requests.pop(request_id, None)

        # Build L2-resident tier info from the retained resident keys.
        tier_info = L2ResidentTierInfo()
        if removed is not None and removed.l2_resident_plan:
            resident_keys: list[ObjectKey] = []
            resident_adapters: list[int] = []
            for adapter_idx, plan_bitmap in removed.l2_resident_plan.items():
                retained_resident = plan_bitmap & result
                for global_i in retained_resident.get_indices_list():
                    resident_keys.append(removed.keys[global_i])
                    resident_adapters.append(adapter_idx)
            if resident_keys:
                tier_info = L2ResidentTierInfo(
                    keys=tuple(resident_keys),
                    adapter_indices=tuple(resident_adapters),
                )

        breakdown: dict[str, float] = {}
        if PROFILE_ENABLED and removed is not None:
            # l2lk: lookup/pin span (submit -> all lookup results in).
            # l2load: load span (load submit -> load done); 0 when nothing
            # was loaded into L1 (e.g. all keys served L2-resident). l1rsv:
            # the L1 write-reserve cost folded into the load phase.
            if removed.prof_lookup_done_at and removed.prof_lookup_submit_at:
                breakdown["l2lk"] = (
                    removed.prof_lookup_done_at - removed.prof_lookup_submit_at
                )
            if removed.prof_load_done_at and removed.prof_load_submit_at:
                breakdown["l2load"] = (
                    removed.prof_load_done_at - removed.prof_load_submit_at
                )
            breakdown["l1rsv"] = removed.prof_l1_reserve_seconds

        with self._prefetch_results_lock:
            self._completed_results[request_id] = result
            if tier_info.keys:
                self._completed_tier_info[request_id] = tier_info
            if breakdown:
                self._completed_breakdowns[request_id] = breakdown

        if removed is not None:
            self._status_in_flight_count -= 1
            if removed.phase == PrefetchPhase.LOOKUP:
                self._status_lookup_phase_count -= 1
            elif removed.phase == PrefetchPhase.PLAN_AND_LOAD:
                self._status_load_phase_count -= 1
        logger.debug(
            "Prefetch request %d completed: %d retained keys (%d L2-resident)",
            request_id,
            result.popcount(),
            len(tier_info.keys),
        )

    def _cleanup_in_flight_requests(self) -> None:
        """Release resources for any in-flight requests during shutdown."""
        l1_mgr = self._l1_manager
        for request in self._in_flight_requests.values():
            if request.phase == PrefetchPhase.PLAN_AND_LOAD:
                if request.write_reserved_keys:
                    l1_mgr.finish_write(request.write_reserved_keys)
                    l1_mgr.delete(request.write_reserved_keys)
                self._unlock_all_plan_keys(request)
                self._unlock_l2_resident_keys(request)
            elif request.phase == PrefetchPhase.LOOKUP:
                self._unlock_all_lookups(request)
            logger.warning(
                "Cleaning up in-flight prefetch request %d (%d keys).",
                request.request_id,
                len(request.keys),
            )
        self._in_flight_requests.clear()
