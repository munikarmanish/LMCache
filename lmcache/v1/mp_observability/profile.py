# SPDX-License-Identifier: Apache-2.0
"""Per-request latency-breakdown profiler for the MP retrieve path.

Gated entirely by the ``LMC_PROFILE`` environment variable: when unset (the
default) every entry point here is a cheap no-op so production paths pay
nothing. When set to a truthy value (``1``/``true``/``yes``/``on``) the engine
accumulates per-stage timings for each request and emits a single compact
``PROFILE`` log line when the retrieve completes.

The retrieve TTFT-critical path spans several MP requests and the background
prefetch thread, all keyed by the external ``request_id``:

    LOOKUP        -> hash + submit prefetch task            (stage ``look``)
    prefetch loop -> L2 lookup/pin (lookup_and_lock)        (stage ``l2lk``)
                  -> L1 write reserve                       (stage ``l1rsv``)
                  -> L2 load / copy into L1                 (stage ``l2load``)
    QUERY_STATUS  -> vLLM polls until prefetch done         (stage ``pf_wait``)
    RETRIEVE      -> L1 read-lock acquire                   (stage ``ret_l1``)
                  -> H2D fill (incl. L2-resident submit_h2d)(stage ``ret_h2d``)
                  -> scatter to paged KV (kernel enqueue)   (stage ``ret_scat``)
                  -> finish_read + unpin                    (stage ``unpin``)

Because the stages run on different threads, the profiler stores absolute
start/end timestamps per stage rather than nesting timers. The ``PROFILE``
line reports each stage's duration in milliseconds plus the total wall time
from lookup submit to retrieve completion.
"""

# Standard
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator
import contextlib
import os
import threading
import time

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


def _profiling_enabled() -> bool:
    """Return whether ``LMC_PROFILE`` requests profiling.

    Truthy values are ``1``, ``true``, ``yes``, ``on`` (case-insensitive).
    Anything else (including unset) disables profiling.
    """
    return os.environ.get("LMC_PROFILE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# Resolved once at import; profiling state is fixed for the process lifetime.
PROFILE_ENABLED: bool = _profiling_enabled()


# Stage order for the emitted log line. The first column reported is always
# the all-up wall time; the rest follow this fixed order so successive lines
# line up visually.
_STAGE_ORDER: tuple[str, ...] = (
    "look",  # lookup handler: hash + submit_prefetch_task
    "l2lk",  # L2 lookup/pin: submit_lookup_and_lock -> result
    "l1rsv",  # L1 write-buffer reserve
    "l2load",  # L2 load (copy into L1); ~0 for L2-resident (CXL) keys
    "pf_wait",  # prefetch submit -> retrieve sees it done (incl. poll lag)
    "ret_l1",  # retrieve: read_prefetched_results (L1 read-lock acquire)
    "ret_h2d",  # retrieve: H2D fill — actual device DMA time (CUDA events)
    "ret_scat",  # retrieve: scatter to paged KV — actual device kernel time
    "h2d_cpu",  # retrieve: H2D fill CPU-side launch overhead
    "scat_cpu",  # retrieve: scatter CPU-side launch overhead
    "unpin",  # retrieve: finish_read + release pins / submit_unlock
)


@dataclass
class _RequestProfile:
    """Accumulated per-stage timings for a single request.

    ``durations`` maps a stage name to its total elapsed seconds (summed if a
    stage is entered more than once). ``submit_time`` anchors the all-up wall
    time. ``n_chunks`` / ``payload_bytes`` describe the retrieve payload and
    are filled in by the retrieve handler.
    """

    submit_time: float = field(default_factory=time.perf_counter)
    durations: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    n_chunks: int = 0
    payload_bytes: int = 0


class RequestProfiler:
    """Thread-safe per-request stage-timing accumulator.

    Shared across engine modules and the prefetch thread (all of which hold
    the same instance via the engine context). When profiling is disabled
    every method is a fast no-op. The profiler holds at most one in-flight
    entry per ``request_id`` and drops it when the ``PROFILE`` line is logged.
    """

    def __init__(self) -> None:
        self._enabled = PROFILE_ENABLED
        self._lock = threading.Lock()
        self._profiles: dict[str, _RequestProfile] = {}

    @property
    def enabled(self) -> bool:
        """Whether profiling is active (``LMC_PROFILE`` truthy)."""
        return self._enabled

    def begin(self, request_id: str) -> None:
        """Start (or restart) profiling for ``request_id``.

        Anchors the all-up wall clock. Called at the very start of the lookup
        handler. A no-op when profiling is disabled.

        Args:
            request_id: External request id.
        """
        if not self._enabled or not request_id:
            return
        with self._lock:
            self._profiles[request_id] = _RequestProfile()

    def add(self, request_id: str, stage: str, seconds: float) -> None:
        """Add ``seconds`` to ``stage`` for ``request_id``.

        Used when the caller has already measured the elapsed time (e.g.
        across a span that does not nest cleanly in a ``with`` block). A
        no-op when profiling is disabled or the request is unknown.

        Args:
            request_id: External request id.
            stage: Stage name (one of the documented stages).
            seconds: Elapsed seconds to accumulate into the stage.
        """
        if not self._enabled or not request_id:
            return
        with self._lock:
            prof = self._profiles.get(request_id)
            if prof is not None:
                prof.durations[stage] += seconds

    @contextlib.contextmanager
    def stage(self, request_id: str, stage: str) -> Iterator[None]:
        """Time a ``with`` block and accumulate it into ``stage``.

        A no-op context manager when profiling is disabled (still safe to use
        as ``with profiler.stage(...):``).

        Args:
            request_id: External request id.
            stage: Stage name to accumulate the block's elapsed time into.

        Yields:
            None.
        """
        if not self._enabled or not request_id:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(request_id, stage, time.perf_counter() - start)

    def mark_submit(self, request_id: str) -> None:
        """Record the prefetch-submit instant for the ``pf_wait`` span.

        Stored as the ``_pf_submit`` private stage timestamp (an absolute
        time, resolved into a duration in :meth:`finish`). A no-op when
        profiling is disabled.

        Args:
            request_id: External request id.
        """
        if not self._enabled or not request_id:
            return
        now = time.perf_counter()
        with self._lock:
            prof = self._profiles.get(request_id)
            if prof is not None:
                prof.durations["_pf_submit_at"] = now

    def set_payload(self, request_id: str, n_chunks: int, payload_bytes: int) -> None:
        """Record the retrieve payload size for the log line.

        Args:
            request_id: External request id.
            n_chunks: Number of chunks served by this retrieve.
            payload_bytes: Total payload bytes moved to GPU.
        """
        if not self._enabled or not request_id:
            return
        with self._lock:
            prof = self._profiles.get(request_id)
            if prof is not None:
                prof.n_chunks = n_chunks
                prof.payload_bytes = payload_bytes

    def finish(self, request_id: str) -> None:
        """Emit the compact ``PROFILE`` line and drop the request entry.

        Called at the end of the retrieve handler (the end of the
        TTFT-critical path). Computes the all-up wall time from lookup submit
        and the ``pf_wait`` span (prefetch submit -> retrieve start). A no-op
        when profiling is disabled or the request was never begun.

        Args:
            request_id: External request id.
        """
        if not self._enabled or not request_id:
            return
        with self._lock:
            prof = self._profiles.pop(request_id, None)
        if prof is None:
            return

        now = time.perf_counter()
        total_ms = (now - prof.submit_time) * 1000.0

        durations = dict(prof.durations)
        pf_submit_at = durations.pop("_pf_submit_at", None)
        # pf_wait = time the request spent between the prefetch being
        # submitted and the retrieve handler picking it up. It folds in the
        # async L2 lookup/load (which run concurrently on the prefetch thread)
        # and any vLLM poll latency. The per-stage L2 timings (l2lk/l2load)
        # are reported separately and overlap this span.
        if pf_submit_at is not None:
            ret_start = durations.pop("_ret_start_at", None)
            end = ret_start if ret_start is not None else now
            durations["pf_wait"] = end - pf_submit_at
        durations.pop("_ret_start_at", None)

        gb = prof.payload_bytes / (1024.0**3)
        stage_str = " ".join(
            f"{stage}={durations.get(stage, 0.0) * 1000.0:.1f}"
            for stage in _STAGE_ORDER
        )
        logger.info(
            "PROFILE n=%d gb=%.4f total=%.1f %s",
            prof.n_chunks,
            gb,
            total_ms,
            stage_str,
        )

    def discard(self, request_id: str) -> None:
        """Drop a request's profile without emitting a line.

        Used on the failed-retrieve path (the ``PROFILE`` line is only emitted
        on success) so the entry does not leak. A no-op when profiling is
        disabled or the request is unknown.

        Args:
            request_id: External request id.
        """
        if not self._enabled or not request_id:
            return
        with self._lock:
            self._profiles.pop(request_id, None)

    def mark_retrieve_start(self, request_id: str) -> None:
        """Record the retrieve-handler entry instant (closes ``pf_wait``).

        Args:
            request_id: External request id.
        """
        if not self._enabled or not request_id:
            return
        now = time.perf_counter()
        with self._lock:
            prof = self._profiles.get(request_id)
            if prof is not None:
                prof.durations["_ret_start_at"] = now
