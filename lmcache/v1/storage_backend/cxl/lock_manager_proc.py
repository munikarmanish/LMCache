# SPDX-License-Identifier: Apache-2.0
"""Standalone-process CXL lock-manager arbiter.

Launches `_native/cxl_lock_manager` (a small C program) as a sidecar
process. The C arbiter runs the same sweep as the Python `LockManager`
but without a GIL, so it can't be starved by Python worker threads
hammering the same process. Profile data showed the Python arbiter's
sweep time ballooned from ~10 ms to ~215 ms under donor commit load —
this process boundary fixes that.

Usage:
    proc = ProcessLockManager(dev_path="/dev/dax0.0",
                              pool_size_override=137438953472)
    proc.start()      # spawns the C binary
    ...
    proc.stop()       # SIGTERM, then SIGKILL after a grace period
"""

# Standard
from dataclasses import dataclass
from typing import Optional
import os
import shutil
import signal
import subprocess
import threading
import time

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_NATIVE_DIR = os.path.join(_HERE, "_native")
_BIN_NAME = "cxl_lock_manager"
_SRC_NAME = "cxl_lock_manager.c"


def _bin_path() -> str:
    return os.path.join(_NATIVE_DIR, _BIN_NAME)


def _src_path() -> str:
    return os.path.join(_NATIVE_DIR, _SRC_NAME)


def _needs_rebuild() -> bool:
    """True if the binary is missing or older than the source."""
    b = _bin_path()
    s = _src_path()
    if not os.path.exists(b):
        return True
    if not os.path.exists(s):
        return False  # binary present, source missing — trust the binary
    return os.path.getmtime(s) > os.path.getmtime(b)


def ensure_built() -> str:
    """Build `cxl_lock_manager` if missing or stale. Returns the binary path."""
    binp = _bin_path()
    if not _needs_rebuild():
        return binp
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        raise RuntimeError(
            "no C compiler in PATH; cannot build cxl_lock_manager. "
            "Install gcc or pre-build the binary manually."
        )
    cmd = [
        cc, "-O2", "-march=native", "-pthread",
        _src_path(), "-o", binp,
    ]
    logger.info("building cxl_lock_manager: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"failed to build cxl_lock_manager:\n"
            f"stdout: {res.stdout}\nstderr: {res.stderr}"
        )
    return binp


@dataclass
class ProcessLockManagerConfig:
    dev_path: str
    # NOTE: we deliberately do NOT plumb pool_size_override here. The
    # Python side uses it to cap the mapping for cudaHostRegister's
    # per-call limit, but the C lock manager never touches GPU and only
    # reads the global_locks region near the start of the pool, so it
    # should map the device's full size as reported by sysfs/daxctl.
    # Passing a smaller override caused validation mismatches against
    # what the header records.
    report_interval_s: float = 10.0
    use_flush: bool = True
    # SIGTERM grace before SIGKILL on stop().
    stop_grace_s: float = 5.0


class ProcessLockManager:
    """Sidecar subprocess running the C lock-manager arbiter.

    Lifecycle:
      - start(): builds binary if needed, spawns subprocess. Returns
        once the subprocess has emitted its startup line on stderr.
      - stop(): SIGTERM, wait up to grace_s, then SIGKILL.

    Output: the subprocess writes its startup banner and periodic
    sweep statistics to its stderr. We attach a daemon reader thread
    that forwards each line to the LMCache logger (prefixed) so the
    operator sees them in the unified log.
    """

    def __init__(self, config: ProcessLockManagerConfig):
        self._config = config
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_stop = threading.Event()

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("ProcessLockManager already started")
        binp = ensure_built()
        cmd = [
            binp,
            "--dev", self._config.dev_path,
            "--report-interval-s", str(self._config.report_interval_s),
        ]
        if not self._config.use_flush:
            cmd += ["--no-flush"]

        logger.info("starting cxl_lock_manager subprocess: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
        )

        # Reader thread: forward subprocess stderr to logger.
        self._reader_thread = threading.Thread(
            target=self._read_loop,
            name="cxl-lock-mgr-reader",
            daemon=True,
        )
        self._reader_thread.start()

        # Wait briefly for the startup banner so we surface immediate
        # failures (bad magic, mmap failure, etc.) before returning.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"cxl_lock_manager exited immediately "
                    f"(rc={self._proc.returncode}); check log output above"
                )
            time.sleep(0.05)
            # No explicit ready signal — the banner is the only thing
            # the subprocess emits before the first 10 s of sweeps. We
            # accept that start() returning means "process is alive."
            break

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        try:
            for raw in self._proc.stderr:
                if self._reader_stop.is_set():
                    return
                line = raw.rstrip()
                if line:
                    logger.info("[cxl-lock-mgr] %s", line)
        except Exception:
            logger.exception("cxl-lock-mgr reader thread failed")

    def stop(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=self._config.stop_grace_s)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "cxl_lock_manager did not exit within %.1fs; killing",
                    self._config.stop_grace_s,
                )
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    logger.error("cxl_lock_manager refused to die")
        self._reader_stop.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def __enter__(self) -> "ProcessLockManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
