#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""RDMA scalability sweep: does the NIC degrade as software scales it out?

Drives ``perftest`` (``ib_{write,read,send}_bw``) between two hosts and sweeps
ONE software knob at a time while holding the rest fixed, recording bandwidth,
message rate, requester CPU, NIC error/retransmit counters, and (optionally)
the latency seen by a concurrent 1-QP "victim" flow. The output is a CSV with
one row per (sweep value, message size, repeat) and a per-size summary table
with each point normalized to the best point of its curve, so a degradation
is visible as a ratio dropping below 1.0.

WHAT CAN DEGRADE, AND WHICH SWEEP ISOLATES IT
---------------------------------------------
  --sweep qps        Number of RC queue pairs in ONE process (``-q``). Each
                     QP has on-NIC context; the NIC caches a bounded number of
                     them (ICM cache). Past the cache, every WQE fetches
                     context over PCIe -> message rate, then bandwidth, drops.
                     Sensitive at small messages, amortized at large ones,
                     so run several --sizes. By default tx-depth is FIXED per
                     QP, so total outstanding work grows with qps; pass
                     --const-outstanding N to hold qps*tx_depth = N and
                     separate "more QPs" from "more in flight".
  --sweep procs      Number of concurrent client/server PROCESS pairs, each
                     with its own PD/CQ/MR and one QP (or --qps). Adds the
                     CPU/doorbell/completion-polling side to the QP-count
                     question. Each process is pinned to its own core.
  --sweep tx-depth   Outstanding WQEs on one QP. The bandwidth-delay product:
                     shows how much concurrency ONE connection needs to fill
                     200 Gbps, i.e. the floor below which a lone connection is
                     latency-bound rather than link-bound.
  --sweep outs       (read only) Outstanding RDMA READs per QP (``-o``). The
                     device caps this (max_qp_rd_atom=16 here), which is WHY a
                     one-sided-READ consumer such as the nixl_peer adapter
                     needs several QPs to saturate the link. Pair with the qps
                     sweep under --verb read.
  --sweep size       Message size on one QP: the baseline curve every other
                     sweep is read against.
  --sweep post-list  WQEs per doorbell (``-l``). Doorbell batching; a
                     software lever on per-message CPU/PCIe cost.
  --sweep cq-mod     Completions per CQE (``-Q``). Completion coalescing.
  --sweep inline     Inline threshold (``-I``): payload rides in the WQE
                     instead of being DMA'd -- small-message latency lever.
  --sweep mtu        Path MTU (``-m``).
  --sweep conn       Transport (``-c``): RC vs DC (vs UC for write/send). DC
                     is the transport built to make connection count cheap;
                     if the qps sweep shows an RC cliff, the same sweep with
                     --conn DC says whether the cliff is RC state or the NIC.

Fixed knobs (apply to every point): --verb, --conn, --tx-depth, --qps (for the
procs sweep), --hugepages, --mr-per-qp, --bidir, --cuda-client/--cuda-server,
--extra-args. Two runs of the same sweep differing in one fixed knob is how
you attribute a degradation. The two that matter most:

  --hugepages   Registered memory scales with qps (perftest allocates
                2*size per QP). The NIC also caches address translations
                (MTT); 4 KiB pages overflow it far sooner than 2 MiB pages.
                If the qps cliff moves or vanishes with --hugepages, it was
                translation-cache pressure, not QP-context pressure.
  --mr-per-qp   One memory region per QP instead of one shared MR, adding
                per-QP key (MPT) cache pressure on top of the translations.

VICTIM-FLOW LATENCY (--probe-lat)
---------------------------------
Bandwidth alone hides the failure mode a serving system cares about: a
latency-sensitive flow sharing the NIC with N bulk connections. With
--probe-lat, each point also runs ``ib_write_lat`` (1 QP, --probe-size bytes)
on its own port, started once the bulk traffic is actually flowing (detected
from the port packet counters, since connection setup for thousands of QPs
takes seconds), and records its typical/avg/p99/p99.9 latency. The probe
takes --probe-iters ping-pong samples (perftest only reports percentiles in
iteration mode), so it is a snapshot from inside the bulk window, not the
whole window: keep it small enough to finish before the bulk run ends.

NIC COUNTERS
------------
Every row carries the delta of the RoCE error/congestion counters on BOTH
ends (``c_*`` = client/requester, ``s_*`` = server/responder): retransmits and
ack timeouts, out-of-sequence, CNPs sent/handled (ECN congestion control
kicking in), port_xmit_wait (the port had data but no credit), out_of_buffer.
A bandwidth drop WITH retransmits is a different diagnosis from one without.

TOPOLOGY ASSUMED (scripts/cxl/README.md)
----------------------------------------
Client = this host (g5). Server = --server-host (ssh alias, default g6),
reachable for RDMA at --server-ip (default 192.168.0.2 = g6's mlx5_0). Both
sides use --dev mlx5_0 and GID index 3 (RoCEv2/IPv4); ``show_gids`` to check.
Requires passwordless ssh to the server host and perftest on both.

USAGE
-----
  cd scripts/cxl
  # Baseline curve, then the headline question: RC QP count.
  ./bench_rdma_scaling.py --sweep size
  ./bench_rdma_scaling.py --sweep qps --probe-lat
  ./bench_rdma_scaling.py --sweep qps --verb read --probe-lat
  # Attribute a cliff: translation cache vs QP context vs transport.
  ./bench_rdma_scaling.py --sweep qps --hugepages     # needs vm.nr_hugepages
  ./bench_rdma_scaling.py --sweep qps --const-outstanding 1024
  ./bench_rdma_scaling.py --sweep qps --conn DC
  # Processes instead of QPs; GPU memory instead of DRAM.
  ./bench_rdma_scaling.py --sweep procs --sizes 4096,1048576
  ./bench_rdma_scaling.py --sweep qps --cuda-client 0 --cuda-server 0
  # Print the exact perftest commands instead of running them.
  ./bench_rdma_scaling.py --sweep qps --dry-run

Results land in results/rdma_scaling-<label>.csv. Repeats (--repeats) are
separate rows; the summary shows the median.
"""

# Future
from __future__ import annotations

# Standard
import argparse
import csv
import dataclasses
import datetime as dt
import enum
import os
import pathlib
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

# hw_counters / counters whose per-run delta is recorded on both ends. The
# first group are the "something went wrong" signals; the last three are
# sanity/volume signals (they prove the verb actually ran on the responder).
TRACKED_COUNTERS: tuple[str, ...] = (
    "local_ack_timeout_err",
    "packet_seq_err",
    "out_of_sequence",
    "implied_nak_seq_err",
    "rnr_nak_retry_err",
    "req_transport_retries_exceeded",
    "roce_adp_retrans",
    "duplicate_request",
    "np_cnp_sent",
    "rp_cnp_handled",
    "np_ecn_marked_roce_packets",
    "out_of_buffer",
    "req_cqe_error",
    "resp_cqe_error",
    "port_xmit_wait",
    "port_xmit_discards",
    "port_rcv_errors",
    "rx_read_requests",
    "rx_write_requests",
)

# Derived "trouble" columns shown in the summary table.
RETRANS_COUNTERS = ("local_ack_timeout_err", "packet_seq_err", "roce_adp_retrans")
CNP_COUNTERS = ("np_cnp_sent", "rp_cnp_handled")


class Verb(enum.Enum):
    """RDMA operation under test; selects the perftest binary."""

    WRITE = "write"
    READ = "read"
    SEND = "send"

    @property
    def binary(self) -> str:
        """Name of the perftest bandwidth binary for this verb."""
        return f"ib_{self.value}_bw"


class Sweep(enum.Enum):
    """Which single knob varies across the run."""

    QPS = "qps"
    PROCS = "procs"
    TX_DEPTH = "tx-depth"
    OUTS = "outs"
    SIZE = "size"
    POST_LIST = "post-list"
    CQ_MOD = "cq-mod"
    INLINE = "inline"
    MTU = "mtu"
    CONN = "conn"


DEFAULT_VALUES: dict[Sweep, list[str]] = {
    Sweep.QPS: [str(2**k) for k in range(0, 13)],  # 1 .. 4096
    Sweep.PROCS: [str(2**k) for k in range(0, 7)],  # 1 .. 64
    Sweep.TX_DEPTH: [str(2**k) for k in range(0, 11)],  # 1 .. 1024
    Sweep.OUTS: ["1", "2", "4", "8", "16"],
    Sweep.SIZE: [str(2**k) for k in range(1, 24)],  # 2 B .. 8 MiB
    Sweep.POST_LIST: [str(2**k) for k in range(0, 8)],  # 1 .. 128
    Sweep.CQ_MOD: ["1", "2", "4", "8", "16", "32", "64", "100"],
    Sweep.INLINE: ["0", "32", "64", "128", "256"],
    Sweep.MTU: ["256", "512", "1024", "2048", "4096"],
    Sweep.CONN: ["RC", "DC"],
}

# perftest exits non-zero and prints this when the server is not up yet.
_CONNECT_FAIL = re.compile(r"Couldn't connect|Unable to open file descriptor", re.I)


@dataclasses.dataclass(frozen=True)
class Point:
    """One fully-specified perftest configuration (both ends)."""

    verb: Verb
    conn: str
    size: int
    qps: int
    tx_depth: int
    procs: int
    post_list: int
    cq_mod: int
    inline: int
    mtu: int
    outs: int
    hugepages: bool
    mr_per_qp: bool
    bidir: bool
    duration: int

    def perftest_args(self, dev: str, gid: int, port: int) -> list[str]:
        """perftest argv (minus binary, minus peer address) shared by both ends.

        Args:
            dev: RDMA device name on that end.
            gid: GID index on that end.
            port: TCP rendezvous port for this process pair.

        Returns:
            Argument list to append to the perftest binary.
        """
        args = [
            "-d",
            dev,
            "-x",
            str(gid),
            "-p",
            str(port),
            "-s",
            str(self.size),
            "-q",
            str(self.qps),
            "-t",
            str(self.tx_depth),
            "-D",
            str(self.duration),
            "-c",
            self.conn,
            "-m",
            str(self.mtu),
            "-l",
            str(self.post_list),
            "-Q",
            str(self.cq_mod),
            "-F",
            "--report_gbits",
            "--cpu_util",
        ]
        if self.inline > 0:
            args += ["-I", str(self.inline)]
        if self.verb is Verb.READ:
            args += ["-o", str(self.outs)]
        if self.verb in (Verb.READ, Verb.WRITE):
            args.append("--perform_warm_up")
        if self.hugepages:
            args.append("--use_hugepages")
        if self.mr_per_qp:
            args.append("--mr_per_qp")
        if self.bidir:
            args.append("-b")
        return args

    def registered_bytes(self) -> int:
        """Approximate bytes perftest registers per process (2 x size per QP,
        rounded up to a page)."""
        return max(self.size, 4096) * 2 * self.qps


@dataclasses.dataclass
class BwResult:
    """Parsed output of one perftest bandwidth process."""

    bw_avg_gbps: float
    bw_peak_gbps: float
    msg_rate_mpps: float
    cpu_util: float


@dataclasses.dataclass
class LatResult:
    """Parsed output of one ``ib_write_lat`` run."""

    t_typical_us: float
    t_avg_us: float
    t_p99_us: float
    t_p999_us: float
    t_max_us: float


@dataclasses.dataclass
class Hosts:
    """Where the two ends run and how to reach them."""

    server_host: str
    server_ip: str
    client_dev: str
    server_dev: str
    client_gid: int
    server_gid: int
    ssh_opts: list[str]

    def ssh(self, cmd: str, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
        """Run a shell command on the server host and wait for it.

        Args:
            cmd: Shell command (bash -c semantics on the remote).
            timeout: Seconds before giving up.

        Returns:
            The completed process (stdout/stderr captured, not checked).
        """
        return subprocess.run(
            ["ssh", *self.ssh_opts, self.server_host, cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )


# ---------------------------------------------------------------------------
# perftest output parsing
# ---------------------------------------------------------------------------


def _numeric_row_after_header(text: str) -> list[float]:
    """Return the numeric result row that follows perftest's ``#bytes`` header.

    Args:
        text: Full stdout of a perftest process.

    Returns:
        The floats of the last all-numeric line after the header, or an empty
        list when no such line exists (test failed before reporting).
    """
    seen_header = False
    row: list[float] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#bytes"):
            seen_header = True
            continue
        if not seen_header or not stripped:
            continue
        toks = stripped.split()
        try:
            vals = [float(t) for t in toks]
        except ValueError:
            continue
        if len(vals) >= 5:
            row = vals
    return row


def parse_bw(text: str) -> BwResult:
    """Parse ``ib_*_bw --report_gbits --cpu_util`` output.

    Args:
        text: Full stdout of the process.

    Returns:
        The parsed result.

    Raises:
        ValueError: If no result row is present (the run failed); the message
            carries the tail of the output for diagnosis.
    """
    row = _numeric_row_after_header(text)
    if not row:
        tail = "\n".join(text.strip().splitlines()[-8:])
        raise ValueError(f"no perftest result row; output tail:\n{tail}")
    # bytes, iterations, peak, avg, mpps[, cpu]
    cpu = row[5] if len(row) >= 6 else float("nan")
    return BwResult(
        bw_avg_gbps=row[3], bw_peak_gbps=row[2], msg_rate_mpps=row[4], cpu_util=cpu
    )


def parse_lat(text: str) -> LatResult:
    """Parse ``ib_write_lat`` output.

    Args:
        text: Full stdout of the process.

    Returns:
        The parsed result.

    Raises:
        ValueError: If no result row is present.
    """
    row = _numeric_row_after_header(text)
    # bytes, iters, t_min, t_max, t_typical, t_avg, t_stdev, p99, p99.9
    if len(row) < 9:
        tail = "\n".join(text.strip().splitlines()[-8:])
        raise ValueError(f"no ib_write_lat result row; output tail:\n{tail}")
    return LatResult(
        t_typical_us=row[4],
        t_avg_us=row[5],
        t_p99_us=row[7],
        t_p999_us=row[8],
        t_max_us=row[3],
    )


# ---------------------------------------------------------------------------
# NIC counters
# ---------------------------------------------------------------------------


def _counter_dirs(dev: str) -> list[str]:
    base = f"/sys/class/infiniband/{dev}/ports/1"
    return [f"{base}/hw_counters", f"{base}/counters"]


def read_local_counters(dev: str) -> dict[str, int]:
    """Snapshot the tracked port counters of a local RDMA device.

    Args:
        dev: RDMA device name.

    Returns:
        Mapping counter name -> value for every tracked counter that exists.
    """
    out: dict[str, int] = {}
    for d in _counter_dirs(dev):
        for name in TRACKED_COUNTERS:
            p = pathlib.Path(d) / name
            if p.exists():
                try:
                    out[name] = int(p.read_text().strip())
                except (OSError, ValueError):
                    pass
    return out


def read_remote_counters(hosts: Hosts) -> dict[str, int]:
    """Snapshot the tracked port counters on the server host over ssh.

    Args:
        hosts: Host/device configuration.

    Returns:
        Mapping counter name -> value (empty if the ssh call failed).
    """
    globs = " ".join(f"{d}/*" for d in _counter_dirs(hosts.server_dev))
    cp = hosts.ssh(f"grep -H . {globs} 2>/dev/null")
    out: dict[str, int] = {}
    for line in cp.stdout.splitlines():
        path, _, val = line.rpartition(":")
        name = path.rsplit("/", 1)[-1]
        if name in TRACKED_COUNTERS:
            try:
                out[name] = int(val.strip())
            except ValueError:
                pass
    return out


def counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Per-counter ``after - before`` for counters present in both snapshots."""
    return {
        k: after[k] - before[k] for k in TRACKED_COUNTERS if k in before and k in after
    }


def _port_packets(dev: str) -> int:
    """Sum of port xmit+rcv packets (used to detect when bulk traffic starts)."""
    base = pathlib.Path(f"/sys/class/infiniband/{dev}/ports/1/counters")
    total = 0
    for name in ("port_xmit_packets", "port_rcv_packets"):
        try:
            total += int((base / name).read_text().strip())
        except (OSError, ValueError):
            pass
    return total


# ---------------------------------------------------------------------------
# Running one point
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RunPlan:
    """Everything needed to launch one point: commands for both ends."""

    server_script: str
    client_cmds: list[list[str]]
    ports: list[int]
    lat_port: int
    lat_server_cmd: list[str]
    lat_client_cmd: list[str]


def build_plan(
    pt: Point,
    hosts: Hosts,
    base_port: int,
    cpu_base: int,
    cuda_client: int,
    cuda_server: int,
    extra_args: list[str],
    probe: bool,
    probe_size: int,
    probe_iters: int,
) -> RunPlan:
    """Build the server-side shell script and client argv lists for a point.

    Args:
        pt: The configuration.
        hosts: Host/device configuration.
        base_port: First TCP rendezvous port; process i uses base_port+i.
        cpu_base: First CPU core; process i is pinned to cpu_base+i on both ends.
        cuda_client: CUDA device for the client buffer, or -1 for host memory.
        cuda_server: CUDA device for the server buffer, or -1 for host memory.
        extra_args: Extra perftest args appended on both ends.
        probe: Whether to include the latency victim flow.
        probe_size: Message size for the victim flow.
        probe_iters: Ping-pong samples the victim flow takes.

    Returns:
        The plan.
    """
    ports = [base_port + i for i in range(pt.procs)]
    lat_port = base_port + 500

    server_lines = [
        "set -u",
        "tmp=$(mktemp -d /tmp/rdma_scaling.XXXXXX)",
    ]
    client_cmds: list[list[str]] = []
    for i, port in enumerate(ports):
        common = pt.perftest_args(hosts.server_dev, hosts.server_gid, port) + extra_args
        s_args = list(common)
        if cuda_server >= 0:
            s_args.append(f"--use_cuda={cuda_server}")
        s_cmd = ["taskset", "-c", str(cpu_base + i), pt.verb.binary, *s_args]
        server_lines.append(f"{shlex.join(s_cmd)} > $tmp/{port} 2>&1 &")

        c_args = pt.perftest_args(hosts.client_dev, hosts.client_gid, port) + extra_args
        if cuda_client >= 0:
            c_args.append(f"--use_cuda={cuda_client}")
        client_cmds.append(
            [
                "taskset",
                "-c",
                str(cpu_base + i),
                pt.verb.binary,
                *c_args,
                hosts.server_ip,
            ]
        )

    lat_common = [
        "-d",
        "DEV",
        "-x",
        "GID",
        "-p",
        str(lat_port),
        "-s",
        str(probe_size),
        "-n",
        str(probe_iters),
        "-F",
    ]
    lat_core = cpu_base + pt.procs
    lat_server_cmd = ["taskset", "-c", str(lat_core), "ib_write_lat", *lat_common]
    lat_server_cmd[lat_server_cmd.index("DEV")] = hosts.server_dev
    lat_server_cmd[lat_server_cmd.index("GID")] = str(hosts.server_gid)
    lat_client_cmd = [
        "taskset",
        "-c",
        str(lat_core),
        "ib_write_lat",
        *lat_common,
        hosts.server_ip,
    ]
    lat_client_cmd[lat_client_cmd.index("DEV")] = hosts.client_dev
    lat_client_cmd[lat_client_cmd.index("GID")] = str(hosts.client_gid)
    if probe:
        server_lines.append(f"{shlex.join(lat_server_cmd)} > $tmp/{lat_port} 2>&1 &")

    server_lines.append("wait")
    all_ports = ports + ([lat_port] if probe else [])
    for port in all_ports:
        server_lines.append(f'echo "=====PORT {port}"; cat $tmp/{port}')
    server_lines.append("rm -rf $tmp")

    return RunPlan(
        server_script="\n".join(server_lines),
        client_cmds=client_cmds,
        ports=ports,
        lat_port=lat_port,
        lat_server_cmd=lat_server_cmd,
        lat_client_cmd=lat_client_cmd,
    )


def _port_regex(ports: list[int]) -> str:
    return "ib_[a-z]+_(bw|lat) .*-p (" + "|".join(str(p) for p in ports) + ")( |$)"


def kill_stale(hosts: Hosts, ports: list[int]) -> None:
    """Kill perftest processes bound to our ports on both ends (stale runs)."""
    pat = _port_regex(ports)
    subprocess.run(["pkill", "-f", pat], capture_output=True)
    try:
        hosts.ssh(f"pkill -f {shlex.quote(pat)}", timeout=15)
    except subprocess.TimeoutExpired:
        pass


def wait_for_listen(hosts: Hosts, ports: list[int], timeout: float) -> None:
    """Block until every port in ``ports`` is listening on the server host.

    Args:
        hosts: Host configuration.
        ports: TCP ports the perftest servers rendezvous on.
        timeout: Seconds before giving up.

    Raises:
        RuntimeError: If some port is still not listening after the timeout.
    """
    want = set(ports)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        cp = hosts.ssh("ss -Hltn", timeout=15)
        have: set[int] = set()
        for line in cp.stdout.splitlines():
            toks = line.split()
            if len(toks) >= 4 and ":" in toks[3]:
                try:
                    have.add(int(toks[3].rsplit(":", 1)[1]))
                except ValueError:
                    pass
        if want <= have:
            return
        time.sleep(0.2)
    raise RuntimeError(
        f"perftest servers not listening on {sorted(want)} after {timeout}s"
    )


def wait_for_traffic(dev: str, timeout: float, pkts_per_interval: int = 10_000) -> bool:
    """Block until the local port shows bulk traffic (>= pkts_per_interval per
    100 ms), i.e. perftest finished connection setup and started the test.

    Args:
        dev: Local RDMA device.
        timeout: Seconds before giving up.
        pkts_per_interval: Packet-rate threshold per 100 ms poll.

    Returns:
        True when traffic was detected, False on timeout.
    """
    deadline = time.monotonic() + timeout
    prev = _port_packets(dev)
    while time.monotonic() < deadline:
        time.sleep(0.1)
        cur = _port_packets(dev)
        if cur - prev >= pkts_per_interval:
            return True
        prev = cur
    return False


@dataclasses.dataclass
class PointOutcome:
    """Everything measured for one execution of one point."""

    bw: BwResult  # aggregated over processes (sum bw/mpps, mean cpu)
    per_proc: list[BwResult]
    lat: LatResult | None
    client_delta: dict[str, int]
    server_delta: dict[str, int]
    elapsed_s: float


def _split_server_blocks(text: str) -> dict[int, str]:
    """Split the concatenated server output into ``{port: output}``."""
    blocks: dict[int, str] = {}
    cur: int | None = None
    buf: list[str] = []
    for line in text.splitlines():
        m = re.match(r"=====PORT (\d+)$", line.strip())
        if m:
            if cur is not None:
                blocks[cur] = "\n".join(buf)
            cur = int(m.group(1))
            buf = []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        blocks[cur] = "\n".join(buf)
    return blocks


def run_point(
    pt: Point, plan: RunPlan, hosts: Hosts, probe: bool, log: pathlib.Path
) -> PointOutcome:
    """Execute one point end to end and return its measurements.

    Args:
        pt: The configuration.
        plan: Commands built by ``build_plan``.
        hosts: Host configuration.
        probe: Whether the victim-flow latency probe is enabled.
        log: File to append raw perftest output to.

    Returns:
        The measured outcome.

    Raises:
        RuntimeError: If servers fail to come up or any client fails.
    """
    all_ports = plan.ports + ([plan.lat_port] if probe else [])
    kill_stale(hosts, all_ports)

    c_before = read_local_counters(hosts.client_dev)
    s_before = read_remote_counters(hosts)

    # Generous setup allowance: perftest exchanges every QP's state over TCP
    # before the timed window, and registering GiBs of memory is not free.
    setup_s = 60 + 0.03 * pt.qps * pt.procs + pt.registered_bytes() * pt.procs / 2e9
    total_timeout = pt.duration + setup_s

    t0 = time.monotonic()
    # The whole server-side script travels as one quoted argument (ssh joins
    # its argv with spaces and the remote shell re-parses it), so no stdin
    # pipe is involved and communicate() below has nothing to flush.
    server = subprocess.Popen(
        [
            "ssh",
            *hosts.ssh_opts,
            hosts.server_host,
            "bash",
            "-c",
            shlex.quote(plan.server_script),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    clients: list[subprocess.Popen[str]] = []
    lat_proc: subprocess.Popen[str] | None = None
    try:
        wait_for_listen(hosts, all_ports, timeout=30)
        for cmd in plan.client_cmds:
            clients.append(
                subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
            )

        lat_text = ""
        if probe:
            if wait_for_traffic(hosts.client_dev, timeout=setup_s):
                lat_proc = subprocess.Popen(
                    plan.lat_client_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            else:
                sys.stderr.write(
                    "  [probe] bulk traffic never detected; skipping latency probe\n"
                )
                kill_stale(hosts, [plan.lat_port])

        client_texts: list[str] = []
        for p in clients:
            out, _ = p.communicate(timeout=total_timeout)
            client_texts.append(out)
        if lat_proc is not None:
            lat_text, _ = lat_proc.communicate(timeout=total_timeout)
        server_text, _ = server.communicate(timeout=total_timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        for p in clients + ([lat_proc] if lat_proc else []):
            p.kill()
        server.kill()
        kill_stale(hosts, all_ports)
        raise
    elapsed = time.monotonic() - t0

    c_after = read_local_counters(hosts.client_dev)
    s_after = read_remote_counters(hosts)

    with log.open("a") as f:
        f.write(f"\n##### {dt.datetime.now().isoformat()} {pt}\n")
        for port, txt in zip(plan.ports, client_texts, strict=True):
            f.write(f"--- client port {port}\n{txt}\n")
        if lat_text:
            f.write(f"--- lat client port {plan.lat_port}\n{lat_text}\n")
        f.write(f"--- server (all ports)\n{server_text}\n")

    per_proc: list[BwResult] = []
    for cmd, txt in zip(plan.client_cmds, client_texts, strict=True):
        try:
            per_proc.append(parse_bw(txt))
        except ValueError as e:
            raise RuntimeError(f"client failed: {shlex.join(cmd)}\n{e}") from e

    # Sanity: the server must have reported too, else the client's number is
    # a half-run (e.g. server died mid-test).
    s_blocks = _split_server_blocks(server_text)
    for port in plan.ports:
        if port not in s_blocks or not _numeric_row_after_header(s_blocks[port]):
            raise RuntimeError(
                f"server on port {port} produced no result:\n"
                f"{s_blocks.get(port, server_text)[-600:]}"
            )

    agg = BwResult(
        bw_avg_gbps=sum(r.bw_avg_gbps for r in per_proc),
        bw_peak_gbps=sum(r.bw_peak_gbps for r in per_proc),
        msg_rate_mpps=sum(r.msg_rate_mpps for r in per_proc),
        cpu_util=statistics.fmean(r.cpu_util for r in per_proc),
    )
    lat = parse_lat(lat_text) if lat_text else None
    return PointOutcome(
        bw=agg,
        per_proc=per_proc,
        lat=lat,
        client_delta=counter_delta(c_before, c_after),
        server_delta=counter_delta(s_before, s_after),
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Sweep construction
# ---------------------------------------------------------------------------


def _round_up_multiple(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def make_points(
    args: argparse.Namespace, sweep: Sweep, values: list[str], sizes: list[int]
) -> list[tuple[str, Point]]:
    """Expand the sweep into ``(sweep_value, Point)`` pairs, sizes outermost so
    a partial run still yields complete curves for the sizes it finished.

    Args:
        args: Parsed CLI arguments (the fixed knobs).
        sweep: The varying knob.
        values: Its values, as strings.
        sizes: Message sizes (ignored for the size sweep).

    Returns:
        Ordered list of points with their sweep value.

    Raises:
        ValueError: On combinations perftest cannot run.
    """
    verb = Verb(args.verb)
    if verb is Verb.READ and args.conn == "UC":
        raise ValueError("RDMA READ is not defined over UC")

    base = dict(
        verb=verb,
        conn=args.conn,
        size=0,
        qps=args.qps,
        tx_depth=args.tx_depth,
        procs=1,
        post_list=args.post_list,
        cq_mod=args.cq_mod,
        inline=0,
        mtu=args.mtu,
        outs=args.outs,
        hugepages=args.hugepages,
        mr_per_qp=args.mr_per_qp,
        bidir=args.bidir,
        duration=args.duration,
    )

    def finish(kw: dict[str, object]) -> Point:
        pl = int(kw["post_list"])
        kw["tx_depth"] = _round_up_multiple(max(int(kw["tx_depth"]), pl), pl)
        return Point(**kw)  # type: ignore[arg-type]

    out: list[tuple[str, Point]] = []
    if sweep is Sweep.SIZE:
        for v in values:
            kw = dict(base, size=int(v))
            out.append((v, finish(kw)))
        return out

    for size in sizes:
        for v in values:
            kw = dict(base, size=size)
            if sweep is Sweep.QPS:
                q = int(v)
                kw["qps"] = q
                if args.const_outstanding > 0:
                    kw["tx_depth"] = max(1, args.const_outstanding // q)
            elif sweep is Sweep.PROCS:
                kw["procs"] = int(v)
            elif sweep is Sweep.TX_DEPTH:
                kw["tx_depth"] = int(v)
            elif sweep is Sweep.OUTS:
                if verb is not Verb.READ:
                    raise ValueError("--sweep outs only applies to --verb read")
                kw["outs"] = int(v)
            elif sweep is Sweep.POST_LIST:
                kw["post_list"] = int(v)
            elif sweep is Sweep.CQ_MOD:
                kw["cq_mod"] = int(v)
            elif sweep is Sweep.INLINE:
                kw["inline"] = int(v)
            elif sweep is Sweep.MTU:
                kw["mtu"] = int(v)
            elif sweep is Sweep.CONN:
                if verb is Verb.READ and v == "UC":
                    continue
                kw["conn"] = v
            out.append((v, finish(kw)))
    return out


# ---------------------------------------------------------------------------
# CSV + summary
# ---------------------------------------------------------------------------

POINT_FIELDS = [f.name for f in dataclasses.fields(Point)]
CSV_FIELDS = (
    ["label", "sweep", "sweep_value", "repeat", "ts"]
    + POINT_FIELDS
    + ["cuda_client", "cuda_server"]
    + ["bw_avg_gbps", "bw_peak_gbps", "msg_rate_mpps", "cpu_util", "cpu_cores"]
    + ["per_proc_bw_gbps"]
    + ["lat_typ_us", "lat_avg_us", "lat_p99_us", "lat_p999_us", "lat_max_us"]
    + [f"c_{c}" for c in TRACKED_COUNTERS]
    + [f"s_{c}" for c in TRACKED_COUNTERS]
    + ["elapsed_s"]
)


def _fmt_size(n: int) -> str:
    if n >= 1 << 20 and n % (1 << 20) == 0:
        return f"{n >> 20}MiB"
    if n >= 1 << 10 and n % (1 << 10) == 0:
        return f"{n >> 10}KiB"
    return f"{n}B"


def print_summary(rows: list[dict[str, object]], sweep: Sweep, probe: bool) -> None:
    """Print one table per message size: the median over repeats of each sweep
    value, normalized to the best value in that curve.

    Args:
        rows: CSV rows collected so far.
        sweep: The varying knob (names the first column).
        probe: Whether latency columns are present.
    """
    by_size: dict[int, dict[str, list[dict[str, object]]]] = {}
    order: dict[int, list[str]] = {}
    for r in rows:
        s = int(r["size"])
        v = str(r["sweep_value"])
        by_size.setdefault(s, {}).setdefault(v, []).append(r)
        if v not in order.setdefault(s, []):
            order[s].append(v)

    def med(rs: list[dict[str, object]], key: str) -> float:
        vals = [float(r[key]) for r in rs if r[key] not in ("", None)]
        return statistics.median(vals) if vals else float("nan")

    def isum(rs: list[dict[str, object]], keys: tuple[str, ...]) -> int:
        tot = 0
        for r in rs:
            for k in keys:
                for side in ("c_", "s_"):
                    val = r.get(side + k)
                    if val not in ("", None):
                        tot += int(val)  # type: ignore[arg-type]
        return tot

    hdr = f"{sweep.value:>10} {'Gbps':>9} {'rel':>6} {'Mpps':>9} {'cores':>6}"
    if probe:
        hdr += f" {'lat_typ':>8} {'lat_p99':>8}"
    hdr += f" {'retrans':>8} {'cnp':>7} {'xmit_wait':>10}"
    for size in sorted(by_size):
        print(f"\n=== size {_fmt_size(size)} ===")
        print(hdr)
        curve = by_size[size]
        best = max(med(curve[v], "bw_avg_gbps") for v in order[size])
        for v in order[size]:
            rs = curve[v]
            bw = med(rs, "bw_avg_gbps")
            line = (
                f"{v:>10} {bw:9.2f} {bw / best if best else float('nan'):6.2f} "
                f"{med(rs, 'msg_rate_mpps'):9.3f} {med(rs, 'cpu_cores'):6.2f}"
            )
            if probe:
                line += f" {med(rs, 'lat_typ_us'):8.2f} {med(rs, 'lat_p99_us'):8.2f}"
            xw = 0
            for r in rs:
                for side in ("c_", "s_"):
                    val = r.get(side + "port_xmit_wait")
                    if val not in ("", None):
                        xw += int(val)  # type: ignore[arg-type]
            line += (
                f" {isum(rs, RETRANS_COUNTERS):8d} {isum(rs, CNP_COUNTERS):7d} {xw:10d}"
            )
            print(line)
    print(
        "\nrel = Gbps / best Gbps in that curve. cores = requester CPU in"
        " core-equivalents (perftest reports a machine-wide share)."
        " retrans = ack timeouts + seq errors +"
        " adaptive retrans (both ends, summed over repeats); cnp = ECN congestion"
        " notifications; xmit_wait = ticks the port had data but no credit."
    )


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def preflight(args: argparse.Namespace, hosts: Hosts, points: list[Point]) -> None:
    """Fail early on missing binaries, ssh, or hugepages.

    Args:
        args: CLI arguments.
        hosts: Host configuration.
        points: All points that will run.

    Raises:
        RuntimeError: With an actionable message.
    """
    verb = Verb(args.verb)
    need = [verb.binary] + (["ib_write_lat"] if args.probe_lat else [])
    for b in need + ["taskset"]:
        if shutil.which(b) is None:
            raise RuntimeError(
                f"{b} not on PATH locally (install perftest / util-linux)"
            )
    cp = hosts.ssh(f"which {' '.join(need)} taskset ss; hostname", timeout=20)
    if cp.returncode != 0:
        raise RuntimeError(
            f"ssh {hosts.server_host} failed or binaries missing there:\n"
            f"{cp.stdout}{cp.stderr}"
        )
    if args.hugepages:
        max_bytes = max(p.registered_bytes() * p.procs for p in points)
        need_pages = -(-max_bytes // (2 << 20)) + 16
        for where, txt in (
            ("local", pathlib.Path("/proc/meminfo").read_text()),
            (hosts.server_host, hosts.ssh("cat /proc/meminfo").stdout),
        ):
            m = re.search(r"HugePages_Free:\s+(\d+)", txt)
            free = int(m.group(1)) if m else 0
            if free < need_pages:
                raise RuntimeError(
                    f"--hugepages: {where} has {free} free 2MiB hugepages, sweep needs "
                    f"~{need_pages}. Provision on BOTH hosts first:\n"
                    f"  sudo sysctl vm.nr_hugepages={need_pages}"
                )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Full rationale and sweep descriptions: see the module docstring "
        "(head -120 of this file).",
    )
    p.add_argument("--sweep", required=True, choices=[s.value for s in Sweep])
    p.add_argument(
        "--values",
        default="",
        help="comma-separated sweep values (default: a power-of-two ladder per sweep)",
    )
    p.add_argument(
        "--sizes",
        default="64,4096,65536,1048576",
        help="comma-separated message sizes (bytes) to run every non-size sweep at",
    )
    p.add_argument("--verb", default="write", choices=[v.value for v in Verb])
    p.add_argument(
        "--conn", default="RC", choices=["RC", "UC", "DC"], help="fixed transport"
    )
    p.add_argument(
        "--qps", type=int, default=1, help="fixed QPs per process (non-qps sweeps)"
    )
    p.add_argument(
        "--tx-depth", type=int, default=128, help="fixed WQEs in flight per QP"
    )
    p.add_argument(
        "--const-outstanding",
        type=int,
        default=0,
        help="qps sweep: hold qps*tx_depth at this value instead of fixing tx_depth",
    )
    p.add_argument("--post-list", type=int, default=1)
    p.add_argument(
        "--cq-mod", type=int, default=100, help="perftest's default CQ moderation"
    )
    p.add_argument("--mtu", type=int, default=4096)
    p.add_argument(
        "--outs", type=int, default=16, help="fixed outstanding READs per QP"
    )
    p.add_argument(
        "--hugepages", action="store_true", help="--use_hugepages on both ends"
    )
    p.add_argument("--mr-per-qp", action="store_true", help="--mr_per_qp on both ends")
    p.add_argument("--bidir", action="store_true", help="-b: both ends transmit")
    p.add_argument(
        "--cuda-client", type=int, default=-1, help="client buffer on this GPU"
    )
    p.add_argument(
        "--cuda-server", type=int, default=-1, help="server buffer on this GPU"
    )
    p.add_argument(
        "--extra-args", default="", help="extra perftest args for both ends (quoted)"
    )
    p.add_argument("--duration", type=int, default=5, help="seconds per point (-D)")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument(
        "--probe-lat",
        action="store_true",
        help="concurrent 1-QP ib_write_lat victim flow",
    )
    p.add_argument(
        "--probe-size", type=int, default=64, help="victim-flow message size"
    )
    p.add_argument(
        "--probe-iters",
        type=int,
        default=20000,
        help="victim-flow ping-pong samples (must finish inside the bulk window)",
    )
    p.add_argument(
        "--server-host", default="g6", help="ssh target running the server side"
    )
    p.add_argument(
        "--server-ip", default="192.168.0.2", help="server's IP on the RDMA fabric"
    )
    p.add_argument("--dev", default="mlx5_0", help="client RDMA device")
    p.add_argument("--server-dev", default="mlx5_0")
    p.add_argument("--gid", type=int, default=3, help="client GID index (RoCEv2/IPv4)")
    p.add_argument("--server-gid", type=int, default=3)
    p.add_argument("--base-port", type=int, default=18600)
    p.add_argument(
        "--cpu-base", type=int, default=8, help="first core to pin to (both ends)"
    )
    p.add_argument("--label", default="", help="results/rdma_scaling-<label>.csv")
    p.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    args = p.parse_args()

    sweep = Sweep(args.sweep)
    values = args.values.split(",") if args.values else DEFAULT_VALUES[sweep]
    sizes = _int_list(args.sizes)
    label = (
        args.label
        or f"{sweep.value}-{args.verb}-{args.conn}-{dt.datetime.now():%Y%m%d-%H%M%S}"
    )

    try:
        points = make_points(args, sweep, values, sizes)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    control_dir = pathlib.Path(tempfile.gettempdir()) / f"rdma_scaling-{os.getuid()}"
    control_dir.mkdir(exist_ok=True)
    hosts = Hosts(
        server_host=args.server_host,
        server_ip=args.server_ip,
        client_dev=args.dev,
        server_dev=args.server_dev,
        client_gid=args.gid,
        server_gid=args.server_gid,
        ssh_opts=[
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={control_dir}/cm-%r@%h:%p",
            "-o",
            "ControlPersist=120",
        ],
    )
    extra = shlex.split(args.extra_args)

    if args.dry_run:
        for v, pt in points:
            plan = build_plan(
                pt,
                hosts,
                args.base_port,
                args.cpu_base,
                args.cuda_client,
                args.cuda_server,
                extra,
                args.probe_lat,
                args.probe_size,
                args.probe_iters,
            )
            print(f"\n##### {sweep.value}={v} size={_fmt_size(pt.size)}")
            print(f"# on {hosts.server_host}:")
            print(plan.server_script)
            print("# on this host (after the servers are listening):")
            for cmd in plan.client_cmds:
                print(shlex.join(cmd))
            if args.probe_lat:
                print("# once bulk traffic is flowing:")
                print(shlex.join(plan.lat_client_cmd))
        return 0

    try:
        preflight(args, hosts, [pt for _, pt in points])
    except RuntimeError as e:
        print(f"preflight failed: {e}", file=sys.stderr)
        return 1

    results_dir = SCRIPT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / f"rdma_scaling-{label}.csv"
    log_path = results_dir / f"rdma_scaling-{label}.log"
    print(
        f"sweep={sweep.value} verb={args.verb} conn={args.conn} points={len(points)} "
        f"repeats={args.repeats} duration={args.duration}s -> {csv_path}"
    )

    # perftest reports CPU as a share of ALL cores (/proc/stat), so one busy
    # polling core on a 256-core box reads 0.4%. Core-equivalents are legible.
    ncpu = os.cpu_count() or 1
    rows: list[dict[str, object]] = []
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for idx, (v, pt) in enumerate(points, 1):
            plan = build_plan(
                pt,
                hosts,
                args.base_port,
                args.cpu_base,
                args.cuda_client,
                args.cuda_server,
                extra,
                args.probe_lat,
                args.probe_size,
                args.probe_iters,
            )
            for rep in range(args.repeats):
                tag = (
                    f"[{idx}/{len(points)} {sweep.value}={v} "
                    f"size={_fmt_size(pt.size)} rep={rep}]"
                )
                try:
                    o = run_point(pt, plan, hosts, args.probe_lat, log_path)
                except RuntimeError as e:
                    print(f"{tag} FAILED: {e}", file=sys.stderr)
                    continue
                except KeyboardInterrupt:
                    print(
                        "\ninterrupted; partial results are in the CSV", file=sys.stderr
                    )
                    print_summary(rows, sweep, args.probe_lat)
                    return 130
                row: dict[str, object] = {
                    "label": label,
                    "sweep": sweep.value,
                    "sweep_value": v,
                    "repeat": rep,
                    "ts": dt.datetime.now().isoformat(timespec="seconds"),
                    **{
                        k: (getattr(pt, k).value if k == "verb" else getattr(pt, k))
                        for k in POINT_FIELDS
                    },
                    "cuda_client": args.cuda_client,
                    "cuda_server": args.cuda_server,
                    "bw_avg_gbps": f"{o.bw.bw_avg_gbps:.3f}",
                    "bw_peak_gbps": f"{o.bw.bw_peak_gbps:.3f}",
                    "msg_rate_mpps": f"{o.bw.msg_rate_mpps:.4f}",
                    "cpu_util": f"{o.bw.cpu_util:.2f}",
                    "cpu_cores": f"{o.bw.cpu_util / 100 * ncpu:.2f}",
                    "per_proc_bw_gbps": ";".join(
                        f"{r.bw_avg_gbps:.2f}" for r in o.per_proc
                    ),
                    "lat_typ_us": f"{o.lat.t_typical_us:.2f}" if o.lat else "",
                    "lat_avg_us": f"{o.lat.t_avg_us:.2f}" if o.lat else "",
                    "lat_p99_us": f"{o.lat.t_p99_us:.2f}" if o.lat else "",
                    "lat_p999_us": f"{o.lat.t_p999_us:.2f}" if o.lat else "",
                    "lat_max_us": f"{o.lat.t_max_us:.2f}" if o.lat else "",
                    "elapsed_s": f"{o.elapsed_s:.1f}",
                }
                for c in TRACKED_COUNTERS:
                    row[f"c_{c}"] = o.client_delta.get(c, "")
                    row[f"s_{c}"] = o.server_delta.get(c, "")
                w.writerow(row)
                f.flush()
                rows.append(row)
                lat_s = (
                    f" lat_typ={o.lat.t_typical_us:.2f}us p99={o.lat.t_p99_us:.2f}us"
                    if o.lat
                    else ""
                )
                print(
                    f"{tag} {o.bw.bw_avg_gbps:8.2f} Gbps "
                    f"{o.bw.msg_rate_mpps:8.3f} Mpps "
                    f"cpu={o.bw.cpu_util / 100 * ncpu:5.2f} cores{lat_s} "
                    f"({o.elapsed_s:.0f}s)"
                )

    print_summary(rows, sweep, args.probe_lat)
    print(f"\ncsv: {csv_path}\nraw: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
