#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# bg_nic_load.sh — Synthetic background RDMA load to contend for the NIC
# that the nixl_peer adapter uses, standing in for the TP/PP collectives
# and P/D-disaggregation traffic this 2-node setup does not have.
#
# WHY THIS EXISTS
# ---------------
# The CXL L2 adapter reaches remote KV cache over the CXL fabric; the
# nixl_peer adapter reaches it over RDMA through the NIC. On an idle NIC
# the RDMA path is fast, so the two look comparable. In a real serving
# deployment the NIC is NOT idle: it carries TP/PP collectives, P/D KV
# transfer, and model/weight traffic. This script recreates that
# contention so the comparison reflects a realistic deployment.
#
# TOPOLOGY (see also: scripts/cxl/README.md)
#   mlx5_0 -> ens3f0np0  200 Gbps  192.168.0.x   <- NIXL/UCX runs here
#   mlx5_1 -> ens3f1np1  100 Gbps  192.168.1.x
#   Both ports are on ONE BlueField-3 (PCI d6:00.0/.1), behind a single
#   PCIe Gen5 x16 uplink. So there are two distinct contention modes:
#
#   MODE "port"  (--dev mlx5_0, default)
#       Background traffic shares the SAME 200G port as NIXL. Contends
#       for port bandwidth AND the PCIe uplink AND the DPU packet
#       engines. This is the faithful analogue of P/D transfer or TP
#       collectives sharing a NIC with the KV path.
#
#   MODE "pcie"  (--dev mlx5_1)
#       Background traffic uses the OTHER port, so it does not consume
#       NIXL's port bandwidth, but still shares the PCIe x16 uplink and
#       the DPU's processing capacity. Use this to show that even
#       traffic on a "different" link degrades NIXL, because the DPU is
#       one device -- while CXL is unaffected either way.
#
# CXL traffic touches NEITHER of these resources, which is the point.
#
# USAGE
#   # 1. On the SERVER node (the one that will be the donor, e.g. g5):
#   ./bg_nic_load.sh server
#
#   # 2. On the CLIENT node (e.g. g6), pick a load level:
#   ./bg_nic_load.sh client --peer 192.168.0.1 --load 50
#
#   # Sweep contention intensity (recommended -- one load point cannot
#   # tell you WHERE the crossover is):
#   for L in 0 25 50 75 100; do
#       ./bg_nic_load.sh client --peer 192.168.0.1 --load $L --duration 120 &
#       # ... run the TTFT benchmark against the nixl node here ...
#   done
#
# LOAD CONTROL
#   --load N   Approximate percentage of line rate to consume (0-100).
#              Implemented by rate-limiting perftest. 0 means "start
#              nothing" so the same harness can collect the baseline.
#              100 means unthrottled (saturate the link).
#
# NOTE ON QoS: this deliberately does NOT configure DSCP/PFC priority.
# The default is a fair-share fight for the link, which is the
# pessimistic-but-honest case. If you want to model a deployment where
# collectives are prioritized over KV traffic, add --tclass.

set -euo pipefail

ROLE="${1:-}"
shift || true

DEV="mlx5_0"          # NIXL's port by default -> direct contention
GID_IDX=3             # RoCEv2 IPv4 GID; check `show_gids` if this fails
PORT=18515
PEER=""
LOAD=100
DURATION=60
SIZE=1048576          # 1 MiB messages: bandwidth-shaped, like collectives
QPS=4                 # multiple QPs, as a real collective would use

usage() { sed -n '2,60p' "$0"; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)      DEV="$2"; shift 2 ;;
        --peer)     PEER="$2"; shift 2 ;;
        --load)     LOAD="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --size)     SIZE="$2"; shift 2 ;;
        --qps)      QPS="$2"; shift 2 ;;
        --port)     PORT="$2"; shift 2 ;;
        --gid)      GID_IDX="$2"; shift 2 ;;
        -h|--help)  usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

if ! command -v ib_write_bw >/dev/null 2>&1; then
    echo "ERROR: ib_write_bw not found (install perftest)" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Line rate of the chosen device, used to turn --load into a rate limit.
# ---------------------------------------------------------------------------
netdev="$(ibdev2netdev 2>/dev/null | awk -v d="$DEV" '$1==d {print $5}')"
if [[ -z "$netdev" ]]; then
    echo "ERROR: could not map $DEV to a netdev (check ibdev2netdev)" >&2
    exit 1
fi
speed_mbps="$(cat "/sys/class/net/${netdev}/speed" 2>/dev/null || echo 0)"
if [[ "$speed_mbps" -le 0 ]]; then
    echo "ERROR: could not read link speed for $netdev" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Rate limiting. perftest's --rate_limit is in the units given by
# --rate_units; "g" is Gbps. We target LOAD% of line rate.
# ---------------------------------------------------------------------------
rate_args=()
if [[ "$LOAD" -eq 0 ]]; then
    echo "[bg_nic_load] --load 0: generating no background traffic (baseline)."
    exit 0
elif [[ "$LOAD" -lt 100 ]]; then
    target_gbps="$(awk -v s="$speed_mbps" -v l="$LOAD" 'BEGIN{printf "%.1f", s/1000.0*l/100.0}')"
    # rate_limit_type=SW on purpose. The default (HW) offloads pacing to
    # the NIC's own scheduler, which both smooths the traffic into
    # something less like bursty collective traffic and leans on the very
    # DPU resource we are trying to contend for. SW pacing keeps the
    # burstiness and leaves the packet engines to fight for.
    rate_args=(--rate_limit="$target_gbps" --rate_units=g --rate_limit_type=SW)
    echo "[bg_nic_load] dev=$DEV ($netdev, ${speed_mbps} Mb/s) target=${target_gbps} Gbps (${LOAD}%)"
else
    echo "[bg_nic_load] dev=$DEV ($netdev, ${speed_mbps} Mb/s) target=UNTHROTTLED (100%)"
fi

common=(
    -d "$DEV"
    -x "$GID_IDX"
    -p "$PORT"
    -s "$SIZE"
    -q "$QPS"
    -D "$DURATION"
    --report_gbits
)

case "$ROLE" in
    server)
        echo "[bg_nic_load] server: waiting for client on port $PORT (${DURATION}s)"
        exec ib_write_bw "${common[@]}" "${rate_args[@]}"
        ;;
    client)
        if [[ -z "$PEER" ]]; then
            echo "ERROR: client role requires --peer <server-ip-on-that-fabric>" >&2
            echo "       e.g. --peer 192.168.0.1 for mlx5_0, 192.168.1.1 for mlx5_1" >&2
            exit 1
        fi
        echo "[bg_nic_load] client: driving load against $PEER for ${DURATION}s"
        exec ib_write_bw "${common[@]}" "${rate_args[@]}" "$PEER"
        ;;
    *)
        usage
        ;;
esac
