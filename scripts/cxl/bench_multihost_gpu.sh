#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run bench_h2d.py on several hosts at the same time, against the same shared
# CXL pool, and sum the aggregate bandwidths.
#
# This is the GPU-side counterpart to bench_multihost_bw.sh (which drives MLC
# and therefore measures the CPU's path to the pool). GPU reads are peer-to-peer
# DMA from the PCIe endpoint and do not take the same route through the host
# mesh, so the two can and do disagree -- running both is how you tell a device
# limit apart from a host-path limit.
#
# What the numbers mean:
#   sum(solo) ~= concurrent   -> the pool serves each host independently; GPU
#                                bandwidth scales with the number of hosts.
#   concurrent ~= one host    -> the hosts contend on a shared ceiling (the
#                                switch, the endpoint port, or the modules).
#
# Requirements on every host: the LMCache checkout at the same path with an
# up-to-date scripts/cxl/bench_h2d.py, the lmcache venv, /dev/dax0.0 in DEVDAX
# mode (not system-ram -- that is the MLC configuration), and passwordless ssh.
#
# Usage:
#     scripts/cxl/bench_multihost_gpu.sh
#     scripts/cxl/bench_multihost_gpu.sh --hosts g5,g6 --payload-kib 8192
#     scripts/cxl/bench_multihost_gpu.sh --gap-gib 0      # all GPUs on module 1
#     scripts/cxl/bench_multihost_gpu.sh --skip-solo

set -uo pipefail

HOSTS_CSV="g5,g6"
PAYLOAD_KIB=8192
ITERS=20000
GAP_GIB=128
GPUS="all"
DIRECTION="h2d"
DEV="/dev/dax0.0"
REPO="/home/manish/code/LMCache"
PY="/home/manish/.virtualenvs/lmcache/bin/python"
SSH_USER="${USER}"
SKIP_SOLO=0

usage() { sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hosts)       HOSTS_CSV="$2"; shift 2 ;;
        --payload-kib) PAYLOAD_KIB="$2"; shift 2 ;;
        --iters)       ITERS="$2"; shift 2 ;;
        --gap-gib)     GAP_GIB="$2"; shift 2 ;;
        --gpus)        GPUS="$2"; shift 2 ;;
        --dir)         DIRECTION="$2"; shift 2 ;;
        --dev)         DEV="$2"; shift 2 ;;
        --repo)        REPO="$2"; shift 2 ;;
        --python)      PY="$2"; shift 2 ;;
        --user)        SSH_USER="$2"; shift 2 ;;
        --skip-solo)   SKIP_SOLO=1; shift ;;
        -h|--help)     usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

IFS=',' read -r -a HOSTS <<< "$HOSTS_CSV"
NHOSTS=${#HOSTS[@]}

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/cxl-mhgpu.XXXXXX")"
trap 'rm -rf "$WORKDIR"' EXIT
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=5)

SELF_NAMES=" $(hostname) $(hostname -s) $(hostname -f 2>/dev/null) localhost "
is_local_host() { [[ "$SELF_NAMES" == *" $1 "* ]]; }

# The bench command for one host. --gpus/--offset-gap-gib make each host drive
# both of its GPUs, one per CXL module. We pull the CXL row's aggregate column
# out of the table: the row reads
#     <payload> <solo> <aggr> <speedup>x <effic>%  <per-gpu...>
# and we want $3 (aggr) from the line inside the [CXL] section.
bench_cmd() {
    local gap_arg=""
    [[ "$GAP_GIB" != "0" ]] && gap_arg="--offset-gap-gib $GAP_GIB"
    # Emit "<aggr> <start_epoch> <end_epoch>". The timestamps bracket the whole
    # bench process, so the caller can verify the hosts' windows really did
    # overlap instead of assuming the barrier lined them up.
    cat <<EOF
cd $REPO
__t0=\$(date +%s.%N)
__v=\$($PY scripts/cxl/bench_h2d.py \
    --cxl-dev $DEV --gpus $GPUS --dir $DIRECTION \
    --min-kib $PAYLOAD_KIB --max-kib $PAYLOAD_KIB --iters $ITERS $gap_arg 2>/dev/null \
  | awk '/\[CXL\]/{c=1} c && /^ *[0-9]+ (KiB|MiB) /{print \$4; exit}')
__t1=\$(date +%s.%N)
echo "\$__v \$__t0 \$__t1"
EOF
}

# Launch the bench on each host in \$@ so their measurement windows overlap.
# Every host waits for the same absolute epoch second before starting; the
# headroom must cover ssh + python + torch import + CUDA init, which is much
# slower to start than MLC, hence 25s rather than 5s.
run_group() {
    local tag="$1"; shift
    local -a group=("$@")
    local start_at=$(( $(date +%s) + 30 ))
    local pids=()
    for h in "${group[@]}"; do
        (
            barrier="while [ \$(date +%s) -lt $start_at ]; do sleep 0.05; done"
            if is_local_host "$h"; then
                out=$(bash -c "$barrier; $(bench_cmd)" 2>"$WORKDIR/$tag.$h.err")
            else
                out=$(ssh "${SSH_OPTS[@]}" "${SSH_USER}@${h}" \
                          "$barrier; $(bench_cmd)" 2>"$WORKDIR/$tag.$h.err")
            fi
            echo "$h ${out:-FAILED}" > "$WORKDIR/$tag.$h"
        ) &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
}

sum_tag() {
    # One awk over all files (see check_overlap): a split invocation would
    # print a partial sum per chunk instead of one total.
    find "$WORKDIR" -maxdepth 1 -name "$1.*" ! -name '*.err' -exec cat {} + 2>/dev/null \
        | awk '{ if ($2 ~ /^[0-9.]+$/) { s += $2; n++ } }
               END { printf "%.2f %d\n", s, n }'
}

report_group() {
    for f in "$WORKDIR/$1."*; do
        [[ "$f" == *.err ]] && continue
        [[ -e "$f" ]] || continue
        read -r h v _t0 _t1 < "$f"
        if [[ "$v" == FAILED || -z "$v" ]]; then
            printf "  %-8s %14s\n" "$h" "FAILED"
            [[ -s "$WORKDIR/$1.$h.err" ]] && sed 's/^/      /' "$WORKDIR/$1.$h.err" | head -4
        else
            printf "  %-8s %11s GB/s  (both GPUs on that host)\n" "$h" "$v"
        fi
    done
}

# Report how much of the hosts' run windows actually coincided. The aggregate
# is only meaningful if every host was transferring at the same time; a low
# overlap means the hosts ran mostly sequentially and the "concurrent" total is
# really a sum of near-solo runs.
check_overlap() {
    # cat the files into one awk invocation: piping through xargs can split
    # them across several awk runs, each seeing a single host and so never
    # computing an intersection.
    find "$WORKDIR" -maxdepth 1 -name "$1.*" ! -name '*.err' -exec cat {} + 2>/dev/null \
        | awk '
            # n must be initialised before it indexes the arrays: an unset n
            # subscripts by the empty string, leaving t0[0]/t1[0] as zeros and
            # collapsing the intersection below. +0 forces numeric comparison.
            BEGIN { n = 0 }
            { if ($3 ~ /^[0-9.]+$/) { t0[n] = $3+0; t1[n] = $4+0; n++ } }
            END {
                if (n < 2) { exit }
                lo = t0[0]; hi = t1[0];
                for (i = 1; i < n; i++) {
                    if (t0[i] > lo) lo = t0[i];
                    if (t1[i] < hi) hi = t1[i];
                }
                span = 0;
                for (i = 0; i < n; i++) { d = t1[i]-t0[i]; if (d > span) span = d }
                ov = hi - lo;
                if (ov < 0) ov = 0;
                printf "  window overlap        : %.1fs of %.1fs longest run (%.0f%%)\n",
                       ov, span, (span > 0 ? ov/span*100 : 0);
                if (span > 0 && ov/span < 0.8)
                    print  "  WARNING: the hosts did not overlap for most of their runs, so\n" \
                           "  the concurrent total is closer to a sum of solo runs than to a\n" \
                           "  real contention measurement. Raise --iters.";
            }'
}

echo "=============================================================="
echo " Multi-host GPU bandwidth against the shared CXL pool"
echo "=============================================================="
echo "hosts    : ${HOSTS[*]}  (n=$NHOSTS)"
echo "per host : --gpus $GPUS  --offset-gap-gib $GAP_GIB  ${PAYLOAD_KIB} KiB x $ITERS  ($DIRECTION)"
echo "device   : $DEV (must be devdax)"
echo

solo_total=0; solo_n=0
if [[ $SKIP_SOLO -eq 0 ]]; then
    echo "--- SOLO: one host at a time ---"
    for h in "${HOSTS[@]}"; do run_group "solo" "$h"; done
    report_group solo
    read -r solo_total solo_n <<< "$(sum_tag solo)"
    echo "  ------------------------------"
    printf "  %-8s %11s GB/s  (sum of solo runs = ideal linear scaling)\n" "SUM" "$solo_total"
    echo
fi

echo "--- CONCURRENT: all $NHOSTS hosts at once ---"
run_group "conc" "${HOSTS[@]}"
report_group conc
read -r conc_total conc_n <<< "$(sum_tag conc)"
echo "  ------------------------------"
printf "  %-8s %11s GB/s  (aggregate across %s hosts)\n" "TOTAL" "$conc_total" "${conc_n:-0}"
check_overlap conc
echo

echo "=============================================================="
echo " VERDICT"
echo "=============================================================="
if [[ $SKIP_SOLO -eq 1 ]]; then
    echo "  Aggregate across $NHOSTS hosts: $conc_total GB/s"
    echo "  (no solo baseline; re-run without --skip-solo to judge scaling)"
    exit 0
fi
if [[ "${solo_n:-0}" -ne "$NHOSTS" || "${conc_n:-0}" -ne "$NHOSTS" ]]; then
    echo "  INCONCLUSIVE: ${solo_n:-0}/$NHOSTS hosts reported solo and"
    echo "  ${conc_n:-0}/$NHOSTS concurrent. A verdict needs every host in both."
    exit 1
fi
awk -v solo="$solo_total" -v conc="$conc_total" -v n="$NHOSTS" 'BEGIN {
    if (solo <= 0 || conc <= 0) { print "  a run failed; see above"; exit 1 }
    printf "  sum of solo runs      : %10.2f GB/s\n", solo;
    printf "  concurrent aggregate  : %10.2f GB/s\n", conc;
    printf "  scaling efficiency    : %9.1f%%\n", conc/solo*100;
    printf "  effective speedup     : %10.2fx over one host (n=%d)\n", conc/(solo/n), n;
    print "";
    if (conc/solo >= 0.85)
        print "  => SCALES with hosts: the pool is not the ceiling for GPU P2P\n" \
              "     reads, so the earlier CPU-side ~26 GB/s was a host-path limit.";
    else if (conc/solo <= 0.60)
        print "  => DOES NOT SCALE: the hosts share one ceiling (switch/endpoint\n" \
              "     port/modules). That ceiling is the pool-wide GPU read budget.";
    else
        print "  => PARTIAL SCALING: some shared-resource contention.";
}'
