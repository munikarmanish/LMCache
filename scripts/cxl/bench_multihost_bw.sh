#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Does aggregate CXL read bandwidth scale with the number of parallel hosts?
#
# Each host runs MLC against its own CXL NUMA node (node 1) at the same
# wall-clock moment; the per-host numbers are summed. Running the hosts one at
# a time (the "solo" baseline) and then together (the "concurrent" run) tells
# you whether they share a bottleneck:
#
#   aggregate ~= sum(solo)   -> the hosts have independent paths; CXL read
#                               bandwidth scales with host count.
#   aggregate ~= one host's  -> the hosts contend on a shared resource (the
#                               solo number      CXL switch / device); adding hosts buys nothing.
#
# Both outcomes are real results. On a topology where every host reaches the
# pool through ONE switch, the flat outcome is the expected one and is the
# thing worth knowing before designing around cross-host CXL bandwidth.
#
# Requirements on every host: mlc on PATH (or --mlc), passwordless ssh from
# the host running this script, and /dev/dax0.0 reconfigured to system-ram so
# the CXL memory appears as a NUMA node:
#
#     sudo sh -c 'echo offline > /sys/devices/system/memory/auto_online_blocks'
#     sudo daxctl reconfigure-device --mode=system-ram dax0.0
#     sudo sh -c 'echo online_movable > /sys/devices/system/memory/auto_online_blocks'
#
# (LMCache itself needs devdax; convert back with --mode=devdax afterwards.)
#
# Usage:
#     scripts/cxl/bench_multihost_bw.sh                      # g5 + g6
#     scripts/cxl/bench_multihost_bw.sh --hosts g5,g6,g7
#     scripts/cxl/bench_multihost_bw.sh --skip-solo          # concurrent only
#     scripts/cxl/bench_multihost_bw.sh --node 1 --secs 5

set -uo pipefail

HOSTS_CSV="g5,g6"
CXL_NODE=1
SECS=5
BUF_KIB=100000          # MLC default; >= this reproduces the settled number
MLC="mlc"
SSH_USER="${USER}"
SKIP_SOLO=0
REPEATS=1

usage() {
    sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hosts)     HOSTS_CSV="$2"; shift 2 ;;
        --node)      CXL_NODE="$2"; shift 2 ;;
        --secs)      SECS="$2"; shift 2 ;;
        --buf-kib)   BUF_KIB="$2"; shift 2 ;;
        --mlc)       MLC="$2"; shift 2 ;;
        --user)      SSH_USER="$2"; shift 2 ;;
        --repeats)   REPEATS="$2"; shift 2 ;;
        --skip-solo) SKIP_SOLO=1; shift ;;
        -h|--help)   usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

IFS=',' read -r -a HOSTS <<< "$HOSTS_CSV"
NHOSTS=${#HOSTS[@]}
if [[ $NHOSTS -lt 1 ]]; then
    echo "need at least one host" >&2
    exit 1
fi

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/cxl-multihost.XXXXXX")"
trap 'rm -rf "$WORKDIR"' EXIT

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=5)

# True when $1 names this machine, so we can run the command directly instead
# of ssh'ing to ourselves.
SELF_NAMES=" $(hostname) $(hostname -s) $(hostname -f 2>/dev/null) localhost "
is_local_host() {
    [[ "$SELF_NAMES" == *" $1 "* ]]
}

# Run MLC on one host, reading from the CXL NUMA node, and echo MB/s.
#
# --bandwidth_matrix reports a full node-to-node matrix; the value we want is
# the row for node 0 (where the CPUs are), column for the CXL node. `-R` is
# read-only traffic. The matrix row looks like:
#        0    187011.8    26217.2
# so the CXL column is field (CXL_NODE + 2).
remote_bw_cmd() {
    local col=$((CXL_NODE + 2))
    cat <<EOF
export PATH="\$HOME/bin:\$PATH"
$MLC --bandwidth_matrix -R -t$SECS -b$BUF_KIB 2>/dev/null \
  | awk '/^ +0[^0-9]/ { print \$$col; exit }'
EOF
}

# Fire MLC on every host in \$@ at once and write "<host> <mbps>" lines to
# \$WORKDIR/<tag>.<host>. A barrier keeps the measurement windows overlapping:
# each host sleeps until a shared start time, so the runs coincide even though
# ssh setup costs differ.
run_group() {
    local tag="$1"; shift
    local -a group=("$@")
    # Absolute epoch second when every host begins measuring. 5s of headroom
    # covers ssh handshake + MLC startup on all hosts.
    local start_at=$(( $(date +%s) + 5 ))
    local pids=()

    local barrier="while [ \$(date +%s) -lt $start_at ]; do sleep 0.05; done"
    for h in "${group[@]}"; do
        (
            # Run locally when the target is this machine -- ssh'ing to
            # ourselves needs host keys we may not have, and the extra hop
            # buys nothing.
            if is_local_host "$h"; then
                out=$(bash -c "$barrier
                               $(remote_bw_cmd)" 2>"$WORKDIR/$tag.$h.err")
            else
                out=$(ssh "${SSH_OPTS[@]}" "${SSH_USER}@${h}" \
                    "$barrier
                     $(remote_bw_cmd)" 2>"$WORKDIR/$tag.$h.err")
            fi
            echo "$h ${out:-FAILED}" > "$WORKDIR/$tag.$h"
        ) &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
}

# Sum the MB/s values for a tag, printing "<total> <n_ok>".
sum_tag() {
    local tag="$1"
    # Only the per-host result files, never the .err siblings.
    find "$WORKDIR" -maxdepth 1 -name "$tag.*" ! -name '*.err' -print0 2>/dev/null \
        | xargs -0 -r awk '
            { if ($2 ~ /^[0-9.]+$/) { s += $2; n++ } }
            END { printf "%.1f %d\n", s, n }'
}

report_group() {
    local tag="$1"
    for f in "$WORKDIR/$tag."*; do
        [[ "$f" == *.err ]] && continue
        [[ -e "$f" ]] || continue
        read -r h v < "$f"
        if [[ "$v" == FAILED || -z "$v" ]]; then
            printf "  %-8s %14s\n" "$h" "FAILED"
            local errf="$WORKDIR/$tag.$h.err"
            [[ -s "$errf" ]] && sed 's/^/      /' "$errf" | head -3
        else
            printf "  %-8s %11s MB/s\n" "$h" "$v"
        fi
    done
}

echo "=============================================================="
echo " CXL aggregate read bandwidth vs. host count"
echo "=============================================================="
echo "hosts        : ${HOSTS[*]}  (n=$NHOSTS)"
echo "CXL numa node: $CXL_NODE"
echo "MLC          : $MLC --bandwidth_matrix -R -t$SECS -b$BUF_KIB"
echo

solo_total=0
if [[ $SKIP_SOLO -eq 0 ]]; then
    echo "--- SOLO: one host at a time (baseline, no cross-host contention) ---"
    for h in "${HOSTS[@]}"; do
        run_group "solo" "$h"
    done
    report_group solo
    read -r solo_total solo_n <<< "$(sum_tag solo)"
    echo "  ------------------------------"
    printf "  %-8s %11s MB/s  (sum of solo runs = ideal linear scaling)\n" \
        "SUM" "$solo_total"
    echo
fi

conc_total=0
for r in $(seq 1 "$REPEATS"); do
    if [[ $REPEATS -gt 1 ]]; then
        echo "--- CONCURRENT: all $NHOSTS hosts at once (repeat $r/$REPEATS) ---"
    else
        echo "--- CONCURRENT: all $NHOSTS hosts reading at the same time ---"
    fi
    rm -f "$WORKDIR"/conc.* 2>/dev/null
    run_group "conc" "${HOSTS[@]}"
    report_group conc
    read -r conc_total conc_n <<< "$(sum_tag conc)"
    echo "  ------------------------------"
    printf "  %-8s %11s MB/s  (aggregate across %s hosts)\n" \
        "TOTAL" "$conc_total" "${conc_n:-0}"
    echo
done

echo "=============================================================="
echo " VERDICT"
echo "=============================================================="
if [[ $SKIP_SOLO -eq 1 ]]; then
    echo "Aggregate across $NHOSTS hosts: $conc_total MB/s"
    echo "(no solo baseline; re-run without --skip-solo to judge scaling)"
    exit 0
fi

if [[ "${solo_n:-0}" -ne "$NHOSTS" || "${conc_n:-0}" -ne "$NHOSTS" ]]; then
    echo "  INCONCLUSIVE: only ${solo_n:-0}/$NHOSTS hosts reported in the solo"
    echo "  run and ${conc_n:-0}/$NHOSTS in the concurrent run. A verdict needs"
    echo "  every host in both. Fix the failures above and re-run."
    exit 1
fi

awk -v solo="$solo_total" -v conc="$conc_total" -v n="$NHOSTS" '
BEGIN {
    if (solo <= 0 || conc <= 0) {
        print "Could not compute a ratio (a run failed). See output above.";
        exit 1;
    }
    per_host = solo / n;
    ratio    = conc / solo;
    speedup  = conc / per_host;
    printf "  sum of solo runs      : %10.1f MB/s\n", solo;
    printf "  concurrent aggregate  : %10.1f MB/s\n", conc;
    printf "  scaling efficiency    : %9.1f%%   (concurrent / sum-of-solo)\n", ratio*100;
    printf "  effective speedup     : %10.2fx  over a single host (n=%d)\n", speedup, n;
    print  "";
    if (ratio >= 0.85)
        print "  => SCALES. Hosts have independent paths to the pool; aggregate\n" \
              "     CXL read bandwidth grows with host count.";
    else if (ratio <= 0.60)
        print "  => DOES NOT SCALE. The hosts contend on a shared resource (the\n" \
              "     CXL switch or the device itself). Aggregate is capped near a\n" \
              "     single host'\''s bandwidth -- adding hosts does not add bandwidth.";
    else
        print "  => PARTIAL SCALING. Some shared-resource contention; aggregate\n" \
              "     grows sublinearly with host count.";
}'
