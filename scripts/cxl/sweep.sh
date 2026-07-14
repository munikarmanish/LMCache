#!/usr/bin/env bash
# sweep.sh — Drive the kv-cache-tester workloads across adapter x strategy x
# working-set and distil each arm's per-tier hit/miss counts into one CSV.
#
# This is the harness for the "CXL vs NIXL/RDMA cross-node reuse" experiment.
# It does NOT launch the nodes or the router — bring those up first with
# launch_node.sh (both nodes) and, for Sweep B, launch_router.sh. This script
# only: clears the cache, runs one tester arm, then greps the arm's window of
# MP-server log for the (N L1, M L2) prefetch breakdown and writes a CSV row.
#
# Two sweeps (select with the first arg):
#
#   A  capacity headline. working_set_tester in FIXED mode at concurrency 1,
#      one invocation walking the working-set band (0.4M–1.0M tok, under the
#      CXL pool ceiling). Fixed mode needs no --max-ttft, so steady serial load
#      isolates the capacity effect (chunk still in CXL vs recomputed) with no
#      concurrency-search / SLA confound. Hits one node directly (no router).
#      Above a single node's 64 GB L1 (~0.52M tok) that node overflows local
#      DRAM and must fall to CXL (hit) where RDMA would recompute — the gap.
#      Per-working-set TTFT/throughput lands in the tester's own kvct/ output;
#      the CSV carries a whole-run tier tally.
#
#   B  cross-node dedup + fan-out. Drives working_set_tester (fixed mode, same
#      as A) through the ROUTER under the anti_affinity strategy, which routes
#      each repeated prompt AWAY from the node that cached it — forcing the OTHER
#      node to serve it. CXL's second node reads the one shared CXL copy directly;
#      NIXL's must RDMA-fetch from the peer every time (and the chunk was deleted
#      on the peer's eviction → often recompute). That asymmetry is the shared-
#      substrate win. Requires BOTH nodes up in <adapter> mode AND the router
#      already running anti_affinity (./launch_router.sh anti_affinity), which
#      this script checks via /health. Concurrency defaults to 8 (must be >1 so
#      both nodes see load). node1's MP log is rsync'd from c2 to tally both
#      nodes' hits. B PINS ONE working set (WS_B, default 0.8M) — it does not
#      sweep: the cross-node effect is about the same chunk read by both nodes,
#      not capacity, so the working set is a precondition (upper band) not the
#      variable. Run: ./sweep.sh B <cxl|nixl> anti_affinity
#
#   C  realistic trace-replay capstone. Replays real agentic-coding traces
#      (KVCT_DIR/traces) through the ROUTER, mixing every effect A and B
#      isolated: realistic reuse, cross-conversation sharing (warm prefix),
#      eviction aging, and real inter-request timing. Answers "does CXL vs NIXL
#      matter under a realistic workload" — run AFTER A/B, which explain the
#      mechanisms. Same router + dual-node-tally plumbing as B; all tester seeds
#      pinned (TRACE_SEED) so cxl and nixl arms replay an identical sequence.
#      CHUNK_SIZE must match LMCache's chunk_size (256). Run:
#      ./sweep.sh C <cxl|nixl> anti_affinity
#
# The token<->GB bridge uses THIS testbed's measured anchor for
# Llama-3.1-8B-Instruct: 1.2188 GB / 10k tokens => 1000 tok = 0.12188 GB.
# With L1=64 GB/node x2 = 128 GB and CXL=128 GB shared:
#   L1 union      ~ 1.05e6 tok
#   L1+CXL union  ~ 2.10e6 tok
# so the CXL-win band is roughly [1.05M, 2.10M] working-set tokens.
#
# Usage:
#   ./sweep.sh A <cxl|nixl>                 # capacity sweep, single node
#   ./sweep.sh B <cxl|nixl> anti_affinity   # cross-node sweep, via router
#   ./sweep.sh C <cxl|nixl> anti_affinity   # trace-replay capstone, via router
#
# Examples:
#   ./sweep.sh A cxl
#   ./sweep.sh A nixl
#   ./sweep.sh B cxl anti_affinity          # needs both nodes + router up
#   ./sweep.sh B nixl anti_affinity
#   ./sweep.sh C cxl anti_affinity          # trace capstone; needs nodes+router
#   ./sweep.sh C nixl anti_affinity
#
# Env overrides:
#   KVCT_DIR           path to the kv-cache-tester checkout (has *_tester.py).
#   PY                 python to run the tester with (default: kv-cache-tester's
#                      own `uv run python`; set PY=python for the active venv).
#   ROUTER_HOST        host:port the tester hits (default: A=node0 direct
#                      192.168.128.31:8010, B/C=router 192.168.128.31:8000).
#   NODE1_SSH          ssh target for node1's MP log (B/C; default manish@c2).
#   NODE1_LOG_REMOTE   remote path of node1's MP log (B/C; default mirrors
#                      this dir: <HERE>/logs/node1-<adapter>.log).
#   -- Sweep C (trace replay) --
#   TRACE_DIR          trace directory (default $KVCT_DIR/traces).
#   CHUNK_SIZE         tester block size; MUST match LMCache chunk_size (256).
#   TRACE_MAX_CONTEXT  cap per-request context tokens (default 120000).
#   TRACE_START_USERS / TRACE_MAX_USERS  user-session ramp (default 2 / 8).
#   TRACE_WARM_PREFIX_PCT  shared prefix fraction 0-1 (default 0.5).
#   TRACE_CACHE_MAX_AGE    block eviction age seconds (default 600).
#   TRACE_TIME_SCALE   inter-request think-time scale (default 0.1 = 10x faster).
#   TRACE_MAX_TTFT     SLA gate seconds (default 10.0).
#   TRACE_SEED         seed for trace+prompt+content; pin across arms (default 42).
#   CONTEXT_A          per-request context tokens (default 30000).
#   WS_POINTS          Sweep A working-set points, space-separated tokens;
#                      first/last/count set the tester's min/max/increments
#                      (default 0.3M–1.0M, under the 128 GB CXL pool ceiling).
#   WS_B               Sweep B pinned working set in tokens (default 800000);
#                      must be in the upper band (>~0.52M single-node L1, <~1.05M
#                      CXL ceiling) for the cross-node effect to appear.
#   FIXED_CONCURRENCY  fixed-mode concurrency (default: A=1 serial, B=8 so both
#                      nodes see concurrent load).
#   TEST_DURATION      total tester wall-clock seconds (default 600); divided
#                      across working-set growth sections, so higher = more warm
#                      requests per period, diluting cold-transition medians.
#   INIT_STRATEGY      working-set growth: 'min' (default) generates unwarmed
#                      prompts on the fly (cold transitions); 'max' pre-warms the
#                      full set up front so no period sees a cold prompt (clean
#                      steady-state capacity sweep). See the in-body note.
#   TOKENIZER          HF tokenizer id (default meta-llama/Llama-3.1-8B-Instruct).
#   OUT_ROOT           results root (default ./results/sweep).
#
# NOTE: the CXL backend has NO live-chunk capacity eviction — a working set
# above the ~1.05M-token (128 GB) pool fills it and then silently DROPS further
# stores (NoRegionAvailable). Keep WS_POINTS under ~1.0M or results past the
# fill point reflect a frozen, order-dependent CXL, not a warm one.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# --- topology (must match launch_node.sh) ----------------------------------
NODE0_HOST=192.168.128.31
NODE1_HOST=192.168.128.32
VLLM_PORT=8010
LMC_HTTP_PORT=8090
ROUTER_PORT=8000

# --- args ------------------------------------------------------------------
SWEEP="${1:?usage: $0 <A|B|C> <cxl|nixl> [strategy]}"
ADAPTER="${2:?usage: $0 <A|B|C> <cxl|nixl> [strategy]}"
STRATEGY="${3:-max_prefix}"   # only used / recorded for sweeps B and C

case "$ADAPTER" in cxl|nixl) ;; *) echo "adapter must be cxl|nixl" >&2; exit 2 ;; esac
case "$SWEEP"   in A|B|C) ;;   *) echo "sweep must be A|B|C" >&2; exit 2 ;; esac

# --- knobs -----------------------------------------------------------------
KVCT_DIR="${KVCT_DIR:-$HOME/code/kv-cache-tester}"
PY="${PY:-python}"
CONTEXT_A="${CONTEXT_A:-30000}"
# Sweep-A working-set points (tokens); first/last/count drive the tester's
# --min/--max/--increments.
#
# HARD CEILING: the CXL pool is 511 x 256 MiB = 127.75 GB ~ 1.048M tokens, and
# the backend has NO capacity-eviction of live chunks — once the pool fills,
# alloc() raises NoRegionAvailable and every further store is dropped (silent
# CXL freeze). So we keep the working set UNDER the pool: max point 1.0M tok
# (~122 GB) stays just under the ceiling. Sweep 0.3M–1.0M: a single node's L1
# is 64 GB (~0.52M tok), so above ~0.52M one node overflows local DRAM (would
# recompute) while CXL still holds everything — that is the CXL-win band. The
# 0.3M–0.5M points below the knee are the ties baseline (both fit one DRAM).
# Override the whole list with WS_POINTS="a b c ...".
WS_POINTS="${WS_POINTS:-300000 400000 500000 600000 700000 800000 900000 1000000}"
OUT_ROOT="${OUT_ROOT:-$HERE/results/sweep}"

if [[ ! -d "$KVCT_DIR" ]]; then
    echo "ERROR: kv-cache-tester not found at KVCT_DIR=$KVCT_DIR" >&2
    echo "       git clone https://github.com/callanjfox/kv-cache-tester and/or set KVCT_DIR" >&2
    exit 1
fi

LOG_DIR="$HERE/logs"
# The MP-server log this adapter writes to on node0 (per launch_node.sh naming).
# Sweep A hits node0 directly, so node0's log carries every retrieval. Sweep B
# routes across both nodes; we tally node0 AND node1 logs (node1's must be
# reachable — copy it over or NFS-mount; here we assume node1's log is synced
# into logs/ as node1-<adapter>.log, else only node0 is counted with a warn).
MPLOG0="$LOG_DIR/node0-${ADAPTER}.log"
MPLOG1="$LOG_DIR/node1-${ADAPTER}.log"

if [[ ! -f "$MPLOG0" ]]; then
    echo "ERROR: MP log $MPLOG0 not found — is node0 up in '$ADAPTER' mode?" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"
STAMP="$(date +%Y%m%d-%H%M%S)"
# Router sweeps (B, C) tag the run dir with the strategy; A does not. Build the
# suffix explicitly rather than inline `$(... && echo ...)` — under `set -e` that
# substitution returns non-zero for sweep A (the [[ ]] test is false), and a
# failing command substitution in a bare assignment aborts the whole script.
RUN_SUFFIX=""
if [[ "$SWEEP" == B || "$SWEEP" == C ]]; then
    RUN_SUFFIX="_${STRATEGY}"
fi
RUN_DIR="$OUT_ROOT/${SWEEP}_${ADAPTER}${RUN_SUFFIX}_${STAMP}"
mkdir -p "$RUN_DIR"
CSV="$RUN_DIR/tiers.csv"
echo "sweep,adapter,strategy,context,working_set,requests,chunks_l1,chunks_l2,chunks_total,retrievals,recompute_frac" > "$CSV"

# ---------------------------------------------------------------------------
# tally_tiers <mplog...> <since_line0> <since_line1>
#   Sum the (N L1, M L2) prefetch breakdowns that appeared in each MP log AFTER
#   its pre-run line count, i.e. only lines produced by the arm we just ran.
#   Emits: "<sum_l1> <sum_l2> <retrieval_count>".
#
#   The line we parse (storage_manager.py) looks like:
#     Prefetch request completed (L1+L2): 124/124 retained keys (124 L1, 0 L2) ...
#   For NIXL the "L2" count is RDMA-peer hits; for CXL it is CXL-load hits.
#   Chunks NOT retained (request chunk-count minus retained) are recomputes;
#   the tester-side TTFT captures their cost, so here we only need the hit mix
#   plus the retained/total ratio the same line already carries as "R/T".
# ---------------------------------------------------------------------------
tally_tiers() {
    local log="$1" since="$2"
    # New lines only (tail from the pre-run offset), matching the prefetch line.
    tail -n "+$((since + 1))" "$log" 2>/dev/null | awk '
        # ... retained keys (124 L1, 0 L2) ...
        match($0, /\(([0-9]+) L1, ([0-9]+) L2\)/, m) {
            l1 += m[1]; l2 += m[2]; n += 1
        }
        # ... 124/248 retained keys ...  (retained/total for recompute frac)
        match($0, /: ([0-9]+)\/([0-9]+) retained keys/, r) {
            ret += r[1]; tot += r[2]
        }
        END { printf "%d %d %d %d %d\n", l1+0, l2+0, n+0, ret+0, tot+0 }
    '
}

tally_and_record() {
    # tally_and_record <ws-label> <off0> <off1>
    #   Tally per-tier hits from the MP log lines produced since the given
    #   offsets and append one CSV row labelled with <ws-label>.
    local ws="$1" off0="$2" off1="$3"
    local l1 l2 n ret tot
    read -r l1_0 l2_0 n0 ret0 tot0 <<< "$(tally_tiers "$MPLOG0" "$off0")"
    l1_1=0; l2_1=0; n1=0; ret1=0; tot1=0
    if [[ -f "$MPLOG1" ]]; then
        read -r l1_1 l2_1 n1 ret1 tot1 <<< "$(tally_tiers "$MPLOG1" "$off1")"
    elif [[ "$SWEEP" == B ]]; then
        echo "  WARN: $MPLOG1 absent — node1 hits NOT counted (sync node1's MP log into logs/)." >&2
    fi

    l1=$(( l1_0 + l1_1 )); l2=$(( l2_0 + l2_1 )); n=$(( n0 + n1 ))
    ret=$(( ret0 + ret1 )); tot=$(( tot0 + tot1 ))
    local total_chunks=$(( l1 + l2 ))
    local recompute_frac="0"
    if [[ "$tot" -gt 0 ]]; then
        recompute_frac="$(awk -v r="$ret" -v t="$tot" 'BEGIN{printf "%.4f", (t-r)/t}')"
    fi

    echo "$SWEEP,$ADAPTER,$STRATEGY,$CONTEXT_A,$ws,$n,$l1,$l2,$total_chunks,$n,$recompute_frac" >> "$CSV"
    echo "  tallied: L1=$l1 L2=$l2 retrievals=$n retained=$ret/$tot recompute_frac=$recompute_frac"
}

log_offset() {
    # Line count of a log file, or 0 if it does not exist. Guard the existence
    # test first: `wc -l < missing` fails at redirection (before wc runs) and
    # the `|| echo 0` fallback does not catch that, leaking a shell error.
    [[ -f "$1" ]] || { echo 0; return; }
    wc -l < "$1"
}

# ---------------------------------------------------------------------------
# Shared tester knobs (both sweeps drive working_set_tester in fixed mode).
# ---------------------------------------------------------------------------
# Pin the tokenizer to the served model. The tester auto-detects it from
# /v1/models, but pinning makes client-side token counting deterministic and
# independent of that silent auto-detect (which would otherwise fall back to the
# tester's Qwen default if detection failed). NOTE: this does NOT fix the
# tester's "Unknown model" warning or its GB estimates — those come from a
# separate hardcoded KV_CACHE_SIZES table (70B/Qwen only) and are display-only;
# the exact token counts in its working_set_size column are what we plot from.
TOKENIZER="${TOKENIZER:-meta-llama/Llama-3.1-8B-Instruct}"
# Total tester wall-clock, divided across the working-set growth sections, so
# longer duration = more requests per section AFTER each growth event, diluting
# the cold-transition front in each period's median TTFT.
TEST_DURATION="${TEST_DURATION:-300}"
# Working-set growth strategy (the tester's --init-strategy):
#   min (default) — start small; at each growth event GENERATE NEW prompts on
#     the fly. The tester never pre-warms these (its own comment: "NOT
#     initialized - cache misses!"), so the first touch of each grown-in prompt
#     is cold — the source of the cold-transition periods that spike per-period
#     median TTFT. Recover the warm value with plot_sweep.py's min-TTFT line.
#   max — generate AND pre-warm ALL prompts up front; growth merely activates
#     more of the already-warm pool. No period ever sees a cold prompt, so the
#     per-working-set TTFT reflects steady-state tier residency. Caveat: max
#     pre-warms the FULL max-working-set (~122 GB at 1.0M tok) before measuring,
#     right up against the 128 GB CXL ceiling. Set INIT_STRATEGY=max for a
#     transition-free sweep (recommended for the headline figure).
INIT_STRATEGY="${INIT_STRATEGY:-min}"

# Derive the tester's linear min/max/increments from WS_POINTS (first, last,
# count-1) so overriding WS_POINTS still controls the band.
read -r -a _pts <<< "$WS_POINTS"
WS_MIN="${_pts[0]}"
WS_MAX="${_pts[${#_pts[@]}-1]}"
WS_STEPS=$(( ${#_pts[@]} - 1 ))
(( WS_STEPS < 1 )) && WS_STEPS=1

# ---------------------------------------------------------------------------
# run_tester <endpoint> <fixed_concurrency> <tester-log-path>
#   Drive one working_set_tester fixed-mode invocation against <endpoint>,
#   tee'ing to both the terminal and the log. Shared by A and B; only the
#   endpoint (node vs router) and concurrency differ.
# ---------------------------------------------------------------------------
run_tester() {
    local endpoint="$1" concurrency="$2" tlog="$3"
    ( cd "$KVCT_DIR" && $PY working_set_tester.py \
        --api-endpoint "http://$endpoint" \
        --context-sizes "$CONTEXT_A" \
        --min-working-set-size "$WS_MIN" \
        --max-working-set-size "$WS_MAX" \
        --working-set-increments "$WS_STEPS" \
        --mode fixed \
        --fixed-concurrency "$concurrency" \
        --tokenizer "$TOKENIZER" \
        --test-duration "$TEST_DURATION" \
        --init-strategy "$INIT_STRATEGY" \
        --output-dir "$RUN_DIR/kvct" ) 2>&1 | tee "$tlog"
}

# ---------------------------------------------------------------------------
# run_trace_tester <endpoint> <tester-log-path>
#   Drive one trace_replay_tester run (Sweep C) against <endpoint>, tee'ing to
#   both terminal and log. Replays real agentic traces through the router with
#   cross-conversation sharing (--warm-prefix-pct) and realistic timing
#   (--time-scale). All seeds are pinned so the cxl and nixl arms replay an
#   identical sequence — the only fair way to compare adapters on a trace.
#   --chunk-size MUST match LMCache's chunk_size or block boundaries misalign
#   and cache identity breaks for BOTH adapters.
# ---------------------------------------------------------------------------
run_trace_tester() {
    local endpoint="$1" tlog="$2"
    ( cd "$KVCT_DIR" && $PY trace_replay_tester.py \
        --api-endpoint "http://$endpoint" \
        --trace-directory "$TRACE_DIR" \
        --output-dir "$RUN_DIR/kvct" \
        --tokenizer "$TOKENIZER" \
        --chunk-size "$CHUNK_SIZE" \
        --max-context "$TRACE_MAX_CONTEXT" \
        --start-users "$TRACE_START_USERS" \
        --max-users "$TRACE_MAX_USERS" \
        --warm-prefix-pct "$TRACE_WARM_PREFIX_PCT" \
        --cache-max-age "$TRACE_CACHE_MAX_AGE" \
        --time-scale "$TRACE_TIME_SCALE" \
        --test-duration "$TEST_DURATION" \
        --max-ttft "$TRACE_MAX_TTFT" \
        --seed "$TRACE_SEED" \
        --trace-seed "$TRACE_SEED" \
        --prompt-seed "$TRACE_SEED" ) 2>&1 | tee "$tlog"
}

# ---------------------------------------------------------------------------
# require_router <expected-strategy>
#   For router sweeps (B, C): confirm the router is reachable and report its
#   LIVE strategy (from /health). If the CLI arg disagrees with the live
#   strategy, relabel the run dir + CSV to match reality (the script cannot set
#   the router's strategy — launch_router.sh does — so the router is the source
#   of truth). Sets the global STRATEGY. Exits non-zero if unreachable, health
#   unparseable, or the live strategy != <expected-strategy>.
# ---------------------------------------------------------------------------
require_router() {
    local want="$1"
    local health
    health="$(curl -fsS "http://$ENDPOINT/health" 2>/dev/null || true)"
    if [[ -z "$health" ]]; then
        echo "ERROR: router not reachable at http://$ENDPOINT/health — start it" \
             "with: ./launch_router.sh $want" >&2
        exit 4
    fi
    local live
    live="$(sed -n 's/.*"strategy":"\([a-z_]*\)".*/\1/p' <<< "${health// /}")"
    if [[ -z "$live" ]]; then
        echo "ERROR: could not parse strategy from router /health: $health" >&2
        exit 4
    fi
    if [[ "$live" != "$want" ]]; then
        echo "ERROR: Sweep $SWEEP needs the router on $want, but it is on" \
             "'$live'. Restart: ./launch_router.sh $want" >&2
        exit 4
    fi
    if [[ "$STRATEGY" != "$live" ]]; then
        echo "NOTE: relabeling run '$STRATEGY' -> '$live' (router's live strategy)."
        local new_dir="$OUT_ROOT/${SWEEP}_${ADAPTER}_${live}_${STAMP}"
        mv "$RUN_DIR" "$new_dir"
        RUN_DIR="$new_dir"
        CSV="$RUN_DIR/tiers.csv"
        STRATEGY="$live"
    fi
}

# ---------------------------------------------------------------------------
# sync_node1_log
#   For router sweeps: rsync node1's MP log from c2 into logs/node1-<adapter>.log
#   so tally_and_record (which reads MPLOG1) counts BOTH nodes' hits. Warns but
#   does not fail on rsync error (node1 hits then simply go uncounted).
# ---------------------------------------------------------------------------
sync_node1_log() {
    echo ">>> syncing node1 MP log from $NODE1_SSH:$NODE1_LOG_REMOTE"
    if rsync -q -e "ssh -o BatchMode=yes -o ConnectTimeout=5" \
        "$NODE1_SSH:$NODE1_LOG_REMOTE" "$MPLOG1" 2>/dev/null; then
        echo "    synced -> $MPLOG1"
    else
        echo "    WARN: rsync of node1 log failed — node1 hits will NOT be" \
             "counted. Check ssh to $NODE1_SSH and the path $NODE1_LOG_REMOTE." >&2
    fi
}

# ---------------------------------------------------------------------------
# Drive the selected sweep.
# ---------------------------------------------------------------------------
if [[ "$SWEEP" == A ]]; then
    # Sweep A — capacity headline, single-host, no router. working_set_tester in
    # FIXED mode at concurrency 1: purpose-built for the working-set axis, needs
    # no --max-ttft, and steady serial load isolates the capacity effect (chunk
    # still in CXL vs recomputed). Hits node0 directly. Snapshot the MP-log
    # offset once before and tally the whole run after; per-point attribution
    # comes from the tester's own kvct/ output. Clear L1 up front (cold DRAM
    # start) but NOT CXL — retaining evicted chunks across the sweep is the
    # effect under test.
    ENDPOINT="${ROUTER_HOST:-$NODE0_HOST:$VLLM_PORT}"
    FIXED_CONCURRENCY="${FIXED_CONCURRENCY:-1}"

    echo ">>> Sweep A: adapter=$ADAPTER ctx=$CONTEXT_A ws=[$WS_MIN..$WS_MAX]/$WS_STEPS conc=$FIXED_CONCURRENCY init=$INIT_STRATEGY dur=${TEST_DURATION}s -> $ENDPOINT"
    "$HERE/clear_cache.sh" >/dev/null 2>&1 || true

    off0="$(log_offset "$MPLOG0")"
    off1="$(log_offset "$MPLOG1")"
    TLOG="$RUN_DIR/tester_sweepA.log"
    run_tester "$ENDPOINT" "$FIXED_CONCURRENCY" "$TLOG"
    tally_and_record "${WS_MIN}-${WS_MAX}" "$off0" "$off1"

elif [[ "$SWEEP" == B ]]; then
    # Sweep B — cross-node dedup + fan-out. Drives the tester through the ROUTER
    # (not a single node) under the anti_affinity strategy, which routes each
    # repeated prompt AWAY from the node that already cached it, forcing the
    # OTHER node to serve it. For CXL that second node reads the shared CXL pool
    # directly (one stored copy serves both nodes); for NIXL it must RDMA-fetch
    # from the peer every time (and the chunk was deleted on the peer's eviction,
    # so it may be gone → recompute). That asymmetry is the shared-substrate win.
    #
    # Requires: both nodes up in "$ADAPTER" mode AND the router already running
    # with --strategy anti_affinity (launch_router.sh anti_affinity). Concurrency
    # must be > 1 so both nodes see concurrent load; default 8. node1's MP log is
    # rsync'd from c2 before tallying so tally_and_record counts BOTH nodes.
    ENDPOINT="${ROUTER_HOST:-$NODE0_HOST:$ROUTER_PORT}"
    FIXED_CONCURRENCY="${FIXED_CONCURRENCY:-8}"
    NODE1_SSH="${NODE1_SSH:-manish@c2}"
    NODE1_LOG_REMOTE="${NODE1_LOG_REMOTE:-$HERE/logs/node1-${ADAPTER}.log}"

    # Sweep B pins ONE working set, it does not sweep. The cross-node effect is
    # about the same chunk being read by both nodes, not about capacity — so the
    # working set is a PRECONDITION, not the variable. It must sit in the upper
    # band: above a single node's 64 GB L1 (~0.52M tok) so the node the router
    # bounces a prompt TO has evicted its own copy and must fetch cross-node
    # (CXL shared-load vs NIXL peer-fetch); and under the 128 GB CXL ceiling
    # (~1.05M tok) so the shared pool holds it. Below ~0.52M the bounced-to node
    # still has it in L1 → no cross-node fetch → CXL and NIXL look identical
    # (dead zone). Default 0.8M sits squarely in the band. Override with WS_B=.
    # We pin by collapsing the tester's sweep to a single point (min=max, 1 step).
    WS_B="${WS_B:-800000}"
    WS_MIN="$WS_B"
    WS_MAX="$WS_B"
    WS_STEPS=1

    require_router anti_affinity

    echo ">>> Sweep B: adapter=$ADAPTER strategy=$STRATEGY ctx=$CONTEXT_A ws=$WS_B (pinned) conc=$FIXED_CONCURRENCY init=$INIT_STRATEGY dur=${TEST_DURATION}s -> router $ENDPOINT"
    "$HERE/clear_cache.sh" >/dev/null 2>&1 || true

    # node0 offset now; node1 offset from the REMOTE log via ssh (line count).
    off0="$(log_offset "$MPLOG0")"
    off1_remote="$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$NODE1_SSH" \
        "wc -l < '$NODE1_LOG_REMOTE' 2>/dev/null || echo 0" 2>/dev/null || echo 0)"

    TLOG="$RUN_DIR/tester_sweepB.log"
    run_tester "$ENDPOINT" "$FIXED_CONCURRENCY" "$TLOG"
    sync_node1_log
    tally_and_record "${WS_MIN}-${WS_MAX}" "$off0" "$off1_remote"

else
    # Sweep C — realistic trace-replay capstone. Replays real agentic-coding
    # traces (KVCT_DIR/traces, 739 traces) through the ROUTER, mixing all the
    # effects A and B isolated: realistic reuse (hash_ids), cross-conversation
    # sharing (--warm-prefix-pct), eviction aging (--cache-max-age) and real
    # inter-request timing (--time-scale). Answers "does CXL vs NIXL matter under
    # a realistic workload", not "which mechanism" — run it AFTER A and B, which
    # explain the mechanisms. Same router + dual-node-tally plumbing as B.
    #
    # Requires: both nodes up in "$ADAPTER" mode AND the router running (any
    # strategy that induces cross-node sharing; anti_affinity is the default we
    # check for, matching B). All tester seeds are pinned (TRACE_SEED) so the cxl
    # and nixl arms replay an IDENTICAL sequence — the only fair comparison.
    ENDPOINT="${ROUTER_HOST:-$NODE0_HOST:$ROUTER_PORT}"
    NODE1_SSH="${NODE1_SSH:-manish@c2}"
    NODE1_LOG_REMOTE="${NODE1_LOG_REMOTE:-$HERE/logs/node1-${ADAPTER}.log}"

    # Trace-replay knobs. Defaults sized to this testbed (64 GB L1/node ~0.52M
    # tok, 128 GB CXL ~1.05M tok, Llama-8B) and the bundled traces (median ~130k
    # context/request): ~8 concurrent user-sessions of ~130k tok overflow one
    # node's DRAM into the CXL band. CHUNK_SIZE MUST match LMCache's chunk_size
    # (256) or block boundaries misalign and cache identity breaks for BOTH
    # adapters — the single most important knob here.
    TRACE_DIR="${TRACE_DIR:-$KVCT_DIR/traces}"
    CHUNK_SIZE="${CHUNK_SIZE:-256}"
    TRACE_MAX_CONTEXT="${TRACE_MAX_CONTEXT:-120000}"
    TRACE_START_USERS="${TRACE_START_USERS:-2}"
    TRACE_MAX_USERS="${TRACE_MAX_USERS:-8}"
    TRACE_WARM_PREFIX_PCT="${TRACE_WARM_PREFIX_PCT:-0.5}"
    TRACE_CACHE_MAX_AGE="${TRACE_CACHE_MAX_AGE:-600}"
    TRACE_TIME_SCALE="${TRACE_TIME_SCALE:-0.1}"
    TRACE_MAX_TTFT="${TRACE_MAX_TTFT:-10.0}"
    TRACE_SEED="${TRACE_SEED:-42}"

    if [[ ! -d "$TRACE_DIR" ]]; then
        echo "ERROR: trace directory not found: $TRACE_DIR (set TRACE_DIR=)" >&2
        exit 1
    fi

    require_router anti_affinity

    echo ">>> Sweep C: adapter=$ADAPTER strategy=$STRATEGY traces=$TRACE_DIR users=$TRACE_START_USERS..$TRACE_MAX_USERS warm_prefix=$TRACE_WARM_PREFIX_PCT time_scale=$TRACE_TIME_SCALE dur=${TEST_DURATION}s seed=$TRACE_SEED -> router $ENDPOINT"
    "$HERE/clear_cache.sh" >/dev/null 2>&1 || true

    off0="$(log_offset "$MPLOG0")"
    off1_remote="$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$NODE1_SSH" \
        "wc -l < '$NODE1_LOG_REMOTE' 2>/dev/null || echo 0" 2>/dev/null || echo 0)"

    TLOG="$RUN_DIR/tester_sweepC.log"
    run_trace_tester "$ENDPOINT" "$TLOG"
    sync_node1_log
    tally_and_record "trace" "$off0" "$off1_remote"
fi

echo
echo "=== done: $RUN_DIR ==="
echo "CSV        : $CSV   (whole-run tier tally)"
if [[ "$SWEEP" == C ]]; then
    echo "tester data: $RUN_DIR/kvct  (trace_replay_tester per-period TTFT/throughput)"
else
    echo "per-ws data: $RUN_DIR/kvct  (working_set_tester per-working-set TTFT/throughput)"
fi
echo "tester log : $TLOG"
column -t -s, "$CSV"
